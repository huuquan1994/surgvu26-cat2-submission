"""Detector verification for the structured tool state and VLM hints.

For the four VETO_CLASSES—where the classifier
fires on installed-but-parked tools — drop a class from `present` iff the detector NEVER proposes it
(fused conf >= T in any of NFR uniform frames). Recall-safe by construction: non-veto classes are never
touched, and a class the detector proposes even weakly is kept.

Detector = ECDet-L 960 x N folds (models/ec960_veto/fold_k.pth), each frame
mapped to the Cat-1 test canvas (pillarbox crop 892:712:194:4 -> 640x512, bottom 40 px blurred) and the
folds fused with WBF (avg, IoU .55, keep >= .05) — the Cat-1 shipping recipe. Folds are loaded ONE AT A
TIME (peak VRAM ~1.2 GiB) and this runs BEFORE LLaVA is loaded, so the T4 budget is unchanged.

Every failure mode returns None from `clip_presence` (caller keeps the unvetoed state): wrong geometry,
missing weights, CUDA error. The container must never sink a case on the veto.
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vendor" / "ecdetseg"))
from wbf import weighted_boxes_fusion  # noqa: E402

WEIGHTS_DIR = HERE / "../models/ec960_veto"
VETO_CLASSES = frozenset({"cadiere forceps", "force bipolar", "grasping retractor", "prograsp forceps"})
T, K, NFR = 0.05, 1, 30
WBF_IOU, RAW_THRESHOLD = 0.55, 0.05
PILLAR_X, PILLAR_Y, PILLAR_W, PILLAR_H = 194, 4, 892, 712
CANVAS_W, CANVAS_H = 640, 512
UI_BLUR_BOTTOM, UI_BLUR_KERNEL = 40, 51
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# detector head order (Cat-1 12-class vocab, scorer space-form) -> structured-router canonical names
DET_CLASSES = [
    "grasping retractor", "cadiere forceps", "bipolar forceps", "force bipolar", "clip applier", "stapler",
    "permanent cautery hook/spatula", "monopolar curved scissors", "vessel sealer",
    "tip-up fenestrated grasper", "needle driver", "prograsp forceps",
]


def transform_image(bgr: np.ndarray) -> np.ndarray:
    """Raw 1280x720 BGR -> 640x512 detector canvas with the UI strip blurred."""
    if bgr.shape[:2] != (720, 1280):
        raise ValueError(f"expected 1280x720, got {bgr.shape[1]}x{bgr.shape[0]}")
    content = bgr[PILLAR_Y:PILLAR_Y + PILLAR_H, PILLAR_X:PILLAR_X + PILLAR_W]
    canvas = cv2.resize(content, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    canvas[-UI_BLUR_BOTTOM:] = cv2.GaussianBlur(canvas[-UI_BLUR_BOTTOM:], (UI_BLUR_KERNEL, UI_BLUR_KERNEL), 0)
    return canvas


def fold_paths():
    return sorted(WEIGHTS_DIR.glob("fold_*.pth"))


def load_fold(path: Path, device: torch.device):
    from engine.core import YAMLConfig
    cfg = YAMLConfig(str(WEIGHTS_DIR / "config.yml"))
    vit = cfg.yaml_cfg["ViTAdapter"]
    vit["ffn_ratio"] = 4 if vit["name"] == "ecvits" else 6
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    # This checkpoint-only embedding is not used to predict boxes.
    state.pop("decoder.denoising_class_embed.weight", None)
    cfg.model.load_state_dict(state)

    class Deploy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()

        def forward(self, x, sizes):
            return self.post(self.model(x), sizes)

    size = cfg.yaml_cfg["eval_spatial_size"]
    return Deploy().to(device).eval().half(), (int(size[0]), int(size[1]))


def predict(model, hw, canvas_bgr: np.ndarray, device, threshold=RAW_THRESHOLD):
    """-> list of (xyxy normalized to [0,1] on the canvas, score, det class id 0..11)."""
    from PIL import Image
    import torchvision.transforms as TT
    # Detector preprocessing: PIL resize, ToTensor, and ImageNet normalization.
    image = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    x = TT.Compose([TT.Resize(hw), TT.ToTensor(), TT.Normalize(MEAN, STD)])(image).unsqueeze(0).to(device).half()
    sizes = torch.tensor([[image.size[0], image.size[1]]], device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        labels, boxes, scores = model(x, sizes)
    out = []
    for b, s, l in zip(boxes[0].float().cpu().numpy(), scores[0].float().cpu().numpy(), labels[0].cpu().numpy()):
        s, l = float(s), int(l)
        if s < threshold or not 1 <= l <= len(DET_CLASSES):
            continue
        x1, y1, x2, y2 = b
        out.append(([x1 / CANVAS_W, y1 / CANVAS_H, x2 / CANVAS_W, y2 / CANVAS_H], s, l - 1))
    return out


def decode_frames(video_path, nfr=NFR):
    from decord import VideoReader, cpu
    vr = VideoReader(str(video_path), ctx=cpu(0))
    idx = np.linspace(0, len(vr) - 1, nfr, dtype=int)   # same indices as utils.read_video_decord
    return vr.get_batch(idx).asnumpy()                    # (nfr, H, W, 3) RGB


def clip_presence(video_path, nfr=NFR, device="cuda", log=print):
    """-> {canonical class: [per-frame max fused conf] * nfr} or None (caller keeps the state as is)."""
    try:
        paths = fold_paths()
        if not paths:
            log(f"[veto] no fold weights under {WEIGHTS_DIR} -- veto skipped"); return None
        t0 = time.time()
        frames = decode_frames(video_path, nfr)
        if frames.shape[1:3] != (720, 1280):
            log(f"[veto] video is {frames.shape[2]}x{frames.shape[1]}, not 1280x720 -- veto skipped"); return None
        canvases = [transform_image(np.ascontiguousarray(f[:, :, ::-1])) for f in frames]
        dev = torch.device(device)
        per_fold = []                                      # per_fold[k][frame] = list of (box, score, cls)
        for p in paths:
            model, hw = load_fold(p, dev)
            per_fold.append([predict(model, hw, c, dev) for c in canvases])
            del model; torch.cuda.empty_cache()
        pf = {c: [0.0] * nfr for c in DET_CLASSES}
        for fi in range(nfr):
            boxes_l, scores_l, labels_l = [], [], []
            for k in range(len(paths)):
                preds = per_fold[k][fi]
                boxes_l.append(np.clip(np.asarray([b for b, _, _ in preds], dtype=float).reshape(-1, 4), 0, 1))
                scores_l.append(np.asarray([s for _, s, _ in preds], dtype=float))
                labels_l.append(np.asarray([c for _, _, c in preds], dtype=int))
            if sum(len(s) for s in scores_l) == 0:
                continue
            _b, fs, fl = weighted_boxes_fusion(boxes_l, scores_l, labels_l, iou_thr=WBF_IOU,
                                               skip_box_thr=RAW_THRESHOLD, conf_type="avg")
            for s, c in zip(fs, fl):
                name = DET_CLASSES[int(c)]
                if float(s) > pf[name][fi]:
                    pf[name][fi] = float(s)
        log(f"[veto] detector: {len(paths)} folds x {nfr} frames in {time.time()-t0:.1f}s; "
            f"max conf {{{', '.join(f'{c}: {max(v):.2f}' for c, v in pf.items() if max(v) >= T)}}}")
        return pf
    except Exception:                                     # never sink a case on the veto
        log("[veto] detector failed -- veto skipped:\n" + traceback.format_exc()); return None


def det_present(pf: dict, t=T, k=K) -> set:
    return {c for c in DET_CLASSES if sum(1 for x in pf.get(c, ()) if x >= t) >= k}


def apply_veto(present: set, pf: dict | None, veto=VETO_CLASSES) -> set:
    """Drop a VETO class from `present` iff the detector never proposed it. None pf -> unchanged."""
    if pf is None:
        return set(present)
    dp = det_present(pf)
    return {c for c in present if (c not in veto) or (c in dp)}


def filter_tool_hints(tool_frames, pf: dict | None, veto=VETO_CLASSES):
    """Apply the same bounded four-class veto to EfficientNet's textual VLM hints."""
    if pf is None:
        return list(tool_frames), set()
    dp = det_present(pf)
    removed = set()
    filtered = []
    for frame in tool_frames:
        kept = []
        for tool in (part.strip() for part in str(frame).split(",")):
            if not tool:
                continue
            if tool in veto and tool not in dp:
                removed.add(tool)
            else:
                kept.append(tool)
        filtered.append(", ".join(kept))
    return filtered, removed
