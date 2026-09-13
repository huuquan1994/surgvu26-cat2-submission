"""Grand Challenge entrypoint for the final Category 2 submission."""
from pathlib import Path
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import LlavaOnevisionForConditionalGeneration

import json
from surgvu_models import OrganClassifier, SurgToolClassifier
import utils_llava_ov as utils
from structured_router import route_structured, sanitize_final_answer

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

STRUCT_NUM_FRAMES = 30
STRUCT_VOTE_K = 8

LLAVA_MODEL_ID = "../models/llava-onevision-qwen2-7b-ov-hf"
ORGANS_MODEL_PATH = "../models/organs_classifier/efficientnet_v2_s_v3/epoch=29_test_loss=0.04914_f1_avg=0.98064.ckpt"
TOOLS_MODEL_PATH = "../models/tools_classifier/efficientnet_v2_s_smoothing_0.025/epoch=11_test_loss=0.12283_f1_avg=0.97795.ckpt"

MAX_NEW_TOKENS = 2048
NUM_FRAMES = 5

# Target tools
target_tools = [
    "needle driver",
    "monopolar curved scissors",
    "force bipolar",
    "clip applier",
    "cadiere forceps",
    "bipolar forceps",
    "vessel sealer",
    "permanent cautery hook/spatula",
    "prograsp forceps",
    "stapler",
    "grasping retractor",
    "tip-up fenestrated grasper",
]

# Target organs
target_organs = [
  'abdominal wall', 'bladder', 'gallbladder', 'pelvic lymph nodes',
  'rectum', 'sigmoid colon', 'unspecified', 'uterine horn',
]

target_tools = sorted(target_tools)
target_organs = sorted(target_organs)

def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()
    print("Inputs: ", interface_key)
    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "endoscopic-robotic-surgery-video",
            "visual-context-question",
        ): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input
    video_path = INPUT_PATH / "endoscopic-robotic-surgery-video.mp4"

    input_question = load_json_file(
        location=INPUT_PATH / "visual-context-question.json",
    )

    # Run the detector first so its memory never overlaps with LLaVA.
    import detector_veto
    veto_pf = detector_veto.clip_presence(video_path, nfr=detector_veto.NFR)

    # define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the LLaVA model
    llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="cuda")

    llava_processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)

    # Load tools and organs classifiers
    # tools classifier
    tools_model = SurgToolClassifier()
    tools_model.load_from_ckpt(TOOLS_MODEL_PATH)

    tools_model = tools_model.to("cuda")
    tools_model = tools_model.eval()

    # organs classifier
    organs_model = OrganClassifier()
    organs_model.load_from_ckpt(ORGANS_MODEL_PATH)

    organs_model = organs_model.to("cuda")
    organs_model = organs_model.eval()

    # predict tools and organs
    organ_name = utils.predict_organs(video_path, organs_model, target_organs)
    tools_all_frames = utils.predict_tools(
        video_path, tools_model, target_tools, num_frames=NUM_FRAMES)
    if veto_pf is not None:
        tools_all_frames, hint_removed = detector_veto.filter_tool_hints(
            tools_all_frames, veto_pf)
        print(f"[veto] filtered VLM tool hints; removed: {sorted(hint_removed)}")

    tools_description = f"The organ being manipulated is the {organ_name}."
    for idx in range(NUM_FRAMES):
        print(f"Frame {idx}: Detected tools: {tools_all_frames[idx]}")
        tools_description += f"Surgical tools in frame {idx}: {tools_all_frames[idx]}\n"

    # Structured state: 30-frame tool vote, per-class scores, five-frame fallback, and organ.
    present_5f = set()
    for frame in tools_all_frames:
        for name in (t.strip() for t in str(frame).split(",")):
            if name:
                present_5f.add(name)
    struct_state = {"present": set(), "present_5f": present_5f, "scores": {}, "organ": organ_name}
    tool_score_mat = utils.predict_tool_scores(
        video_path, tools_model, target_tools, num_frames=STRUCT_NUM_FRAMES)
    _tool_mean = tool_score_mat.mean(axis=0)
    struct_state["present"] = {
        target_tools[j] for j in range(len(target_tools))
        if int((tool_score_mat[:, j] > 0.5).sum()) >= STRUCT_VOTE_K}
    struct_state["scores"] = {target_tools[j]: float(_tool_mean[j]) for j in range(len(target_tools))}
    print("Structured state present tools:", sorted(struct_state["present"]))
    before = set(struct_state["present"])
    struct_state["present"] = detector_veto.apply_veto(before, veto_pf)
    print(f"[veto] {'applied' if veto_pf is not None else 'SKIPPED (no detector output)'}; "
          f"removed: {sorted(before - struct_state['present'])}; "
          f"present after veto: {sorted(struct_state['present'])}")
    print("Structured state present_5f:", sorted(struct_state["present_5f"]))

    # Load video
    video_clip = utils.read_video_decord(
        str(video_path), num_frames=NUM_FRAMES, crop_side=True, crop_bottom=True)

    # Inference
    vlm_description = utils.get_video_description(
        model=llava_model, processor=llava_processor,
        video_clip=video_clip, tools_present=tools_description,
        max_new_tokens=MAX_NEW_TOKENS)

    video_description = tools_description + vlm_description

    print("Video description: ", video_description)

    gen_answer = utils.get_answer(
        model=llava_model, processor=llava_processor,
        video_description=video_description, video_clip=video_clip,
        question_data=input_question)

    # Route answer types using structured state; fall back to the raw VLM answer on failure.
    try:
        final_answer, qtype = route_structured(
            input_question, struct_state, gen_answer, description=vlm_description)
        if final_answer is None:
            final_answer = gen_answer
    except Exception as exc:                           # never let post-processing sink a case
        print("Router failed, using raw answer:", repr(exc))
        final_answer, qtype = gen_answer, "raw"

    # Never emit an empty or sentinel answer, including after a router exception.
    final_answer, sanitize_source = sanitize_final_answer(final_answer, vlm_description)
    if sanitize_source:
        qtype = sanitize_source

    print("Question: ", json.dumps(input_question, indent=4))
    print("Raw answer: ", gen_answer)
    print(f"[structured] source={qtype}")
    print(f"Routed [{qtype}]: ", final_answer)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "visual-context-response.json",
        content=final_answer,
    )

    return 0

def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    print('These are the inputs:' , inputs)
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

if __name__ == "__main__":
    raise SystemExit(run())
