# SurgVU26 Category 2 Submission — Team Capybara

**Member:** Quan Huu Cap  
**Affiliation:** [Aillis, Inc.](https://aillis.jp/), Tokyo, Japan

Inference container for our Category 2 surgical visual question-answering entry in the
SurgVU Challenge 2026. The final submission achieved **0.687276 BERTScore-F1** and ranked
first on the final leaderboard.

The method extends our winning 2025 pipeline with detector-verified tool evidence and
question-aware answer selection. It combines LLaVA-OneVision-7B, EfficientNetV2-S tool
and organ classifiers, and five ECDet-L models.

## Solution report

The submitted method description is available in the
[Category 2 solution report](SurgVU26-Cat2-Capybara-Report.pdf).

## Requirements

- Linux
- Docker with NVIDIA GPU support
- An NVIDIA GPU and driver compatible with PyTorch 2.8.0 and CUDA 12.8
- The public LLaVA-OneVision checkpoint and our trained classifier/detector checkpoints

## Download the models

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
then download LLaVA-OneVision into `models/`:

```bash
hf download llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --local-dir ./models/llava-onevision-qwen2-7b-ov-hf
```

Our ECDet-L detector checkpoints are hosted in the **private** Hugging Face repository
[quancap/surgvu26-cat2-models](https://huggingface.co/quancap/surgvu26-cat2-models).

**Access must be granted upon request.** After access is granted, authenticate and download:

```bash
hf auth login
hf download quancap/surgvu26-cat2-models \
  --include "ec960_veto/*" \
  --include "tools_classifier/*" \
  --include "organs_classifier/*" \
  --local-dir ./models
```

The tool and organ EfficientNetV2-S checkpoints are publicly available through the
[SurgVU25 Category 2 model download instructions](https://github.com/huuquan1994/surgvu25-cat2-submission/blob/main/Models/README.md).
Place those `tools_classifier/` and `organs_classifier/` directories under `models/` if using
that alternative. The ECDet-L checkpoints still require access to our private Hugging Face repo.

The resulting layout must be:

```text
models/
├── llava-onevision-qwen2-7b-ov-hf/
├── tools_classifier/
│   └── efficientnet_v2_s_smoothing_0.025/
│       └── epoch=11_test_loss=0.12283_f1_avg=0.97795.ckpt
├── organs_classifier/
│   └── efficientnet_v2_s_v3/
│       └── epoch=29_test_loss=0.04914_f1_avg=0.98064.ckpt
└── ec960_veto/
    ├── config.yml
    ├── fold_0.pth
    ├── fold_1.pth
    ├── fold_2.pth
    ├── fold_3.pth
    └── fold_4.pth
```

## Build the Docker image

From the repository root:

```bash
docker build --platform=linux/amd64 -t surgvu26-cat2-submission .
```

The final inference settings are fixed in the source code.

## Run the container

Prepare the Grand Challenge input directory:

```text
input/
├── endoscopic-robotic-surgery-video.mp4
├── visual-context-question.json
└── inputs.json
```

`visual-context-question.json` contains a bare JSON string. `inputs.json` contains the two
Grand Challenge interface slugs, `endoscopic-robotic-surgery-video` and
`visual-context-question`.

Then run:

```bash
mkdir -p output
chmod 777 output

docker run --rm --gpus all \
  --network none \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  surgvu26-cat2-submission
```

The answer is written as a bare JSON string to:

```text
output/visual-context-response.json
```

## Final model configuration

- LLaVA-OneVision-7B in 4-bit NF4 with FP16 compute, using 5 uniformly sampled frames
- EfficientNetV2-S tool classifier: 5-frame text hints and a separate 30-frame state pass
- EfficientNetV2-S organ classifier: majority vote over 30 frames
- ECDet-L: 5 folds at 960-pixel input resolution, fused with Weighted Boxes Fusion
- Detector verification: confidence 0.05, applied to four classifier-prone tool classes
- Question-aware routing: structured tool/organ state for grounded questions and VLM fallback
- Commercial tool names, minimal yes/no mirrors, and short singular tool-entity answers

The VLM path crops the side pillarboxes and bottom interface strip. Classifier frames retain the
image geometry and blur the interface text. Detector frames use the Category 1 640×512 geometry
with the bottom 40 pixels blurred by a 51×51 Gaussian kernel.
