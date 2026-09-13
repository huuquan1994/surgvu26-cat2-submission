# Adapted for inference-only use in the SurgVU26 submission.
"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RF-DETR (https://github.com/roboflow/rf-detr)
Copyright (c) 2025 Roboflow. All Rights Reserved.
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
---------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
from torch import Tensor


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    (x0, y0, x1, y1) = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return torch.stack(b, dim=-1)
