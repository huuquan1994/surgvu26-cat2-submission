# Adapted for inference-only use in the SurgVU26 submission.
"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ..core import register

__all__ = ["PostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
    ) -> None:
        super().__init__()
        # Checkpoint config flag: selects sigmoid score decoding, not a loss.
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        (logits, boxes) = (outputs["pred_logits"], outputs["pred_boxes"])
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            (scores, index) = torch.topk(
                scores.flatten(1), self.num_top_queries, dim=-1
            )
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )
        else:
            scores = F.softmax(logits)[:, :, :-1]
            (scores, labels) = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                (scores, index) = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )
        return (labels, boxes, scores)

    def deploy(self):
        self.eval()
        return self
