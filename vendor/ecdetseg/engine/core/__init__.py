# Adapted for inference-only use in the SurgVU26 submission.
"""Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from .workspace import GLOBAL_CONFIG, create, register
from .yaml_config import YAMLConfig

__all__ = ["GLOBAL_CONFIG", "create", "register", "YAMLConfig"]
