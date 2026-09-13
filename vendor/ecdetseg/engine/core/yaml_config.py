# Adapted for inference-only use in the SurgVU26 submission.
"""EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import copy
from .workspace import create
from .yaml_utils import load_config, merge_config


class YAMLConfig:
    """Load the model and postprocessor definitions for checkpoint inference."""

    def __init__(self, cfg_path):
        self.yaml_cfg = copy.deepcopy(load_config(cfg_path, cfg={}))
        self._model = None
        self._postprocessor = None

    @property
    def global_cfg(self):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self):
        if self._model is None:
            self._model = create(self.yaml_cfg["model"], self.global_cfg)
        return self._model

    @property
    def postprocessor(self):
        if self._postprocessor is None:
            self._postprocessor = create(
                self.yaml_cfg["postprocessor"], self.global_cfg
            )
        return self._postprocessor
