# Third-party inference code

- `ecdetseg/engine/` is an inference-only subset of EdgeCrafter ECDet
  (github.com/capsule2077/edgecrafter), licensed under Apache-2.0 as indicated by its source
  headers. The retained code defines and loads the detector used for tool verification.
- `task2_runtime/wbf.py` is adapted from `ensemble-boxes` 1.0.9 by ZFTurbo.
  Its MIT license is included in `ensemble-boxes-LICENSE`.

Dataset loaders, augmentation, training losses, optimizers, and solvers are not included.
