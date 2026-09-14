# Vendored tool pins

`tools/RWKV-LM` and `tools/YuE` are plain git clones and are **not** tracked in this repo.
Recreate them with:

```bash
git clone https://github.com/BlinkDL/RWKV-LM.git tools/RWKV-LM
git -C tools/RWKV-LM checkout 9a75f9f037afa4418ee6283b584b92b1adb89ca1
git -C tools/RWKV-LM apply ../../tools/patches/rwkv-lm-9a75f9f-train_temp.patch

git clone https://github.com/multimodal-art-projection/YuE.git tools/YuE
git -C tools/YuE checkout 0edaf2f4053ef4731334b8329834b107977f9637
```

`rwkv-lm-9a75f9f-train_temp.patch` holds the local edits to `RWKV-v7/train_temp`
(CUDA kernels, `src/dataset.py`, `src/model.py`, `src/trainer.py`) used by `scripts/train_stage1.sh`.
