#!/usr/bin/env python3
"""Unit test for the lyric-cursor CTC loss (src/model.py lyric_ctc_batch / aux_cursor_loss) on real line-format val docs.
1. extracts the lines of one 8192-token window and prints the decoded words of the first few next to their frame counts;
2. forward + backward of a 2-layer model with two aux taps; the aux loss must be finite and give gradients to the
   blocks and the heads; an untrained head should sit near the uniform-CTC value.
usage (rwkv_py312, one GPU): python scripts/test_aux_cursor.py
"""
import os, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]; TT = ROOT / "tools/RWKV-LM/RWKV-v7/train_temp"
for k, v in dict(RWKV_MY_TESTING="x070", RWKV_KERNEL="", RWKV_HEAD_SIZE="64", RWKV_HEAD_L2WRAP_CE_CHUNK="0", RWKV_FLOAT_MODE="bf16",
                 RWKV_JIT_ON="1", RWKV_CTXLEN="8192", RWKV_AUX_W="0.1", RWKV_AUX_LAYERS="0,1",
                 RWKV_AUX_VOCAB=str(ROOT / "data/binidx/yue2_line_aux_vocab.npy")).items():
    os.environ.setdefault(k, v)
sys.path.insert(0, str(TT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tools/RWKV-LM/RWKV-v5/tokenizer"))
cwd = os.getcwd(); os.chdir(TT)
import numpy as np, torch  # noqa: E402
from src.model import RWKV, lyric_ctc_batch, AUX_VOCAB  # noqa: E402
os.chdir(cwd)
from src.binidx import MMapIndexedDataset  # noqa: E402
from build_stage1_manifest import VOCAB_TXT  # noqa: E402
from rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402

tok = TRIE_TOKENIZER(str(VOCAB_TXT)); ds = MMapIndexedDataset(str(ROOT / "data/binidx/yue2_line_stage1_val"))
docs = np.concatenate([np.asarray(ds[i]) for i in range(3)]).astype(np.int64)
idx = docs[None, 1000:1000 + 8192]                                    # starts mid-document: the first partial line must be skipped
frames, cols, tgts, flen, tlen = lyric_ctc_batch(idx)
print(f"{len(flen)} complete lines in the window; frames p50 {int(np.median(flen))}, targets p50 {int(np.median(tlen))}")
for j in range(4):
    classes = [cols[j][t] for t in tgts[j]]
    words = tok.decode([int(AUX_VOCAB[c - 2]) for c in classes if c >= 2])
    print(f"  line {j}: {flen[j]} frames, {tlen[j]} targets, unk {sum(c == 1 for c in classes)}: {words!r}")
assert len(flen) > 5 and all(f >= 2 * t for f, t in zip(flen, tlen))

args = SimpleNamespace(n_layer=2, n_embd=512, dim_att=512, dim_ffn=2048, head_size=64, vocab_size=98816, ctx_len=8192, my_testing="x070", grad_cp=0)
m = RWKV(args).cuda().bfloat16()
x = torch.tensor(idx, device="cuda")
m._forward_features(x); aux = m.aux_cursor_loss(x); aux.backward()
V = len(AUX_VOCAB) + 2
print(f"aux CTC loss {aux.item():.3f} (untrained; per-target uniform reference ~ {np.log(V):.2f} nats x frames/targets), "
      f"head grad {m.aux_heads[0].proj.weight.grad.abs().sum().item():.3e}, block0 att.output grad {m.blocks[0].att.output.weight.grad.abs().sum().item():.3e} (key is 0 at init: output proj is zero-init)")
assert torch.isfinite(aux) and m.blocks[0].att.output.weight.grad.abs().sum() > 0
print("PASS")
