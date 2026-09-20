#!/usr/bin/env python3
"""Teacher-forced diagnostics for the R3 stage-1 model (PLAN §8.6 G3 gate): does it condition on the lyrics/caption?

For N documents of data/binidx/yue2_stage1_val (text EOD SOA YUE2CODEC codes EOA 0) computes per-token CE in the
text region and the code region under three prompts:
  matched   : the document's own text
  shuffled  : the text of another val document (same format, wrong lyrics/caption)
  none      : no text at all (SOA YUE2CODEC codes)
If code-region loss(matched) ≈ loss(shuffled), the model ignores the prompt — generation cannot follow lyrics.
Also reports code loss by position bucket (first 250 frames / middle / last) and the R1-a token repeat rate.
Run in rwkv_py312 (source /root/envs/env_rwkv), one GPU.
usage: rwkv_eval_loss.py --ckpt out/stage1_yue2/rwkv-final.pth [--n 40] [--max-len 8192]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tools/RWKV-LM/RWKV-v7/train_temp"))
from yue2_layout import EOD, SOA, EOA, YUE2CODEC, YUE2_BASE, N_CODES  # noqa: E402
from rwkv_generate import RWKV7, load_kernel, DTYPE  # noqa: E402
from src.binidx import MMapIndexedDataset  # noqa: E402


@torch.no_grad()
def full_logits_loss(model, idx):
    """CE per position for idx[1:] given idx[:-1] (GPT mode, whole sequence, head applied in chunks)."""
    z = model.z; dev = model.dev; state = model.new_state()
    x = z["emb.weight"][torch.tensor(idx[:-1], device=dev)]; v_first = torch.empty_like(x)
    for i in range(model.n_layer):
        b = f"blocks.{i}."
        xx = F.layer_norm(x, (model.n_embd,), weight=z[b + "ln1.weight"], bias=z[b + "ln1.bias"])
        xx, state[i * 3], v_first = model.tmix(i, xx, state[i * 3], v_first, state[i * 3 + 1], True); x = x + xx
        xx = F.layer_norm(x, (model.n_embd,), weight=z[b + "ln2.weight"], bias=z[b + "ln2.bias"])
        dx = torch.cat((state[i * 3 + 2].unsqueeze(0), xx[:-1])) - xx; kx = xx + dx * z[b + "ffn.x_k"]
        x = x + (torch.relu(kx @ z[b + "ffn.key.weight"]) ** 2) @ z[b + "ffn.value.weight"]
    x = F.layer_norm(x, (model.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
    tgt = torch.tensor(idx[1:], device=dev); out = torch.empty(len(tgt), device=dev)
    for s in range(0, len(tgt), 1024):
        lg = (x[s:s + 1024] @ z["head.weight"]).float(); out[s:s + 1024] = F.cross_entropy(lg, tgt[s:s + 1024], reduction="none")
    return out.cpu().numpy()


def split(doc):
    doc = [int(t) for t in doc]; e = doc.index(EOD); text = doc[:e]; body = doc[e:]      # body = EOD SOA YUE2CODEC codes EOA 0
    return text, body


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=str(ROOT / "out/stage1_yue2/rwkv-final.pth"))
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--max-len", type=int, default=8192); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); load_kernel(); model = RWKV7(a.ckpt); ds = MMapIndexedDataset(str(ROOT / "data/binidx/yue2_stage1_val"))
    rng = np.random.default_rng(a.seed); ids = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)
    docs = [split(np.asarray(ds[int(i)])) for i in ids]; res = {k: [] for k in ("matched", "shuffled", "none")}; textloss = []; pos = {"first250": [], "mid": [], "last250": []}; t0 = time.time()
    for j, (text, body) in enumerate(docs):
        other = docs[(j + 1) % len(docs)][0]
        variants = {"matched": text + body, "shuffled": other + body, "none": body[1:]}
        for k, seq in variants.items():
            seq = seq[:a.max_len]; l = full_logits_loss(model, seq)
            nt = len(seq) - len(body) if k != "none" else -1          # number of text tokens in this sequence
            code_start = (nt + 3) if k != "none" else 2                 # index (in targets) of the first code token
            codes = l[code_start - 1:code_start - 1 + (len(body) - 5)]  # predictions of the code tokens (excluding EOA/0)
            res[k].append(float(codes.mean()))
            if k == "matched":
                textloss.append(float(l[:nt - 1].mean())); pos["first250"].append(float(codes[:250].mean())); pos["last250"].append(float(codes[-250:].mean())); pos["mid"].append(float(codes[250:-250].mean()) if len(codes) > 500 else float("nan"))
        if j % 5 == 4: print(f"  {j + 1}/{len(docs)} matched {np.mean(res['matched']):.3f} shuffled {np.mean(res['shuffled']):.3f} none {np.mean(res['none']):.3f} {time.time() - t0:.0f}s", flush=True)
    print(f"\nRESULT n={len(docs)} code-token CE (nats): matched {np.mean(res['matched']):.4f} | shuffled-text {np.mean(res['shuffled']):.4f} | no-text {np.mean(res['none']):.4f}")
    print(f"       text-token CE {np.mean(textloss):.4f} | code CE by position: first250 {np.mean(pos['first250']):.3f} mid {np.nanmean(pos['mid']):.3f} last250 {np.mean(pos['last250']):.3f}")
    d = np.array(res["matched"]) - np.array(res["shuffled"]); print(f"       per-doc (matched - shuffled): mean {d.mean():.4f} sd {d.std():.4f} frac<0 {np.mean(d < 0):.2f}  | uniform over {N_CODES} codes = {np.log(N_CODES):.3f} nats")


if __name__ == "__main__":
    main()
