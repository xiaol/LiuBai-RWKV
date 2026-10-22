#!/usr/bin/env python3
"""Convert Mothersuperior safetensors releases (v5/v8/v9) to the .pt layouts our tools load.

  head : tokenizer_head_*.safetensors (8-layer v4-architecture state_dict) -> {"model": state_dict}   (yue2_roundtrip.py Tok, HEAD_PT)
  lora : nar_lora_joint_*.safetensors (layers.{i}.nar_self_attn.{q,k,v,o}_proj.lora_{A,B}, nar_mlp.{gate,up,down}_proj.lora_{A,B},
         vae2llm.*, llm2vae.*) -> {"lora": [A, B, ...] in layer/q,k,v,o,gate,up,down order, "io": {"vae2llm": sd, "llm2vae": sd}, "rank": r}
usage: ms_convert.py head IN.safetensors OUT.pt | ms_convert.py lora IN.safetensors OUT.pt
"""
import sys
import torch
from safetensors.torch import load_file

kind, src, dst = sys.argv[1:4]
sd = load_file(src)
if kind == "head":
    torch.save({"model": sd}, dst)
else:
    n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers."))
    lora = []
    for i in range(n_layers):
        for mod, names in (("nar_self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), ("nar_mlp", ("gate_proj", "up_proj", "down_proj"))):
            for n in names:
                lora += [sd[f"layers.{i}.{mod}.{n}.lora_A"].float(), sd[f"layers.{i}.{mod}.{n}.lora_B"].float()]
    io = {m: {"weight": sd[f"{m}.weight"].float(), "bias": sd[f"{m}.bias"].float()} for m in ("vae2llm", "llm2vae")}
    torch.save({"lora": lora, "io": io, "rank": lora[0].shape[0]}, dst)
print(kind, src, "->", dst)
