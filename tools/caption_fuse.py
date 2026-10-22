#!/usr/bin/env python3
"""Caption stage C: fuse lyrics + measured facts (caption_facts.py) + Omni window captions (caption_omni.py) into the three
caption forms the RWKV song LM trains on, with Qwen3-30B-A3B-Instruct-2507 under vLLM.

  tags        Suno-style comma list (genre, subgenre, mood, vocal type, key instruments, tempo word, era/production)
  description 2-4 sentences in Suno v5/v6 prompt style: arrangement, vocal timbre and delivery, production, energy arc
  sections    one directive per section (aligned to the SongFormer sections that hold lyric lines), e.g.
              {"label": "chorus", "start": 50.2, "directive": "full band, belted male tenor, gang vocals, wide drums"}
Rules given to the LLM: describe only what the audio evidence supports; facts (bpm, key, sections) beat the Omni text when
they disagree; never name artists, songs, shows or albums (style learning, not imitation); Omni's lyric quotes are ignored
(the real lyrics come from the LRC). Output JSON is validated; invalid rows are retried once at temperature 0.
usage (vllm13 venv, 1 GPU fp8 or 2 GPUs bf16):
  CUDA_VISIBLE_DEVICES=0,1 /root/envs/vllm13-venv/bin/python tools/caption_fuse.py --manifest M --facts F --omni O --out C [--tp 2]
"""
import argparse, json, os, re

MODEL = "/models/caption/Qwen3-30B-A3B-Instruct-2507"
SYSTEM = """You write training captions for a text-to-song model. You get one song's lyrics, measured facts and several
time-windowed audio descriptions written by an audio captioner. Return ONLY a JSON object:
{"language": "...", "tags": "...", "description": "...", "sections": [{"i": <section index>, "directive": "..."}]}
- tags: 8-16 lowercase comma-separated tags in the style of Suno style prompts: genre, subgenre, mood, vocal gender/type,
  2-4 signature instruments, tempo feel, production/era. Include "live" only if a crowd/venue is audible.
- description: 2-4 sentences, concrete and audible: arrangement, vocal timbre and delivery, production, how energy develops.
- sections: one entry per listed section index, directive <= 20 words: what changes in instrumentation, vocal delivery
  and dynamics, in plain musical words (sparse, builds, full band, drops out, softer, louder). The loudness hints are for
  you only: never write numbers, "dB" or "rel_db" in a directive.
- Count the singers: if the descriptions mention both a male and a female voice in different windows, it is a duet;
  say which voice leads each section.
- Measured bpm/key/sections are ground truth; if the audio descriptions disagree, trust the measurements.
- The audio captioner sometimes hallucinates lyrics, accents, sound effects or genre jumps; keep only claims that recur
  across windows or match the facts. Never quote lyrics. Never name artists, songs, albums, shows or labels."""


def build_user(song, facts, omni):
    secs = []
    for i, s in enumerate(facts["sections"]):
        first = s["lines"][0][:40] if s["lines"] else ""
        lvl = "much quieter" if s["rel_db"] < -6 else "quieter" if s["rel_db"] < -2 else "louder" if s["rel_db"] > 2 else "average"
        secs.append(f'{i}: {s["label"]} {s["start"]:.0f}-{s["end"]:.0f}s, loudness {lvl}' + (f', first line "{first}"' if first else " (no lyrics)"))
    wins = "\n\n".join(f'[{w["t0"]:.0f}-{w["t1"]:.0f}s] {w["caption"][:1800]}' for w in omni["windows"])
    return (f"Duration {song['duration']:.0f}s. Measured: tempo {facts['bpm']} bpm (beat stability {facts['beat_stability']}), "
            f"key {facts['key']} (confidence margin {facts['key_conf']}).\n\nSections:\n" + "\n".join(secs) +
            f"\n\nLyrics:\n{song['lyrics'][:2500]}\n\nAudio descriptions by window:\n{wins}")


def parse(text, n_sec):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try: o = json.loads(m.group(0))
    except json.JSONDecodeError: return None
    if not isinstance(o.get("tags"), str) or not isinstance(o.get("description"), str) or not isinstance(o.get("sections"), list): return None
    dirs = {d.get("i"): d.get("directive", "") for d in o["sections"] if isinstance(d, dict)}
    if len([i for i in range(n_sec) if dirs.get(i)]) < max(1, n_sec // 2): return None
    dirs = {k: re.sub(r"[,;]?\s*[-+]?\d+(\.\d+)?\s*(dB|rel_db)\b|rel_db\s*[-+]?\d+(\.\d+)?", "", v, flags=re.I).strip(" ,;") for k, v in dirs.items()}
    o["sections"] = dirs; return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--facts", required=True); ap.add_argument("--omni", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tp", type=int, default=1); ap.add_argument("--quant", default="none")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    load = lambda p: {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
    songs, facts, omni = load(args.manifest), load(args.facts), load(args.omni)
    ids = [i for i in songs if i in facts and i in omni]
    llm = LLM(model=MODEL, tensor_parallel_size=args.tp, quantization=None if args.quant == "none" else args.quant,
              gpu_memory_utilization=0.92, max_model_len=16384, seed=1234)
    tok = llm.get_tokenizer()
    prompt = lambda i: tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_user(songs[i], facts[i], omni[i])}],
                                               add_generation_prompt=True, tokenize=False)
    res = {}
    for temp in (0.7, 0.0):
        todo = [i for i in ids if i not in res]
        if not todo: break
        outs = llm.generate([prompt(i) for i in todo], SamplingParams(temperature=temp, top_p=0.8, max_tokens=1500))
        for i, o in zip(todo, outs):
            p = parse(o.outputs[0].text, len(facts[i]["sections"]))
            if p: res[i] = p
    with open(args.out, "w", encoding="utf-8") as f:
        for i in ids:
            if i not in res: print("FAILED", i); continue
            p = res[i]; secs = facts[i]["sections"]
            f.write(json.dumps(dict(id=i, language=p.get("language", ""), tags=p["tags"], description=p["description"],
                                    bpm=facts[i]["bpm"], key=facts[i]["key"],
                                    sections=[dict(label=s["label"], start=s["start"], end=s["end"], directive=p["sections"].get(k, ""))
                                              for k, s in enumerate(secs)]), ensure_ascii=False) + "\n")
    print(f"done {len(res)}/{len(ids)} -> {args.out}")


if __name__ == "__main__":
    main()
