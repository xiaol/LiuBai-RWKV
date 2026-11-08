#!/bin/bash
# Whole-song test at every s1 export (user 2026-11-08): wait for out/liubai_song/s1/M_ema_s<STEP>.pt (written right after the
# step-STEP dcp save), stop s1, generate the 16 held-out songs of data/song_test16.json (4 GPUs x 4 songs, timed lyrics,
# CFG 3, 24 steps), score them (Qwen3-ASR per-line transcripts + letter CER), update the listening page, resume s1 from the step-STEP dcp.
# The first pause also runs a tiny-ckpt smoke test and the recon ceiling (true latents decoded, same scorer).
# s1 is resumed whatever happens to the test. Log: logs/song_eval_chain.log. Results: out/song_eval/s1_<k>k/score.json.
cd /root/x/rwkv-music
S1=out/liubai_song/s1; EV=out/song_eval; PY=venvs/yue2/bin/python; T=data/song_test16.json; mkdir -p $EV logs
s1_pids() { ps -eo pid,args | grep "[t]rain_liubai_song.py --out $S1 " | awk '{print $1}'; }
gen4() {  # gen4 OUT LOGNAME args... : one part per GPU, returns non-zero unless all 4 parts wrote their json
  local o=$1 ln=$2; shift 2
  for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r $PY tools/gen_song.py --out $o --test $T --part $r --parts 4 "$@" > logs/$ln.part$r.log 2>&1 & done
  wait; for r in 0 1 2 3; do [ -f $o/gen.part$r.json ] || { echo "$(date) $ln part $r FAILED: $(tail -2 logs/$ln.part$r.log)"; return 1; }; done
}
score() {  # score GPU DIR LOG : Qwen3-ASR transcripts (qwen-asr venv), then CER + loudness (venvs/yue2)
  CUDA_VISIBLE_DEVICES=$1 /root/envs/qwen-asr-venv/bin/python tools/transcribe_song_qwen.py $2 > $3 2>&1 && $PY tools/score_song.py $2 >> $3 2>&1
}
resume_s1() {
  local step=$1
  for r in 0 1 2 3; do mv logs/liubai_song_s1.rank$r.log logs/liubai_song_s1.rank$r.to$step.log 2>/dev/null; done
  echo "$(date) resuming s1 from $step"
  (GPUS=0,1,2,3 nohup bash scripts/run_liubai_song.sh $S1 --steps 200000 >> logs/resume_song_s1.log 2>&1 &)
}
for STEP in ${STEPS:-$(seq 20000 10000 200000)}; do
  CK=$S1/M_ema_s$STEP.pt
  echo "$(date) waiting for $CK"
  until [ -f $CK ] && $PY -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1])" $CK 2>/dev/null; do sleep 60; done
  sleep 60
  if [ $STEP -lt 200000 ]; then
    echo "$(date) stopping s1 at $STEP"; for p in $(s1_pids); do kill $p 2>/dev/null; done
    for i in $(seq 60); do [ -z "$(s1_pids)" ] && break; sleep 10; done
    [ -n "$(s1_pids)" ] && { echo "$(date) s1 did not stop, kill -9"; for p in $(s1_pids); do kill -9 $p; done; sleep 30; }
  else
    while [ -n "$(s1_pids)" ]; do sleep 60; done
  fi
  sleep 20; G=$EV/s1_$((STEP / 1000))k; t0=$(date +%s); ok=1
  if [ ! -f $EV/smoke_ok ]; then
    rm -rf /tmp/song_smoke; ok=0
    gen4 /tmp/song_smoke song_eval_smoke --ckpt /tmp/msvae_lm/song_tiny3/M_ema_s50.pt --max-sec 12 --steps 2 &&
      score 0 /tmp/song_smoke logs/song_eval_smoke.score.log &&
      grep -q "^== " logs/song_eval_smoke.score.log && ok=1 && touch $EV/smoke_ok
    echo "$(date) smoke test ok=$ok"
  fi
  if [ $ok = 1 ]; then
    rm -rf $G; gen4 $G song_eval_s1_$STEP --ckpt $CK || ok=0
    echo "$(date) generation $STEP ok=$ok ($(( $(date +%s) - t0 )) s)"
  fi
  if [ $ok = 1 ] && [ ! -f $EV/recon/score.json ]; then
    gen4 $EV/recon song_eval_recon --recon && score 1 $EV/recon logs/song_eval_recon.score.log &
  fi
  [ $ok = 1 ] && { score 0 $G logs/song_eval_s1_$STEP.score.log || ok=0; }
  wait
  echo "$(date) score $STEP ok=$ok ($(( $(date +%s) - t0 )) s): $(grep '^== ' logs/song_eval_s1_$STEP.score.log 2>/dev/null)"
  [ $STEP -lt 200000 ] && resume_s1 $STEP
  if [ $ok = 1 ]; then
    cols=""; [ -f $EV/recon/score.json ] && cols="recon=$EV/recon"
    for d in $(ls -d $EV/s1_*k | sort -t_ -k2 -n); do [ -f $d/score.json ] && cols="$cols $(basename $d)=$d"; done
    python3 tools/song_page.py $EV/player.html $cols
  fi
done
echo "$(date) chain done"
