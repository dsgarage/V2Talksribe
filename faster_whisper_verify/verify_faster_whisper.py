#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, io, json, time
from faster_whisper import WhisperModel

def ts(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60); ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(segs, path):
    with io.open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text'].strip()}\n\n")

def write_vtt(segs, path):
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            a = ts(s['start']).replace(",", "."); b = ts(s['end']).replace(",", ".")
            f.write(f"{a} --> {b}\n{s['text'].strip()}\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--outdir", default=None, help="Output directory (default: same as audio file with folder name)")
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--language", default="ja")              # 日本語前提（自動検出なら "auto"）
    ap.add_argument("--device", default="cpu")               # cpu 推奨（安定）
    ap.add_argument("--compute-type", default="int8")        # 速度優先
    ap.add_argument("--beam-size", type=int, default=3)      # 小さめで速く
    args = ap.parse_args()

    # ファイル名（拡張子なし）を取得
    base = os.path.splitext(os.path.basename(args.audio))[0]

    # 出力ディレクトリを決定：指定されない場合は元ファイルと同じ階層にファイル名のフォルダを作成
    if args.outdir is None:
        audio_dir = os.path.dirname(os.path.abspath(args.audio))
        outdir = os.path.join(audio_dir, base)
    else:
        outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)

    print("=== Faster-Whisper Verify ===")
    print(f"Audio: {args.audio}")
    print(f"Model: {args.model_size}  Device: {args.device}  compute_type: {args.compute_type}  beam: {args.beam_size}")

    t0 = time.time()
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)

    # 言語設定（"auto"/"none" なら自動検出）
    lang = None if str(args.language).lower() in ("auto","none","") else args.language

    t1 = time.time()
    segments, info = model.transcribe(
        args.audio,
        language=lang,
        beam_size=args.beam_size,
        vad_filter=False,
        word_timestamps=False
    )
    seg_list = [{"start": float(s.start or 0.0), "end": float(s.end or 0.0), "text": (s.text or "").strip()} for s in segments]
    t2 = time.time()

    json_path = os.path.join(outdir, f"{base}.json")
    srt_path  = os.path.join(outdir, f"{base}.srt")
    vtt_path  = os.path.join(outdir, f"{base}.vtt")
    with io.open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "audio": os.path.abspath(args.audio),
            "model": args.model_size,
            "device": args.device,
            "compute_type": args.compute_type,
            "language": args.language,
            "segments": seg_list
        }, ensure_ascii=False, indent=2))
    write_srt(seg_list, srt_path)
    write_vtt(seg_list, vtt_path)

    dur = getattr(info, "duration", None)
    print("=== Done ===")
    print(f"Segments: {len(seg_list)}")
    if dur: print(f"Audio duration: {dur/60:.2f} min")
    print(f"Load model: {(t1-t0):.1f}s, Transcribe: {(t2-t1)/60:.2f} min")
    if dur: print(f"RTF: {((t2-t1)/dur):.2f} (lower is faster)")

    print(f"Outputs:\n  JSON: {json_path}\n  SRT : {srt_path}\n  VTT : {vtt_path}")

if __name__ == "__main__":
    main()
