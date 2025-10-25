#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, subprocess, sys
import soundfile as sf, numpy as np, librosa

def has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def ffmpeg_convert(src, dst):
    cmd = ["ffmpeg","-y","-i",src,"-ar","16000","-ac","1","-acodec","pcm_s16le",dst]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False

def py_convert(src, dst):
    y, sr = librosa.load(src, sr=None, mono=True)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
    sf.write(dst, y.astype(np.float32), 16000)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if has_ffmpeg():
        if ffmpeg_convert(args.inp, args.out):
            print(f"Preprocessed (ffmpeg) → {args.out}")
        else:
            print("ffmpeg failed, falling back to Python...")
            py_convert(args.inp, args.out)
            print(f"Preprocessed (Python) → {args.out}")
    else:
        py_convert(args.inp, args.out)
        print(f"Preprocessed (Python) → {args.out}")

if __name__ == "__main__":
    main()
