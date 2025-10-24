#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
talksribe.py
Mac向けCUIの話者分離つき文字起こしツール（ローカル完結）。
- WebRTC VAD で発話区間検出
- SpeechBrain ECAPA 埋め込み + sklearn で話者クラスタリング
- faster-whisper でASR（日本語対応）
- JSON / SRT / VTT 出力、進捗表示、ログ
"""

from __future__ import annotations
import os
import io
import sys
import math
import json
import wave
import queue
import time
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import soundfile as sf
import librosa
import webrtcvad
from tqdm import tqdm
import typer
from rich.console import Console
from rich.table import Table
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ASR
from faster_whisper import WhisperModel

# Speaker embedding（ECAPA）
from speechbrain.pretrained import EncoderClassifier

app = typer.Typer(add_completion=False, help="話者分離つき文字起こし（CUI）")
console = Console()

# -----------------------------
# データモデル
# -----------------------------

class Segment(BaseModel):
    start: float
    end: float

class Utterance(BaseModel):
    start: float
    end: float
    text: str
    speaker: str
    words: Optional[List[Dict]] = None

class TranscriptionResult(BaseModel):
    audio_path: str
    sample_rate: int
    num_channels: int
    duration: float
    language: Optional[str] = None
    model_size: str
    speakers: List[str]
    utterances: List[Utterance]
    segments: List[Dict] = Field(default_factory=list)  # raw ASR segments
    diarization: List[Dict] = Field(default_factory=list)  # diarization intervals

# -----------------------------
# ユーティリティ
# -----------------------------

def read_audio_any(path: str) -> Tuple[np.ndarray, int]:
    """
    48kHz/32-bit float WAVなどを安全にロードし、np.float32のndarrayとsrを返す。
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        # ステレオ→モノラル
        audio = np.mean(audio, axis=1)
    return audio, sr

def ensure_mono_16k_int16(audio: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    VAD/webrtcvad 用に 16kHz/mono/16-bit PCM に変換
    """
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr
    # クリップしてint16へ
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16, sr

def frame_generator(pcm16: np.ndarray, sr: int, frame_ms: int = 30):
    """
    30msフレーム生成（webrtcvadの要件: 10/20/30ms）
    """
    frame_len = int(sr * frame_ms / 1000)
    num_frames = len(pcm16) // frame_len
    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        yield pcm16[start:end], start / sr, end / sr

def vad_collector(pcm16: np.ndarray, sr: int, aggressiveness: int = 2,
                  frame_ms: int = 30, padding_ms: int = 300) -> List[Segment]:
    """
    WebRTC VADにより発話区間を抽出し、[start, end] のリストを返す。
    """
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(pcm16, sr, frame_ms=frame_ms))
    if not frames:
        return []

    padding_frames = int(padding_ms / frame_ms)
    ring_buffer = queue.Queue(maxsize=padding_frames)

    triggered = False
    voiced_frames = []
    segments = []

    for frame, t0, t1 in frames:
        is_speech = vad.is_speech(frame.tobytes(), sample_rate=sr)

        if not triggered:
            ring_buffer.put((frame, t0, t1, is_speech))
            if sum(1 for _ in list(ring_buffer.queue) if _[3]) > 0.9 * ring_buffer.maxsize:
                triggered = True
                # ring buffer 内から開始
                while not ring_buffer.empty():
                    f, st, et, sp = ring_buffer.get()
                    voiced_frames.append((f, st, et))
        else:
            voiced_frames.append((frame, t0, t1))
            ring_buffer.put((frame, t0, t1, is_speech))
            if sum(1 for _ in list(ring_buffer.queue) if not _[3]) > 0.9 * ring_buffer.maxsize:
                # 終了
                seg_start = voiced_frames[0][1]
                seg_end = voiced_frames[-1][2]
                segments.append(Segment(start=float(seg_start), end=float(seg_end)))
                triggered = False
                voiced_frames = []
                ring_buffer = queue.Queue(maxsize=padding_frames)

    # 終端処理
    if voiced_frames:
        seg_start = voiced_frames[0][1]
        seg_end = voiced_frames[-1][2]
        segments.append(Segment(start=float(seg_start), end=float(seg_end)))

    # 短すぎる区間を除外（< 0.3s）
    segments = [s for s in segments if (s.end - s.start) >= 0.3]
    return segments

def slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    s = int(start * sr)
    e = int(end * sr)
    e = min(e, len(audio))
    return audio[s:e]

def sec_to_srt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(utterances: List[Utterance], path: str):
    with io.open(path, "w", encoding="utf-8") as f:
        for i, u in enumerate(utterances, 1):
            f.write(f"{i}\n")
            f.write(f"{sec_to_srt_ts(u.start)} --> {sec_to_srt_ts(u.end)}\n")
            f.write(f"{u.speaker}: {u.text}\n\n")

def write_vtt(utterances: List[Utterance], path: str):
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for u in utterances:
            start = sec_to_srt_ts(u.start).replace(",", ".")
            end = sec_to_srt_ts(u.end).replace(",", ".")
            f.write(f"{start} --> {end}\n")
            f.write(f"{u.speaker}: {u.text}\n\n")

# -----------------------------
# 話者埋め込み & クラスタリング
# -----------------------------

class SpeakerEmbedder:
    def __init__(self, device: str = "cpu"):
        # ECAPA TDNN
        self.clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "spkrec-ecapa-voxceleb"),
        )

    def embed(self, audio_16k: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        単一波形（16k, mono, float32[-1,1]）→ 192次元程度の埋め込み（L2正規化）
        """
        import torch
        wav_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0)
        with torch.no_grad():
            emb = self.clf.encode_batch(wav_tensor).squeeze(0).squeeze(0).cpu().numpy()
        # 正規化
        norm = np.linalg.norm(emb) + 1e-12
        return emb / norm

def estimate_n_speakers(embs: np.ndarray, k_min: int = 2, k_max: int = 6) -> int:
    if embs.shape[0] < k_min:
        return 1
    best_k = 2
    best_score = -1
    for k in range(k_min, min(k_max, embs.shape[0]) + 1):
        try:
            lab = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average").fit_predict(embs)
            score = silhouette_score(embs, lab, metric="cosine")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    return best_k

def cluster_speakers(embs: np.ndarray, n_speakers: Optional[int] = None) -> np.ndarray:
    if embs.shape[0] == 1:
        return np.array([0])
    if n_speakers is None:
        n_speakers = estimate_n_speakers(embs)
    labels = AgglomerativeClustering(n_clusters=n_speakers, affinity="cosine", linkage="average").fit_predict(embs)
    return labels

# -----------------------------
# ASR（faster-whisper）
# -----------------------------

def asr_transcribe_chunks(
    model: WhisperModel,
    audio: np.ndarray,
    sr: int,
    chunks: List[Segment],
    language: Optional[str] = None,
    beam_size: int = 5,
) -> List[Dict]:
    """
    各チャンクをASR、結果を連結しraw segmentsとして返す
    """
    results = []
    for seg in tqdm(chunks, desc="ASR", unit="seg"):
        a = slice_audio(audio, sr, seg.start, seg.end)
        # faster-whisperはfloat32/16k/monoが扱いやすい
        if sr != 16000:
            a = librosa.resample(a, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
            cur_sr = 16000
        else:
            cur_sr = sr
        # 推論
        segments, info = model.transcribe(a, language=language, beam_size=beam_size, vad_filter=False)
        for s in segments:
            # s.start/s.end はチャンク内相対 → 全体時刻へ
            results.append({
                "start": float(seg.start + (s.start or 0.0)),
                "end": float(seg.start + (s.end or 0.0)),
                "text": s.text.strip(),
                "words": [{"start": float(seg.start + (w.start or 0.0)),
                           "end": float(seg.start + (w.end or 0.0)),
                           "word": w.word} for w in (s.words or [])]
            })
    return results

# -----------------------------
# マージ & 出力整形
# -----------------------------

def assign_speakers_to_segments(segments: List[Segment], labels: np.ndarray) -> List[Dict]:
    diar = []
    for seg, lab in zip(segments, labels):
        diar.append({"start": seg.start, "end": seg.end, "speaker": f"SPK_{lab+1}"})
    return diar

def fuse_asr_with_diarization(raw_asr: List[Dict], diar: List[Dict]) -> List[Utterance]:
    """
    ASRセグメント（細かい）を話者区間に合わせ、話者発話単位にまとめる。
    """
    utterances: List[Utterance] = []
    for d in diar:
        d_start, d_end, spk = d["start"], d["end"], d["speaker"]
        texts = []
        words = []
        s_min = d_end
        s_max = d_start
        for seg in raw_asr:
            if seg["end"] <= d_start or seg["start"] >= d_end:
                continue
            texts.append(seg["text"])
            s_min = min(s_min, seg["start"])
            s_max = max(s_max, seg["end"])
            words.extend([w for w in seg.get("words", []) if (w["end"] > d_start and w["start"] < d_end)])
        if texts:
            u = Utterance(start=float(s_min), end=float(s_max), text=" ".join(texts), speaker=spk, words=words)
            utterances.append(u)
    # 連続する同一話者を結合して読みやすく
    merged: List[Utterance] = []
    for u in utterances:
        if merged and merged[-1].speaker == u.speaker and (u.start - merged[-1].end) < 0.8:
            merged[-1].end = u.end
            merged[-1].text = (merged[-1].text + " " + u.text).strip()
            if u.words:
                merged[-1].words = (merged[-1].words or []) + u.words
        else:
            merged.append(u)
    return merged

# -----------------------------
# メイン処理
# -----------------------------

@app.command()
def transcribe(
    audio: str = typer.Option(..., help="入力音声ファイル（WAV/AIFF/M4Aなど可）"),
    outdir: str = typer.Option("./outputs", help="出力ディレクトリ"),
    model_size: str = typer.Option("small", help="faster-whisperモデルサイズ（tiny/base/small/medium/large-v2）"),
    language: Optional[str] = typer.Option("ja", help="言語コード。自動検出はNone"),
    beam_size: int = typer.Option(5, help="ASRビーム幅"),
    vad_aggr: int = typer.Option(2, help="VADアグレッシブ度（0〜3）"),
    max_speakers: int = typer.Option(6, help="自動推定の上限話者数"),
    num_speakers: Optional[int] = typer.Option(None, help="話者数を固定する場合に指定"),
    device: str = typer.Option("cpu", help="ASR/Speaker埋め込みのデバイス（cpu 推奨）"),
    compute_type: str = typer.Option("int8", help="faster-whisper 計算型（int8/int8_float16/float16/float32など）"),
):
    """
    長時間録音でも堅牢に動作する CUI 文字起こし + 話者分離
    """
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio))[0]

    console.rule("[bold]ロード")
    wav, sr_in = read_audio_any(audio)
    duration = len(wav) / sr_in
    console.print(f"入力: {audio}  sr={sr_in}Hz  dur={duration/60:.1f}分")

    # VAD用に 16k/int16 へ
    pcm16, sr16 = ensure_mono_16k_int16(wav.copy(), sr_in, target_sr=16000)

    console.rule("[bold]VAD（発話区間抽出）")
    segments = vad_collector(pcm16, sr16, aggressiveness=vad_aggr, frame_ms=30, padding_ms=300)
    console.print(f"検出区間: {len(segments)}")

    if not segments:
        console.print("[red]発話区間が検出できませんでした。録音レベルやファイル形式をご確認ください。[/red]")
        raise typer.Exit(code=1)

    # 話者埋め込み
    console.rule("[bold]話者埋め込み")
    embedder = SpeakerEmbedder(device="cpu")  # CPUで安定
    # 埋め込みは16k/float32
    wav16_float = pcm16.astype(np.float32) / 32768.0

    embs = []
    for seg in tqdm(segments, desc="Embed", unit="seg"):
        a = slice_audio(wav16_float, sr16, seg.start, seg.end)
        # 短すぎる区間は埋め込みが不安定→少し拡張
        if len(a) < int(0.6 * sr16):
            pad = int(0.6 * sr16) - len(a)
            a = np.pad(a, (0, pad))
        e = embedder.embed(a, sr=sr16)
        embs.append(e)
    embs = np.vstack(embs)

    console.rule("[bold]話者クラスタリング")
    if num_speakers is not None:
        labels = cluster_speakers(embs, n_speakers=num_speakers)
        nspk = num_speakers
    else:
        est = estimate_n_speakers(embs, k_min=2, k_max=max_speakers)
        labels = cluster_speakers(embs, n_speakers=est)
        nspk = len(np.unique(labels))
    speakers = [f"SPK_{i+1}" for i in range(nspk)]
    console.print(f"推定話者数: {nspk}")

    diar = assign_speakers_to_segments(segments, labels)

    console.rule("[bold]ASR（faster-whisper）")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_asr = asr_transcribe_chunks(model, wav, sr_in, segments, language=language, beam_size=beam_size)

    console.rule("[bold]整形/結合")
    utterances = fuse_asr_with_diarization(raw_asr, diar)

    result = TranscriptionResult(
        audio_path=os.path.abspath(audio),
        sample_rate=sr_in,
        num_channels=1,
        duration=duration,
        language=language,
        model_size=model_size,
        speakers=speakers,
        utterances=utterances,
        segments=raw_asr,
        diarization=diar,
    )

    # 保存
    json_path = os.path.join(outdir, f"{base}.json")
    srt_path = os.path.join(outdir, f"{base}.srt")
    vtt_path = os.path.join(outdir, f"{base}.vtt")

    with io.open(json_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2, ensure_ascii=False))

    write_srt(utterances, srt_path)
    write_vtt(utterances, vtt_path)

    console.rule("[bold green]完了")
    table = Table(show_header=True, header_style="bold")
    table.add_column("項目")
    table.add_column("値", overflow="fold")
    table.add_row("入力", audio)
    table.add_row("総時間", f"{duration/60:.2f} 分")
    table.add_row("話者数", str(nspk))
    table.add_row("出力(JSON)", json_path)
    table.add_row("出力(SRT)", srt_path)
    table.add_row("出力(VTT)", vtt_path)
    console.print(table)

if __name__ == "__main__":
    app()
