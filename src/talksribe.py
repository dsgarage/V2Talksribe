#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, io, queue, math
from typing import List, Optional, Tuple, Dict
import numpy as np, soundfile as sf, librosa, webrtcvad
from tqdm import tqdm
import typer
from rich.console import Console
from rich.table import Table
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from faster_whisper import WhisperModel
_HAVE_RESEMBLYZER=_HAVE_SPECTRAL=_HAVE_SILERO=False
try:
    from resemblyzer import VoiceEncoder; _HAVE_RESEMBLYZER=True
except ImportError:
    pass
try:
    from spectralcluster import SpectralClusterer; _HAVE_SPECTRAL=True
except ImportError:
    pass
try:
    import torch; _HAVE_SILERO=True
except ImportError:
    pass
console=Console()
app=typer.Typer(add_completion=False, help="二段階VAD+話者分離 文字起こしCUI")
class Segment(BaseModel): start: float; end: float
class Utterance(BaseModel):
    start: float; end: float; text: str; speaker: str; words: Optional[List[Dict]]=None
class TranscriptionResult(BaseModel):
    audio_path:str; sample_rate:int; num_channels:int; duration:float
    language:Optional[str]=None; model_size:str; speakers:List[str]
    utterances:List[Utterance]; segments:List[Dict]=Field(default_factory=list); diarization:List[Dict]=Field(default_factory=list)
def read_audio_any(p:str)->Tuple[np.ndarray,int]:
    a,sr=sf.read(p,dtype="float32",always_2d=False); 
    if hasattr(a,'ndim') and a.ndim==2: a=np.mean(a,axis=1); return a,sr
    return a,sr
def ensure_mono_sr(a:np.ndarray,sr:int,target_sr:int=16000)->Tuple[np.ndarray,int]:
    if sr!=target_sr: a=librosa.resample(a,orig_sr=sr,target_sr=target_sr,res_type="kaiser_best"); sr=target_sr
    return a.astype(np.float32),sr
def float_to_int16(a:np.ndarray)->np.ndarray: a=np.clip(a,-1.0,1.0); return (a*32767.0).astype(np.int16)
def slice_audio(a:np.ndarray,sr:int,start:float,end:float)->np.ndarray:
    s=int(start*sr); e=min(int(end*sr),len(a)); return a[s:e]
def sec_to_srt_ts(sec:float)->str:
    h=int(sec//3600); m=int((sec%3600)//60); s=int(sec%60); ms=int((sec-int(sec))*1000); return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
def write_srt(utts:List[Utterance],p:str):
    with io.open(p,"w",encoding="utf-8") as f:
        for i,u in enumerate(utts,1):
            f.write(f"{i}\n{sec_to_srt_ts(u.start)} --> {sec_to_srt_ts(u.end)}\n{u.speaker}: {u.text}\n\n")
def write_vtt(utts:List[Utterance],p:str):
    with io.open(p,"w",encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for u in utts:
            f.write(f"{sec_to_srt_ts(u.start).replace(',','.')} --> {sec_to_srt_ts(u.end).replace(',','.')}\n{u.speaker}: {u.text}\n\n")
def coarse_activity_segments(wav:np.ndarray,sr:int,frame_ms:int=50,hop_ms:int=25,percentile:float=75.0,ratio:float=0.3,min_chunk_sec:float=10.0,pad_sec:float=0.5)->List[Segment]:
    frame=int(sr*frame_ms/1000.0); hop=int(sr*hop_ms/1000.0)
    if len(wav)<frame: return [Segment(start=0.0,end=len(wav)/sr)]
    energies=[float(np.sum(wav[i:i+frame]*wav[i:i+frame])) for i in range(0,len(wav)-frame,hop)]
    if len(energies)==0: return [Segment(start=0.0,end=len(wav)/sr)]
    thr=float(np.percentile(np.asarray(energies,dtype=np.float32),percentile))*ratio
    mask=[1 if e>thr else 0 for e in energies]
    segs=[]; cur=None
    for idx,v in enumerate(mask):
        t=idx*hop/sr
        if v and cur is None: cur=t
        elif (not v) and (cur is not None): segs.append(Segment(start=float(cur),end=float(t+frame/sr))); cur=None
    if cur is not None: segs.append(Segment(start=float(cur),end=float(len(wav)/sr)))
    def pad(seg): return Segment(start=max(0.0,seg.start-pad_sec),end=min(len(wav)/sr,seg.end+pad_sec))
    segs=[pad(s) for s in segs if (s.end-s.start)>=min_chunk_sec]
    merged=[]
    for s in segs:
        if merged and (s.start-merged[-1].end)<1.0: merged[-1].end=max(merged[-1].end,s.end)
        else: merged.append(s)
    return merged
def precise_vad_webrtc(pcm16:np.ndarray,sr:int,aggr:int=2,frame_ms:int=30,padding_ms:int=300)->List[Segment]:
    vad=webrtcvad.Vad(aggr)
    frame_len=int(sr*frame_ms/1000); padding_frames=int(padding_ms/frame_ms)
    ring=queue.Queue(maxsize=padding_frames); triggered=False; voiced=[]; segs=[]
    for i in range(0,len(pcm16)-frame_len,frame_len):
        frame=pcm16[i:i+frame_len]; t0=i/sr; t1=(i+frame_len)/sr; speech=vad.is_speech(frame.tobytes(),sample_rate=sr)
        if not triggered:
            ring.put((frame,t0,t1,speech))
            if sum(1 for _ in list(ring.queue) if _[3])>0.9*ring.maxsize:
                triggered=True
                while not ring.empty():
                    _,st,et,_=ring.get(); voiced.append((st,et))
        else:
            voiced.append((t0,t1)); ring.put((frame,t0,t1,speech))
            if sum(1 for _ in list(ring.queue) if not _[3])>0.9*ring.maxsize:
                segs.append(Segment(start=float(voiced[0][0]),end=float(voiced[-1][1]))); triggered=False; voiced=[]; ring=queue.Queue(maxsize=padding_frames)
    if voiced: segs.append(Segment(start=float(voiced[0][0]),end=float(voiced[-1][1])))
    return [s for s in segs if (s.end-s.start)>=0.3]
def precise_vad_silero(wav16:np.ndarray,sr:int)->List[Segment]:
    if not _HAVE_SILERO: raise RuntimeError("Silero VADには torch が必要です。`pip install torch`")
    model,utils=torch.hub.load(repo_or_dir='snakers4/silero-vad',model='silero_vad',verbose=False)
    (get_speech_timestamps,_,_,_,_)=utils
    ts=get_speech_timestamps(wav16.astype(np.float32),model,sampling_rate=sr)
    return [Segment(start=t['start']/sr,end=t['end']/sr) for t in ts]
def diarize_resemblyzer(wav16:np.ndarray,sr16:int,segs:List[Segment],max_speakers:int):
    if not (_HAVE_RESEMBLYZER and _HAVE_SPECTRAL): raise RuntimeError("`pip install resemblyzer spectralcluster` が必要")
    encoder=VoiceEncoder(); X=[]
    for s in tqdm(segs,desc="Embed(resemblyzer)",unit="seg"):
        a=slice_audio(wav16,sr16,s.start,s.end); 
        if len(a)<int(0.6*sr16): a=np.pad(a,(0,int(0.6*sr16)-len(a)))
        X.append(encoder.embed_utterance(a.astype(np.float32)))
    import numpy as np; X=np.vstack(X)
    labels=SpectralClusterer(min_clusters=2,max_clusters=max_speakers,p_percentile=0.95,gaussian_blur_sigma=1.0).predict(X)
    n=len(np.unique(labels)); speakers=[f"SPK_{i+1}" for i in range(n)]
    return speakers,labels
def diarize_ecapa(wav16:np.ndarray,sr16:int,segs:List[Segment],max_speakers:int,num_speakers:Optional[int]):
    from speechbrain.pretrained import EncoderClassifier; import numpy as np, torch
    clf=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir=os.path.join(os.path.expanduser("~"),".cache","speechbrain","spkrec-ecapa-voxceleb"),run_opts={"device":"cpu"})
    X=[]
    for s in tqdm(segs,desc="Embed(ECAPA)",unit="seg"):
        a=slice_audio(wav16,sr16,s.start,s.end); 
        if len(a)<int(0.6*sr16): a=np.pad(a,(0,int(0.6*sr16)-len(a)))
        w=torch.from_numpy(a.astype(np.float32)).unsqueeze(0)
        with torch.no_grad(): e=clf.encode_batch(w).squeeze(0).squeeze(0).cpu().numpy()
        e=e/(np.linalg.norm(e)+1e-12); X.append(e)
    X=np.vstack(X)
    if num_speakers is None:
        best_k,best=-1,-1
        for k in range(2,min(max_speakers,X.shape[0])+1):
            try:
                lab=AgglomerativeClustering(n_clusters=k,affinity="cosine",linkage="average").fit_predict(X)
                score=silhouette_score(X,lab,metric="cosine")
                if score>best: best_k, best= k, score
            except Exception:
                continue
        nspk=best_k if best_k!=-1 else 2
    else: nspk=num_speakers
    labels=AgglomerativeClustering(n_clusters=nspk,affinity="cosine",linkage="average").fit_predict(X)
    speakers=[f"SPK_{i+1}" for i in range(nspk)]
    return speakers,labels
def assign_speakers_to_segments(segs:List[Segment],labels)->List[Dict]:
    return [{"start":s.start,"end":s.end,"speaker":f"SPK_{lab+1}"} for s,lab in zip(segs,labels)]
def fuse_asr_with_diarization(raw_asr:List[Dict],diar:List[Dict])->List[Utterance]:
    utts=[]; 
    for d in diar:
        ds,de,spk=d["start"],d["end"],d["speaker"]; texts=[]; words=[]; s_min,de_min=de,ds
        for seg in raw_asr:
            if seg["end"]<=ds or seg["start"]>=de: continue
            texts.append(seg["text"]); s_min=min(s_min,seg["start"]); de_min=max(de_min,seg["end"])
            words.extend([w for w in seg.get("words",[]) if (w["end"]>ds and w["start"]<de)])
        if texts: utts.append(Utterance(start=float(s_min),end=float(de_min),text=" ".join(texts),speaker=spk,words=words))
    merged=[]
    for u in utts:
        if merged and merged[-1].speaker==u.speaker and (u.start-merged[-1].end)<0.8:
            merged[-1].end=u.end; merged[-1].text=(merged[-1].text+" "+u.text).strip()
            merged[-1].words=(merged[-1].words or [])+(u.words or [])
        else: merged.append(u)
    return merged
def asr_transcribe_chunks(model:WhisperModel,audio:np.ndarray,sr:int,chunks:List[Segment],language:Optional[str],beam_size:int)->List[Dict]:
    out=[]
    for seg in tqdm(chunks,desc="ASR",unit="seg"):
        a=slice_audio(audio,sr,seg.start,seg.end)
        cur=a if sr==16000 else librosa.resample(a,orig_sr=sr,target_sr=16000,res_type="kaiser_best")
        segments,info=model.transcribe(cur,language=language,beam_size=beam_size,vad_filter=False)
        for s in segments:
            out.append({"start":float(seg.start+(s.start or 0.0)),"end":float(seg.start+(s.end or 0.0)),"text":s.text.strip(),
                        "words":[{"start":float(seg.start+(w.start or 0.0)),"end":float(seg.start+(w.end or 0.0)),"word":w.word} for w in (s.words or [])]})
    return out
@app.command()
def transcribe(audio: str=typer.Option(...), outdir: str=typer.Option("./outputs"), model_size: str=typer.Option("small"),
               language: Optional[str]=typer.Option("ja"), beam_size: int=typer.Option(5),
               vad_backend: str=typer.Option("silero"), diar_backend: str=typer.Option("resemblyzer"),
               max_speakers: int=typer.Option(6), num_speakers: Optional[int]=typer.Option(None),
               device: str=typer.Option("cpu"), compute_type: str=typer.Option("int8"),
               coarse_percentile: float=typer.Option(75.0), coarse_ratio: float=typer.Option(0.3),
               coarse_min_chunk: float=typer.Option(10.0), coarse_pad: float=typer.Option(0.5)):
    os.makedirs(outdir,exist_ok=True); base=os.path.splitext(os.path.basename(audio))[0]
    console.rule("[bold]ロード"); wav,sr_in=read_audio_any(audio); duration=len(wav)/sr_in; console.print(f"入力: {audio}  sr={sr_in}Hz  dur={duration/60:.1f}分")
    console.rule("[bold]Stage1: 粗VAD"); wav16,sr16=ensure_mono_sr(wav.copy(),sr_in,16000)
    coarse=coarse_activity_segments(wav16,sr16,percentile=coarse_percentile,ratio=coarse_ratio,min_chunk_sec=coarse_min_chunk,pad_sec=coarse_pad)
    console.print(f"粗チャンク数: {len(coarse)}"); 
    if not coarse: console.print("[red]音活動検出なし。しきい値を調整してください。[/red]"); raise typer.Exit(code=1)
    all_diar=[]; all_asr=[]; speakers_global=[]
    model=WhisperModel(model_size,device=device,compute_type=compute_type)
    for idx,cseg in enumerate(coarse,1):
        console.rule(f"[bold]Stage2: チャンク {idx}/{len(coarse)}  {cseg.start:.2f}-{cseg.end:.2f}s")
        sub=slice_audio(wav16,sr16,cseg.start,cseg.end)
        if vad_backend=="silero": precise=precise_vad_silero(sub,sr16)
        elif vad_backend=="webrtc": precise=precise_vad_webrtc(float_to_int16(sub),sr16,2,30,300)
        else: raise typer.BadParameter("vad_backend は silero|webrtc")
        precise_abs=[Segment(start=cseg.start+s.start,end=cseg.start+s.end) for s in precise]
        console.print(f"  精密VADセグメント: {len(precise_abs)}")
        if not precise_abs: continue
        if diar_backend=="resemblyzer":
            spk,labels=diarize_resemblyzer(wav16,sr16,precise_abs,max_speakers=max_speakers); diar=assign_speakers_to_segments(precise_abs,labels)
        elif diar_backend=="ecapa":
            spk,labels=diarize_ecapa(wav16,sr16,precise_abs,max_speakers=max_speakers,num_speakers=num_speakers); diar=assign_speakers_to_segments(precise_abs,labels)
        elif diar_backend=="pyannote":
            spk=["SPK_1"]; diar=[{"start":s.start,"end":s.end,"speaker":"SPK_1"} for s in precise_abs]
        else: raise typer.BadParameter("diar_backend は resemblyzer|ecapa|pyannote")
        for s in spk:
            if s not in speakers_global: speakers_global.append(s)
        sub_full=slice_audio(wav,sr_in,cseg.start,cseg.end)
        rel=[Segment(start=s.start-cseg.start,end=s.end-cseg.start) for s in precise_abs]
        raw=asr_transcribe_chunks(model,sub_full,sr_in,rel,language,beam_size)
        for r in raw: r["start"]+=cseg.start; r["end"]+=cseg.start
        all_diar.extend(diar); all_asr.extend(raw)
    all_diar=sorted(all_diar,key=lambda d:d["start"]); all_asr=sorted(all_asr,key=lambda d:d["start"])
    console.rule("[bold]整形/結合"); utts=fuse_asr_with_diarization(all_asr,all_diar)
    res=TranscriptionResult(audio_path=os.path.abspath(audio),sample_rate=sr_in,num_channels=1,duration=duration,language=language,model_size=model_size,
                             speakers=speakers_global or ["SPK_1"], utterances=utts, segments=all_asr, diarization=all_diar)
    jp=os.path.join(outdir,f"{base}.json"); sp=os.path.join(outdir,f"{base}.srt"); vp=os.path.join(outdir,f"{base}.vtt")
    with io.open(jp,"w",encoding="utf-8") as f: f.write(res.model_dump_json(indent=2,ensure_ascii=False))
    write_srt(utts,sp); write_vtt(utts,vp)
    console.rule("[bold green]完了"); t=Table(show_header=True,header_style="bold"); t.add_column("項目"); t.add_column("値",overflow="fold")
    t.add_row("入力",audio); t.add_row("総時間",f"{duration/60:.2f} 分"); t.add_row("話者数",str(len(res.speakers))); t.add_row("出力(JSON)",jp); t.add_row("出力(SRT)",sp); t.add_row("出力(VTT)",vp); console.print(t)
@app.command()
def realtime(
    model_size: str = typer.Option("small", help="Whisperモデルサイズ"),
    language: Optional[str] = typer.Option("ja", help="言語コード (ja/en等)"),
    device: str = typer.Option("cpu", help="デバイス (cpu/cuda/mps)"),
    compute_type: str = typer.Option("int8", help="計算精度"),
    beam_size: int = typer.Option(3, help="ビームサーチ幅(小さめ推奨)"),
    chunk_sec: float = typer.Option(0.5, help="VADチャンク長(秒)"),
    vad_threshold: float = typer.Option(0.5, help="VAD発話判定しきい値"),
    input_device: Optional[int] = typer.Option(None, help="入力デバイスID (--devicesで確認)"),
    outfile: Optional[str] = typer.Option(None, help="結果保存先テキストファイル"),
):
    """マイク入力からリアルタイム文字起こしを行う。"""
    from realtime import realtime_pipeline, save_transcript
    lines = realtime_pipeline(
        model_size=model_size, language=language, device=device,
        compute_type=compute_type, beam_size=beam_size,
        chunk_sec=chunk_sec, vad_threshold=vad_threshold,
        input_device=input_device,
    )
    if outfile and lines:
        save_transcript(lines, outfile)

@app.command()
def devices():
    """利用可能なオーディオ入力デバイスを一覧表示する。"""
    from realtime import list_devices
    list_devices()

@app.command()
def menubar():
    """macOS メニューバーでリアルタイム文字起こしを実行する。"""
    from menubar import main as menubar_main
    menubar_main()

if __name__ == "__main__":
    app()
