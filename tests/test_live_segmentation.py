# -*- coding: utf-8 -*-
"""ライブの切り方・幻聴の除去・テンポのテスト。マイクも本物のモデルも本物の保存先も使わない。

周波数で印を付けた音声と、それを読む偽の Transcriber を使う(利得と正規化で振幅が変わるため、
振幅ではなく周波数で印を付ける)。エンジンはワーカーを動かさず _process_audio を直接呼ぶ。
"""
import os
import queue
import time
import unittest

import numpy as np

import engine as engine_module
from live_level import frame_rms_db
from segmenter import SegInfo
from settings import load_vocabulary as real_load_vocabulary
from tests.fakes import FakeCaptureFactory
from tests.test_engine_recovery import EngineTestBase

SR = 16000


def noise(sec, rng):
    return rng.normal(0, 0.0004, int(sec * SR)).astype(np.float32)


def tone(sec, k, amp=0.02):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * (200 + 20 * k) * t)).astype(np.float32)


def words(ks, rng):
    """単語 k は 0.8 秒の正弦波、間は 0.2 秒の雑音。"""
    parts = []
    for k in ks:
        parts += [tone(0.8, k), noise(0.2, rng)]
    return np.concatenate(parts)


class MarkerVAD:
    backend = "webrtc"

    def load(self):
        pass

    def reset(self):
        pass

    def speech_prob(self, x):
        return 1.0 if frame_rms_db(x) > -50 else 0.0

    def is_speech(self, x):
        return self.speech_prob(x) >= 0.5


class MarkerTranscriber:
    model_size = "base"

    def __init__(self):
        self.language = None
        self.beam_size = 1
        self.hotwords = None
        self.no_speech_threshold = 0.6
        self.log_prob_threshold = -1.0
        self.calls = []        # {"len_sec","prompt","opts","t"}
        self.partial_sleep = 0.0

    def collect_segments(self, audio, initial_prompt=None, **opts):
        self.calls.append({"len_sec": len(audio) / SR, "prompt": initial_prompt, "opts": dict(opts),
                           "t": time.monotonic()})
        if opts.get("without_timestamps") and self.partial_sleep:
            time.sleep(self.partial_sleep)
        x = np.abs(audio)
        nb = len(x) // 160
        if nb == 0:
            return []
        env = x[:nb * 160].reshape(nb, 160).max(axis=1)
        above = env > 0.3 * env.max()
        segs = []
        i = 0
        while i < nb:
            if not above[i]:
                i += 1
                continue
            j = i
            while j < nb and above[j]:
                j += 1
            if (j - i) * 0.01 >= 0.3:
                mid = (i + j) // 2
                a, b = max(i * 160, (mid - 10) * 160), min(j * 160, (mid + 10) * 160)
                seg = audio[a:b]
                spec = np.abs(np.fft.rfft(seg))
                f = int(np.argmax(spec)) * SR / len(seg)
                k = int(round((f - 200) / 20))
                segs.append(SegInfo(i * 0.01, j * 0.01, f"w{k}", 0.0, -0.1))
            i = j
        return segs

    def collect_texts(self, audio, emit_until_sec=None, **opts):
        return [s.text for s in self.collect_segments(audio)]


class ThankYouTranscriber(MarkerTranscriber):
    def __init__(self, no_speech, logprob):
        super().__init__()
        self.no_speech = no_speech
        self.logprob = logprob

    def collect_segments(self, audio, initial_prompt=None, **opts):
        self.calls.append({"len_sec": len(audio) / SR, "prompt": initial_prompt, "opts": dict(opts),
                           "t": time.monotonic()})
        return [SegInfo(0.0, len(audio) / SR, "Thank you.", self.no_speech, self.logprob)]


class LiveSegmentationTest(EngineTestBase):
    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(0)

    def make(self, transcriber=None, **tuning):
        eng = self.make_engine(run=False)
        eng.settings.live_tuning.vad_on_agc = False
        for k, v in tuning.items():
            setattr(eng.settings.live_tuning, k, v)
        eng.tuning = eng.settings.live_tuning
        eng.transcriber = transcriber or MarkerTranscriber()
        eng.vad = MarkerVAD()
        eng.is_recording = True
        eng.save_path = os.path.join(self.tmpdir, "out.txt")
        eng.partial_enabled = False
        return eng

    def feed(self, eng, audio, chunk_sec=0.5, sleep=0.0):
        n = int(chunk_sec * SR)
        for i in range(0, len(audio), n):
            eng._process_audio(audio[i:i + n])
            if sleep:
                time.sleep(sleep)

    def transcripts(self, eng):
        out = []
        while True:
            try:
                ev = eng.events.get_nowait()
            except queue.Empty:
                return out
            if ev[0] == "transcript":
                out.append(ev[2])

    # (a)
    def test_a_long_speech_no_loss_no_dup(self):
        eng = self.make()
        eng.chunk_sec = 0.5
        audio = np.concatenate([noise(2, self.rng), words(range(45), self.rng), noise(2, self.rng)])
        self.feed(eng, audio, 0.5)
        eng._finalize_all("end")
        got = " ".join(self.transcripts(eng)).split()
        self.assertEqual(got, [f"w{k}" for k in range(45)])

    # (b)
    def test_b_preroll(self):
        eng = self.make()
        audio = np.concatenate([noise(2, self.rng), tone(2.5, 0), noise(4, self.rng)])
        self.feed(eng, audio)
        self.assertEqual(len(eng.transcriber.calls), 1)
        self.assertGreaterEqual(eng.transcriber.calls[0]["len_sec"], 2.5 + 0.25 + 1.2 - 0.3)

    # (c)
    def test_c_silence_end_independent_of_chunk(self):
        for chunk_sec in (0.5, 1.0):
            with self.subTest(chunk_sec=chunk_sec):
                eng = self.make()
                fed = [0]
                at_final = []
                orig_proc = eng._process_audio
                orig_fin = eng._finalize_utterance

                def proc(a, orig_proc=orig_proc, fed=fed):
                    fed[0] += len(a)
                    return orig_proc(a)

                def fin(reason, orig_fin=orig_fin, fed=fed, at_final=at_final):
                    at_final.append(fed[0])
                    return orig_fin(reason)

                eng._process_audio = proc
                eng._finalize_utterance = fin
                audio = np.concatenate([noise(2, self.rng), tone(3, 0), noise(4, self.rng)])
                self.feed(eng, audio, chunk_sec)
                self.assertEqual(len(at_final), 1)
                end_of_tone = 5 * SR
                d = (at_final[0] - end_of_tone) / SR
                self.assertGreaterEqual(d, 1.2 - 0.26)
                self.assertLessEqual(d, 1.2 + 0.26 + chunk_sec)

    # (d)
    def test_d_short_utterances_are_joined(self):
        eng = self.make()
        audio = np.concatenate([noise(2, self.rng), tone(1.0, 0), noise(1.3, self.rng),
                                tone(1.0, 5), noise(3, self.rng)])
        self.feed(eng, audio)
        self.assertEqual(len(eng.transcriber.calls), 1)
        self.assertEqual(self.transcripts(eng), ["w0 w5"])

    # (e)
    def test_e_no_voice_is_not_transcribed(self):
        eng = self.make()
        audio = noise(12, self.rng)
        for fi in (12, 24, 36):           # フレームの境目をまたがない位置にクリックを置く
            s = fi * 4096 + 1000
            audio[s:s + 800] = 0.5
        self.feed(eng, audio)
        self.assertEqual(eng.transcriber.calls, [])
        self.assertGreaterEqual(eng.live_stats["utt_skipped"], 1)

    # (f)
    def test_f_hallucination_dropped_only_when_weak(self):
        eng = self.make(ThankYouTranscriber(0.5, -0.6))
        self.feed(eng, np.concatenate([noise(2, self.rng), tone(1.0, 0), noise(3, self.rng)]))
        self.assertEqual(self.transcripts(eng), [])
        self.assertEqual(eng.live_stats["halluc_dropped"], 1)
        eng2 = self.make(ThankYouTranscriber(0.0, -0.1))
        self.feed(eng2, np.concatenate([noise(2, self.rng), tone(3.0, 0), noise(3, self.rng)]))
        self.assertEqual(self.transcripts(eng2), ["Thank you."])

    # (g)
    def test_g_vocabulary(self):
        path = os.path.join(self.tmpdir, "vocabulary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# メモ\nTalksribe\nBlackHole\n")
        eng = self.make()
        eng.capture_factory = FakeCaptureFactory()
        orig = engine_module.load_vocabulary
        engine_module.load_vocabulary = lambda *a, **k: real_load_vocabulary(path)
        self.addCleanup(setattr, engine_module, "load_vocabulary", orig)
        eng._begin_stream()
        self.assertEqual(eng.transcriber.hotwords, "Talksribe、BlackHole")
        self.assertEqual(eng.transcriber.no_speech_threshold, eng.tuning.live_no_speech_threshold)
        eng._close_capture()

    # (h)
    def test_h_context(self):
        for chars, expect in ((5, "w1 w2"), (0, None)):
            with self.subTest(context_chars=chars):
                eng = self.make(context_chars=chars)
                audio = np.concatenate([noise(2, self.rng), words([0, 1, 2], self.rng), noise(3, self.rng),
                                        words([5, 6], self.rng), noise(3, self.rng)])
                self.feed(eng, audio)
                calls = eng.transcriber.calls
                self.assertEqual(len(calls), 2)
                self.assertIsNone(calls[0]["prompt"])
                self.assertEqual(calls[1]["prompt"], expect)

    # (i)
    def test_i_partial_interval_follows_cost(self):
        eng = self.make(partial_cost_factor=4.0)
        eng.partial_enabled = True
        eng.partial_interval_sec = 0.0
        eng.transcriber.partial_sleep = 0.2
        audio = np.concatenate([noise(1, self.rng), tone(12, 0)])
        self.feed(eng, audio, 0.25, sleep=0.05)
        ts = [c["t"] for c in eng.transcriber.calls if c["opts"].get("without_timestamps")]
        self.assertGreaterEqual(len(ts), 2)
        for a, b in zip(ts, ts[1:]):
            self.assertGreaterEqual(b - a, 0.8)

    # (j)
    def test_j_partial_skipped_when_behind(self):
        eng = self.make()
        eng.partial_enabled = True
        eng.partial_interval_sec = 0.0
        eng._stream_t0 = time.monotonic() - 10
        eng._audio_recv_sec = 0.0
        self.feed(eng, np.concatenate([noise(1, self.rng), tone(3, 0)]))
        self.assertEqual([c for c in eng.transcriber.calls if c["opts"].get("without_timestamps")], [])
        self.assertGreaterEqual(eng.live_stats["partials_skipped_backlog"], 1)

    # (k)
    def test_k_temperature_and_tokens(self):
        eng = self.make()
        eng.partial_enabled = True
        eng.partial_interval_sec = 0.0
        self.feed(eng, np.concatenate([noise(1, self.rng), tone(3, 0), noise(4, self.rng)]))
        partial = [c for c in eng.transcriber.calls if c["opts"].get("without_timestamps")]
        final = [c for c in eng.transcriber.calls if not c["opts"].get("without_timestamps")]
        self.assertGreaterEqual(len(partial), 1)
        self.assertEqual(len(final), 1)
        for c in eng.transcriber.calls:
            self.assertEqual(c["opts"]["temperature"], (0.0,))
            self.assertEqual(c["opts"]["max_new_tokens"], int(min(220, 16 + 15 * c["len_sec"])))
        self.assertEqual(partial[0]["opts"]["beam_size"], 1)
        self.assertIs(partial[0]["opts"]["without_timestamps"], True)

    # 窓の続きでなければ dedupe_prefix は掛からない
    def test_second_utterance_prefix_kept(self):
        eng = self.make()
        audio = np.concatenate([noise(2, self.rng), words([8, 9, 10], self.rng), noise(3, self.rng),
                                words([9, 10, 11], self.rng), noise(3, self.rng)])
        self.feed(eng, audio)
        self.assertEqual(self.transcripts(eng), ["w8 w9 w10", "w9 w10 w11"])


if __name__ == "__main__":
    unittest.main()
