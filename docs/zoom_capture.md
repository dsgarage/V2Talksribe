# Zoom 会議の音声を Talksribe で文字起こしする

Zoom の相手の声を文字起こしするには、Zoom の再生音を仮想オーディオデバイス
**BlackHole 2ch** に流し込み、Talksribe の入力をその BlackHole にします。

## 前提（このマシンでは確認済み）

- `BlackHole 2ch` 導入済み（入力 device 1 / 出力 device 1）
- `MultiOutput`（複数出力装置）作成済み（出力 device 6）
  - スピーカーと BlackHole の両方へ同時出力する装置。これで「自分でも聞こえる」かつ
    「Talksribe で拾える」を両立します。

`MultiOutput` の中身を確認・作り直したいときは
`アプリケーション > ユーティリティ > Audio MIDI設定` を開き、左下の `+` →
「複数出力装置を作成」で `MacBook Airのスピーカー` と `BlackHole 2ch` の両方に
チェックを入れます（マスター装置はスピーカー側）。

## 手順

1. **Zoom の再生先を MultiOutput にする**
   - Zoom → 設定 → オーディオ → スピーカー で `MultiOutput` を選択。
   - （Zoom 側で選べない場合は、macOS の `システム設定 > サウンド > 出力` を
     `MultiOutput` にしても可。ただし全システム音が BlackHole に乗ります。）

2. **Talksribe の入力を BlackHole にする**
   - メニューバーの Talksribe → 入力デバイス → `BlackHole 2ch`（device 1）を選択。

3. **録音を開始**
   - Zoom 会議で相手が話すと、その音声が文字起こしされます。

4. **会議後は元に戻す**
   - Zoom / システムのスピーカーを `MacBook Airのスピーカー` に。
   - Talksribe の入力を `MacBook Airのマイク`（device 2, 既定）に戻す。

## 拾える音・拾えない音

- ○ 相手（リモート参加者）の声 … Zoom の再生音として BlackHole に乗るため拾えます。
- × 自分の声 … 自分のマイク入力は BlackHole に流れません。
  - 自分の声も同時に文字起こししたい場合は、Audio MIDI設定で
    「機器セット（Aggregate Device）」に BlackHole + 内蔵マイクをまとめ、
    それを Talksribe の入力にする構成が必要です（上級者向け）。

## トラブルシュート

- **無音で文字起こしされない**
  - Talksribe の入力が `BlackHole 2ch` になっているか。
  - Zoom の再生先が `MultiOutput`（または BlackHole を含む出力）になっているか。
  - 入力音量の実測は次で確認できます:
    ```bash
    /Users/daisuketsukada/.pyenv/versions/3.11.0/bin/python3 - <<'PY'
    import sounddevice as sd, numpy as np
    sr=16000; rec=sd.rec(int(3*sr),samplerate=sr,channels=1,device=1,dtype='float32'); sd.wait()
    print("BlackHole RMS=%.5f peak=%.5f"%(float(np.sqrt(np.mean(rec**2))), float(np.max(np.abs(rec)))))
    PY
    ```
    Zoom で音が鳴っている最中に RMS が 0 でなければ経路は正常です。
