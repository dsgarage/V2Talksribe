# Talksribe 初心者クイックスタート

メニューバー常駐のリアルタイム文字起こしアプリです。3 ステップで使えます。

## 1. 起動する

```bash
bash scripts/run_dev.sh
```

画面右上のメニューバーに Talksribe アイコンが出れば起動成功です。

## 2. リアルタイム文字起こし（自分の声）

1. メニューバーの Talksribe アイコンをクリック。
2. 入力デバイスが `MacBook Airのマイク`（既定）になっていることを確認。
3. 「録音を開始」を選ぶと、話した内容がその場で文字になります。
4. 結果は `~/Documents/Talksribe/` にテキスト保存されます。

> ポイント: 入力が `ZoomAudioDevice` だと通話中以外は無音で、何も出ません。
> 迷ったら `MacBook Airのマイク` を選びます。

## 3. 録音ファイルを文字起こしする

音声/動画ファイルを丸ごと文字起こしできます。

```bash
python3 scripts/transcribe_file.py <ファイルパス> --model base --lang ja
```

- `--model` … `tiny`(速い) / `base`(標準) / `small`(高精度)
- `--lang` … `ja` 日本語 / `en` 英語 / 省略で自動判定
- 出力は `~/Documents/Talksribe/{時刻}_{ファイル名}.txt`

## 困ったとき

| 症状 | 確認すること |
|---|---|
| 文字起こしされない | 入力デバイスが `MacBook Airのマイク` か。マイク許可が出ているか |
| アプリが二重に起動した | `pgrep -fl menubar.py` で確認し、古い PID を `kill <PID>` |
| ログを見たい | `~/Documents/Talksribe/debug.log` |
| Zoom の音声を拾いたい | `docs/zoom_capture.md` を参照 |

## 電力プロファイル（精度と速さの切り替え）

メニューから選べます。

- 省電力（eco） … tiny モデル・軽い
- 標準（balanced） … base モデル
- 高精度（accuracy） … small モデル・重いが正確
