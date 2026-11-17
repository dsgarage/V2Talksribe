#!/bin/bash
# Talksribe の診断。メニューバーに項目が見えないときに実行し、出力を Issue に残す。
# プロセスは止めない(USR1 でスタックを書き出させるだけ)。
set -u
DIR="$HOME/Documents/Talksribe"
PIDS=$(pgrep -f "src/menubar.py" || true)
if [ -z "$PIDS" ]; then
  echo "Talksribe(src/menubar.py)は動いていません"
else
  for P in $PIDS; do
    ps -o pid,lstart,etime,%cpu,rss,command -p "$P"
    # USR1 の既定の動作は「終了」。faulthandler を登録した新しいコードのプロセス
    # (debug.log に "startup: pid=<P> " の行がある)にだけ送る
    if grep -q "startup: pid=$P " "$DIR/debug.log" "$DIR"/debug.log.* 2>/dev/null; then
      kill -USR1 "$P" 2>/dev/null && echo "USR1 を送りました: $P"
    else
      echo "pid $P は古いコードの可能性があるため USR1 を送りません(送ると終了します)"
    fi
  done
  sleep 1
fi
echo "--- diag (直近 5 行) ---";        grep "diag:" "$DIR/debug.log" 2>/dev/null | tail -5
echo "--- sleep/wake (直近 5 行) ---";  grep -E "system (sleep|wake)" "$DIR/debug.log" 2>/dev/null | tail -5
echo "--- faulthandler.log (末尾 40 行) ---"; tail -40 "$DIR/faulthandler.log" 2>/dev/null
echo "--- stderr.log (末尾 20 行) ---";       tail -20 "$DIR/stderr.log" 2>/dev/null
