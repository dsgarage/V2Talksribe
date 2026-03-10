#!/bin/bash
# Install a source-launcher app into /Applications.
set -euo pipefail

APP_NAME="Talksribe"
APP_PATH="/Applications/${APP_NAME}.app"
PYTHON="/Users/daisuketsukada/.pyenv/versions/3.11.0/bin/python3"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"

echo "=== ${APP_NAME} ビルド開始 ==="

# 既存アプリを削除
if [ -d "$APP_PATH" ]; then
    echo "既存の ${APP_PATH} を削除..."
    rm -rf "$APP_PATH"
fi

# .app バンドルを手動構築
echo "アプリバンドルを構築..."
mkdir -p "$APP_PATH/Contents/MacOS"
mkdir -p "$APP_PATH/Contents/Resources"

# ランチャースクリプト（ソースツリーを直接実行）
cat > "$APP_PATH/Contents/MacOS/${APP_NAME}" << 'LAUNCHER'
#!/bin/bash
ROOT_DIR="/Users/daisuketsukada/Documents/dsgarageScript/V2TApp"
export PYTHONPATH="$ROOT_DIR/src"
exec /Users/daisuketsukada/.pyenv/versions/3.11.0/bin/python3 "$ROOT_DIR/src/menubar.py"
LAUNCHER
chmod +x "$APP_PATH/Contents/MacOS/${APP_NAME}"

# Info.plist
cat > "$APP_PATH/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Talksribe</string>
    <key>CFBundleDisplayName</key>
    <string>Talksribe</string>
    <key>CFBundleIdentifier</key>
    <string>com.dsgarage.talksribe</string>
    <key>CFBundleVersion</key>
    <string>1.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.1.0</string>
    <key>CFBundleExecutable</key>
    <string>Talksribe</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Talksribe はリアルタイム文字起こしのためにマイクを使用します。</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
</dict>
</plist>
PLIST

# Quarantine 属性を除去
xattr -r -d com.apple.quarantine "$APP_PATH" 2>/dev/null || true

# アドホック署名
codesign --force --deep --sign - "$APP_PATH" 2>/dev/null || true

echo ""
echo "=== ビルド完了 ==="
echo "  場所: ${APP_PATH}"
echo "  実行コード: ${SRC_DIR}/menubar.py"
echo ""

# 起動テスト
echo "起動テスト中..."
> /tmp/talksribe_launch.log
open "$APP_PATH"
sleep 5

if pgrep -f "menubar.py" > /dev/null; then
    echo "✓ 起動成功！メニューバーに「TS」が表示されているか確認してください。"
else
    # open が失敗した場合、直接実行にフォールバック
    echo "open コマンドで起動できませんでした。直接実行を試みます..."
    "$APP_PATH/Contents/MacOS/${APP_NAME}" &
    sleep 5
    if pgrep -f "menubar.py" > /dev/null; then
        echo "✓ 直接実行で起動成功！メニューバーを確認してください。"
        echo ""
        echo "※ このアプリは open コマンドでは起動できないため、"
        echo "  以下のエイリアスを ~/.zshrc に追加することを推奨します:"
        echo ""
        echo "  alias talksribe='/Applications/Talksribe.app/Contents/MacOS/Talksribe &'"
        echo ""
    else
        echo "△ 起動に失敗しました。ログ: /tmp/talksribe_launch.log"
        cat /tmp/talksribe_launch.log 2>/dev/null
    fi
fi

echo ""
echo "--- 便利な設定 ---"
echo "自動起動: システム設定 → 一般 → ログイン項目 → 「+」→ Talksribe"
echo "Spotlight: Cmd+Space → 「Talksribe」"
echo "停止: メニューバーの TS → 終了"
