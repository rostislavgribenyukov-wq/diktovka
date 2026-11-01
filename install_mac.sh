#!/bin/bash
# Установщик «Диктовки» для macOS.
# Запуск: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/rostislavgribenyukov-wq/diktovka/main/install_mac.sh)"
set -e

REPO="https://raw.githubusercontent.com/rostislavgribenyukov-wq/diktovka/main"
DIR="$HOME/Diktovka"
mkdir -p "$DIR"
cd "$DIR"

echo "=== Диктовка: русский голосовой ввод (macOS) ==="
echo "Папка: $DIR"

# 1. Файлы программы — рядом со скриптом или с GitHub
for f in diktovka.py VERSION; do
  if [ -f "$(dirname "$0")/$f" ]; then cp "$(dirname "$0")/$f" "$DIR/$f";
  else curl -fsSL "$REPO/$f" -o "$DIR/$f"; fi
done

# 2. uv — маленький менеджер Python (ставим локально, систему не трогаем)
export UV_PYTHON_INSTALL_DIR="$DIR/python"
export UV_CACHE_DIR="$DIR/.cache"
UV="$DIR/uv/uv"
if [ ! -x "$UV" ]; then
  echo "Скачиваю uv…"
  curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz" -o "$DIR/uv.tgz"
  mkdir -p "$DIR/uv" && tar -xzf "$DIR/uv.tgz" -C "$DIR/uv" --strip-components=1 && rm "$DIR/uv.tgz"
fi

# 3. Python 3.12 и библиотеки
echo "Ставлю Python и библиотеки (пара минут)…"
"$UV" venv "$DIR/venv" --python 3.12 --allow-existing -q
PY="$DIR/venv/bin/python"
"$UV" pip install --python "$PY" -q faster-whisper mlx-whisper sounddevice numpy pyperclip pynput pillow pyobjc

# 4. Модель распознавания (около 1,6 ГБ, один раз)
echo "Скачиваю модель распознавания Whisper (около 1,6 ГБ, один раз)…"
"$PY" "$DIR/diktovka.py" --download || { echo "Модель не загрузилась — подробности в diktovka.log"; exit 1; }

# 5. Приложение «Диктовка» в /Applications (чтобы macOS давала разрешения понятному имени)
APP="/Applications/Diktovka.app"
mkdir -p "$APP/Contents/MacOS"
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Диктовка</string>
  <key>CFBundleIdentifier</key><string>com.halo.diktovka</string>
  <key>CFBundleExecutable</key><string>diktovka</string>
</dict></plist>
EOF
cat > "$APP/Contents/MacOS/diktovka" <<EOF
#!/bin/sh
exec "$PY" "$DIR/diktovka.py" "\$@"
EOF
chmod +x "$APP/Contents/MacOS/diktovka"
codesign --force --deep --sign - "$APP" 2>/dev/null || true

# 6. Автозапуск
PLIST="$HOME/Library/LaunchAgents/com.halo.diktovka.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.halo.diktovka</string>
  <key>ProgramArguments</key><array><string>$APP/Contents/MacOS/diktovka</string></array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>
EOF
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo ""
echo "Готово!"
echo "Дальше важное: Системные настройки → Конфиденциальность и безопасность →"
echo "  разрешите «Диктовке» Микрофон и Универсальный доступ (переключатели слева)."
echo "Потом зажмите правый ⌘, говорите, отпустите — текст вставится под курсор."
echo "Настройки: $DIR/config.json. Журнал: $DIR/diktovka.log"
