# Установщик «Диктовки» – русский голосовой ввод для Windows.
# Запуск: powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/rostislavgribenyukov-wq/diktovka/main/install.ps1 | iex"

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Repo = 'https://raw.githubusercontent.com/rostislavgribenyukov-wq/diktovka/main'

function Say($t, $c = 'Cyan') { Write-Host $t -ForegroundColor $c }

Say "`n=== Диктовка: русский голосовой ввод (Windows) ===`n" 'White'

# 1. Папка – без кириллицы в пути (так надёжнее для библиотек распознавания)
$Dir = 'C:\Diktovka'
try { New-Item -ItemType Directory -Force $Dir | Out-Null; [IO.File]::WriteAllText("$Dir\.w", 'ok'); Remove-Item "$Dir\.w" }
catch { $Dir = Join-Path $env:LOCALAPPDATA 'Diktovka'; New-Item -ItemType Directory -Force $Dir | Out-Null }
Say "Папка: $Dir"

# Если старая копия запущена – выключаем, чтобы обновить файлы
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like "*diktovka.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# 2. Файлы программы: рядом с установщиком или с GitHub
$Here = $PSScriptRoot
foreach ($f in 'diktovka.py', 'uninstall.ps1') {
  if ($Here -and (Test-Path (Join-Path $Here $f))) { Copy-Item (Join-Path $Here $f) "$Dir\$f" -Force }
  else { Invoke-WebRequest "$Repo/$f" -OutFile "$Dir\$f" -UseBasicParsing }
}

# 3. uv – маленький менеджер Python, кладём прямо в папку, систему не трогаем
$env:UV_PYTHON_INSTALL_DIR = "$Dir\python"
$env:UV_CACHE_DIR = "$Dir\.cache"
$Uv = "$Dir\uv\uv.exe"
if (-not (Test-Path $Uv)) {
  Say 'Скачиваю uv…'
  $zip = "$Dir\uv.zip"
  Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip -UseBasicParsing
  Expand-Archive $zip "$Dir\uv" -Force; Remove-Item $zip
}

# 4. Python и библиотеки
Say 'Ставлю Python 3.12 и библиотеки (пара минут)…'
& $Uv venv "$Dir\venv" --python 3.12 --allow-existing -q
if ($LASTEXITCODE) { throw 'Не получилось создать окружение Python' }
$Py = "$Dir\venv\Scripts\python.exe"
$pkgs = @('faster-whisper', 'sounddevice', 'numpy', 'pyperclip', 'pynput', 'pystray', 'pillow')

$gpu = $null
try { $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1) } catch {}
if ($gpu) {
  Say "Нашёл видеокарту: $gpu – ставлю ускорение CUDA (около 1 ГБ)…"
  $pkgs += @('nvidia-cublas-cu12', 'nvidia-cudnn-cu12==9.*')
} else {
  Say 'Видеокарты NVIDIA нет – распознавать будет процессор (медленнее, но работает).' 'Yellow'
}
& $Uv pip install --python $Py -q @pkgs
if ($LASTEXITCODE) { throw 'Не получилось поставить библиотеки' }

# 5. Модель распознавания (около 1,6 ГБ, один раз)
Say 'Скачиваю модель распознавания Whisper (около 1,6 ГБ, один раз)…'
$env:HF_HUB_VERBOSITY = 'error'
& $Py "$Dir\diktovka.py" --download 2>$null
if ($LASTEXITCODE) { throw 'Модель не загрузилась – подробности в diktovka.log' }

# 6. Ярлыки: рабочий стол + автозапуск
$Pyw = "$Dir\venv\Scripts\pythonw.exe"
$sh = New-Object -ComObject WScript.Shell
foreach ($p in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Startup'))) {
  $l = $sh.CreateShortcut((Join-Path $p 'Диктовка.lnk'))
  $l.TargetPath = $Pyw; $l.Arguments = "`"$Dir\diktovka.py`""; $l.WorkingDirectory = $Dir
  $l.IconLocation = 'C:\Windows\System32\SndVol.exe,0'; $l.Description = 'Русский голосовой ввод – F8'
  $l.Save()
}

# 7. Запуск
Start-Process $Pyw "`"$Dir\diktovka.py`"" -WorkingDirectory $Dir
Say "`nГотово!" 'Green'
Say 'Поставь курсор в любое поле ввода, нажми F8, говори, нажми F8 ещё раз – текст вставится.' 'White'
Say 'Значок микрофона – возле часов. Запускается сама при включении компьютера.' 'White'
Say "Клавиша и настройки – в $Dir\config.json. Удалить – $Dir\uninstall.ps1`n" 'White'
