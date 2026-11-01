# Диктовка – русский голосовой ввод в любое окно (macOS и Windows).
# Зажать клавишу → говорить → отпустить → текст вставляется под курсор.
# Распознаёт локально: Whisper (MLX на Apple Silicon / faster-whisper в остальных случаях).
# Автообновление с GitHub. Автор порта: Ростислав Грибенюков.

import os, sys, json, time, math, struct, threading, subprocess, traceback

APP_VERSION = "1.1.0"
REPO = "rostislavgribenyukov-wq/diktovka"   # отсюда скачиваются обновления
PLATFORM = "mac" if sys.platform == "darwin" else "win"

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
import warnings; warnings.filterwarnings("ignore")
LOG = os.path.join(BASE, "diktovka.log")
CFG_PATH = os.path.join(BASE, "config.json")
LOCK_PATH = os.path.join(BASE, ".lock")

DEFAULT_CFG = {
    "hotkey": "cmd_r",           # macOS: cmd_r (правый ⌘). Windows: "f8", "ctrl+alt+space", …
    "language": "ru",
    "model": "large-v3-turbo",   # на слабом ПК можно "small"
    "device": "auto",            # auto / cuda / cpu (cuda — только Windows с NVIDIA)
    "engine": "auto",            # auto / mlx / fwhisper
    "beep": True,                # звуки вкл/выкл
    "volume": 0.5,               # громкость звуков 0–1
    "auto_send": False,          # True – после вставки нажать Enter (сразу отправить)
    "update_channel": "main",    # ветка GitHub с обновлениями
    "prompt": "Привет. Это голосовое сообщение на русском языке, с точками и запятыми.",
}


def log(*a):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a) + "\n")
    except Exception:
        pass


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH, encoding="utf-8-sig") as f:
                saved = json.load(f)
            saved = {k: v for k, v in saved.items() if k in DEFAULT_CFG}
            cfg.update(saved)
        except Exception as e:
            log("config.json не прочитан:", e)
    else:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CFG, f, ensure_ascii=False, indent=2)
    if PLATFORM == "win" and cfg["hotkey"] == "cmd_r":
        cfg["hotkey"] = "f8"     # на Windows нет правого ⌘
    return cfg


# ---------- распознавание ----------

def load_model(cfg):
    if PLATFORM == "mac" and cfg.get("engine", "auto") in ("auto", "mlx"):
        try:
            import mlx_whisper
            return ("mlx", mlx_whisper)
        except Exception as e:
            log("mlx недоступен, беру faster-whisper:", e)
    from faster_whisper import WhisperModel
    dev = cfg.get("device", "auto")
    if PLATFORM == "win" and dev == "auto":
        try:
            import ctranslate2
            dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            dev = "cpu"
    if dev == "cuda":
        try:
            _add_cuda_dlls()
            m = WhisperModel(cfg["model"], device="cuda", compute_type="float16")
            return ("fwhisper-cuda", m)
        except Exception as e:
            log("CUDA не завелась, перехожу на процессор:", e)
    threads = max(2, (os.cpu_count() or 4) - 1)
    m = WhisperModel(cfg["model"], device="cpu", compute_type="int8", cpu_threads=threads)
    return ("fwhisper", m)


def _add_cuda_dlls():
    # библиотеки CUDA из pip-пакетов nvidia-* — подсказываем Windows, где их искать
    if PLATFORM != "win":
        return
    try:
        import site
        roots = site.getsitepackages() + [site.getusersitepackages()]
    except Exception:
        roots = []
    roots.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    for r in roots:
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            d = os.path.join(r, "nvidia", sub, "bin")
            if os.path.isdir(d):
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


def trim_silence(audio, sr=16000, thresh=0.008, pad=0.15):
    # обрезаем тишину по краям — быстрее и точнее
    import numpy as np
    if len(audio) == 0:
        return audio
    idx = np.where(np.abs(audio) > thresh)[0]
    if len(idx) == 0:
        return audio
    s = max(0, idx[0] - int(pad * sr))
    e = min(len(audio), idx[-1] + int(pad * sr))
    return audio[s:e]


def transcribe(model, audio, cfg):
    audio = trim_silence(audio)
    kind, m = model
    if kind == "mlx":
        r = m.transcribe(
            audio, path_or_hf_repo=f"mlx-community/whisper-{cfg['model']}",
            language=cfg["language"], initial_prompt=cfg.get("prompt") or None,
        )
        return (r.get("text") or "").strip()
    segs, _ = m.transcribe(
        audio, language=cfg["language"], beam_size=3, vad_filter=True,
        initial_prompt=cfg.get("prompt") or None, condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segs).strip()


# ---------- звуки: синтезируем приятные короткие сигналы (одинаковые на обеих ОС) ----------

_SND_CACHE = {}


def _synth_wav(path, f0, f1, dur):
    # мягкий тон с затуханием и лёгким сдвигом частоты — «поп», а не системный бип
    sr = 22050
    n = int(sr * dur)
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 36 + n * 2) + b"WAVEfmt " +
                 struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
                 b"data" + struct.pack("<I", n * 2))
        frames = bytearray()
        for i in range(n):
            t = i / sr
            f = f0 + (f1 - f0) * (t / dur)
            env = math.exp(-4.5 * t / dur) * min(1.0, t / 0.008)
            frames += struct.pack("<h", int(32767 * 0.6 * env * math.sin(2 * math.pi * f * t)))
        fh.write(frames)


def _sound_file(kind):
    if kind in _SND_CACHE:
        return _SND_CACHE[kind]
    specs = {"start": (740, 990, 0.16), "stop": (520, 392, 0.14),
             "ready": (660, 880, 0.22), "error": (220, 165, 0.30)}
    p = os.path.join(BASE, f"snd_{kind}.wav")
    if not os.path.exists(p):
        try:
            _synth_wav(p, *specs.get(kind, specs["ready"]))
        except Exception as e:
            log("звук:", e)
            return None
    _SND_CACHE[kind] = p
    return p


def play_sound(cfg, kind):
    if not cfg.get("beep"):
        return
    p = _sound_file(kind)
    if not p:
        return
    try:
        vol = float(cfg.get("volume", 0.5))
        if PLATFORM == "mac":
            subprocess.Popen(["afplay", "-v", str(vol), p],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import winsound
            winsound.PlaySound(p, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass


# ---------- вставка и фокус (по платформам) ----------

def frontmost_name():
    if PLATFORM == "mac":
        try:
            from AppKit import NSWorkspace
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return app.localizedName() if app else None
        except Exception:
            return None
    return None  # на Windows окно запоминается по hwnd


def activate_target(target_hwnd):
    if PLATFORM == "mac":
        try:
            from AppKit import NSWorkspace, NSApplicationActivateIgnoringOtherApps
            for app in NSWorkspace.sharedWorkspace().runningApplications():
                if app.localizedName() == target_hwnd and not app.isHidden():
                    app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                    return True
        except Exception as e:
            log("activate:", e)
        return False
    # Windows: target_hwnd — HWND
    try:
        import ctypes
        u = ctypes.windll.user32
        if u.GetForegroundWindow() != target_hwnd:
            u.SetForegroundWindow(target_hwnd)
        return True
    except Exception as e:
        log("activate:", e)
        return False


def current_front_hwnd():
    if PLATFORM != "win":
        return None
    try:
        import ctypes
        return ctypes.windll.user32.GetForegroundWindow()
    except Exception:
        return None


def send_paste():
    if PLATFORM == "mac":
        _send_key_mac(9, command=True)    # V
    else:
        from pynput import keyboard
        kb = keyboard.Controller()
        with kb.pressed(keyboard.Key.ctrl):
            kb.press("v")
            kb.release("v")


def send_enter():
    if PLATFORM == "mac":
        _send_key_mac(36, command=False)
    else:
        from pynput import keyboard
        keyboard.Controller().press(keyboard.Key.enter)
        keyboard.Controller().release(keyboard.Key.enter)


def _send_key_mac(keycode, command=False):
    from Quartz import (CGEventCreateKeyboardEvent, CGEventSetFlags,
                        CGEventPost, kCGHIDEventTap, kCGEventFlagMaskCommand)
    for down in (True, False):
        ev = CGEventCreateKeyboardEvent(None, keycode, down)
        if command:
            CGEventSetFlags(ev, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, ev)


def single_instance():
    if PLATFORM == "mac":
        import fcntl
        f = open(LOCK_PATH, "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True
    import ctypes
    k = ctypes.windll.kernel32
    k.CreateMutexW(None, False, "Global\\Diktovka_AIM")
    return k.GetLastError() != 183


# ---------- интерфейс macOS (AppKit): значок в меню-баре + стеклянная плашка ----------

if PLATFORM == "mac":
    import objc
    from AppKit import NSView, NSColor, NSBezierPath

    MIC_IDLE = os.path.join(BASE, "mic_idle.png")
    MIC_REC = os.path.join(BASE, "mic_rec.png")
    APP_ICON = os.path.join(BASE, "app_icon.png")

    class _DotView(NSView):
        """Цветная точка-индикатор состояния; при записи пульсирует."""
        def initWithFrame_(self, frame):
            self = objc.super(_DotView, self).initWithFrame_(frame)
            self._c = NSColor.whiteColor()
            self._timer = None
            self._t0 = 0.0
            return self

        def setColor_(self, c):
            self._c = c
            self.setNeedsDisplay_(True)

        def startPulsing(self):
            if self._timer is None:
                from Foundation import NSTimer
                self._t0 = time.time()
                self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                    1 / 20, True, lambda t: self._pulse())

        def stopPulsing(self):
            if self._timer is not None:
                self._timer.invalidate()
                self._timer = None
            self.setAlphaValue_(1.0)

        def _pulse(self):
            p = (time.time() - self._t0) * 2 * math.pi / 1.4  # период 1,4 с
            self.setAlphaValue_(0.55 + 0.45 * (0.5 + 0.5 * math.sin(p)))

        def drawRect_(self, rect):
            self._c.set()
            NSBezierPath.bezierPathWithOvalInRect_(self.bounds()).fill()

    class _LevelBar(NSView):
        """Плавная полоса уровня звука: системные цвета, адаптируется под тему."""
        def initWithFrame_(self, frame):
            self = objc.super(_LevelBar, self).initWithFrame_(frame)
            self._v = 0.0
            return self

        def setValue_(self, v):
            self._v = max(0.0, min(1.0, v))
            self.setNeedsDisplay_(True)

        def drawRect_(self, rect):
            b = self.bounds()
            h = b.size.height
            NSColor.labelColor().colorWithAlphaComponent_(0.12).set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(b, h / 2, h / 2).fill()
            if self._v > 0.01:
                w = b.size.width * self._v
                if self._v < 0.6:
                    c = NSColor.systemGreenColor()
                elif self._v < 0.85:
                    c = NSColor.systemOrangeColor()
                else:
                    c = NSColor.systemRedColor()
                c.colorWithAlphaComponent_(0.95).set()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    ((b.origin.x, b.origin.y), (w, h)), h / 2, h / 2).fill()

    class _SpinnerView(NSView):
        """Плавный круговой индикатор «обработка»."""
        def initWithFrame_(self, frame):
            self = objc.super(_SpinnerView, self).initWithFrame_(frame)
            self._angle = 0.0
            self._timer = None
            return self

        def startAnimating(self):
            if self._timer is None:
                from Foundation import NSTimer
                self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                    1 / 30, True, lambda t: self._tick())

        def stopAnimating(self):
            if self._timer is not None:
                self._timer.invalidate()
                self._timer = None

        def isAnimating(self):
            return self._timer is not None

        def _tick(self):
            self._angle = (self._angle + 11) % 360
            self.setNeedsDisplay_(True)

        def drawRect_(self, rect):
            b = self.bounds()
            c = (b.origin.x + b.size.width / 2, b.origin.y + b.size.height / 2)
            r = min(b.size.width, b.size.height) / 2 - 2
            path = NSBezierPath.bezierPath()
            path.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                c, r, self._angle, self._angle + 285, False)
            path.setLineWidth_(2.2)
            path.setLineCapStyle_(1)
            NSColor.labelColor().set()
            path.stroke()

    def _draw_mic_glyph(d, cx, top, s, color, w):
        # микрофон-глиф: капсула + дуга + ножка; s — высота глифа
        body_w = s * 0.50
        body_h = s * 0.58
        d.rounded_rectangle((cx - body_w / 2, top + s - body_h, cx + body_w / 2, top + s),
                            radius=body_w / 2, fill=color)
        arc_r = s * 0.44
        ay = top + (s - body_h) + body_h * 0.55
        d.arc((cx - arc_r, ay - arc_r, cx + arc_r, ay + arc_r), 15, 165, fill=color, width=w)
        d.line((cx, ay + arc_r - w / 2, cx, top), fill=color, width=w)

    def make_icons():
        try:
            from PIL import Image, ImageDraw
            SS = 8
            size = 18 * SS
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            _draw_mic_glyph(d, size / 2, size * 0.12, size * 0.76, (0, 0, 0, 255), int(size * 0.09))
            img = img.resize((18, 18), Image.LANCZOS)
            img.save(MIC_IDLE)
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            m = size * 0.05
            d.ellipse((m, m, size - m, size - m), fill=(224, 59, 59, 255))
            _draw_mic_glyph(d, size / 2, size * 0.30, size * 0.40, (255, 255, 255, 255), int(size * 0.055))
            img = img.resize((18, 18), Image.LANCZOS)
            img.save(MIC_REC)
            size = 512
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            r = size * 0.22
            d.rounded_rectangle((0, 0, size, size), radius=r, fill=(64, 107, 217, 255))
            d.rounded_rectangle((0, 0, size, size * 0.55), radius=r, fill=(91, 141, 239, 255))
            d.rectangle((0, size * 0.3, size, size * 0.55), fill=(91, 141, 239, 255))
            d.rounded_rectangle((0, size * 0.3, size, size * 0.6), radius=r * 0.5, fill=(91, 141, 239, 255))
            d.rounded_rectangle((0, 0, size, size), radius=r, outline=(30, 50, 110, 120), width=3)
            _draw_mic_glyph(d, size / 2, size * 0.26, size * 0.48, (255, 255, 255, 255), int(size * 0.055))
            img.save(APP_ICON)
        except Exception as e:
            log("иконки:", e)

    class CocoaUI:
        """Значок микрофона в меню-баре и плашка-индикатор. Все вызовы из любых потоков безопасны."""
        OW, OH = 300, 62

        def __init__(self, cfg):
            self.cfg = cfg
            self.app = None
            from AppKit import (NSStatusBar, NSImage, NSMenu, NSMenuItem, NSFont,
                                NSApplication, NSVariableStatusItemLength)
            make_icons()
            self._img_idle = NSImage.alloc().initWithContentsOfFile_(MIC_IDLE)
            self._img_rec = NSImage.alloc().initWithContentsOfFile_(MIC_REC)
            if self._img_idle is not None:
                self._img_idle.setTemplate_(True)
            if self._img_rec is not None:
                self._img_rec.setTemplate_(False)

            self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
            if self._img_idle is not None:
                self.item.button().setImage_(self._img_idle)
            self.item.setToolTip_("Диктовка – русский голосовой ввод")

            menu = NSMenu.alloc().init()
            self.m_record = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Запись", "toggle:", "")
            self.m_record.setTarget_(self)
            m_cfg = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Настройки (config.json)", "openCfg:", "")
            m_cfg.setTarget_(self)
            m_log = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Журнал (diktovka.log)", "openLog:", "")
            m_log.setTarget_(self)
            m_quit = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Выход", "quit:", "")
            m_quit.setTarget_(self)
            for m in (self.m_record, m_cfg, m_log, NSMenuItem.separatorItem(), m_quit):
                menu.addItem_(m)
            self.item.setMenu_(menu)

            self._panel = None
            self._label = None
            self._level = None
            self._dot = None
            self._spinner = None

        def toggle_(self, sender):
            try:
                if self.app:
                    self.app.toggle()
            except Exception:
                log("menu:", traceback.format_exc()[-300:])

        def openCfg_(self, sender):
            subprocess.Popen(["open", CFG_PATH])

        def openLog_(self, sender):
            subprocess.Popen(["open", LOG])

        def quit_(self, sender):
            os._exit(0)

        def _main(self, fn):
            from Foundation import NSOperationQueue
            def safe():
                try:
                    fn()
                except Exception:
                    log("UI:", traceback.format_exc()[-300:])
            NSOperationQueue.mainQueue().addOperationWithBlock_(safe)

        def _make_panel(self):
            from AppKit import (NSPanel, NSColor, NSTextField, NSView, NSScreen,
                                NSFont, NSFontWeightMedium)
            W, H = self.OW, self.OH
            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                ((0, 0), (W, H)), 1 << 7, 2, False)
            panel.setFloatingPanel_(True)
            panel.setLevel_(25)  # NSStatusWindowLevel — над обычными окнами, под строкой меню
            panel.setBackgroundColor_(NSColor.clearColor())
            panel.setOpaque_(False)
            panel.setHasShadow_(True)
            panel.setIgnoresMouseEvents_(True)
            panel.setAlphaValue_(0.0)

            # честная прозрачность: тонкий тёмный слой (крутится alpha ниже)
            bg = NSView.alloc().initWithFrame_(((0, 0), (W, H)))
            bg.setWantsLayer_(True)
            bg.layer().setBackgroundColor_(NSColor.blackColor().colorWithAlphaComponent_(0.28).CGColor())
            bg.layer().setCornerRadius_(22)
            bg.layer().setMasksToBounds_(True)
            bg.layer().setBorderWidth_(0.5)
            bg.layer().setBorderColor_(NSColor.whiteColor().colorWithAlphaComponent_(0.16).CGColor())
            panel.setContentView_(bg)

            dot = _DotView.alloc().initWithFrame_(((20, 26), (10, 10)))
            bg.addSubview_(dot)
            spinner = _SpinnerView.alloc().initWithFrame_(((17, 23), (16, 16)))
            spinner.setHidden_(True)
            bg.addSubview_(spinner)
            label = NSTextField.alloc().initWithFrame_(((38, 19), (W - 50, 24)))
            try:
                label.setFont_(NSFont.systemFontOfSize_weight_(13, NSFontWeightMedium))
            except Exception:
                label.setFont_(NSFont.systemFontOfSize_(13))
            label.setAlignment_(0)
            bg.addSubview_(label)
            level = _LevelBar.alloc().initWithFrame_(((16, 11), (W - 32, 5)))
            bg.addSubview_(level)

            screen = NSScreen.mainScreen().frame()
            panel.setFrameOrigin_(((screen.size.width - W) / 2, 120))
            self._panel, self._label, self._level, self._dot, self._spinner = panel, label, level, dot, spinner

        def _dot_color(self, color):
            if color == "red":
                return NSColor.systemRedColor()
            if color == "green":
                return NSColor.systemGreenColor()
            if color == "blue":
                return NSColor.systemBlueColor()
            if color == "error":
                return NSColor.systemOrangeColor()
            return NSColor.tertiaryLabelColor()

        def say(self, text, color=None, level=None):
            STATE = {"red": "record", "blue": "busy", "green": "ok", "error": "error", None: "info"}
            state = STATE.get(color, "info")

            def op():
                from AppKit import NSAnimationContext
                if self._panel is None:
                    self._make_panel()
                panel, label = self._panel, self._label
                # белый текст с мягкой тёмной тенью — читается на любом фоне
                from AppKit import NSShadow
                shadow = NSShadow.alloc().init()
                shadow.setShadowOffset_((0, -1.5))
                shadow.setShadowBlurRadius_(4)
                shadow.setShadowColor_(NSColor.blackColor().colorWithAlphaComponent_(0.8))
                attrs = {"NSFont": label.font(), "NSColor": NSColor.whiteColor(), "NSShadow": shadow}
                from Foundation import NSMutableAttributedString
                astr = NSMutableAttributedString.alloc().initWithString_attributes_(text, attrs)
                label.setAttributedStringValue_(astr)
                # симметрия: группа «точка + текст» строго по центру плашки
                tw = astr.size().width
                d, gap = 10.0, 8.0
                total = d + gap + tw
                x0 = max(16.0, (self.OW - total) / 2)
                cy = 34.0
                self._dot.setFrame_(((x0, cy - d / 2), (d, d)))
                self._spinner.setFrame_(((x0 - 3, cy - 8), (16, 16)))
                label.setFrame_(((x0 + d + gap, cy - 12), (min(tw + 4, self.OW - 16), 24)))
                self._dot.setColor_(self._dot_color(color))
                busy = state == "busy"
                if self._spinner.isAnimating() != busy:
                    (self._spinner.startAnimating if busy else self._spinner.stopAnimating)()
                self._spinner.setHidden_(not busy)
                self._dot.setHidden_(busy)
                self._dot.stopPulsing()
                if state == "record":
                    self._dot.startPulsing()
                self._level.setHidden_(state != "record")
                if level is not None:
                    self._level.setValue_(level)
                if not panel.isVisible():
                    panel.setAlphaValue_(0.0)
                    panel.orderFront_(None)
                    def fade_in(ctx):
                        ctx.setDuration_(0.18)
                        panel.animator().setAlphaValue_(1.0)
                    NSAnimationContext.runAnimationGroup_(fade_in)
                else:
                    panel.orderFront_(None)
            self._main(op)

        def hide_panel(self):
            def op():
                from AppKit import NSAnimationContext
                panel = self._panel
                if panel is not None and panel.isVisible():
                    def fade_out(ctx):
                        ctx.setDuration_(0.15)
                        panel.animator().setAlphaValue_(0.0)
                    NSAnimationContext.runAnimationGroup_(fade_out)
                    panel.performSelector_withObject_afterDelay_("orderOut:", None, 0.2)
            self._main(op)

        def set_level(self, v):
            if self._level is not None:
                self._main(lambda: self._level.setValue_(v))

        def set_recording(self, on):
            def op():
                img = self._img_rec if on else self._img_idle
                if img is not None:
                    self.item.button().setImage_(img)
            self._main(op)

        def set_status(self, text):
            self._main(lambda: self.item.button().setTitle_(text or ""))

        def run(self):
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
            nsapp = NSApplication.sharedApplication()
            nsapp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
            log("интерфейс: значок в меню-баре, плашка-индикатор")
            nsapp.run()


# ---------- интерфейс Windows (tkinter + значок в трее) ----------

else:
    import queue

    class TkUI:
        """Стеклянная плашка внизу экрана + значок в системном трее."""

        COLORS = {"red": "#ff6b6b", "blue": "#6bb8ff", "green": "#7ee787", "": "#ffffff"}

        def __init__(self, cfg):
            self.cfg = cfg
            self.app = None
            self._q = queue.Queue()
            self._panel = None
            self._dot = self._text = self._lvl_bg = self._lvl_fg = self._spin = None
            self._spin_on = False
            self._pulse_on = False

        # --- вызовы из ядра (App) — потокобезопасно, через очередь ---
        def say(self, text, color="", level=None):
            self._q.put(("say", text, color, level))

        def hide_panel(self):
            self._q.put(("hide",))

        def set_level(self, v):
            self._q.put(("level", v))

        def set_recording(self, on):
            self._q.put(("rec", on))

        def set_status(self, s):
            pass  # в трее статус не нужен, всё показывает плашка

        def _build(self):
            import tkinter as tk
            self.tk = tk
            root = tk.Tk()
            root.withdraw()
            self.root = root
            W, H = 360, 64
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            p = tk.Toplevel(root)
            p.overrideredirect(True)
            p.attributes("-topmost", True)
            try:
                p.attributes("-alpha", 0.88)
            except Exception:
                pass
            p.geometry(f"{W}x{H}+{(sw - W) // 2}+{sh - H - 120}")
            c = tk.Canvas(p, width=W, height=H, highlightthickness=0, bd=0, bg="#151517")
            c.pack()
            self._c, self._W = c, W
            # полоса уровня голоса: серая подложка + цветной отрезок
            self._lvl_bg = c.create_line(28, 50, W - 28, 50, fill="#3a3a3f", width=3,
                                         capstyle=tk.ROUND)
            self._lvl_fg = c.create_line(28, 50, 28, 50, fill="#7ee787", width=3,
                                         capstyle=tk.ROUND)
            # спиннер «распознаю»
            self._spin = c.create_arc(W / 2 - 9, 14, W / 2 + 9, 32, start=0, extent=285,
                                      style="arc", outline="#6bb8ff", width=3)
            c.itemconfigure(self._spin, state="hidden")
            # текст и точка-светофор (позиции выставит _layout_group)
            self._text = c.create_text(W / 2, 24, text="", fill="#ffffff",
                                       font=("Segoe UI", 11))
            self._dot = c.create_oval(0, 0, 12, 12, fill="#9a9aa0", outline="")
            self._panel = p

        def _layout_group(self):
            # ровная симметрия: «точка + текст» — единая группа, строго по центру
            c, W = self._c, self._W
            bb = c.bbox(self._text)
            if not bb:
                return
            w = bb[2] - bb[0]
            gap, d = 12, 12
            total = w + gap + d
            x0 = W / 2 - total / 2
            c.coords(self._dot, x0, 18, x0 + d, 18 + d)
            c.coords(self._text, x0 + d + gap + w / 2, 24)

        def _pulse(self):
            if not self._pulse_on:
                return
            cur = self._c.itemcget(self._dot, "fill")
            self._c.itemconfigure(self._dot,
                                  fill="#ff8a80" if cur == "#ff3d3d" else "#ff3d3d")
            self.root.after(450, self._pulse)

        def _spin_tick(self):
            if not self._spin_on:
                return
            c = self._c
            c.itemconfigure(self._spin,
                            start=(float(c.itemcget(self._spin, "start")) - 12) % 360)
            self.root.after(30, self._spin_tick)

        def _set_level(self, v):
            W = self._W
            v = max(0.0, min(1.0, v))
            self._c.coords(self._lvl_fg, 28, 50, 28 + v * (W - 56), 50)
            self._c.itemconfigure(self._lvl_fg,
                                  fill="#ff6b6b" if v > 0.85 else
                                       ("#ffd54f" if v > 0.6 else "#7ee787"))

        def _poll(self):
            try:
                while True:
                    item = self._q.get_nowait()
                    op = item[0]
                    if op == "say":
                        _, text, color, level = item
                        self._c.itemconfigure(self._text, text=text,
                                              fill=self.COLORS.get(color, "#ffffff"))
                        self._layout_group()
                        if level is not None:
                            self._set_level(level)
                        busy = "спозна" in text
                        if busy != self._spin_on:
                            self._spin_on = busy
                            self._c.itemconfigure(
                                self._spin, state="normal" if busy else "hidden")
                            if busy:
                                self._spin_tick()
                        self._panel.deiconify()
                        self._panel.lift()
                    elif op == "hide":
                        self._pulse_on = self._spin_on = False
                        self._c.itemconfigure(self._spin, state="hidden")
                        if self._panel is not None:
                            self._panel.withdraw()
                    elif op == "level":
                        self._set_level(item[1])
                    elif op == "rec":
                        self._pulse_on = bool(item[1])
                        self._c.itemconfigure(self._dot,
                                              fill="#ff3d3d" if item[1] else "#9a9aa0")
                        if item[1]:
                            self._pulse()
                    elif op == "tray_toggle" and self.app is not None:
                        self.app.toggle()
                    elif op == "tray_quit":
                        self._quit()
            except queue.Empty:
                pass
            self.root.after(40, self._poll)

        def _quit(self):
            log("выход из трея")
            os._exit(0)

        def _tray(self):
            try:
                import pystray
                from PIL import Image, ImageDraw
                img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                d = ImageDraw.Draw(img)
                d.ellipse((10, 10, 54, 54), fill=(255, 61, 61, 255))
                menu = pystray.Menu(
                    pystray.MenuItem("Запись (вкл/выкл)", lambda *a: self._q.put(("tray_toggle",))),
                    pystray.MenuItem("Настройки (config.json)", lambda *a: os.startfile(CFG_PATH)),
                    pystray.MenuItem("Журнал", lambda *a: os.startfile(LOG)),
                    pystray.MenuItem("Выход", lambda *a: self._q.put(("tray_quit",))),
                )
                pystray.Icon("diktovka", img, "Диктовка", menu).run()
            except Exception as e:
                log("трей:", e)

        def run(self):
            self._build()
            threading.Thread(target=self._tray, daemon=True).start()
            self.root.after(40, self._poll)
            log("интерфейс: плашка + значок в трее")
            self.root.mainloop()


# ---------- ядро: запись, распознавание, вставка (одинаково на обеих ОС) ----------

class App:
    def __init__(self, cfg, ui):
        self.cfg = cfg
        self.ui = ui
        self.model = None
        self.recording = False
        self.busy = False
        self.chunks = []
        self.stream = None
        self.target_app = None
        self.lock = threading.Lock()
        self.key = {"cmd_r": "правый ⌘", "cmd_l": "левый ⌘"}.get(cfg["hotkey"], cfg["hotkey"])
        self._peak = 0.0

    # --- запись по удержанию клавиши ---
    def hold_start(self):
        with self.lock:
            if self.busy or self.recording:
                return
            if self.model is None:
                self.ui.say("Загружаю модель…", "blue")
                return
            self.start()

    def hold_stop(self):
        with self.lock:
            if not self.recording or self.busy:
                return
            self.recording = False
            self.busy = True
            threading.Thread(target=self.finish, daemon=True).start()

    # --- переключение кликом по меню/трею ---
    def toggle(self):
        with self.lock:
            if self.busy:
                return
            if self.model is None:
                self.ui.say("Загружаю модель…", "blue")
                return
            if not self.recording:
                self.start()
            else:
                self.recording = False
                self.busy = True
                threading.Thread(target=self.finish, daemon=True).start()

    def start(self):
        import sounddevice as sd
        if PLATFORM == "mac":
            self.target_app = frontmost_name()
        else:
            self.target_app = current_front_hwnd()
        self.chunks = []
        self._peak = 0.0

        def cb(indata, frames, t, status):
            import numpy as np
            p = float(np.abs(indata).max())
            if p > self._peak:
                self._peak = p
            self.chunks.append(indata[:, 0].copy())

        try:
            self.stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=cb)
            self.stream.start()
        except Exception as e:
            log("микрофон:", e)
            self.ui.say("Нет микрофона — проверьте разрешения в настройках системы", "red")
            threading.Timer(5, self.ui.hide_panel).start()
            return
        self.recording = True
        log("запись началась")
        play_sound(self.cfg, "start")
        self.ui.set_recording(True)
        self.ui.say(f"Говорите…   отпустите {self.key} — вставить", "red")
        threading.Thread(target=self._level_loop, daemon=True).start()
        # сторож: если отпускание клавиши потерялось — автостоп через 3 минуты
        watchdog = threading.Timer(180.0, self._watchdog_stop)
        watchdog.daemon = True
        watchdog.start()

    def _watchdog_stop(self):
        with self.lock:
            stuck = self.recording and not self.busy
        if stuck:
            log("автостоп: отпускание клавиши не дошло")
            self.hold_stop()

    def _level_loop(self):
        # плавно показываем уровень голоса на плашке, ~12 раз в секунду
        while self.recording:
            v = min(1.0, self._peak)
            self.ui.set_level(v)
            self._peak *= 0.55
            time.sleep(0.08)

    def finish(self):
        import numpy as np
        try:
            self.stream.stop(); self.stream.close()
            play_sound(self.cfg, "stop")
            self.ui.set_recording(False)
            audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype="float32")
            peak = float(np.abs(audio).max()) if len(audio) else 0.0
            log(f"запись остановлена, {len(audio)/16000:.1f} с, уровень {peak:.4f}")
            if peak > 1.0:  # перегрузка микрофона – приводим к рабочему диапазону
                audio = (audio / peak * 0.98).astype("float32")
            if len(audio) < 16000 * 0.3:
                self.ui.say("Пусто – ничего не записалось")
                threading.Timer(2.5, self.ui.hide_panel).start()
                return
            # защита от галлюцинаций модели на тишине: без речи распознавание не запускаем
            speech = trim_silence(audio)
            if len(speech) < int(16000 * 0.4) or float(np.abs(speech).max()) < 0.05:
                log("нет речи — распознавание пропущено")
                self.ui.say("Не расслышал — зажмите клавишу и говорите чуть дольше")
                threading.Timer(3, self.ui.hide_panel).start()
                return
            self.ui.say("Распознаю…", "blue", level=1.0)
            t0 = time.time()
            text = transcribe(self.model, audio, self.cfg)
            log(f"{len(audio)/16000:.1f} с речи → {time.time()-t0:.1f} с: {text}")
            if not text:
                play_sound(self.cfg, "error")
                self.ui.say("Не расслышал – попробуйте ещё раз")
                threading.Timer(3, self.ui.hide_panel).start()
                return
            self.paste(text)
            self.ui.say("Вставлено", "green")
            threading.Timer(2, self.ui.hide_panel).start()
        except Exception:
            log(traceback.format_exc())
            play_sound(self.cfg, "error")
            self.ui.say("Ошибка — подробности в журнале", "red")
        finally:
            self.busy = False

    def paste(self, text):
        import pyperclip
        log(f"вставка: цель={self.target_app!r}, текст={text!r}")
        try:
            old = pyperclip.paste()
        except Exception:
            old = None
        pyperclip.copy(text)
        if PLATFORM == "mac":
            if self.target_app and frontmost_name() != self.target_app:
                activate_target(self.target_app)
                time.sleep(0.15)
            log(f"активное приложение сейчас: {frontmost_name()!r}")
        elif self.target_app:
            activate_target(self.target_app)
            time.sleep(0.15)
        time.sleep(0.05)
        try:
            send_paste()
        except Exception as e:
            log("вставка не удалась:", e)
            if PLATFORM == "mac":
                self.ui.say("Нужно разрешение «Универсальный доступ» для Диктовки", "red")
            else:
                self.ui.say("Вставка не удалась — запустите Диктовку от имени администратора", "red")
            return
        log("команда вставки отправлена")
        if self.cfg.get("auto_send"):
            time.sleep(0.15)
            try:
                send_enter()
            except Exception as e:
                log("enter:", e)
        time.sleep(1.2)
        if old is not None:
            try:
                pyperclip.copy(old)
            except Exception:
                pass

    def load(self):
        try:
            self.ui.set_status("")
            t0 = time.time()
            self.model = load_model(self.cfg)
            kind = self.model[0]
            where = {"mlx": "MLX (Apple Silicon)",
                     "fwhisper-cuda": "CUDA (NVIDIA)",
                     "fwhisper": "процессор"}[kind]
            log(f"модель {self.cfg['model']} загружена ({where}) за {time.time()-t0:.1f} с")
            # прогрев: первая транскрипция компилирует ядра — делаем её на старте, а не при диктовке
            import numpy as np
            warm = np.zeros(16000 * 2, dtype="float32")
            t1 = time.time()
            transcribe(self.model, warm, self.cfg)
            log(f"прогрев модели за {time.time()-t1:.1f} с")
            play_sound(self.cfg, "ready")
            self.ui.set_status("")
            self.ui.say(f"Диктовка готова — зажмите {self.key} и говорите", "green")
            threading.Timer(4, self.ui.hide_panel).start()
        except Exception:
            log(traceback.format_exc())
            self.ui.say("Модель не загрузилась — см. журнал", "red")


# ---------- горячая клавиша (pynput, одинаково на обеих ОС) ----------

NAMED_KEYS = {
    "space": "space", "enter": "enter", "return": "enter", "tab": "tab",
    "escape": "esc", "esc": "esc", "delete": "delete", "backspace": "backspace",
    "insert": "insert", "home": "home", "end": "end", "pageup": "page_up",
    "pagedown": "page_down", "up": "up", "down": "down", "left": "left", "right": "right",
}
# "cmd" = любой Command; "cmd_r" = только правый (и так же для ctrl/alt/shift)
MOD_KEYS = {
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "option": "alt",
    "shift": "shift", "cmd": "cmd", "command": "cmd", "win": "cmd", "super": "cmd",
    "ctrl_l": "ctrl_l", "ctrl_r": "ctrl_r", "alt_l": "alt_l", "alt_r": "alt_r",
    "shift_l": "shift_l", "shift_r": "shift_r", "cmd_l": "cmd_l", "cmd_r": "cmd_r",
}
MOD_SIDES = {"ctrl": 1, "alt": 1, "shift": 1, "cmd": 1}


def parse_hotkey(spec):
    # 'f8', 'cmd_r' или 'ctrl+alt+space' -> (модификаторы, основная клавиша)
    from pynput import keyboard
    mods, key = set(), None
    for part in spec.lower().replace(" ", "").split("+"):
        if not part:
            continue
        if part in MOD_KEYS:
            mods.add(MOD_KEYS[part])
        elif part in NAMED_KEYS:
            key = getattr(keyboard.Key, NAMED_KEYS[part])
        elif part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 20:
            key = getattr(keyboard.Key, part)
        elif len(part) == 1:
            key = keyboard.KeyCode.from_char(part)
    if key is None and not mods:
        raise ValueError(f"Не понял клавишу «{spec}» в config.json")
    return mods, key


def mods_satisfied(mods, state):
    for m in mods:
        if m in MOD_SIDES:  # «cmd» = левый или правый
            if not ({m, m + "_l", m + "_r"} & state):
                return False
        elif m not in state:
            return False
    return True


def hotkey_loop(app):
    # глобальная клавиша через pynput
    if PLATFORM == "mac":
        fail_msg = "Горячая клавиша не работает — нет разрешения «Универсальный доступ»"
    else:
        fail_msg = "Горячая клавиша не работает — запустите Диктовку от имени администратора"
    try:
        from pynput import keyboard
        mods, key = parse_hotkey(app.cfg["hotkey"])
        state = set()

        def norm(k):
            if k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l):
                return "ctrl_l"
            if k == keyboard.Key.ctrl_r:
                return "ctrl_r"
            if k in (keyboard.Key.alt, keyboard.Key.alt_l):
                return "alt_l"
            if k in (keyboard.Key.alt_r, keyboard.Key.alt_gr):
                return "alt_r"
            if k in (keyboard.Key.shift, keyboard.Key.shift_l):
                return "shift_l"
            if k == keyboard.Key.shift_r:
                return "shift_r"
            if k in (keyboard.Key.cmd, keyboard.Key.cmd_l):
                return "cmd_l"
            if k == keyboard.Key.cmd_r:
                return "cmd_r"
            return None

        def on_press(k):
            n = norm(k)
            if n:
                state.add(n)
                # горячая клавиша – сам модификатор (например, правый ⌘): зажали – записываем
                if key is None and mods_satisfied(mods, state) and not app.recording and not app.busy:
                    app.hold_start()
                return
            if key is None or not mods_satisfied(mods, state):
                return  # ждём пока нажаты все модификаторы
            try:
                if key == k or (isinstance(key, keyboard.KeyCode) and key.char == getattr(k, "char", None)):
                    app.toggle()
            except Exception:
                pass

        def on_release(k):
            n = norm(k)
            if not n:
                return
            state.discard(n)
            # отпустили клавишу-модификатор – заканчиваем запись
            if key is None and app.recording:
                wanted = set()
                for m in mods:
                    wanted |= {m, m + "_l", m + "_r"} if m in MOD_SIDES else {m}
                if n in wanted:
                    app.hold_stop()

        for attempt in range(1000):
            try:
                with keyboard.Listener(on_press=on_press, on_release=on_release,
                                       suppress=False) as listener:
                    listener.join()
                log(f"слушатель клавиш остановлен (попытка {attempt + 1}) — перезапуск")
            except Exception:
                log(traceback.format_exc()[-300:])
            time.sleep(2)
        app.ui.say(fail_msg, "red")
    except Exception:
        log(traceback.format_exc())
        app.ui.say(fail_msg, "red")


# ---------- автообновление с GitHub ----------

def _version_tuple(s):
    out = []
    for part in str(s).strip().split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def check_update(cfg):
    # сравниваем VERSION в ветке репозитория; если там новее — меняем файл и перезапускаемся
    try:
        import urllib.request
        channel = cfg.get("update_channel", "main")
        url = f"https://raw.githubusercontent.com/{REPO}/{channel}/VERSION"
        with urllib.request.urlopen(url, timeout=10) as r:
            remote = r.read().decode("utf-8", "replace").strip()
        log(f"проверка обновлений: локально {APP_VERSION}, на сервере {remote}")
        if not remote or _version_tuple(remote) <= _version_tuple(APP_VERSION):
            return
        self_url = f"https://raw.githubusercontent.com/{REPO}/{channel}/diktovka.py"
        tmp = os.path.join(BASE, ".diktovka_update")
        with urllib.request.urlopen(self_url, timeout=120) as r:
            data = r.read()
        if len(data) < 10 * 1024 or not data.lstrip().startswith(b"# \xd0\x94\xd0\xb8\xd0\xba\xd1\x82\xd0\xbe\xd0\xb2\xd0\xba\xd0\xb0"):
            log("обновление отклонено: файл подозрительный")
            if os.path.exists(tmp):
                os.remove(tmp)
            return
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, os.path.abspath(__file__))
        log(f"обновление до {remote} скачано — перезапуск")
        if PLATFORM == "mac":
            try:
                uid = os.getuid()
                subprocess.run(["launchctl", "kickstart", "-k", f"gui/{uid}/com.halo.diktovka"],
                               timeout=10)
                time.sleep(3)
            except Exception:
                pass
        else:
            try:
                subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0) & 0)
            except Exception:
                pass
            os._exit(0)
        os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)])
    except Exception as e:
        log("обновление:", e)


def update_loop(cfg):
    while True:
        try:
            check_update(cfg)
        except Exception:
            log(traceback.format_exc()[-300:])
        time.sleep(6 * 3600)


# ---------- запуск ----------

def main():
    cfg = load_cfg()

    if "--download" in sys.argv:          # скачать модель и проверить (вызывает установщик)
        m = load_model(cfg)
        where = {"mlx": "MLX", "fwhisper-cuda": "CUDA", "fwhisper": "процессор"}[m[0]]
        print(f"OK: модель {cfg['model']} работает ({where})")
        return
    if "--test" in sys.argv:              # проверка на файле: --test file.wav
        import numpy as np, subprocess
        wav = sys.argv[sys.argv.index("--test") + 1]
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", wav, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                             capture_output=True).stdout
        audio = np.frombuffer(raw, dtype="float32")
        m = load_model(cfg)
        where = {"mlx": "MLX", "fwhisper-cuda": "CUDA", "fwhisper": "процессор"}[m[0]]
        t0 = time.time()
        text = transcribe(m, audio, cfg)
        print(f"{where}, {len(audio)/16000:.1f} с речи распознано за {time.time()-t0:.1f} с: {text}")
        return
    if "--demo" in sys.argv:              # сквозная проверка: запись → распознавание → вставка
        if PLATFORM != "mac":
            print("Демо-режим пока только на macOS")
            return
        import numpy as np
        import subprocess as sp
        import sounddevice as sd
        print("Демо: записываю 7 секунд. Сейчас через колонки прозвучит фраза…")
        model = load_model(cfg)
        chunks = []

        def cb(indata, frames, t, status):
            chunks.append(indata[:, 0].copy())

        stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=cb)
        stream.start()
        time.sleep(0.3)
        sp.Popen(["say", "Привет! Это автоматическая проверка русской диктовки."],
                 stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        sd.sleep(7000)
        stream.stop(); stream.close()
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        print(f"записано {len(audio)/16000:.1f} с, уровень {peak:.3f}")
        if peak > 1.0:
            audio = (audio / peak * 0.98).astype("float32")
        t0 = time.time()
        text = transcribe(model, audio, cfg)
        print(f"распознано за {time.time()-t0:.1f} с: {text!r}")
        import pyperclip
        pyperclip.copy(text)
        target = frontmost_name()
        print(f"активное приложение: {target!r} — вставляю")
        time.sleep(0.2)
        send_paste()
        print("вставлено")
        return

    if PLATFORM == "win" and sys.stdout is None:   # pythonw: вывод некуда
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull

    if not single_instance():
        log("уже запущена – значок в меню-баре/трее")
        return

    ui = CocoaUI(cfg) if PLATFORM == "mac" else TkUI(cfg)   # интерфейс — в главном потоке
    app = App(cfg, ui)
    ui.app = app
    threading.Thread(target=hotkey_loop, args=(app,), daemon=True).start()
    threading.Thread(target=app.load, daemon=True).start()
    threading.Thread(target=update_loop, args=(cfg,), daemon=True).start()
    log("запущена, клавиша", cfg["hotkey"], "версия", APP_VERSION)
    ui.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
