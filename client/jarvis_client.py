"""JARVIS Windows client.

Owns the microphone, the speaker, and the screen. Talks to the Pipecat agent
over a raw WebSocket (no ICE, no SDP, no HTTPS -- this is loopback) and also
hosts a small HTTP endpoint the agent calls when the LLM invokes a tool.

Wire format, both directions: headerless mono PCM.
  up   16-bit signed little-endian @ 16 kHz
  down 16-bit signed little-endian @ 24 kHz
"""

import asyncio
import base64
import io
import json
import os
import queue
import subprocess
import threading
import time
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import requests
import sounddevice as sd
import websockets

import window_targeting as wt

WS_URL = "ws://127.0.0.1:8765"
UPLOAD_URL = "http://127.0.0.1:8080/api/command"

IN_RATE = 16000
OUT_RATE = 24000
IN_CHUNK = 320          # 20 ms @ 16 kHz
EXEC_PORT = 8766

# How long to keep trying to reach an agent before giving up, and the backoff
# schedule between attempts. Sized against start-agent.sh, which allows 60 s
# for a cold start.
RECONNECT_DEADLINE_SECS = float(os.environ.get("JARVIS_RECONNECT_SECS", "180"))
RECONNECT_BACKOFF = (0.5, 1.0, 2.0, 3.0, 5.0)

AUDIO_OUT_Q: queue.Queue[np.ndarray] = queue.Queue()
MUTE = threading.Event()

# Set JARVIS_CLIENT_VERBOSE=1 for a per-frame audio counter, so "is it actually
# streaming" is answerable without a debugger. Off by default: 50 lines/second
# buries everything else.
CLIENT_VERBOSE = os.environ.get("JARVIS_CLIENT_VERBOSE", "0") == "1"


def _log(msg: str) -> None:
    """Timestamped status line, to match the agent's log format."""
    print(f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}"
          f"  CLIENT  {msg}", flush=True)


def _audio_stats() -> None:
    """Report what the live path is actually moving, every 5s."""
    sent = audio_stats["sent"]
    got = audio_stats["got"]
    audio_stats["sent"] = audio_stats["got"] = 0
    _log(f"audio  up {sent} frames / 5s ({sent * 0.02:.1f}s)   "
         f"down {got} chunks / 5s ({got * 1024 / OUT_RATE:.1f}s)")


def _stats_loop() -> None:
    while True:
        time.sleep(5)
        _audio_stats()


audio_stats = {"sent": 0, "got": 0}


# --------------------------------------------------------------------------
# screen + action execution
# --------------------------------------------------------------------------
def capture_screen_b64(max_side: int = 512) -> str:
    """Grab the screen and return a small JPEG as base64.

    max_side is 512 rather than 800 because the vision model charges by pixels: on
    the reference machine the same screenshot cost 445 ms of prompt evaluation at
    800 px against 117 ms at 512 px, and the model still reads the layout and the
    application names. Raise it (or pass a larger value) if small UI text starts
    coming back wrong.
    """
    import mss
    from PIL import Image

    try:
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            img.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=60)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        return f"__error__:{e}"


def active_windows() -> list[str]:
    try:
        cmd = (
            "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} "
            "| Select-Object -ExpandProperty MainWindowTitle -First 8"
        )
        res = subprocess.run(["powershell", "-Command", cmd],
                             capture_output=True, text=True, timeout=10)
        return [t.strip() for t in res.stdout.splitlines() if t.strip()]
    except Exception:
        return []


def _ps(script: str) -> str:
    wrapped = f"[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; {script}"
    res = subprocess.run(["powershell", "-NoProfile", "-Command", wrapped],
                         capture_output=True, text=True, timeout=20,
                         encoding="utf-8", errors="replace")
    return (res.stdout or res.stderr).strip()


def _vol(delta: int, times: int = 5) -> dict:
    """Nudge system volume. 175 = up, 174 = down."""
    ch = "175" if delta > 0 else "174"
    _ps(f"$w = New-Object -ComObject WScript.Shell; 1..{times} | "
        f"% {{ $w.SendKeys([char]{ch}) }}")
    return {"ok": True,
            "message": "Volume turned up." if delta > 0 else "Volume turned down."}


def execute(action: str, target: str = "") -> dict:
    """The single place that is allowed to touch the machine.

    Browser-relative actions resolve a real browser window first. Previously
    they sent ctrl+t to whatever had focus, which is how "open a new tab"
    opened a tab in the IDE running the test harness. Every action now reports
    what it actually did, and browser actions fail loudly rather than claiming
    success when no browser was found.
    """
    import pyautogui

    pyautogui.FAILSAFE = True

    def need_browser():
        win, launched = wt.ensure_browser()
        if win is None:
            return None, {
                "ok": False,
                "error": "No browser window found, and none could be launched. "
                         "Open Chrome, Edge or Firefox and try again.",
            }
        msg = f"({wt.describe_windows()})"
        if launched:
            msg = f"launched a browser. {msg}"
        return win, msg

    if action == "LIST_WINDOWS":
        wins = active_windows()
        return {"ok": True, "message": "; ".join(wins) if wins else "No windows reported."}

    if action == "LIST_BROWSERS":
        return {"ok": True, "message": wt.describe_windows()}

    if action == "DESCRIBE_SCREEN":
        b64 = capture_screen_b64()
        if b64.startswith("__error__:"):
            return {"ok": False, "error": b64.split(":", 1)[1]}
        if not b64:
            return {"ok": False, "error": "screen capture returned nothing"}
        try:
            r = requests.post(UPLOAD_URL, json={
                "user_text": f"Look at my screen and answer: {target}",
                "screen_context": "Active Windows: " + ", ".join(active_windows()),
                "image_b64": b64,
                # The agent speaks the reply itself with its own (ONNX) Kokoro.
                # Asking the backend for audio as well made it synthesise 5-6 s of
                # torch TTS whose WAV was then thrown away, on the critical path.
                "want_audio": False,
            }, timeout=120)
            r.raise_for_status()
            return {"ok": True, "message": r.json().get("text", "")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    try:
        # ---- browser-relative: focus a real browser first -------------------
        if action in ("OPEN_TAB", "CLOSE_TAB", "FOCUS_SEARCH_BAR",
                      "NEXT_TAB", "PREVIOUS_TAB", "GO_BACK", "GO_FORWARD"):
            win, err = need_browser()
            if win is None:
                return err
            name = wt.display_name(win["name"])

            def current_title() -> str:
                titles = [w["clean"] for w in wt.browser_windows(refresh=True)]
                return titles[0] if titles else "unknown"

            def send(*keys):
                with keyboard_action():
                    pyautogui.hotkey(*keys)
                time.sleep(wt.VERIFY_SETTLE_SECS)

            if action == "OPEN_TAB":
                send("ctrl", "t")
                return {"ok": True,
                        "message": f"Opened a new tab in {name}. "
                                   f"It now shows: {current_title()}"}
            if action == "CLOSE_TAB":
                send("ctrl", "w")
                return {"ok": True,
                        "message": f"Closed a tab in {name}. "
                                   f"It now shows: {current_title()}"}
            if action == "NEXT_TAB":
                send("ctrl", "tab")
                return {"ok": True,
                        "message": f"Switched to the next tab in {name}. "
                                   f"It now shows: {current_title()}"}
            if action == "PREVIOUS_TAB":
                send("ctrl", "shift", "tab")
                return {"ok": True,
                        "message": f"Switched to the previous tab in {name}. "
                                   f"It now shows: {current_title()}"}
            if action == "GO_BACK":
                send("alt", "left")
                return {"ok": True,
                        "message": f"Went back in {name}. "
                                   f"It now shows: {current_title()}"}
            if action == "GO_FORWARD":
                send("alt", "right")
                return {"ok": True,
                        "message": f"Went forward in {name}. "
                                   f"It now shows: {current_title()}"}
            with keyboard_action():
                pyautogui.hotkey("ctrl", "l")
            return {"ok": True, "message": f"Focused the address bar in {name}."}

        if action == "OPEN_BROWSER":
            # ensure_browser, not focus_browser: with no browser running,
            # focus_browser returns None and the old code then reported
            # "Opened a browser" without having opened anything.
            win, launched = wt.ensure_browser()
            if win is None:
                return {"ok": False,
                        "error": "Could not find or launch a browser. "
                                 f"Tried the usual install paths. {wt.describe_windows()}"}
            name = wt.display_name(win["name"])
            verb = "Opened" if launched else "Brought"
            return {"ok": True,
                    "message": f"{verb} {name}. {wt.describe_windows()}"}

        if action == "VISUAL_CLICK":
            w, h = pyautogui.size()
            norm = _ground(target, w, h)
            if norm is None:
                return {"ok": False, "error": f"could not locate {target!r}"}
            pyautogui.moveTo(int(norm[0]), int(norm[1]), duration=0.2)
            pyautogui.click()
            return {"ok": True,
                    "message": f"Clicked at ({int(norm[0])}, {int(norm[1])}) "
                               f"looking for {target!r}."}

        # ---- application windows --------------------------------------------
        if action == "LIST_APPS":
            return {"ok": True, "message": wt.describe_apps()}

        if action == "FOCUS_APP":
            if not target.strip():
                return {"ok": False, "error": "FOCUS_APP needs an app name"}
            win = wt.find_app(target)
            if win is None:
                return {"ok": False,
                        "error": f"no open window matching {target!r}. "
                                 f"{wt.describe_apps()}"}
            if not wt.focus_hwnd(win["hwnd"]):
                return {"ok": False, "error": f"could not focus {win['name']}"}
            time.sleep(0.25)
            return {"ok": True,
                    "message": f"Switched to {win['name']}: {win['title'][:60]}"}

        if action in ("MINIMIZE_WINDOW", "MAXIMIZE_WINDOW", "RESTORE_WINDOW"):
            # No target means the app named in the request; falling back to the
            # foreground window keeps "minimise it" working after the user has
            # already clicked.
            state = {"MINIMIZE_WINDOW": "minimize",
                     "MAXIMIZE_WINDOW": "maximize",
                     "RESTORE_WINDOW": "restore"}[action]
            win = wt.find_app(target) if target.strip() else None
            if win is None and not target.strip():
                win = wt.foreground_window()
            if win is None:
                return {"ok": False,
                        "error": f"no window to {state}"
                                 + (f" matching {target!r}" if target.strip() else "")
                                 + f". {wt.describe_apps()}"}
            if not wt.set_window_state(win["hwnd"], state):
                return {"ok": False, "error": f"could not {state} {win['name']}"}
            time.sleep(0.2)
            return {"ok": True,
                    "message": f"{state.capitalize()}d {win['name']}."}

        # ---- typing and keys -------------------------------------------------
        if action == "TYPE_TEXT":
            text = target.strip()
            if not text:
                return {"ok": False, "error": "TYPE_TEXT needs the text to type"}
            with keyboard_action():
                pyautogui.typewrite(text, interval=0.01)
            return {"ok": True, "message": f"Typed: {text}"}
        if action == "PRESS_KEY":
            key = _norm_key(target)
            if key is None:
                return {"ok": False,
                        "error": f"unknown key {target!r}. "
                                 f"Supported: {', '.join(sorted(KEY_ALIASES))}"}
            with keyboard_action():
                pyautogui.press(key)
            return {"ok": True, "message": f"Pressed {key}."}

        # ---- non-browser, focus-independent --------------------------------
        if action == "NAVIGATE_YOUTUBE":
            webbrowser.open("https://youtube.com")
            return {"ok": True, "message": "Opened YouTube in the default browser."}
        if action == "SCROLL_DOWN":
            pyautogui.scroll(-600)
            return {"ok": True, "message": "Scrolled down."}
        if action == "SCROLL_UP":
            pyautogui.scroll(600)
            return {"ok": True, "message": "Scrolled up."}
        if action == "SCROLL_LEFT":
            pyautogui.hscroll(-600)
            return {"ok": True, "message": "Scrolled left."}
        if action == "SCROLL_RIGHT":
            pyautogui.hscroll(600)
            return {"ok": True, "message": "Scrolled right."}
        if action == "LOWER_BRIGHTNESS":
            _ps("(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightnessMethods)"
                ".WmiSetBrightness(1, 30)")
            return {"ok": True, "message": "Lowered the screen brightness."}
        if action == "RAISE_BRIGHTNESS":
            _ps("(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightnessMethods)"
                ".WmiSetBrightness(1, 80)")
            return {"ok": True, "message": "Raised the screen brightness."}
        if action == "LOWER_VOLUME":
            return _vol(-1)
        if action == "RAISE_VOLUME":
            return _vol(+1)
        if action == "MUTE_VOLUME":
            _ps("$w = New-Object -ComObject WScript.Shell; $w.SendKeys([char]173)")
            return {"ok": True, "message": "Muted the volume."}
        if action == "OPEN_TERMINAL":
            os_start("start wt")
            return {"ok": True, "message": "Opened a new terminal window."}
        if action == "LOCK_WORKSTATION":
            os_start("rundll32.exe user32.dll,LockWorkStation")
            return {"ok": True, "message": "Locking the workstation now."}

        # ---- media transport -------------------------------------------------
        # pyautogui's own media keys (present in KEYBOARD_KEYS), so no extra
        # dependency and the standard Windows media path is used. Focus-independent
        # by design: "pause the music" has to work whatever is in front, which is
        # also why these need no target.
        if action in MEDIA_KEYS:
            with keyboard_action():
                pyautogui.press(MEDIA_KEYS[action])
            return {"ok": True, "message": MEDIA_MESSAGES[action]}

        # ---- clipboard -------------------------------------------------------
        if action == "CLIPBOARD_READ":
            text = _clipboard_get()
            if text is None:
                return {"ok": False, "error": "could not read the clipboard"}
            if not text.strip():
                return {"ok": True, "message": "The clipboard is empty."}
            return {"ok": True, "message": f"The clipboard contains: {text[:400]}"}
        if action == "CLIPBOARD_WRITE":
            text = target.strip()
            if not text:
                return {"ok": False, "error": "CLIPBOARD_WRITE needs the text to copy"}
            if not _clipboard_set(text):
                return {"ok": False, "error": "could not write to the clipboard"}
            return {"ok": True, "message": f"Copied {text[:80]!r} to the clipboard."}

        # ---- open a URL ------------------------------------------------------
        if action == "OPEN_URL":
            url = _normalise_url(target)
            if url is None:
                return {"ok": False,
                        "error": f"{target!r} is not an http or https address"}
            webbrowser.open(url)
            return {"ok": True, "message": f"Opened {url} in the default browser."}

        # ---- screenshot to a file -------------------------------------------
        if action == "SAVE_SCREENSHOT":
            path, err = _save_screenshot(target.strip())
            if path is None:
                return {"ok": False, "error": err}
            return {"ok": True, "message": f"Saved a screenshot to {path}"}

        # ---- window snapping -------------------------------------------------
        if action in ("SNAP_LEFT", "SNAP_RIGHT"):
            side = "left" if action == "SNAP_LEFT" else "right"
            win = wt.find_app(target) if target.strip() else wt.foreground_window()
            if win is None:
                return {"ok": False,
                        "error": "no window to snap"
                                 + (f" matching {target!r}" if target.strip() else "")
                                 + f". {wt.describe_apps()}"}
            if target.strip() and not wt.focus_hwnd(win["hwnd"]):
                return {"ok": False, "error": f"could not focus {win['name']}"}
            time.sleep(0.25)
            with keyboard_action():
                pyautogui.hotkey("win", side)
            time.sleep(wt.VERIFY_SETTLE_SECS)
            return {"ok": True, "message": f"Snapped {win['name']} to the {side} half."}

        return {"ok": False, "error": f"unhandled action {action!r}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def os_start(cmd: str) -> None:
    import os

    os.system(cmd)


@contextmanager
def keyboard_action():
    """Run a keyboard-only action with pyautogui's failsafe suspended.

    pyautogui.FAILSAFE aborts any call -- including press(), hotkey() and
    typewrite() -- when the cursor sits in a screen corner. That is the right
    default for someone driving the machine by hand, and the wrong one here: the
    user is talking, not pointing, so their cursor is frequently parked in a
    corner, and every keystroke the agent tried would fail with
    FailSafeException. Observed doing exactly that to ctrl+shift+tab.

    Only keyboard actions are wrapped. Anything that moves or clicks the mouse
    keeps the failsafe, since that is the case it actually protects against.

    pyautogui is imported here rather than at module scope to match the rest of
    the file, which defers every heavy import until it is needed.
    """
    import pyautogui

    was = pyautogui.FAILSAFE
    pyautogui.FAILSAFE = False
    try:
        yield
    finally:
        pyautogui.FAILSAFE = was


MEDIA_KEYS = {
    "MEDIA_PLAY_PAUSE": "playpause",
    "MEDIA_NEXT": "nexttrack",
    "MEDIA_PREV": "prevtrack",
    "MEDIA_STOP": "stop",
}
MEDIA_MESSAGES = {
    "MEDIA_PLAY_PAUSE": "Toggled play/pause.",
    "MEDIA_NEXT": "Skipped to the next track.",
    "MEDIA_PREV": "Went back a track.",
    "MEDIA_STOP": "Stopped playback.",
}


def _clipboard_get() -> str | None:
    """Read the clipboard as text, or None if it could not be read.

    win32clipboard rather than pyperclip because the clipboard is routinely held
    open for a few milliseconds by whatever is copying; the retry loop turns that
    from an exception the user hears about into a short wait. An empty string is a
    real answer (the clipboard holds something that is not text, or nothing), which
    is why "could not read" and "empty" are different returns.
    """
    import win32clipboard

    for _ in range(5):
        try:
            win32clipboard.OpenClipboard()
            try:
                if not win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                    return ""
                return win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
        except Exception:
            time.sleep(0.1)
    return None


def _clipboard_set(text: str) -> bool:
    import win32clipboard

    for _ in range(5):
        try:
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
            finally:
                win32clipboard.CloseClipboard()
            return True
        except Exception:
            time.sleep(0.1)
    return False


def _normalise_url(target: str) -> str | None:
    """Accept only http(s), adding the scheme to a bare host.

    Deliberately narrow. The target comes from a language model, and handing an
    arbitrary string to webbrowser.open() means whatever the shell association
    decides to do with it -- local files, custom protocol handlers, and their side
    effects. Anything that is not plainly a web address is refused instead of
    guessed at.
    """
    q = (target or "").strip().strip("<>\"'")
    if not q:
        return None
    low = q.lower()
    if low.startswith(("http://", "https://")):
        return q
    if "://" in low:
        return None
    if " " not in q and "." in q:
        return "https://" + q
    return None


def _screenshot_dir() -> str:
    """First writable screenshot directory, preferring the obvious one.

    ~/Pictures/Jarvis is what anyone would expect, but a redirected or
    OneDrive-backed Pictures folder can be a reparse point whose target is missing
    -- the directory exists while creating a child inside it fails with WinError 2,
    which is what happens on the reference machine. So each candidate is tried in
    turn and the one that worked is what gets reported back, rather than trusting
    the first and failing in the user's ear.
    """
    import tempfile

    candidates = [
        os.environ.get("JARVIS_SCREENSHOT_DIR", ""),
        os.path.join(os.path.expanduser("~"), "Pictures", "Jarvis"),
        os.path.join(os.path.expanduser("~"), "Jarvis screenshots"),
        os.path.join(tempfile.gettempdir(), "Jarvis screenshots"),
    ]
    last = "no candidate tried"
    for cand in candidates:
        if not cand:
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".jarvis-write-test")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(probe)
            return cand
        except Exception as e:
            last = f"{cand} ({type(e).__name__}, {e})"
    raise OSError(f"no writable screenshot directory; last failure: {last}")


def _save_screenshot(label: str) -> tuple[str | None, str]:
    """Save a full-resolution PNG, returning (path, error).

    Separate from capture_screen_b64 on purpose: that one shrinks to 512 px of
    JPEG because it is going to a vision model, and a shrunk JPEG is not what
    someone means by "take a screenshot".
    """
    import re

    import mss
    from PIL import Image

    try:
        out_dir = _screenshot_dir()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "screenshot"
        path = os.path.join(out_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-{safe[:40]}.png")
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            img.save(path, format="PNG")
        return path, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# Spoken key name -> pyautogui key name. The agent hears "escape" or "esc" or
# "return", never "escape". Kept as an explicit table so an unrecognised word
# is refused rather than guessed at: pyautogui.press() raises on a bad name, and
# a wrong guess here means a key the user did not ask for.
KEY_ALIASES = {
    "enter": "enter", "return": "enter", "ret": "enter",
    "escape": "esc", "esc": "esc", "cancel": "esc",
    "tab": "tab", "space": "space", "spacebar": "space",
    "backspace": "backspace", "delete": "delete", "del": "delete",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end", "pageup": "pageup", "pagedown": "pagedown",
    "page up": "pageup", "page down": "pagedown",
}


def _norm_key(target: str):
    """Resolve a spoken key name to a pyautogui key, or None if unknown."""
    q = (target or "").strip().lower()
    if not q:
        return None
    if q in KEY_ALIASES:
        return KEY_ALIASES[q]
    # "page up"/"page down" spoken with no space
    return KEY_ALIASES.get(q.replace(" ", ""))


def _ground(target: str, screen_w: int, screen_h: int):
    """UIA first, VLM fallback. Phase 4 replaces this; kept honest for now."""
    try:
        from pywinauto import Desktop  # noqa: F401

        for wdw in Desktop(backend="uia").windows():
            try:
                for d in wdw.descendants():
                    name = (d.window_text() or "").strip()
                    if name and target.lower() in name.lower():
                        r = d.rectangle()
                        return r.mid_point().x, r.mid_point().y
            except Exception:
                continue
    except Exception:
        pass

    b64 = capture_screen_b64()
    if not b64:
        return None
    try:
        r = requests.post(UPLOAD_URL, json={
            "user_text": f"click {target}",
            "screen_context": "Active Windows: " + ", ".join(active_windows()),
            "image_b64": b64,
            # Only the coordinates are used, so no audio is needed either.
            "want_audio": False,
        }, timeout=120)
        r.raise_for_status()
        b = r.json()
        if b.get("type") == "action" and b.get("norm_x") is not None:
            return b["norm_x"] / 1000.0 * screen_w, b["norm_y"] / 1000.0 * screen_h
    except Exception:
        pass
    return None


class ExecHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._reply(400, {"ok": False, "error": f"bad json: {e}"})
            return
        try:
            result = execute(req.get("action", ""), req.get("target", ""))
        except Exception as e:
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self._reply(200, result)

    def _reply(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_exec_server():
    # 127.0.0.1, not 0.0.0.0. This endpoint runs whatever action it is handed --
    # TYPE_TEXT, VISUAL_CLICK, LOCK_WORKSTATION -- with no authentication at all,
    # so bound to every interface it is a remote-control service for the whole
    # Wi-Fi LAN. The agent only needs loopback (tools.py probes 127.0.0.1:8766
    # first, and the WSL VM shares the host's loopback in mirrored mode).
    srv = ThreadingHTTPServer(("127.0.0.1", EXEC_PORT), ExecHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------
def audio_out_loop():
    """Drains the playout queue. Swallowing chunks on barge-in is what makes
    the agent stop talking the instant the user interrupts."""
    with sd.RawOutputStream(samplerate=OUT_RATE, blocksize=IN_CHUNK // 2,
                            channels=1, dtype="int16") as out:
        while True:
            if MUTE.is_set():
                time.sleep(0.01)
                continue
            try:
                chunk = AUDIO_OUT_Q.get(timeout=0.05)
            except queue.Empty:
                continue
            out.write(chunk.tobytes())


def flush_playout():
    while not AUDIO_OUT_Q.empty():
        try:
            AUDIO_OUT_Q.get_nowait()
        except queue.Empty:
            break


# The live input stream. It MUST be kept here for the life of the program.
#
# mic_frames() used to return only the queue, leaving this as the sole owner of
# the sd.InputStream. CPython collects by refcount, so the stream was destroyed
# the instant that function returned -- PortAudio tore the device down while its
# callback thread was still running, and the callback wrote into freed memory.
# The process died with 0xC0000005 (access violation) a second after printing
# "mic live", which is why the client appeared to close on its own. No Python
# traceback appears because the crash is below the interpreter.
_MIC_STREAM = None


def mic_frames() -> queue.Queue[np.ndarray]:
    global _MIC_STREAM

    q: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    _MIC_STREAM = sd.InputStream(samplerate=IN_RATE, blocksize=IN_CHUNK,
                                 channels=1, dtype="float32", callback=cb)
    _MIC_STREAM.start()
    return q


# --------------------------------------------------------------------------
async def amain():
    start_exec_server()
    _log(f"tool-exec endpoint on 127.0.0.1:{EXEC_PORT}")

    out_thread = threading.Thread(target=audio_out_loop, daemon=True)
    out_thread.start()
    mic = mic_frames()
    _log(f"mic live @ {IN_RATE} Hz, speaker @ {OUT_RATE} Hz")

    # The mic callback runs on a PortAudio thread and lands in a plain
    # queue.Queue. Hop it onto the event loop from a dedicated thread rather
    # than calling the blocking mic.get() inside a coroutine: a blocking get on
    # the loop stalls downlink and playout for a whole 20 ms frame, 50 times a
    # second, which is exactly the latency that makes barge-in feel late.
    #
    # The stop flag matters. This thread outlives the session on purpose, so
    # that a reconnect does not have to reopen the microphone, but a bare
    # `while True` kept calling call_soon_threadsafe after asyncio.run had
    # closed the loop -- every disconnect ended in a RuntimeError traceback
    # from a thread nobody was watching. mic.get() has a timeout so the flag is
    # actually polled instead of being stuck in an untimed block.
    loop = asyncio.get_running_loop()
    up_q: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            try:
                chunk = mic.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                loop.call_soon_threadsafe(up_q.put_nowait, chunk)
            except RuntimeError:
                # Loop closed underneath us during interpreter shutdown.
                return

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()

    if CLIENT_VERBOSE:
        threading.Thread(target=_stats_loop, daemon=True).start()

    # Reconnect instead of exiting. The agent is restarted constantly during
    # development, and WSL's mirrored port relay drops the odd connection even
    # when the agent was up beforehand -- both look identical to the client: a
    # completed handshake followed by an abnormal close. Exiting turned both
    # into "run me again", so this retries with backoff and only gives up once
    # it looks like the agent is genuinely not there.
    #
    # The budget is wall-clock, not a retry count, because the thing being waited
    # on is a cold agent: start-agent.sh alone allows 60 s for the models to come
    # up. A fixed list of attempts summed to less than that and gave up while the
    # agent was still loading.
    deadline = time.monotonic() + RECONNECT_DEADLINE_SECS
    delay = RECONNECT_BACKOFF[0]
    attempt = 0
    last = "no attempt made"

    try:
        while True:
            # Anything captured while disconnected would otherwise be replayed
            # as the first frames of the new session and trip the VAD on audio
            # the agent never heard in that context.
            while not up_q.empty():
                up_q.get_nowait()

            try:
                await session(up_q)
                return
            except (websockets.exceptions.ConnectionClosed,
                    ConnectionRefusedError, OSError) as e:
                last = f"{type(e).__name__}: {e}"
                attempt += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                _log(f"disconnected ({last}) -- retry {attempt} "
                     f"in {delay:.0f}s, {remaining:.0f}s of patience left")
                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 2, RECONNECT_BACKOFF[-1])

        _log(f"giving up after {attempt} attempts "
             f"({RECONNECT_DEADLINE_SECS:.0f}s). last error: {last}")
        _log("is the agent running? ./start-agent.sh in WSL.")
    finally:
        # Only on real shutdown. While retrying, the pump must keep running so
        # the mic stays open across reconnects.
        stop.set()


async def session(up_q: asyncio.Queue) -> None:
    """One connection lifetime. Returns when the session ends cleanly."""
    async with websockets.connect(WS_URL, max_size=None) as ws:
        _log(f"connected to {WS_URL}")

        async def uplink():
            while True:
                data = await up_q.get()
                pcm16 = (np.clip(data, -1, 1) * 32767).astype(np.int16)
                audio_stats["sent"] += 1
                await ws.send(pcm16.tobytes())

        down_q: asyncio.Queue = asyncio.Queue()

        async def downlink():
            # A bare `async for` here is fine: ConnectionClosed propagates and
            # amain turns it into a reconnect.
            async for message in ws:
                if isinstance(message, bytes):
                    audio_stats["got"] += 1
                    await down_q.put(message)
                else:
                    _log(f"<- {message}")

        async def playout():
            buffer = bytearray()
            while True:
                chunk = await down_q.get()
                if MUTE.is_set():
                    flush_playout()
                    continue
                buffer.extend(chunk)
                step = 1024
                while len(buffer) >= step:
                    block = bytes(buffer[:step])
                    del buffer[:step]
                    AUDIO_OUT_Q.put(np.frombuffer(block, dtype=np.int16).copy())

        await asyncio.gather(uplink(), downlink(), playout())



def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default=None)
    ap.add_argument("--no-mic", action="store_true",
                    help="use the bundled test clip instead of the microphone")
    ap.add_argument("--exec-only", action="store_true",
                    help="only host the tool-exec endpoint (for testing tools headlessly)")
    ap.add_argument("--clip", default=None)
    args = ap.parse_args()

    if args.ws:
        globals()["WS_URL"] = args.ws

    if args.exec_only:
        srv = start_exec_server()
        _log(f"tool-exec endpoint on 127.0.0.1:{EXEC_PORT} (ctrl-c to stop)")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            _log("exiting")
        return

    if args.no_mic or args.clip:
        try:
            asyncio.run(replay(args.clip))
        except KeyboardInterrupt:
            _log("exiting")
        return

    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        _log("exiting")
    except websockets.exceptions.ConnectionClosed:
        _log("disconnected.")
    except ConnectionRefusedError:
        _log(f"no agent at {WS_URL}. start it with ./start-agent.sh in WSL.")
    except OSError as e:
        _log(f"network error: {e}")


async def replay(clip: str | None):
    """Feed a WAV through the real pipeline and print what comes back."""
    import soundfile as sf

    path = clip or os.environ.get("JARVIS_TEST_CLIP", "")
    if not path:
        raise SystemExit(
            "replay needs a 16 kHz mono WAV: pass --clip PATH or set JARVIS_TEST_CLIP"
        )
    data, sr = sf.read(path, dtype="float32")
    data = data.reshape(-1)
    if sr != IN_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), IN_RATE)
        data = resample_poly(data, IN_RATE // g, int(sr) // g).astype(np.float32)

    start_exec_server()
    received = bytearray()
    t0 = time.time()
    first_audio = None

    async with websockets.connect(WS_URL, max_size=None) as ws:
        async def reader():
            nonlocal first_audio
            async for message in ws:
                if isinstance(message, bytes):
                    if first_audio is None:
                        first_audio = time.time() - t0
                        print(f"[client] first audio after {first_audio:.2f}s")
                    received.extend(message)
                else:
                    print(f"[client] <- {message}")

        task = asyncio.create_task(reader())
        step = 320
        for i in range(0, len(data), step):
            block = data[i:i + step]
            pcm16 = (np.clip(block, -1, 1) * 32767).astype(np.int16)
            await ws.send(pcm16.tobytes())
            await asyncio.sleep(0.02)   # realtime, so the VAD sees real pacing
        # trailing silence so the turn closes
        for _ in range(75):
            await ws.send(np.zeros(step, dtype=np.int16).tobytes())
            await asyncio.sleep(0.02)

        for _ in range(400):
            await asyncio.sleep(0.05)
            if task.done():
                break
        task.cancel()

    if received:
        arr = np.frombuffer(bytes(received), dtype=np.int16)
        path_out = "replay_out.wav"
        sf.write(path_out, arr, OUT_RATE, subtype="PCM_16")
        print(f"[client] received {len(arr)} samples "
              f"({len(arr)/OUT_RATE:.2f}s) -> {path_out}")
    else:
        print("[client] received no audio")


if __name__ == "__main__":
    main()
