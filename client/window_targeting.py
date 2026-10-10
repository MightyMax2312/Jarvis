"""Window targeting for the Windows client.

Every browser-relative action (open/close tab, focus the address bar) is only
meaningful against a real browser window. Sending ctrl+t to whatever happens to
be focused is how "open a new tab" ended up opening a tab in the IDE that was
running my own test harness.

So browser actions resolve a browser window via UI Automation first, bring it
to the front, and only then send the key. If no browser can be found, the
action fails loudly instead of firing into the void.
"""

import os
import re
import subprocess
import time

# Get-Process reports ProcessName WITHOUT the .exe suffix, so these are bare
# names, not regexes.
BROWSER_PATTERNS = [
    ("chrome", "chrome"),
    ("msedge", "msedge"),
    ("firefox", "firefox"),
    ("brave", "brave"),
    ("opera", "opera"),
    ("vivaldi", "vivaldi"),
    ("librewolf", "librewolf"),
    ("thunderbird", "thunderbird"),
]

# Some of these run as background helpers with no page (e.g. the Logitech
# configurator uses msedge). A title with a " - " separator implies a real tab.
# "Picture in picture" is here because a Chrome launched on a fresh agent
# profile opens one alongside the tab strip, and it is not a page to navigate.
BROWSER_HELPER_HINTS = ("logitech", "logi", "onedrive", "microsoft edge update",
                        "picture in picture")

# Which browser the agent should prefer and launch. Override with
# JARVIS_BROWSER=chrome|msedge|firefox|brave.
#
# Chrome is the default, not Edge. This was Edge until it proved unable to open
# a window on demand: an already-running Edge with no visible window swallows a
# command-line launch and produces nothing, and Edge does not honour
# --new-window the way Chromium does (reported on r/edge and in the Microsoft
# Q&A thread "Launch Edge in a new window from the command line"). The agent
# spent its full 8 s per-browser budget waiting for a window that never came,
# then fell through to Chrome, which appeared in about a second.
PREFERRED_BROWSER = os.environ.get("JARVIS_BROWSER", "chrome").lower()

# Launch targets in fallback order. Absolute paths because a browser is usually
# NOT on PATH, so subprocess.Popen(["chrome.exe"]) raises FileNotFoundError
# even when the browser is installed. ensure_browser() tries PREFERRED first.
BROWSER_LAUNCH = [
    ("msedge", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ("msedge", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ("chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ("chrome", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ("firefox", r"C:\Program Files\Mozilla Firefox\firefox.exe"),
    ("firefox", r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"),
]

# A throwaway profile directory for the agent's own launches.
#
# This is the fix that actually works, rather than a flag. Chromium-family
# browsers are single-instance by profile: launching a second one hands the URL
# to the running process over IPC and exits. If that process is running with no
# visible window -- which happens constantly on this box, e.g. Edge alive with
# only a vendor helper window -- the launch "succeeds" and nothing appears, so
# the agent burns its whole timeout. Pointing --user-data-dir at a separate
# directory gives the launch its own process, and a window really shows up.
AGENT_PROFILE_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "JarvisAgent", "BrowserProfile",
)

_LOGIN_STRIP = re.compile(
    r"\s*[-–—(]\s*(sign in|log in|private|incognito|personalization|"
    r"notification|update|profile|welcome|default)\b.*$",
    re.IGNORECASE,
)

# Edge's window title contains U+200B where the ® should be, and some apps
# embed other zero-width characters. They survive UTF-8 correctly but read as
# stray gaps, so drop them before reporting a title to the user.
#
# Built from explicit code points rather than literal characters: invisible
# characters in source are unreviewable and easy to mangle in transit.
INVISIBLE_CODEPOINTS = (
    0x00AD,  # soft hyphen
    0x180E,  # mongolian vowel separator
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F,  # zero-width + bidi marks
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidi embedding/override
    0x2060, 0x2061, 0x2062, 0x2063, 0x2064,  # word joiner, invisible operators
    0xFEFF,  # BOM / zero-width nbsp
    0xFFF9, 0xFFFA, 0xFFFB, 0xFFFC,  # interlinear annotation
)
_INVISIBLE = re.compile(f"[{''.join(chr(c) for c in INVISIBLE_CODEPOINTS)}]")

# Spoken name for each browser. The agent is talking to a person, so it should
# say "Edge", not "msedge".
DISPLAY_NAME = {
    "msedge": "Edge",
    "chrome": "Chrome",
    "firefox": "Firefox",
    "brave": "Brave",
    "opera": "Opera",
    "vivaldi": "Vivaldi",
    "librewolf": "LibreWolf",
    "thunderbird": "Thunderbird",
}


def display_name(name: str) -> str:
    return DISPLAY_NAME.get(name, name.capitalize())


def _clean_title(title: str) -> str:
    return _INVISIBLE.sub("", title).strip()


def _ps(text: str) -> str:
    """Run a PowerShell snippet and decode stdout as UTF-8.

    Two things are needed for non-ASCII to survive: PowerShell's own output
    encoding must be set to UTF-8 (otherwise it emits the ANSI code page, and
    Edge's "Microsoft® Edge" arrives as "Microsoft? Edge"), and subprocess
    must decode as UTF-8 rather than the locale code page.
    """
    wrapped = f"[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; {text}"
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", wrapped],
        capture_output=True, text=True, timeout=15, encoding="utf-8",
        errors="replace",
    ).stdout


# How long to wait after sending a hotkey before reading titles back. The
# window list is cached, so this only costs a short sleep.
VERIFY_SETTLE_SECS = float(os.environ.get("JARVIS_VERIFY_SECS", "0.35"))

# Window enumeration shells out to PowerShell (~0.3-0.5s). Cache it briefly so a
# single action does not pay for it three times over.
_WINDOW_CACHE_TTL = float(os.environ.get("JARVIS_WINDOW_CACHE_TTL", "1.5"))
_window_cache: tuple[float, list] = (0.0, [])


def cached_windows() -> list:
    global _window_cache
    now = time.time()
    ts, cached = _window_cache
    if now - ts < _WINDOW_CACHE_TTL and cached:
        return cached
    rows = _top_level_windows()
    _window_cache = (now, rows)
    return rows


def _top_level_windows():
    """(exe, title, hwnd) for every visible top-level window with a title.

    pywinauto first: Get-Process attributes MainWindowTitle to whichever single
    process owns a window, which is unreliable for Chromium's multi-process
    model (Edge may expose a real browser window under a helper process, and
    vice versa). pywinauto enumerates actual HWNDs, so it sees the window
    rather than guessing which process owns it.
    """
    rows = []
    try:
        from pywinauto import Desktop

        for wdw in Desktop(backend="win32").windows():
            try:
                if not wdw.is_visible():
                    continue
                title = (wwd.window_text() or "").strip()
                if not title:
                    continue
                exe = ""
                try:
                    pid = wdw.process_id()
                    exe = _process_name(pid)
                except Exception:
                    pass
                rows.append((exe, title, int(wwd.handle)))
            except Exception:
                continue
    except Exception as e:
        print(f"[window_targeting] pywinauto unavailable ({e}); falling back")

    if rows:
        return rows

    ps = (
        "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | "
        "ForEach-Object { \"$($_.ProcessName)|$($_.MainWindowTitle)|"
        "$($_.MainWindowHandle)\" }"
    )
    for line in _ps(ps).splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            exe, title, hwnd = parts[0].strip(), parts[1].strip(), parts[2].strip()
            try:
                hwnd_i = int(hwnd)
            except ValueError:
                continue
            if hwnd_i:
                rows.append((exe, title, hwnd_i))
    return rows


def _process_name(pid: int) -> str:
    import ctypes
    import ctypes.wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = ctypes.wintypes.DWORD(1024)
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(
            handle, 0, buf, ctypes.byref(size)
        ):
            return os.path.basename(buf.value).replace(".exe", "")
        return ""
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def browser_windows(refresh: bool = False):
    """Top-level windows belonging to a known browser, best candidate first."""
    found = []
    for exe, title, hwnd in (_top_level_windows() if refresh else cached_windows()):
        low = exe.lower()
        name = next((n for p, n in BROWSER_PATTERNS if low == p), None)
        if name is None:
            continue
        clean = _clean_title(_LOGIN_STRIP.sub("", title)) or title
        # Skip background helper windows: these exes also host tray/companion
        # apps that have nothing to do with browsing.
        if any(h in clean.lower() for h in BROWSER_HELPER_HINTS) and not _looks_like_page(clean):
            continue
        found.append({
            "exe": exe, "name": name, "title": title,
            "clean": clean, "hwnd": hwnd,
        })
    # Sort: preferred browser first, then windows that look like they have a
    # page open, then the rest.
    found.sort(key=lambda w: (
        0 if w["name"] == PREFERRED_BROWSER else 1,
        0 if _looks_like_page(w["clean"]) else 1,
    ))
    return found


def _looks_like_page(clean_title: str) -> bool:
    """Heuristic: a browser tab title looks like 'Site - Browser Name'."""
    return " - " in clean_title or " – " in clean_title or " — " in clean_title


def focus_hwnd(hwnd: int) -> bool:
    """Bring a window to the foreground."""
    try:
        import win32gui  # pywin32

        win32gui.ShowWindow(hwnd, 9)  # SW_RESTORE
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.user32.ShowWindow(hwnd, 9)
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


# ShowWindow commands, from winuser.h.
SW_MINIMIZE = 6
SW_MAXIMIZE = 3
SW_RESTORE = 9


def foreground_window():
    """The window that currently has focus, as an app record, or None."""
    try:
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return None
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
        title = _clean_title(buf.value or "")
        if not title:
            return None
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        exe = _process_name(pid.value)
        return {"exe": exe, "name": app_display_name(exe),
                "title": title, "hwnd": int(hwnd)}
    except Exception:
        return None


def set_window_state(hwnd: int, state: str) -> bool:
    """minimise | maximise | restore a window by handle."""
    cmd = {"minimize": SW_MINIMIZE, "maximise": SW_MAXIMIZE,
           "maximize": SW_MAXIMIZE, "restore": SW_RESTORE}.get(state.lower())
    if cmd is None:
        return False
    try:
        import ctypes

        return bool(ctypes.windll.user32.ShowWindow(hwnd, cmd))
    except Exception:
        return False


def focus_browser(prefer: str | None = None) -> dict | None:
    """Focus a browser window. Returns its record, or None if none exists.

    Preference order: an explicit `prefer`, then PREFERRED_BROWSER, then
    whatever looks most like a real page.
    """
    wins = browser_windows()
    if not wins:
        return None
    chosen = None
    for want in (prefer, PREFERRED_BROWSER):
        if not want:
            continue
        chosen = next((w for w in wins if w["name"] == want), None)
        if chosen:
            break
    if chosen is None:
        chosen = wins[0]
    if focus_hwnd(chosen["hwnd"]):
        time.sleep(0.25)  # let the foreground settle before sending keys
        return chosen
    return None


def ensure_browser(timeout: float = 3.0, total_budget: float = 12.0):
    """Focus an existing browser, or launch one. Returns (record, launched).

    Two changes make the cold path fast.

    First, the launch gets its own --user-data-dir. Chromium browsers are
    single-instance per profile, so a plain launch hands the URL to whatever is
    already running and exits; if that process has no visible window, nothing
    appears and the agent waits for nothing. A separate profile directory means
    a separate process, so a window reliably appears.

    Second, the per-browser timeout dropped from 8 s to 3 s and the total budget
    from 25 s to 12 s. With a dedicated profile the window appears in well under
    a second, so a long timeout only ever mattered when the launch was going to
    fail anyway -- where it just delayed falling through to the next browser.
    Measured: OPEN_TAB took 11.56 s before, because 8 s was spent on Edge.
    """
    existing = focus_browser()
    if existing:
        return existing, False

    # Try the preferred browser's install locations first, then the rest.
    ordered = [p for p in BROWSER_LAUNCH if p[0] == PREFERRED_BROWSER]
    ordered += [p for p in BROWSER_LAUNCH if p[0] != PREFERRED_BROWSER]

    try:
        os.makedirs(AGENT_PROFILE_DIR, exist_ok=True)
    except OSError as e:
        print(f"[window_targeting] could not create profile dir: {e}")

    deadline_all = time.time() + total_budget
    attempted = []
    for name, path in ordered:
        if not os.path.isfile(path):
            continue
        attempted.append(path)
        if name in ("firefox", "librewolf"):
            args = [path, "-no-remote", "-profile", AGENT_PROFILE_DIR,
                    "about:blank"]
        else:
            # --no-first-run and --no-default-browser-check keep a cold profile
            # from stopping on a welcome page or a default-browser prompt,
            # which is a window we would then focus instead of a blank tab.
            args = [path,
                    f"--user-data-dir={AGENT_PROFILE_DIR}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "about:blank"]
        try:
            subprocess.Popen(
                args, shell=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            print(f"[window_targeting] launch failed: {path}: {e}")
            continue

        deadline = min(time.time() + timeout, deadline_all)
        while time.time() < deadline:
            time.sleep(0.4)
            if [w for w in browser_windows(refresh=True) if w["name"] == name]:
                focused = focus_browser(prefer=name)
                if focused:
                    return focused, True
        if time.time() >= deadline_all:
            break

    print(f"[window_targeting] no browser launched; tried {attempted or 'nothing found on disk'}")
    return None, False


def describe_windows() -> str:
    wins = browser_windows()
    pref = display_name(PREFERRED_BROWSER)
    if not wins:
        return f"No browser windows are open. Preferred browser is {pref}."
    parts = [f"{display_name(w['name'])}: {w['clean']}" for w in wins[:5]]
    return f"Browser windows (preferring {pref}) - " + "; ".join(parts)


# ---------------------------------------------------------------------------
# application windows
# ---------------------------------------------------------------------------
# Spoken name for the non-browser apps worth switching to. Without this the
# agent says "Code" (the process name) when it means VS Code, and "Code" is
# ambiguous with every other editor. Keyed by the bare process name, as
# Get-Process and QueryFullProcessImageNameW both report it.
APP_DISPLAY_NAME = {
    "code": "Visual Studio Code",
    "devenv": "Visual Studio",
    "idea64": "IntelliJ IDEA",
    "pycharm64": "PyCharm",
    "notepad": "Notepad",
    "notepad++": "Notepad++",
    "explorer": "File Explorer",
    "cmd": "Command Prompt",
    "powershell": "PowerShell",
    "pwsh": "PowerShell",
    "windowsterminal": "Windows Terminal",
    "wt": "Windows Terminal",
    "taskmgr": "Task Manager",
    "spotify": "Spotify",
    "discord": "Discord",
    "slack": "Slack",
    "steam": "Steam",
    "word": "Microsoft Word",
    "excel": "Microsoft Excel",
    "powerpnt": "Microsoft PowerPoint",
    "outlook": "Outlook",
    "zoom": "Zoom",
    "teams": "Microsoft Teams",
    "obs64": "OBS Studio",
    "photos": "Photos",
    "calculator": "Calculator",
    "settings": "Settings",
}

# Windows that are not really "apps" to switch to: shell surfaces, overlays and
# tray companions. They are listed by LIST_WINDOWS (which is an honest dump of
# what has a title) but never chosen as a switch target, because focusing a
# tray helper is never what "switch to my music player" means.
NON_APP_EXES = {
    "applicationframehost", "searchhost", "shellexperiencehost",
    "startmenuexperiencehost", "textinputhost", "lockapp", "logonui", "dwm",
    "explorer", "powershell_ise", "windowsinstaller", "taskmgr",
}

# Substrings that mark a title as a shell overlay rather than an application.
# The same heuristic browser_windows() uses, kept separate so the two lists can
# diverge as new junk appears. Matched against the TITLE, not the exe: the
# Logitech configurator runs inside an msedge window titled "Logistage", so an
# exe-name check never catches it.
APP_TITLE_HINTS = BROWSER_HELPER_HINTS + (
    "dimmer", "overlay", "notification", "toast", "input experience",
    "audio control", "quick access",
)


def app_display_name(exe: str, fallback: str = "") -> str:
    return APP_DISPLAY_NAME.get(exe.lower(), fallback or exe.capitalize())


def app_windows():
    """Top-level windows belonging to real applications, not shell helpers."""
    rows = []
    for exe, title, hwnd in cached_windows():
        low = exe.lower()
        clean = _clean_title(title)
        if not low or low in NON_APP_EXES:
            continue
        if any(h in clean.lower() for h in APP_TITLE_HINTS):
            continue
        rows.append({
            "exe": exe,
            "name": app_display_name(exe),
            "title": clean,
            "hwnd": hwnd,
        })
    rows.sort(key=lambda w: w["name"])
    return rows


def find_app(query: str):
    """Best application window matching a spoken phrase, or None.

    Matches on the spoken name first ("visual studio code", "vs code"), then the
    process name, then the window title. The window title is matched last and
    loosely, because a title match alone will happily pick a browser tab that
    merely mentions the app in its text.
    """
    q = (query or "").strip().lower()
    if not q:
        return None
    wins = app_windows()
    if not wins:
        return None

    def score(w):
        name = w["name"].lower()
        exe = w["exe"].lower()
        if q in name or name in q:
            return 0
        if q in exe or exe in q:
            return 1
        if q in w["title"].lower():
            return 2
        return 9

    best = min(wins, key=score, default=None)
    return best if best and score(best) < 9 else None


def describe_apps() -> str:
    wins = app_windows()
    if not wins:
        return "No application windows are open."
    parts = [f"{w['name']}: {w['title'][:40]}" for w in wins[:8]]
    return "Open applications - " + "; ".join(parts)


if __name__ == "__main__":
    print(f"preferred browser: {PREFERRED_BROWSER}")
    print("\nall top-level windows:")
    for exe, title, hwnd in _top_level_windows():
        print(f"  {exe:24} {title[:60]:62} hwnd={hwnd}")
    print("\nbrowsers detected:")
    for w in browser_windows():
        mark = "  <-- preferred" if w["name"] == PREFERRED_BROWSER else ""
        print(f"  {w['name']:10} {w['clean'][:60]:62} hwnd={w['hwnd']}{mark}")
    print("\n" + describe_windows())
