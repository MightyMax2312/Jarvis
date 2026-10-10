"""The desktop-action vocabulary, in one place.

Its own module, with no third-party imports, because the two entry points run
under different virtualenvs and this is the only file both can import:

  * server.py (the legacy /api/command path) runs under jarvis-env -- FastAPI,
    Kokoro, sherpa-onnx;
  * the agent (piped/*) runs under pipecat-env -- pipecat, aiohttp.

Both need the same answer to "which verbs may the router return, and which of
those may it dispatch on its own?". Keeping a single definition makes that
agreement a fact rather than a promise: ACTION_CHOICES used to be copied into
server.py and router.py, and SELF_CONTAINED_ROUTER lived in tools.py.
"""

# The router's closed option set: everything a 0.5B pointer model may pick from.
# It cannot emit text, so it can only choose among these keys, and the values are
# what it actually reads -- they are phrased as descriptions of the action rather
# than as identifiers for that reason.
ACTION_CHOICES: dict[str, str] = {
    # browser
    "NAVIGATE_YOUTUBE": "Open or navigate directly to YouTube",
    "OPEN_BROWSER": "Open the web browser",
    "OPEN_TAB": "Open a new web browser tab",
    "CLOSE_TAB": "Close current active browser tab",
    "NEXT_TAB": "Switch to the next browser tab",
    "PREVIOUS_TAB": "Switch to the previous browser tab",
    "GO_BACK": "Go back to the previous page in the browser",
    "GO_FORWARD": "Go forward to the next page in the browser",
    "FOCUS_SEARCH_BAR": "Click or focus browser address bar or search input field",
    "VISUAL_CLICK": "Click a specific button, link, or image target on screen",
    # windows and applications
    "FOCUS_APP": "Switch to or focus a named application window",
    "LIST_APPS": "List which applications are currently open",
    "LIST_BROWSERS": "List which browsers are open, with their tabs",
    "MINIMIZE_WINDOW": "Minimise a window",
    "MAXIMIZE_WINDOW": "Maximise a window",
    "RESTORE_WINDOW": "Restore a minimised or maximised window",
    "OPEN_TERMINAL": "Open terminal command prompt window",
    # scrolling
    "SCROLL_DOWN": "Scroll active window content downward",
    "SCROLL_UP": "Scroll active window content upward",
    "SCROLL_LEFT": "Scroll active window content horizontally left",
    "SCROLL_RIGHT": "Scroll active window content horizontally right",
    # system
    "LOWER_BRIGHTNESS": "Reduce screen brightness",
    "RAISE_BRIGHTNESS": "Increase screen brightness",
    "LOWER_VOLUME": "Lower audio volume",
    "RAISE_VOLUME": "Raise audio volume",
    "MUTE_VOLUME": "Mute system audio",
    "LOCK_WORKSTATION": "Lock the workstation PC",
    # input
    "TYPE_TEXT": "Type a specific piece of text into the keyboard",
    "PRESS_KEY": "Press a single named key such as enter or escape",
    # media, clipboard, URLs, screenshots, window snapping
    "MEDIA_PLAY_PAUSE": "Play or pause whatever media is playing",
    "MEDIA_NEXT": "Skip to the next track",
    "MEDIA_PREV": "Go back to the previous track",
    "MEDIA_STOP": "Stop media playback",
    "CLIPBOARD_READ": "Read out what is currently on the clipboard",
    "CLIPBOARD_WRITE": "Copy a piece of text to the clipboard",
    "OPEN_URL": "Open a web address in the default browser",
    "SAVE_SCREENSHOT": "Save a screenshot to a file on disk",
    "SNAP_LEFT": "Snap a window to the left half of the screen",
    "SNAP_RIGHT": "Snap a window to the right half of the screen",
    # perception
    "DESCRIBE_SCREEN": "Look at the screen and describe what is visible on it",
    "NONE": "No direct system action matched",
}

# Cheap pre-filter, so pure chat never pays a router call (~300 ms measured).
# Intentionally permissive: the router's own confidence is what decides
# (JEV_CONFIDENCE_THRESHOLD).
#
# "screenshot", "copy", "snap" and "clipboard" were deliberately absent until
# SAVE_SCREENSHOT, CLIPBOARD_READ/WRITE and SNAP_LEFT/RIGHT existed, because none
# of them named a verb: "copy" could only ever have produced a confident wrong
# pick or NONE. Now that the verbs are real, leaving them out was measured to cost
# the fast path completely -- the end-to-end test showed "what is on my clipboard"
# skipping the router and paying 9.3 s of LLM inference, when the router answers
# CLIPBOARD_READ at conf 0.96 in about 150 ms. "paste" and "undo" still name
# nothing and stay out.
ACTION_KEYWORDS = (
    "click", "tap", "press", "pause", "skip", "mute", "play", "stop",
    "lower", "raise", "volume", "brightness", "open", "close", "browser",
    "terminal", "lock", "tab", "scroll", "search", "youtube", "navigate",
    "go to", "back", "forward", "type", "zoom", "minimise", "minimize",
    "maximise", "maximize", "window", "switch", "app", "application",
    "focus", "see", "look", "screen", "display", "show me", "what am i",
    "clipboard", "copy", "screenshot", "snap",
)

# Verbs the router may answer on its own, with no LLM turn and no chance for
# anything to extract an argument. Anything needing an argument -- what to type,
# which app, which button -- is deliberately excluded: the fast path runs the
# action immediately and the model never sees the sentence, so it has no way to
# know the target. Those have to go through the LLM, which reads the whole
# sentence.
#
# OPEN_TERMINAL is deliberately absent. Now that FOCUS_APP exists, "switch to
# Windows Terminal" and "open a terminal" are genuinely different requests, and
# the 0.5B router picked OPEN_TERMINAL at conf 0.912 for the former -- so it
# opened a second terminal instead of switching to the one already open. Only the
# LLM, which reads the whole sentence, gets this distinction right. Kept out of
# the set rather than deleted from the vocabulary so "open a terminal" still
# works, just via the model.
#
# The invariant that matters: everything here must also be in
# tools.SUPPORTED_ACTIONS, or the fast path would dispatch a verb the LLM tool
# layer would refuse. tools.py asserts it at import.
SELF_CONTAINED_ROUTER = {
    "NAVIGATE_YOUTUBE", "OPEN_BROWSER", "OPEN_TAB", "CLOSE_TAB", "NEXT_TAB",
    "PREVIOUS_TAB", "GO_BACK", "GO_FORWARD", "FOCUS_SEARCH_BAR",
    "SCROLL_DOWN", "SCROLL_UP", "SCROLL_LEFT", "SCROLL_RIGHT",
    "LOWER_BRIGHTNESS", "RAISE_BRIGHTNESS", "LOWER_VOLUME", "RAISE_VOLUME",
    "MUTE_VOLUME", "LOCK_WORKSTATION", "LIST_APPS", "LIST_BROWSERS",
    # Media transport, clipboard reads and window snapping need no argument to
    # extract: the verb is the whole request. SAVE_SCREENSHOT, CLIPBOARD_WRITE and
    # OPEN_URL are deliberately absent -- their target carries the label, the text
    # or the address, and the fast path has no way to pull that out of the
    # sentence, so the LLM has to see it.
    "MEDIA_PLAY_PAUSE", "MEDIA_NEXT", "MEDIA_PREV", "MEDIA_STOP",
    "CLIPBOARD_READ", "SNAP_LEFT", "SNAP_RIGHT",
    # The user's own words are the question, so this needs no extraction either.
    "DESCRIBE_SCREEN",
}
