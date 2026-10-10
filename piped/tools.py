"""Tool layer: the LLM's only way to touch the machine.

Tool calls are forwarded to the Windows client, which owns the screen, the
mouse and the keyboard. Pipecat never touches the OS directly -- that keeps
the WSL/Windows privilege boundary clean and means the action executor stays in
the same process as the mic and speaker.
"""

import json
import os
import socket
import time

import aiohttp
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

from .vocab import SELF_CONTAINED_ROUTER

# WSL2 networking topology differs:
#   mirrored mode  (this box) -> the VM shares the host's addresses, so 127.0.0.1
#                   reaches Windows in both directions, and that is the fastest
#                   path.
#   NAT mode        -> Windows is the default gateway, so 192.168.x.1.
# Rather than hardcode one, probe loopback and fall back to the gateway. Two
# overrides exist, at different granularities: JARVIS_CLIENT_HOST replaces the
# host:port that is probed, JARVIS_CLIENT_EXEC the whole URL.
DEFAULT_CLIENT_HOST = "127.0.0.1:8766"
# The NAT-mode fallback is the Windows default gateway on that network. Left empty
# by default so a published copy carries no private address; set
# JARVIS_GATEWAY_FALLBACK when the loopback probe fails (WSL in NAT mode).
GATEWAY_FALLBACK = os.environ.get("JARVIS_GATEWAY_FALLBACK", "")

_CLIENT_HOST_OVERRIDE = os.environ.get("JARVIS_CLIENT_HOST", "")
_CLIENT_EXEC_OVERRIDE = os.environ.get("JARVIS_CLIENT_EXEC", "")
_resolved_client_url: str | None = None


def _windows_host() -> str:
    if _CLIENT_HOST_OVERRIDE:
        return _CLIENT_HOST_OVERRIDE
    for candidate in (DEFAULT_CLIENT_HOST, GATEWAY_FALLBACK):
        if not candidate:
            continue
        host, _, port = candidate.rpartition(":")
        try:
            with socket.create_connection((host, int(port)), timeout=1.5):
                logger.info(f"[tools] desktop client reachable at {candidate}")
                return candidate
        except OSError:
            continue
    logger.warning(
        f"[tools] no desktop client found at {DEFAULT_CLIENT_HOST}"
        + (f" or {GATEWAY_FALLBACK}" if GATEWAY_FALLBACK else "")
        + "; start jarvis_client.py --exec-only (set JARVIS_GATEWAY_FALLBACK if "
          "WSL is in NAT mode rather than mirrored)"
    )
    return DEFAULT_CLIENT_HOST


def client_exec_url() -> str:
    """Where the desktop client lives: resolved on first use, then cached.

    Deliberately lazy. _windows_host() opens real TCP connections, so doing this
    at import time cost up to 3 s of startup and logged a warning before the
    pipeline even existed. _post() drops the cache again when a call fails to
    connect, so a client that starts later -- or that is only reachable on the
    gateway address -- is picked up by the next tool call instead of needing the
    agent restarted.
    """
    global _resolved_client_url
    if _CLIENT_EXEC_OVERRIDE:
        return _CLIENT_EXEC_OVERRIDE
    if _resolved_client_url is None:
        _resolved_client_url = f"http://{_windows_host()}/execute"
    return _resolved_client_url

# Kept in sync with router.ACTION_CHOICES. The LLM may only request these, so a
# hallucinated verb becomes a validation error rather than a stray pyautogui call.
SUPPORTED_ACTIONS = [
    # browser
    "NAVIGATE_YOUTUBE", "OPEN_BROWSER", "OPEN_TAB", "CLOSE_TAB", "NEXT_TAB",
    "PREVIOUS_TAB", "GO_BACK", "GO_FORWARD", "FOCUS_SEARCH_BAR", "VISUAL_CLICK",
    # window / app
    "FOCUS_APP", "LIST_APPS", "LIST_BROWSERS", "MINIMIZE_WINDOW",
    "MAXIMIZE_WINDOW", "RESTORE_WINDOW", "OPEN_TERMINAL",
    # scrolling
    "SCROLL_DOWN", "SCROLL_UP", "SCROLL_LEFT", "SCROLL_RIGHT",
    # system
    "LOWER_BRIGHTNESS", "RAISE_BRIGHTNESS", "LOWER_VOLUME", "RAISE_VOLUME",
    "MUTE_VOLUME", "LOCK_WORKSTATION",
    # input
    "TYPE_TEXT", "PRESS_KEY",
    # media, clipboard, URLs, screenshots, window snapping
    "MEDIA_PLAY_PAUSE", "MEDIA_NEXT", "MEDIA_PREV", "MEDIA_STOP",
    "CLIPBOARD_READ", "CLIPBOARD_WRITE", "OPEN_URL", "SAVE_SCREENSHOT",
    "SNAP_LEFT", "SNAP_RIGHT",
    # perception
    "DESCRIBE_SCREEN",
]

# Which verbs the fast path may run on its own is defined in vocab.py -- a module
# with no third-party imports, so server.py can gate its legacy /api/command path
# on the very same set instead of keeping a second copy of it. The invariant the
# old comment could only promise is now checked at import:
assert SELF_CONTAINED_ROUTER <= set(SUPPORTED_ACTIONS), (
    "fast-path verbs missing from SUPPORTED_ACTIONS: "
    f"{sorted(SELF_CONTAINED_ROUTER - set(SUPPORTED_ACTIONS))}"
)

async def _post(payload: dict, timeout: float = 20.0) -> dict:
    """POST to the desktop client.

    Uses a short-lived session per call. A session created outside the pipeline
    is bound to whatever event loop existed at construction time; sharing one
    across the pipeline risks handing aiohttp a loop it is not running on.
    Tool calls are rare, so the extra handshake costs nothing.
    """
    global _resolved_client_url

    url = client_exec_url()
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.post(url, json=payload) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"client returned {resp.status}: {body[:200]}")
                return json.loads(body)
    except Exception as e:
        # A connection-level failure means the address we cached is wrong: the
        # client was still down when we probed, or it has come up somewhere else.
        # Forget it so the next call probes again. Timeouts are OSError subclasses
        # in Python 3, and so are aiohttp's connector errors, so this covers the
        # reachability cases without discarding a good address merely because a
        # reachable client answered with an HTTP error.
        if isinstance(e, OSError) and not _CLIENT_EXEC_OVERRIDE:
            _resolved_client_url = None
        logger.warning(f"[tool] POST {url} failed: {type(e).__name__}: {e}")
        raise


async def call_client(action: str, target: str = "", timeout: float = 20.0) -> str:
    """Forwards one action to the desktop client and returns the spoken reply.

    Never raises. Everything the agent says is synthesised, so a failure has to
    arrive as a sentence rather than an exception: a plainly spoken "could not
    reach the desktop client" beats a hallucinated success.
    """
    logger.info(f"[tool] -> {action} {target!r}")
    t0 = time.perf_counter()
    try:
        result = await _post({"action": action, "target": target}, timeout=timeout)
    except Exception as e:
        logger.info(f"[tool] <- {action} FAILED in {time.perf_counter()-t0:.2f}s: {e}")
        return f"Could not reach the desktop client: {e}"
    if not result.get("ok"):
        logger.info(f"[tool] <- {action} rejected in {time.perf_counter()-t0:.2f}s: "
                    f"{result.get('error')}")
        return f"{action} failed: {result.get('error', 'unknown error')}"
    reply = result.get("message") or ""
    logger.info(f"[tool] <- {action} ok in {time.perf_counter()-t0:.2f}s: {reply}")
    return reply


# DESCRIBE_SCREEN round-trips a screenshot through the 4B vision model; 20 s is
# not enough on a cold vision call. Defined above perform_action because that is
# where it is read.
ACTION_TIMEOUTS = {"DESCRIBE_SCREEN": 90.0}


async def perform_action(action: str, target: str = "") -> str:
    """Validated action dispatch, shared by the LLM tool handler and fast path.

    The allowlist is the security boundary: a hallucinated verb becomes a
    refusal instead of a stray pyautogui call. Also the single place that knows
    which actions require a target.
    """
    if action not in SUPPORTED_ACTIONS:
        return f"Unsupported action {action}. Supported: {', '.join(SUPPORTED_ACTIONS)}."

    needs_target = {"VISUAL_CLICK", "FOCUS_APP", "TYPE_TEXT", "PRESS_KEY",
                    "MINIMIZE_WINDOW", "MAXIMIZE_WINDOW", "RESTORE_WINDOW",
                    "DESCRIBE_SCREEN", "CLIPBOARD_WRITE", "OPEN_URL"}
    if action in needs_target and not target.strip():
        if action == "DESCRIBE_SCREEN":
            return "DESCRIBE_SCREEN needs a question about the screen."
        return f"{action} needs a target saying what to act on."

    return await call_client(action, target,
                             timeout=ACTION_TIMEOUTS.get(action, 20.0)) \
        or f"{action} done."


def build_tools() -> list:
    """Returns FunctionSchema objects, each carrying its own handler.

    The schemas have to be advertised on the LLMContext: that is where Pipecat
    reads the tool list from when it builds the request, so putting them anywhere
    else produces a chat-only agent with no error -- the model answers "opening a
    tab now" without calling anything. A FunctionSchema that carries its own
    handler needs no llm.register_function(); bot.py hands nothing but the
    schemas to the context, and Pipecat warns that the manual step is redundant.

    Built as explicit FunctionSchema rather than raw callables on purpose:
    passing a callable in the context makes Pipecat treat it as a *direct
    function* (first arg must be named `params`), which is a Pipecat-specific
    protocol, not the OpenAI function-calling schema Ollama expects.

    The handler protocol, verified against pipecat 1.12.0 (llm_service.py:1844):
    the handler is awaited as `item.handler(params)` with a single
    FunctionCallParams object, not unpacked kwargs, and its RETURN VALUE IS
    DISCARDED. So a handler written as (action, target) receives that object as
    `action` and dies on "'FunctionCallParams' object has no attribute 'strip'",
    and one that returns its reply looks correct while leaving the call
    unsettled: no FunctionCallResultFrame, no follow-up LLM turn, and the agent
    goes quiet. Every handler here delivers its result through
    params.result_callback(...).
    """

    # A FunctionSchema that carries a handler is invoked with a single
    # FunctionCallParams object (llm_service.py:117), NOT unpacked kwargs.
    # Handlers written as (action, target) receive that object as `action` and
    # fail with "'FunctionCallParams' object has no attribute 'strip'".
    # A FunctionSchema that carries a handler is invoked as
    #   await item.handler(params)      (llm_service.py:1844)
    # and the RETURN VALUE IS DISCARDED. A handler must deliver its result via
    # params.result_callback(...) instead -- see _missing_function_call_handler
    # at llm_service.py:1881 for the canonical shape. Returning a string from
    # the handler looks correct and silently leaves the call unsettled: no
    # FunctionCallResultFrame, so the follow-up LLM turn never runs and the
    # agent goes quiet.
    async def run_action(params: FunctionCallParams) -> None:
        args = params.arguments or {}
        action = str(args.get("action", "")).strip().upper()
        target = str(args.get("target", "") or "").strip()
        logger.info(f"[tool] run_action(action={action!r}, target={target!r})")
        await params.result_callback(await perform_action(action, target))

    async def list_windows(params: FunctionCallParams) -> None:
        logger.info("[tool] list_windows()")
        # Not in SUPPORTED_ACTIONS: this is a read-only query, not a verb the
        # model may ask for, so it skips the action allowlist deliberately.
        await params.result_callback(
            await call_client("LIST_WINDOWS") or "No windows reported."
        )

    async def describe_screen(params: FunctionCallParams) -> None:
        question = str((params.arguments or {}).get("question", "")).strip()
        logger.info(f"[tool] describe_screen(question={question!r})")
        if not question:
            await params.result_callback("describe_screen needs a question.")
            return
        # 90 s: this one round-trips a screenshot through the 4B vision model.
        await params.result_callback(
            await call_client("DESCRIBE_SCREEN", question, timeout=90.0)
        )

    return [
        FunctionSchema(
            name="run_action",
            description=(
                "Control the user's computer: open, close, or switch browser tabs, "
                "navigate back and forward, switch to another application, "
                "minimise or maximise a window, snap a window to half the screen, "
                "scroll, type text, press a key, change volume or brightness, "
                "control media playback, read or write the clipboard, open a web "
                "address, save a screenshot, open the browser or terminal, list "
                "which applications or browsers are open, or lock the workstation. "
                "Call this when the user asks you to act "
                "on their machine. Do not merely describe the action."
            ),
            properties={
                "action": {
                    "type": "string",
                    "enum": SUPPORTED_ACTIONS,
                    "description": "The desktop action to perform.",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "What to act on. Required for VISUAL_CLICK (describe the "
                        "element), FOCUS_APP (the app name, e.g. 'Visual Studio "
                        "Code'), TYPE_TEXT (the exact text to type), PRESS_KEY "
                        "(a key name, e.g. 'enter'), DESCRIBE_SCREEN (the "
                        "question), and the MINIMIZE/MAXIMIZE/RESTORE window "
                        "actions (the app name). Ignored otherwise."
                    ),
                },
            },
            required=["action"],
            handler=run_action,
        ),
        FunctionSchema(
            name="list_windows",
            description=(
                "List the titles of windows currently open on the user's screen. "
                "Use this before switching or focusing so you use the real name."
            ),
            properties={},
            required=[],
            handler=list_windows,
        ),
        FunctionSchema(
            name="describe_screen",
            description=(
                "Look at the user's screen and answer a question about what is "
                "visible on it."
            ),
            properties={
                "question": {
                    "type": "string",
                    "description": "What to look for on the screen.",
                },
            },
            required=["question"],
            handler=describe_screen,
        ),
    ]
