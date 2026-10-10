"""Entry point: python -m piped.run [--host H] [--port P]"""

import argparse
import asyncio
import sys

from loguru import logger

from .bot import run_bot


def main():
    ap = argparse.ArgumentParser(prog="piped.run")
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to listen on; the default keeps this "
                         "unauthenticated audio/control socket on loopback")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.verbose else "INFO",
        format="{time:HH:mm:ss.SSS} | {level: <7} | {message}",
    )

    try:
        asyncio.run(run_bot(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("bye")


if __name__ == "__main__":
    main()
