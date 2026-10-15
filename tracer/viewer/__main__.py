"""python -m tracer.viewer [--port 5000] [--host 127.0.0.1] [--pat-id ID] [--model gemini-3.1-flash-lite]"""
from __future__ import annotations

import argparse
import sys

from .. import config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--pat-id", default=None, help="stay to open first (default: first row of stays.csv)")
    ap.add_argument("--model", default="gemini-3.1-flash-lite", help="default chat model")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    from .app import create_app
    app = create_app(pat_id=args.pat_id, chat_model=args.model)
    print(f"TRACER viewer: http://{args.host}:{args.port}/  (store: {config.data_root()}, local tz: {config.local_tz_name()})")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
