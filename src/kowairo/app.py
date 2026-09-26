"""Application entry point — GUI by default, CLI file-conversion for testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="kowairo", description="Kowairo — realtime AI voice changer")
    parser.add_argument("--cli", action="store_true",
                        help="GUI を起動せず CLI モードで動作")
    parser.add_argument("--input", type=Path,
                        help="CLI: 入力 WAV ファイル")
    parser.add_argument("--output", type=Path,
                        help="CLI: 出力 WAV ファイル")
    parser.add_argument("--text", type=str,
                        help="CLI: テキスト読み上げ (ASR をスキップ)")
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--model", type=Path,
                        help="CLI: 導入する .aivmx または .zip")
    parser.add_argument("--hub-uuid", type=str, default=None,
                        help="CLI: AivisHub モデル UUID を導入")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--engine-version", default="1.2.0")
    args = parser.parse_args()

    if args.cli or args.input or args.text or args.model or args.hub_uuid:
        from .cli import run_cli

        return run_cli(args)

    from .ui.run import run_gui

    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
