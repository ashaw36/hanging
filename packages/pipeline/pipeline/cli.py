from __future__ import annotations

import argparse
from pathlib import Path

from .run import DEFAULT_PACKAGES, run_pipeline


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Behavior feature MVP demo pipeline")
    p.add_argument("--video", "-i", required=True, help="Input video path")
    p.add_argument("--out", "-o", required=True, help="Output job directory")
    p.add_argument("--asr-model", default="tiny", help="faster-whisper model size")
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick demo")
    p.add_argument("--stride", type=int, default=1, help="Frame stride")
    p.add_argument("--no-asr", action="store_true", help="Disable ASR even if package list includes asr_basic")
    p.add_argument(
        "--packages",
        default=",".join(DEFAULT_PACKAGES),
        help="Comma-separated packages, e.g. visual_base,gaze_basic,audio_acoustic,asr_basic,events_basic[,au_advanced]",
    )
    p.add_argument(
        "--template",
        default=None,
        help="Customer HTML template for CSS (optional)",
    )
    args = p.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    template = Path(args.template) if args.template else root / "docs" / "examples" / "report-preview.html"
    packages = [x.strip() for x in args.packages.split(",") if x.strip()]

    bundle = run_pipeline(
        Path(args.video),
        Path(args.out),
        asr_model=args.asr_model,
        max_frames=args.max_frames,
        frame_stride=args.stride,
        enable_asr=not args.no_asr,
        packages=packages,
        template_html=template if template.exists() else None,
    )
    print(f"OK report_id={bundle['report_id']} out={args.out}")
    print(f"  packages={bundle['config_echo']['packages']}")
    print(f"  report.html / report.json / metrics_summary.csv / full.zip")


if __name__ == "__main__":
    main()
