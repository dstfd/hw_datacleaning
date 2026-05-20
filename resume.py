"""Resume a failed live/ run from the last completed loop artifact.

Usage (from project root):
  python3 -m live.resume 5b4a269f-5f9a-4008-9508-09f90f02057d
  python3 -m live.resume 5b4a269f-5f9a-4008-9508-09f90f02057d --from 3b

Stages: 3b (needs loop3_linking.json), 4 (needs loop3_linking.json after 3b).
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.loops import loop3b_coherence, loop4_confidence  # noqa: E402

RUNS = Path(__file__).resolve().parent / "data" / "runs"
ROOT_OUTPUT = ROOT / "output.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume live pipeline from a run id")
    parser.add_argument("run_id", help="UUID folder under live/data/runs/")
    parser.add_argument(
        "--from",
        dest="from_stage",
        default="3b",
        choices=["3b", "4"],
        help="3b = coherence + assembly; 4 = assembly only (3b already done)",
    )
    parser.add_argument(
        "--copy-as",
        metavar="PATH",
        default=None,
        help="Also copy final output.json here (e.g. output.py-live-{run_id}.json)",
    )
    args = parser.parse_args()

    out_dir = RUNS / args.run_id
    if not out_dir.is_dir():
        print(f"Error: run dir not found: {out_dir}", file=sys.stderr)
        return 1

    linking_path = out_dir / "loop3_linking.json"
    if not linking_path.is_file():
        print(f"Error: missing {linking_path} — cannot resume before loop 3", file=sys.stderr)
        return 1

    print(f"Resume run: {args.run_id}")
    print(f"→ {out_dir}")
    print(f"  from stage: {args.from_stage}\n")

    linking = json.loads(linking_path.read_text())
    t0 = time.time()

    if args.from_stage == "3b":
        loop3b_coherence.run(linking, out_dir)
        linking_path.write_text(json.dumps(linking, indent=2))
        print(f"  loop 3b done ({time.time() - t0:.1f}s)\n")

    loop4_confidence.run(linking, args.run_id, out_dir, ROOT_OUTPUT)
    print(f"  loop 4 done ({time.time() - t0:.1f}s)")
    print(f"\nOutput: {ROOT_OUTPUT}")

    if args.copy_as:
        dest = ROOT / args.copy_as
        shutil.copy(ROOT_OUTPUT, dest)
        print(f"Copy: {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
