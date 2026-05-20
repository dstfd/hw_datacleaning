"""Pipeline driver — runs all 5 loops, writes artifacts under live/data/runs/{runId}."""

from __future__ import annotations
import sys
import time
import uuid
from pathlib import Path

# Show progress immediately when stdout is piped (Cursor background shells).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live.loops import (  # noqa: E402
    loop1_structure, loop2_cleaning, loop2b_bridge,
    loop3_linking, loop3b_coherence, loop4_confidence,
)

XLSX = ROOT / "data" / "recipe_data.xlsx"
RUNS = Path(__file__).resolve().parent / "data" / "runs"
ROOT_OUTPUT = ROOT / "output.json"
NAMED_OUTPUT = ROOT / "output.py.json"  # never overwritten on resume — use output.py-live-{runId}.json


def resume(run_id: str) -> int:
    """Continue a failed run from loop 3b (requires loop3_linking.json)."""
    out_dir = RUNS / run_id
    linking_path = out_dir / "loop3_linking.json"
    if not linking_path.exists():
        print(f"Error: missing {linking_path} — cannot resume.")
        return 1

    print(f"Resuming run: {run_id}")
    print(f"→ artifacts: {out_dir}\n")

    import json

    linking = json.loads(linking_path.read_text(encoding="utf-8"))
    t0 = time.time()

    loop3b_coherence.run(linking, out_dir)
    linking_path.write_text(json.dumps(linking, indent=2), encoding="utf-8")
    t1 = time.time()
    print(f"  loop 3b took {t1-t0:.1f}s\n")

    loop4_confidence.run(linking, run_id, out_dir, ROOT_OUTPUT)
    t2 = time.time()
    print(f"  loop 4 took {t2-t1:.1f}s")

    named = ROOT / f"output.py-live-{run_id}.json"
    named.write_text(ROOT_OUTPUT.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nResume total: {t2-t0:.1f}s")
    print(f"Output: {ROOT_OUTPUT}")
    print(f"Named copy (preserves output.py.json): {named}")
    return 0


def main() -> int:
    run_id = str(uuid.uuid4())
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pipeline run: {run_id}")
    print(f"→ artifacts: {out_dir}")
    print()

    t0 = time.time()
    structure = loop1_structure.run(XLSX, out_dir)
    t1 = time.time()
    print(f"  loop 1 took {t1-t0:.1f}s\n")

    cleaning = loop2_cleaning.run(structure, XLSX, out_dir)
    t2 = time.time()
    print(f"  loop 2 took {t2-t1:.1f}s\n")

    bridge = loop2b_bridge.run(cleaning, XLSX, out_dir)
    t3 = time.time()
    print(f"  loop 2b took {t3-t2:.1f}s\n")

    linking = loop3_linking.run(structure, cleaning, bridge, XLSX, out_dir)
    t4 = time.time()
    print(f"  loop 3 took {t4-t3:.1f}s\n")

    loop3b_coherence.run(linking, out_dir)
    t4b = time.time()
    print(f"  loop 3b took {t4b-t4:.1f}s\n")

    loop4_confidence.run(linking, run_id, out_dir, ROOT_OUTPUT)
    t5 = time.time()
    print(f"  loop 4 took {t5-t4b:.1f}s")
    print(f"\nTotal: {t5-t0:.1f}s")
    print(f"Output: {ROOT_OUTPUT}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--resume":
        raise SystemExit(resume(sys.argv[2]))
    raise SystemExit(main())
