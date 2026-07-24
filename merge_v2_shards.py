#!/usr/bin/env python3
"""Merge resumable result shards and reject conflicting duplicate runs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged = {}
    for directory in args.inputs:
        for path in Path(directory).glob("*_detail.json"):
            payload = json.loads(path.read_text())
            for run in payload.get("runs", []):
                key = (run.get("model"), run.get("task_id"), int(run.get("rep", 0)))
                prior = merged.get(key)
                if prior is not None and prior != run:
                    # Prefer a successful API record over an ERR placeholder.
                    if prior.get("status") == "ERR" and run.get("status") != "ERR":
                        merged[key] = run
                    elif run.get("status") != "ERR":
                        raise RuntimeError(f"Conflicting non-ERR duplicate: {key}")
                else:
                    merged[key] = run

    runs = sorted(merged.values(), key=lambda r: (r.get("model", ""), r.get("task_id", ""), int(r.get("rep", 0))))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps({"runs": runs}, indent=2))
    tmp.replace(out)
    print(f"Merged {len(runs)} unique runs into {out}")


if __name__ == "__main__":
    main()
