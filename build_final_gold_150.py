#!/usr/bin/env python3
"""Build final 103-trace adjudicated gold and merge with the prior 47 gold."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def stage_of(annotation):
    return "unclassifiable" if annotation.get("unclassifiable") else annotation["failure_stage"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--adjudication", required=True)
    parser.add_argument("--old-gold", required=True)
    parser.add_argument("--output", default="final_gold_150")
    args = parser.parse_args()

    raw = json.loads(Path(args.raw).read_text())
    adjudicated = json.loads(Path(args.adjudication).read_text())
    old_gold = json.loads(Path(args.old_gold).read_text())
    adj_map = {item["trace_id"]: item["adjudication"] for item in adjudicated}

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    new_gold = []
    for item in raw:
        trace_id = item["trace_id"]
        annotations = item["annotations"]
        if trace_id in adj_map:
            decision = adj_map[trace_id]
            stage = decision["gold_stage"]
            silent = decision["gold_silent"]
            origin = decision["gold_originating_step"]
            reason = decision["adjudication_reason"]
            decision_source = "consensus_adjudication"
        else:
            stages = [stage_of(annotations[name]) for name in "abc"]
            silents = [annotations[name]["is_silent"] for name in "abc"]
            origins = [annotations[name]["originating_step"] for name in "abc"]
            if len(set(stages)) != 1 or len(set(silents)) != 1 or len(set(origins)) != 1:
                raise RuntimeError(f"Non-adjudicated trace is not unanimous: {trace_id}")
            stage, silent, origin = stages[0], silents[0], origins[0]
            reason = annotations["a"]["evidence"]
            decision_source = "unanimous_raw_annotation"

        new_gold.append(
            {
                "trace_id": trace_id,
                "gold_stage": stage,
                "gold_silent": silent,
                "gold_originating_step": origin,
                "gold_reason": reason,
                "decision_source": decision_source,
                "explicit_failure_message": stage == "output_mismatch" and silent is False,
                "annotations": annotations,
            }
        )

    normalized_old = []
    for item in old_gold:
        silent = item.get("gold_silent")
        if isinstance(silent, str):
            silent = silent.lower() == "true"
        normalized_old.append(
            {
                **item,
                "trace_id": item.get("id", item.get("trace_id")),
                "gold_silent": silent,
                "decision_source": "prior_human_adjudication",
            }
        )

    combined = normalized_old + new_gold
    if len(new_gold) != 103 or len(combined) != 150:
        raise RuntimeError(f"Expected 103 new and 150 combined; got {len(new_gold)} and {len(combined)}")

    report = {
        "new_gold_count": len(new_gold),
        "combined_gold_count": len(combined),
        "new_stage_distribution": dict(Counter(item["gold_stage"] for item in new_gold)),
        "new_silent_distribution": dict(Counter(str(item["gold_silent"]).lower() for item in new_gold)),
        "new_decision_sources": dict(Counter(item["decision_source"] for item in new_gold)),
        "explicit_observable_output_mismatches": sum(item["explicit_failure_message"] for item in new_gold),
        "combined_stage_distribution": dict(Counter(item["gold_stage"] for item in combined)),
        "combined_silent_distribution": dict(Counter(str(item["gold_silent"]).lower() for item in combined)),
    }
    (out / "adjudicated_gold_103.json").write_text(json.dumps(new_gold, indent=2))
    (out / "adjudicated_gold_150.json").write_text(json.dumps(combined, indent=2))
    (out / "gold_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
