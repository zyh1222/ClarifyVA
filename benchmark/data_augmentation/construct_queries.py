"""
construct_queries.py
--------------------
Starting from da-dev-plans.jsonl (explicit query + plan), construct three
query types:
    - explicit: keep the original query (tests over-clarification)
    - ambiguous: remove key dimensions (tests targeted clarification)
    - infeasible: inject contradictions/invalid constraints (tests detection)

Each output includes full ground truth (answers + code + dimension labels)
to evaluate clarification accuracy, efficiency (turns), and final task quality.

Output format (da-dev-constructed.jsonl):
{
        "id": 7,
        "original_query": "Apply linear regression ...",
        "constructed_query": "Apply a model to predict ...",
        "query_type": "explicit" | "ambiguous" | "infeasible",
        "modified_dimensions": {...},          # only for ambiguous/infeasible
        "strategy": "drop_method" | "none",
        "file_name": "test_ave.csv",
        "concepts": [...],
        "level": "medium",
        "original_plan_steps": [...],
        "original_dimension_values": {...},
        "ground_truth_answers": [["key", "value"], ...],
        "ground_truth_code": "import pandas as pd\\n..."
}

Usage:
        export OPENAI_API_KEY="sk-..."
        python construct_queries.py [--input da-dev-plans.jsonl] [--output da-dev-constructed.jsonl]
        python construct_queries.py --type explicit   # explicit only
        python construct_queries.py --type ambiguous  # ambiguous only
        python construct_queries.py --type all        # all three (default)
"""

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent

DATA_DIR = REPO_ROOT / "benchmark" / "dataset"
DEFAULT_PLANS_FILE = DATA_DIR / "da-dev-plans.jsonl"
DEFAULT_OUTPUT_FILE = DATA_DIR / "da-dev-constructed.jsonl"
LABELS_FILE = DATA_DIR / "da-dev-labels.jsonl"
CODE_FILE = DATA_DIR / "da-dev-code.jsonl"

MODEL = os.environ.get("LLM_MODEL", "GPT-5.4")


def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
    )


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Load GT data ──────────────────────────────────────────────────────

def load_ground_truth() -> tuple[dict, dict]:
    """Load labels and code, returning {id: answers}, {id: code}."""
    labels = {}
    if LABELS_FILE.exists():
        for item in load_jsonl(LABELS_FILE):
            labels[item["id"]] = item.get("common_answers", [])
    codes = {}
    if CODE_FILE.exists():
        for item in load_jsonl(CODE_FILE):
            codes[item["id"]] = item.get("code", "")
    return labels, codes


# ── Dimension grouping ────────────────────────────────────────────────

def classify_dimensions(dim_values: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key in dim_values:
        cat = key.split(".")[0] if "." in key else "other"
        groups.setdefault(cat, []).append(key)
    return groups


def pick_dims_to_modify(dim_values: dict, strategy: str) -> list[str]:
    groups = classify_dimensions(dim_values)
    all_dims = list(dim_values.keys())

    if strategy == "drop_analysis_method":
        return [d for d in all_dims if "method" in d or "analysis" in d.split(".")[0]]
    elif strategy == "drop_transform":
        return [d for d in all_dims if d.split(".")[0] in ("transform", "missing")]
    elif strategy == "drop_specific_params":
        return [d for d in all_dims if any(kw in d for kw in ("ddof", "random_state", "split", "threshold", "precision", "rounding"))]
    elif strategy == "drop_columns":
        return [d for d in all_dims if "column" in d or "feature" in d]
    elif strategy == "drop_all_implicit":
        return [d for d in all_dims if d.split(".")[0] in ("missing", "output")]
    elif strategy == "drop_multiple":
        selected = []
        for cat_dims in groups.values():
            if cat_dims and len(selected) < 3:
                selected.append(cat_dims[0])
        return selected
    return []


# ── Construction strategies ───────────────────────────────────────────

AMBIGUOUS_STRATEGIES = [
    "drop_analysis_method",
    "drop_transform",
    "drop_specific_params",
    "drop_columns",
    "drop_all_implicit",
    "drop_multiple",
]

INFEASIBLE_STRATEGIES = [
    "wrong_column",
    "contradictory_constraint",
    "type_mismatch",
    "invalid_parameter",
]


# ── LLM prompts ───────────────────────────────────────────────────────

AMBIGUOUS_SYSTEM = """\
You rewrite data analysis queries to make them ambiguous by removing or changing specific details.

Given:
- An original explicit query
- A list of dimensions to remove (each with its current value)
- The full plan steps for context

Rewrite the query so that:
1. The dimensions listed are NO LONGER specified or inferable from the query
2. The rest of the query remains as intact as possible
3. The result should still be a natural, realistic user request (not obviously broken)
4. Do NOT add any new information or requirements

Techniques you can use:
- Replace specific method names with vague terms ("analyze", "examine", "model")
- Remove column names and replace with generic references ("relevant columns", "the data")
- Drop parameter values (random_state, split ratio, ddof)
- Replace specific operations with vague goals ("process the data appropriately")
- Remove constraint clauses entirely if they specify the dimension

Return JSON: {"constructed_query": "...", "reasoning": "brief explanation of what was removed"}"""


INFEASIBLE_SYSTEM = """\
You rewrite data analysis queries to make them infeasible by introducing errors.

Given:
- An original explicit query
- A strategy for what kind of error to inject
- The original dimension values and plan for context

Rewrite the query so that:
1. The query looks natural but contains a logical error or impossibility
2. The error corresponds to the specified strategy
3. The rest of the query stays similar to the original

Strategies:
- "wrong_column": Reference column names that don't exist in the dataset
- "contradictory_constraint": Add constraints that contradict each other (e.g. "use population std with ddof=1", "use one-hot encoding but treat as ordinal")
- "type_mismatch": Ask for operations incompatible with column types (e.g. "calculate mean of the Name column")
- "invalid_parameter": Use impossible parameter values (e.g. "split 150% for training", "set random_state to -1 and also to 42")

Return JSON:
{
    "constructed_query": "...",
    "injected_dimensions": {"dim_name": {"original": "...", "modified": "..."}},
    "reasoning": "what error was injected"
}"""


# ── Build shared output base ─────────────────────────────────────────-

def make_base_output(item: dict, gt_labels: dict, gt_codes: dict) -> dict:
    """Build base fields shared by all query types."""
    qid = item["id"]
    return {
        "id": qid,
        "original_query": item["question"],
        "file_name": item.get("file_name", ""),
        "concepts": item.get("concepts", []),
        "level": item.get("level", ""),
        "original_plan_steps": item.get("plan_steps", []),
        "original_dimension_values": item.get("dimension_values", {}),
        "ground_truth_answers": gt_labels.get(qid, []),
        "ground_truth_code": gt_codes.get(qid, ""),
    }


# ── explicit ─────────────────────────────────────────────────────────-

def construct_explicit(item: dict, gt_labels: dict, gt_codes: dict) -> dict:
    """Keep the original query to test over-clarification."""
    out = make_base_output(item, gt_labels, gt_codes)
    out.update({
        "constructed_query": item["question"],  # Unchanged.
        "query_type": "explicit",
        "modified_dimensions": {},
        "strategy": "none",
    })
    return out


# ── ambiguous ─────────────────────────────────────────────────────────

def construct_ambiguous(
    client: OpenAI, item: dict, strategy: str,
    gt_labels: dict, gt_codes: dict,
) -> dict | None:
    dims = item.get("dimension_values", {})
    target_dims = pick_dims_to_modify(dims, strategy)
    if not target_dims:
        return None

    dims_to_remove = {d: dims[d] for d in target_dims}

    user_msg = f"""\
## Original Query
{item['question']}

## Dimensions to Remove
{json.dumps(dims_to_remove, indent=2, ensure_ascii=False)}

## Full Plan Steps (for context)
{json.dumps(item.get('plan_steps', []), indent=2, ensure_ascii=False)}"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": AMBIGUOUS_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)

    modified_dimensions = {
        d: {"original": dims[d], "modified": "<removed>"}
        for d in target_dims
    }

    out = make_base_output(item, gt_labels, gt_codes)
    out.update({
        "constructed_query": result["constructed_query"],
        "query_type": "ambiguous",
        "modified_dimensions": modified_dimensions,
        "strategy": strategy,
    })
    return out


# ── infeasible ───────────────────────────────────────────────────────-

def construct_infeasible(
    client: OpenAI, item: dict, strategy: str,
    gt_labels: dict, gt_codes: dict,
) -> dict | None:
    dims = item.get("dimension_values", {})
    if not dims:
        return None

    user_msg = f"""\
## Original Query
{item['question']}

## Strategy
{strategy}

## Original Dimension Values
{json.dumps(dims, indent=2, ensure_ascii=False)}

## Full Plan Steps
{json.dumps(item.get('plan_steps', []), indent=2, ensure_ascii=False)}"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": INFEASIBLE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)

    injected = result.get("injected_dimensions", {})
    modified_dimensions = {}
    for k, v in injected.items():
        if isinstance(v, dict):
            modified_dimensions[k] = v
        else:
            modified_dimensions[k] = {"original": dims.get(k, ""), "modified": str(v)}

    out = make_base_output(item, gt_labels, gt_codes)
    out.update({
        "constructed_query": result["constructed_query"],
        "query_type": "infeasible",
        "modified_dimensions": modified_dimensions,
        "strategy": strategy,
    })
    return out


# ── Strategy selection ─────────────────────────────────────────────---

def select_best_ambiguous_strategy(dims: dict) -> str | None:
    groups = classify_dimensions(dims)
    if any("method" in d or "analysis" in d.split(".")[0] for d in dims):
        return "drop_analysis_method"
    if "transform" in groups or "missing" in groups:
        return "drop_transform"
    if any(kw in d for d in dims for kw in ("column", "feature")):
        return "drop_columns"
    if any(kw in d for d in dims for kw in ("ddof", "split", "random_state")):
        return "drop_specific_params"
    if len(dims) >= 3:
        return "drop_multiple"
    return "drop_all_implicit" if dims else None


def select_best_infeasible_strategy(dims: dict, concepts: list) -> str:
    dim_str = " ".join(dims.keys())
    if "column" in dim_str or "feature" in dim_str:
        return "wrong_column"
    if "Machine Learning" in concepts:
        return "invalid_parameter"
    if any(kw in dim_str for kw in ("ddof", "method", "encoding")):
        return "contradictory_constraint"
    return "type_mismatch"


# ── Main flow ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Construct explicit/ambiguous/infeasible queries")
    parser.add_argument("--input", type=str, default=None, help="Input plans jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    parser.add_argument("--start_id", type=int, default=None)
    parser.add_argument("--end_id", type=int, default=None)
    parser.add_argument("--type", choices=["explicit", "ambiguous", "infeasible", "both", "all"],
                        default="all", help="both=ambiguous+infeasible, all=explicit+ambiguous+infeasible")
    args = parser.parse_args()

    plans_file = Path(args.input) if args.input else DEFAULT_PLANS_FILE
    output_file = Path(args.output) if args.output else DEFAULT_OUTPUT_FILE

    if not plans_file.exists():
        print(f"File not found: {plans_file}")
        sys.exit(1)

    # Load data.
    plans = load_jsonl(plans_file)
    gt_labels, gt_codes = load_ground_truth()

    if args.start_id is not None:
        plans = [p for p in plans if p["id"] >= args.start_id]
    if args.end_id is not None:
        plans = [p for p in plans if p["id"] < args.end_id]

    # Determine which types to generate.
    gen_explicit = args.type in ("explicit", "all")
    gen_ambiguous = args.type in ("ambiguous", "both", "all")
    gen_infeasible = args.type in ("infeasible", "both", "all")

    # Resume support.
    existing_keys: set[str] = set()
    if output_file.exists():
        for item in load_jsonl(output_file):
            key = f"{item['id']}_{item['query_type']}_{item.get('strategy', '')}"
            existing_keys.add(key)
        print(f"Found {len(existing_keys)} existing items; skipping")

    client = None
    if gen_ambiguous or gen_infeasible:
        client = make_client()

    types_str = "+".join(t for t, g in [("E", gen_explicit), ("A", gen_ambiguous), ("I", gen_infeasible)] if g)
    print(f"Pending: {len(plans)} plans, types: {types_str}, model: {MODEL}")
    print("-" * 60)

    ok, fail, skip = 0, 0, 0

    for idx, item in enumerate(plans):
        dims = item.get("dimension_values", {})
        qid = item["id"]
        prefix = f"[{idx+1}/{len(plans)}] id={qid}"

        # ── explicit ──
        if gen_explicit:
            key = f"{qid}_explicit_none"
            if key not in existing_keys:
                result = construct_explicit(item, gt_labels, gt_codes)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                ok += 1
            else:
                skip += 1

        if not dims:
            continue

        # ── ambiguous ──
        if gen_ambiguous:
            strategy = select_best_ambiguous_strategy(dims)
            if strategy:
                key = f"{qid}_ambiguous_{strategy}"
                if key not in existing_keys:
                    print(f"{prefix} ambiguous/{strategy}", end=" ")
                    try:
                        result = construct_ambiguous(client, item, strategy, gt_labels, gt_codes)
                        if result:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            ok += 1
                            print(f"✓ ({len(result['modified_dimensions'])} dims)")
                        else:
                            print("skip")
                            skip += 1
                    except Exception as e:
                        fail += 1
                        print(f"✗ {e}")

        # ── infeasible ──
        if gen_infeasible:
            strategy = select_best_infeasible_strategy(dims, item.get("concepts", []))
            key = f"{qid}_infeasible_{strategy}"
            if key not in existing_keys:
                print(f"{prefix} infeasible/{strategy}", end=" ")
                try:
                    result = construct_infeasible(client, item, strategy, gt_labels, gt_codes)
                    if result:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        ok += 1
                        print(f"✓ ({len(result['modified_dimensions'])} dims)")
                    else:
                        print("skip")
                        skip += 1
                except Exception as e:
                    fail += 1
                    print(f"✗ {e}")

    print(f"\nDone: {ok} generated, {skip} skipped, {fail} failed -> {output_file}")

    # Summary stats.
    if output_file.exists():
        all_items = load_jsonl(output_file)
        from collections import Counter
        type_counter = Counter(i["query_type"] for i in all_items)
        print(f"\nTotal: {len(all_items)}")
        for t, c in type_counter.most_common():
            print(f"  {t}: {c}")

        if any(i["query_type"] != "explicit" for i in all_items):
            strat_counter = Counter(
                i["strategy"] for i in all_items if i["query_type"] != "explicit"
            )
            print("Strategy distribution:")
            for s, c in strat_counter.most_common():
                print(f"  {s}: {c}")


if __name__ == "__main__":
    main()
