"""
summarize_plans.py
------------------
Read da-dev-code.jsonl and call an LLM to generate a natural-language analytical
plan for each task's code, including step-by-step descriptions and decision
dimensions. Write the output to da-dev-plans.jsonl.

Usage:
    export OPENAI_API_KEY="sk-..."
    python summarize_plans.py [--input da-dev-code.jsonl] [--output da-dev-plans.jsonl]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dataset"
QUESTIONS_FILE = DATA_DIR / "da-dev-questions.jsonl"
LABELS_FILE = DATA_DIR / "da-dev-labels.jsonl"
DEFAULT_CODE_FILE = DATA_DIR / "da-dev-code.jsonl"
DEFAULT_OUTPUT_FILE = DATA_DIR / "da-dev-plans.jsonl"

MODEL = os.environ.get("LLM_MODEL", "gpt-4o")


def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
    )


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


SYSTEM_PROMPT = """\
You are an expert at reading Python data-analysis code.

Given a question and its code, produce a JSON object with two fields:

1. "plan_steps": a list of strings describing what the code does step by step, in execution order. Each step should be a specific, self-contained natural-language sentence. Be precise: include column names, function calls, literal parameter values, and any implicit choices the code makes (e.g. silently ignoring NaN, using default ddof, inclusive vs exclusive bin edges). Cover every meaningful operation — do not skip or merge steps.

2. "dimension_values": a flat dict capturing every decision point in the code that could have been made differently. A "decision point" is any place where the code picks one option from multiple plausible alternatives that would change the output.

   Each key should be a short, specific name for the decision — NO category prefix.
   Use the same naming style a data analyst would use: "correlation_method",
   "encoding_method", "train_test_split", "std_ddof", etc.

   Each value should be a string describing the concrete choice made.

   Examples (for illustration only — extract whatever is actually in the code):
     "columns_used": "Fare, Pclass"
     "encoding_method": "pd.get_dummies (one-hot)"
     "bin_edges": "[0, 13, 20, 60, inf] (right-exclusive)"
     "missing_value_strategy": "df.dropna(subset=['Age', 'Embarked'])"
     "statistical_method": "LinearRegression from sklearn"
     "train_test_split": "80/20, random_state=42"
     "std_ddof": "0 (population)"
     "correlation_method": "pearson"
     "outlier_method": "IQR with 1.5x threshold"

   Do NOT invent dimensions that are not in the code. Only report what the code actually does.
   Do NOT include trivial/fixed dimensions like source file name, output rounding, or print formatting
   — these are specified by the task and are not analytical decisions.

Return ONLY valid JSON, no markdown fences, no explanation."""


def build_prompt(question: dict, code: str) -> str:
    return f"""\
Question: {question['question']}
Constraints: {question.get('constraints', 'None')}

Code:
{code}"""


def call_llm(client: OpenAI, question: dict, code: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, code)},
        ],
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--start_id", type=int, default=None)
    parser.add_argument("--end_id", type=int, default=None)
    args = parser.parse_args()

    code_file = Path(args.input) if args.input else DEFAULT_CODE_FILE
    output_file = Path(args.output) if args.output else DEFAULT_OUTPUT_FILE

    if not code_file.exists():
        print(f"File not found: {code_file}")
        sys.exit(1)

    questions = {item["id"]: item for item in load_jsonl(QUESTIONS_FILE)}
    code_items = {item["id"]: item for item in load_jsonl(code_file)}

    # Resume support.
    existing: set[int] = set()
    if output_file.exists():
        for item in load_jsonl(output_file):
            existing.add(item["id"])
        print(f"Found {len(existing)} existing items; skipping")

    all_ids = sorted(code_items.keys())
    if args.start_id is not None:
        all_ids = [i for i in all_ids if i >= args.start_id]
    if args.end_id is not None:
        all_ids = [i for i in all_ids if i < args.end_id]
    todo_ids = [i for i in all_ids if i not in existing]

    print(f"Pending: {len(todo_ids)}, model: {MODEL}")

    client = make_client()
    ok, fail = 0, 0

    for idx, qid in enumerate(todo_ids):
        code = code_items[qid].get("code", "")
        if not code.strip():
            continue
        question = questions.get(qid, {})
        print(f"[{idx+1}/{len(todo_ids)}] id={qid}", end=" ")

        try:
            plan = call_llm(client, question, code)
            result = {
                "id": qid,
                "question": question.get("question", ""),
                "concepts": question.get("concepts", []),
                "level": question.get("level", ""),
                "file_name": question.get("file_name", ""),
                "plan_steps": plan.get("plan_steps", []),
                "dimension_values": plan.get("dimension_values", {}),
                "code": code,
            }
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            ok += 1
            dims = result['dimension_values']
            print(f"✓ {len(result['plan_steps'])} steps, {len(dims)} dims")
        except Exception as e:
            fail += 1
            print(f"✗ {e}")

    print(f"\nDone: {ok} succeeded, {fail} failed -> {output_file}")


if __name__ == "__main__":
    main()
