"""
generate_da_code.py
-------------------
For each task in InfiAgent-DABench, call an LLM to generate minimal Python code
for data analysis. Iterate until the output matches the ground-truth answers,
then save the code to da-dev-code.jsonl.

Usage:
    export OPENAI_API_KEY="sk-..."
    # Optional: export OPENAI_BASE_URL="https://..."
    # Optional: export LLM_MODEL="gpt-4o"
    python generate_da_code.py [--max_retries 5] [--start_id 0] [--end_id 257]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dataset"
TABLE_DIR = DATA_DIR / "da-dev-tables"
QUESTIONS_FILE = DATA_DIR / "da-dev-questions.jsonl"
LABELS_FILE = DATA_DIR / "da-dev-labels.jsonl"
OUTPUT_FILE = DATA_DIR / "da-dev-code.jsonl"

# ── LLM client ────────────────────────────────────────────────────────
def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
    )

MODEL = os.environ.get("LLM_MODEL", "gpt-4o")

# ── Data loading ──────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions() -> dict[int, dict]:
    return {item["id"]: item for item in load_jsonl(QUESTIONS_FILE)}


def load_labels() -> dict[int, dict]:
    return {item["id"]: item for item in load_jsonl(LABELS_FILE)}


def read_csv_preview(file_name: str, n_rows: int = 5) -> str:
    """Read the first n rows of a CSV as a preview for the LLM."""
    csv_path = TABLE_DIR / file_name
    if not csv_path.exists():
        return f"[File {file_name} not found]"
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [f.readline() for _ in range(n_rows + 1)]
    return "".join(lines)


# ── Answer parsing and matching ───────────────────────────────────────
TOLERANCE = 0.02


def _extract_bracket_value(text: str, start: int) -> str | None:
    """Extract a balanced-bracket value starting at text[start] == '['."""
    if start >= len(text) or text[start] != '[':
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def parse_answers_from_output(output: str, gt_answers: list[list[str]]) -> dict[str, str]:
    """
    Extract @key[value] from stdout, supporting nested brackets and repeated keys.
    """
    from collections import defaultdict, Counter

    all_found: dict[str, list[str]] = defaultdict(list)
    for m in re.finditer(r"@([\w]+)\[", output):
        val = _extract_bracket_value(output, m.end() - 1)
        if val is not None:
            all_found[m.group(1)].append(val.strip())

    results = {}
    key_index: dict[str, int] = defaultdict(int)
    for gt_key, _ in gt_answers:
        idx = key_index[gt_key]
        vals = all_found.get(gt_key, [])
        if idx < len(vals):
            result_key = gt_key if idx == 0 else f"{gt_key}__{idx}"
            results[result_key] = vals[idx]
        key_index[gt_key] += 1
    return results


def _is_empty(s: str) -> bool:
    return s in ("", "[]", "{}", "none", "null", "nan")


def _val_close(a: str, b: str) -> bool:
    a, b = a.strip().strip("'\""), b.strip().strip("'\"")
    if a == b:
        return True
    if _is_empty(a) and _is_empty(b):
        return True
    try:
        return abs(float(a) - float(b)) <= TOLERANCE
    except ValueError:
        pass
    def _num_key(k):
        s = str(k)
        m = re.search(r'(\d+)$', s)
        return m.group(1) if m else s

    try:
        obj_a, obj_b = eval(a), eval(b)  # noqa: S307
        if isinstance(obj_a, dict) and isinstance(obj_b, dict):
            na = {str(k): v for k, v in obj_a.items()}
            nb = {str(k): v for k, v in obj_b.items()}
            if set(na) != set(nb):
                na = {_num_key(k): v for k, v in obj_a.items()}
                nb = {_num_key(k): v for k, v in obj_b.items()}
                if set(na) != set(nb) or len(na) != len(obj_a):
                    return False
            return all(
                abs(float(na[k]) - float(nb[k])) <= TOLERANCE
                if isinstance(na[k], (int, float)) and isinstance(nb[k], (int, float))
                else str(na[k]).lower() == str(nb[k]).lower()
                for k in na
            )
        if isinstance(obj_a, (list, tuple)) and isinstance(obj_b, (list, tuple)):
            if len(obj_a) != len(obj_b):
                return False
            return all(
                abs(float(x) - float(y)) <= TOLERANCE
                if isinstance(x, (int, float)) and isinstance(y, (int, float))
                else str(x).lower() == str(y).lower()
                for x, y in zip(obj_a, obj_b)
            )
    except Exception:
        pass
    return a.lower() == b.lower()


def answers_match(extracted: dict[str, str], gt_answers: list[list[str]]) -> bool:
    from collections import defaultdict
    # Unordered match: group by key and greedily pair values.
    gt_groups: dict[str, list[str]] = defaultdict(list)
    for gt_key, gt_val in gt_answers:
        gt_groups[gt_key].append(gt_val)

    ext_groups: dict[str, list[str]] = defaultdict(list)
    for k, v in extracted.items():
        base_key = k.split("__")[0]
        ext_groups[base_key].append(v)

    for key, gt_vals in gt_groups.items():
        ext_vals = list(ext_groups.get(key, []))
        if len(ext_vals) < len(gt_vals):
            return False
        for gv in gt_vals:
            found = False
            for i, ev in enumerate(ext_vals):
                if _val_close(ev, gv):
                    ext_vals.pop(i)
                    found = True
                    break
            if not found:
                return False
    return True


# ── Code execution (sandboxed subprocess) ─────────────────────────────
def execute_code(code: str, timeout: int = 60) -> tuple[bool, str, str]:
    """Execute Python code in a subprocess and return (success, stdout, stderr)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(TABLE_DIR),  # Use the CSV directory as the working dir.
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    finally:
        os.unlink(tmp_path)


# ── LLM interaction ───────────────────────────────────────────────────
def build_initial_prompt(question: dict, csv_preview: str, gt_answers: list[list[str]]) -> str:
    expected_keys = [k for k, _ in gt_answers]
    expected_output_example = "  ".join(f"@{k}[value]" for k in expected_keys)

    return textwrap.dedent(f"""\
    You are a data analysis code generator. Based on the information below,
    produce **as short as possible** Python code to complete the task.

    ## Task
    {question['question']}

    ## Constraints
    {question.get('constraints', 'None')}

    ## Output format
    {question.get('format', 'None')}

    ## Data file
    File name: {question['file_name']} (in the current working directory)

    ### Data preview (first 5 rows)
    ```
    {csv_preview}
    ```

    ## Requirements
    1. Use print() to output results, strictly in this format: {expected_output_example}
    2. Keep the code short; use only pandas/numpy/sklearn or standard libraries
    3. Read "{question['file_name']}" directly (file is in the current directory)
    4. Round numeric results to 2 decimal places
    5. Output pure Python code only; no explanations

    Please generate the code:
    """)


def build_retry_prompt(
    question: dict,
    csv_preview: str,
    gt_answers: list[list[str]],
    prev_code: str,
    exec_success: bool,
    stdout: str,
    stderr: str,
    extracted: dict[str, str],
) -> str:
    gt_str = ", ".join(f"@{k}[{v}]" for k, v in gt_answers)

    mismatch_info = ""
    if exec_success and extracted:
        for k, gt_v in gt_answers:
            ext_v = extracted.get(k, "<missing>")
            match = "✓" if _val_close(ext_v, gt_v) else "✗"
            mismatch_info += f"  {k}: your_output={ext_v}, expected={gt_v} {match}\n"

    return textwrap.dedent(f"""\
    Your previous code did not produce the correct answers. Please fix it.

    ## Task
    {question['question']}

    ## Constraints
    {question.get('constraints', 'None')}

    ## Output format
    {question.get('format', 'None')}

    ## Data file: {question['file_name']}
    ### Data preview
    ```
    {csv_preview}
    ```

    ## Your previous code
    ```python
    {prev_code}
    ```

    ## Execution result
    Success: {exec_success}
    stdout: {stdout[:2000]}
    stderr: {stderr[:2000]}

    ## Answer comparison
    Expected: {gt_str}
    {mismatch_info}

    ## Please fix the code
    - Identify incorrect values and adjust the logic
    - Respect constraints (e.g., population vs sample std, encoding choice)
    - Round numeric results to 2 decimal places
    - Output pure Python code only
    """)


def extract_code_from_response(response_text: str) -> str:
    """Extract a code block from the LLM response."""
    # Try to extract a ```python ... ``` block.
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to extract a ``` ... ``` block.
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no code fences exist, return the full text (may already be code).
    return response_text.strip()


def call_llm(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert Python data analyst. Output code only, no explanations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


# ── Main flow ──────────────────────────────────────────────────────────
def process_one(
    client: OpenAI,
    qid: int,
    question: dict,
    gt_answers: list[list[str]],
    max_retries: int,
) -> dict:
    """Process one task and return a result dict."""
    csv_preview = read_csv_preview(question["file_name"])
    expected_keys = [k for k, _ in gt_answers]

    # First attempt.
    prompt = build_initial_prompt(question, csv_preview, gt_answers)
    response_text = call_llm(client, prompt)
    code = extract_code_from_response(response_text)

    best_code = code
    best_match_count = 0

    for attempt in range(1, max_retries + 1):
        success, stdout, stderr = execute_code(code)
        extracted = parse_answers_from_output(stdout, gt_answers) if success else {}

        # Count matches (unordered).
        from collections import defaultdict
        _gt_grp: dict[str, list[str]] = defaultdict(list)
        for k, v in gt_answers:
            _gt_grp[k].append(v)
        _ext_grp: dict[str, list[str]] = defaultdict(list)
        for k, v in extracted.items():
            _ext_grp[k.split("__")[0]].append(v)
        match_count = 0
        for k, gvs in _gt_grp.items():
            evs = list(_ext_grp.get(k, []))
            for gv in gvs:
                for i, ev in enumerate(evs):
                    if _val_close(ev, gv):
                        evs.pop(i)
                        match_count += 1
                        break
        if match_count > best_match_count:
            best_match_count = match_count
            best_code = code

        print(
            f"  [id={qid}] attempt {attempt}/{max_retries} | "
            f"exec={'OK' if success else 'FAIL'} | "
            f"matched={match_count}/{len(gt_answers)}"
        )

        if answers_match(extracted, gt_answers):
            print(f"  [id={qid}] ✓ All answers matched!")
            return {
                "id": qid,
                "code": code,
                "attempts": attempt,
                "success": True,
                "extracted_answers": extracted,
            }

        if attempt < max_retries:
            # Retry.
            retry_prompt = build_retry_prompt(
                question, csv_preview, gt_answers,
                code, success, stdout, stderr, extracted,
            )
            response_text = call_llm(client, retry_prompt)
            code = extract_code_from_response(response_text)

    # Retries exhausted; keep the best code.
    print(f"  [id={qid}] ✗ Retries exhausted; keep best (matched={best_match_count}/{len(gt_answers)})")
    # Re-run best_code to get final output.
    success, stdout, _ = execute_code(best_code)
    extracted = parse_answers_from_output(stdout, expected_keys) if success else {}
    return {
        "id": qid,
        "code": best_code,
        "attempts": max_retries,
        "success": False,
        "extracted_answers": extracted,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate analysis code for each DA-Bench task")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries per task")
    parser.add_argument("--start_id", type=int, default=None, help="Start ID (inclusive)")
    parser.add_argument("--end_id", type=int, default=None, help="End ID (exclusive)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    client = make_client()
    questions = load_questions()
    labels = load_labels()
    output_path = Path(args.output) if args.output else OUTPUT_FILE

    # Load existing results (supports resume).
    existing_results: dict[int, dict] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_results[item["id"]] = item
        print(f"Loaded {len(existing_results)} existing results; skipping those tasks")

    # Determine the IDs to process.
    all_ids = sorted(set(questions.keys()) & set(labels.keys()))
    if args.start_id is not None:
        all_ids = [i for i in all_ids if i >= args.start_id]
    if args.end_id is not None:
        all_ids = [i for i in all_ids if i < args.end_id]

    print(f"Total tasks to process: {len(all_ids)}, model: {MODEL}")
    print(f"Output file: {output_path}")
    print("-" * 60)

    results = []
    success_count = 0

    for idx, qid in enumerate(all_ids):
        if qid in existing_results:
            results.append(existing_results[qid])
            if existing_results[qid].get("success"):
                success_count += 1
            continue

        question = questions[qid]
        gt = labels[qid]["common_answers"]
        print(f"\n[{idx + 1}/{len(all_ids)}] Processing id={qid}: {question['question'][:80]}...")

        result = process_one(client, qid, question, gt, args.max_retries)
        results.append(result)
        if result["success"]:
            success_count += 1

        # Append after each task (resume-safe).
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Done! Success: {success_count}/{len(all_ids)}")
    print(f"Results saved to: {output_path}")

    # Write a sorted full version.
    sorted_output = output_path.with_suffix(".sorted.jsonl")
    results.sort(key=lambda x: x["id"])
    with open(sorted_output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Sorted version: {sorted_output}")


if __name__ == "__main__":
    main()
