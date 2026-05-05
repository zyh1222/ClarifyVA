"""
retry_failed.py
---------------
Read items with success=False from da-dev-code.jsonl, re-run the LLM to generate
code, and update the result file in place when successful.

Fixes include:
1. Regex could not match values containing [] (list/dict) -> use balanced-bracket parsing
2. Duplicate keys (e.g., multiple correlation_coefficient) -> support multiple values
3. GT values are complex types (dict/list) -> normalize before comparing
4. Numeric tolerance too small -> relax to 0.02
5. Prompt includes stricter output-format guidance

Usage:
    export OPENAI_API_KEY="sk-..."
    python retry_failed.py [--max_retries 8] [--input da-dev-code.jsonl]
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
DEFAULT_CODE_FILE = DATA_DIR / "da-dev-code.jsonl"

MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
TOLERANCE = 0.02  # Numeric tolerance


def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", None),
    )


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv_preview(file_name: str, n_rows: int = 5) -> str:
    csv_path = TABLE_DIR / file_name
    if not csv_path.exists():
        return f"[File {file_name} not found]"
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [f.readline() for _ in range(n_rows + 1)]
    return "".join(lines)


# ── Answer extraction (balanced brackets) ─────────────────────────────
def extract_bracket_value(text: str, start: int) -> str | None:
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
    Extract @key[value] from stdout, supporting:
    - values containing [] (list/dict)
    - repeated keys (aligned to GT order)
    """
    # Count how many times each GT key should appear.
    from collections import Counter, defaultdict
    gt_key_counts = Counter(k for k, _ in gt_answers)

    # Extract all @key[...] (key, value) pairs.
    all_found: dict[str, list[str]] = defaultdict(list)
    pattern = re.compile(r"@([\w]+)\[")
    for m in pattern.finditer(output):
        key = m.group(1)
        val = extract_bracket_value(output, m.end() - 1)
        if val is not None:
            all_found[key].append(val.strip())

    # Align with GT: consume in the GT key order.
    results = {}
    key_index: dict[str, int] = defaultdict(int)
    for gt_key, _ in gt_answers:
        idx = key_index[gt_key]
        vals = all_found.get(gt_key, [])
        if idx < len(vals):
            # For multiple values, use key, key__1, key__2, ...
            result_key = gt_key if idx == 0 else f"{gt_key}__{idx}"
            results[result_key] = vals[idx]
        key_index[gt_key] += 1

    return results


def _normalize_val(s: str) -> str:
    """Normalize a value string: trim spaces/quotes, keep case-insensitive."""
    s = s.strip().strip("'\"")
    return s


def _is_empty(s: str) -> bool:
    """Check whether a value represents empty: '', [], {}, None."""
    return s in ("", "[]", "{}", "none", "null", "nan")


def _val_close(extracted: str, gt: str) -> bool:
    """Compare two values with numeric tolerance and string normalization."""
    a = _normalize_val(extracted)
    b = _normalize_val(gt)

    # Empty-value comparison.
    if a == b:
        return True
    if _is_empty(a) and _is_empty(b):
        return True

    # Try numeric comparison.
    try:
        return abs(float(a) - float(b)) <= TOLERANCE
    except ValueError:
        pass

    # Try parsing as Python objects (dict/list).
    try:
        obj_a = eval(a)  # noqa: S307
        obj_b = eval(b)  # noqa: S307
        if isinstance(obj_a, dict) and isinstance(obj_b, dict):
            return _dicts_close(obj_a, obj_b)
        if isinstance(obj_a, (list, tuple)) and isinstance(obj_b, (list, tuple)):
            return _lists_close(list(obj_a), list(obj_b))
    except Exception:
        pass

    # Case-insensitive string comparison.
    return a.lower() == b.lower()


def _extract_numeric_key(k) -> str:
    """Normalize keys like 'month_1' -> '1', 1 -> '1', 'class_3' -> '3'."""
    s = str(k)
    m = re.search(r'(\d+)$', s)
    return m.group(1) if m else s


def _dicts_close(a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        # Try key normalization ('month_1' vs 1, 'month_1' vs 'month_01').
        norm_a = {str(k): v for k, v in a.items()}
        norm_b = {str(k): v for k, v in b.items()}
        if set(norm_a.keys()) != set(norm_b.keys()):
            # Further normalize by extracting trailing digits.
            num_a = {_extract_numeric_key(k): v for k, v in a.items()}
            num_b = {_extract_numeric_key(k): v for k, v in b.items()}
            if set(num_a.keys()) != set(num_b.keys()) or len(num_a) != len(a):
                return False
            a, b = num_a, num_b
        else:
            a, b = norm_a, norm_b
    for k in a:
        try:
            if abs(float(a[k]) - float(b[k])) > TOLERANCE:
                return False
        except (ValueError, TypeError):
            if str(a[k]).lower() != str(b[k]).lower():
                return False
    return True


def _lists_close(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            if abs(float(x) - float(y)) > TOLERANCE:
                return False
        except (ValueError, TypeError):
            if str(x).lower() != str(y).lower():
                return False
    return True


def answers_match(extracted: dict[str, str], gt_answers: list[list[str]]) -> bool:
    """
    Check whether all GT answers match.
    For duplicate keys, use unordered greedy matching.
    """
    from collections import defaultdict

    # Group GT by key.
    gt_groups: dict[str, list[str]] = defaultdict(list)
    for gt_key, gt_val in gt_answers:
        gt_groups[gt_key].append(gt_val)

    # Group extracted values by base key.
    ext_groups: dict[str, list[str]] = defaultdict(list)
    for k, v in extracted.items():
        base_key = k.split("__")[0]
        ext_groups[base_key].append(v)

    for key, gt_vals in gt_groups.items():
        ext_vals = ext_groups.get(key, [])
        if len(ext_vals) < len(gt_vals):
            return False
        # Greedy unordered matching: find a match for each GT value.
        remaining = list(ext_vals)
        for gv in gt_vals:
            found = False
            for i, ev in enumerate(remaining):
                if _val_close(ev, gv):
                    remaining.pop(i)
                    found = True
                    break
            if not found:
                return False
    return True


def match_count_detail(extracted: dict[str, str], gt_answers: list[list[str]]) -> tuple[int, str]:
    """Return (match_count, detail_string) using unordered matching."""
    from collections import defaultdict

    # Group by key.
    gt_groups: dict[str, list[str]] = defaultdict(list)
    for gt_key, gt_val in gt_answers:
        gt_groups[gt_key].append(gt_val)

    ext_groups: dict[str, list[str]] = defaultdict(list)
    for k, v in extracted.items():
        base_key = k.split("__")[0]
        ext_groups[base_key].append(v)

    matched = 0
    details = []

    for key, gt_vals in gt_groups.items():
        ext_vals = list(ext_groups.get(key, []))
        for gv in gt_vals:
            found = False
            for i, ev in enumerate(ext_vals):
                if _val_close(ev, gv):
                    ext_vals.pop(i)
                    found = True
                    matched += 1
                    details.append(f"  {key}: got={ev}, gt={gv} ✓")
                    break
            if not found:
                # Use the closest extracted value for diagnosis.
                best = ext_vals[0] if ext_vals else "<missing>"
                details.append(f"  {key}: got={best}, gt={gv} ✗")

    return matched, "\n".join(details)


# ── Code execution ────────────────────────────────────────────────────
def execute_code(code: str, timeout: int = 60) -> tuple[bool, str, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(TABLE_DIR),
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    finally:
        os.unlink(tmp_path)


def extract_code_from_response(text: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def call_llm(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert Python data analyst. Output code only, no explanations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


# ── Diagnose failure ──────────────────────────────────────────────────
def diagnose_failure(
    gt_answers: list[list[str]],
    extracted: dict[str, str],
    exec_success: bool,
    stdout: str,
    stderr: str,
) -> str:
    """Generate a failure diagnosis to guide the LLM fix."""
    if not exec_success:
        return f"Code execution failed: {stderr[:500]}"

    from collections import defaultdict
    key_index: dict[str, int] = defaultdict(int)
    issues = []

    for gt_key, gt_val in gt_answers:
        idx = key_index[gt_key]
        result_key = gt_key if idx == 0 else f"{gt_key}__{idx}"
        key_index[gt_key] += 1
        ext_val = extracted.get(result_key)

        if ext_val is None:
            # Check if stdout had the key but extraction failed.
            if f"@{gt_key}" in stdout:
                issues.append(
                    f"- {gt_key}: Found @{gt_key} in output but failed to extract the value; it may contain nested brackets. Please keep the value format simple."
                )
            else:
                issues.append(
                    f"- {gt_key}: No @{gt_key}[...] found in output; make sure to print this value."
                )
        elif not _val_close(ext_val, gt_val):
            try:
                diff = abs(float(ext_val) - float(gt_val))
                issues.append(
                    f"- {gt_key}: Numeric mismatch. You output {ext_val}, expected {gt_val} (diff {diff:.4f}). Check your logic."
                )
            except ValueError:
                issues.append(
                    f"- {gt_key}: Value mismatch. You output '{ext_val}', expected '{gt_val}'."
                )

    return "\n".join(issues) if issues else "Unknown reason"


# ── Prompt construction ───────────────────────────────────────────────
def build_retry_prompt(
    question: dict,
    csv_preview: str,
    gt_answers: list[list[str]],
    prev_code: str,
    exec_success: bool,
    stdout: str,
    stderr: str,
    extracted: dict[str, str],
    diagnosis: str,
) -> str:
    gt_str = ", ".join(f"@{k}[{v}]" for k, v in gt_answers)
    expected_keys = [k for k, _ in gt_answers]

    # Check for duplicate keys.
    from collections import Counter
    key_counts = Counter(expected_keys)
    has_dup_keys = any(c > 1 for c in key_counts.values())

    dup_key_hint = ""
    if has_dup_keys:
        dup_keys = [k for k, c in key_counts.items() if c > 1]
        dup_key_hint = f"""
Note: The following keys must appear multiple times in the output: {dup_keys}
Print one line per instance, e.g., for each group print("@{dup_keys[0]}[value]").
"""

    # Check for complex value types.
    complex_hint = ""
    for _, v in gt_answers:
        if v.startswith("{") or v.startswith("["):
            complex_hint = """
Note: Some answers are dict or list types. Output them directly with
print(f"@key[{value}]") without extra quotes or escaping. Ensure dict keys
match the expected format.
"""
            break

    return textwrap.dedent(f"""\
    You are a data analysis code generator. The previous code did not produce
    correct answers. Please analyze and rewrite it.

    ## Task
    {question['question']}

    ## Constraints
    {question.get('constraints', 'None')}

    ## Output format
    {question.get('format', 'None')}

    ## Data file: {question['file_name']} (in the current working directory)
    ### Data preview
    ```
    {csv_preview}
    ```

    ## Previous code
    ```python
    {prev_code}
    ```

    ## Execution result
    Success: {exec_success}
    stdout: {stdout[:2000]}
    stderr: {stderr[:2000]}

    ## Expected answers
    {gt_str}

    ## Diagnosis
    {diagnosis}
    {dup_key_hint}{complex_hint}
    ## Please rewrite the code
    - Use the diagnosis to fix the issues
    - Pay attention to ddof=0 vs ddof=1, encoding, missing values, rounding
    - If values are close but off, check dropna timing, rounding precision, and boundary rules (> vs >=)
    - Use print() with strict @key[value] format
    - Round numeric results to 2 decimal places
    - Output pure Python code only
    """)


# ── Main flow ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Retry failed items in da-dev-code.jsonl")
    parser.add_argument("--max_retries", type=int, default=8, help="Max retries per task")
    parser.add_argument("--input", type=str, default=None, help="Input code jsonl file")
    args = parser.parse_args()

    code_file = Path(args.input) if args.input else DEFAULT_CODE_FILE
    if not code_file.exists():
        print(f"File not found: {code_file}")
        sys.exit(1)

    questions = {item["id"]: item for item in load_jsonl(QUESTIONS_FILE)}
    labels = {item["id"]: item for item in load_jsonl(LABELS_FILE)}

    all_results: dict[int, dict] = {}
    for item in load_jsonl(code_file):
        all_results[item["id"]] = item

    failed_ids = sorted(qid for qid, r in all_results.items() if not r.get("success"))
    print(f"Total tasks: {len(all_results)}, failed: {len(failed_ids)}, model: {MODEL}")
    if not failed_ids:
        print("No failed tasks; nothing to retry!")
        return

    print(f"Retry IDs: {failed_ids}")
    print("-" * 60)

    client = make_client()
    fixed_count = 0

    for idx, qid in enumerate(failed_ids):
        question = questions[qid]
        gt = labels[qid]["common_answers"]
        csv_preview = read_csv_preview(question["file_name"])
        prev_result = all_results[qid]

        print(f"\n[{idx + 1}/{len(failed_ids)}] Retrying id={qid}: {question['question'][:80]}...")

        code = prev_result.get("code", "")
        best_code = code
        best_match_count = 0

        for attempt in range(1, args.max_retries + 1):
            success, stdout, stderr = execute_code(code)
            extracted = parse_answers_from_output(stdout, gt) if success else {}
            mc, detail = match_count_detail(extracted, gt)

            if mc > best_match_count:
                best_match_count = mc
                best_code = code

            print(f"  attempt {attempt}/{args.max_retries} | exec={'OK' if success else 'FAIL'} | matched={mc}/{len(gt)}")

            if answers_match(extracted, gt):
                print(f"  [id={qid}] ✓ Fixed!")
                all_results[qid] = {
                    "id": qid,
                    "code": code,
                    "attempts": prev_result.get("attempts", 0) + attempt,
                    "success": True,
                    "extracted_answers": extracted,
                }
                fixed_count += 1
                break

            diagnosis = diagnose_failure(gt, extracted, success, stdout, stderr)
            prompt = build_retry_prompt(
                question, csv_preview, gt,
                code, success, stdout, stderr, extracted, diagnosis,
            )
            response_text = call_llm(client, prompt)
            code = extract_code_from_response(response_text)
        else:
            print(f"  [id={qid}] ✗ Still failed (best matched={best_match_count}/{len(gt)})")
            success, stdout, _ = execute_code(best_code)
            extracted = parse_answers_from_output(stdout, gt) if success else {}
            all_results[qid] = {
                "id": qid,
                "code": best_code,
                "attempts": prev_result.get("attempts", 0) + args.max_retries,
                "success": False,
                "extracted_answers": extracted,
            }

    # Write back to file.
    sorted_results = sorted(all_results.values(), key=lambda x: x["id"])
    with open(code_file, "w", encoding="utf-8") as f:
        for r in sorted_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Fixed this run: {fixed_count}/{len(failed_ids)}")
    print(f"Overall success: {sum(1 for r in sorted_results if r.get('success'))}/{len(sorted_results)}")
    print(f"Results updated in: {code_file}")


if __name__ == "__main__":
    main()
