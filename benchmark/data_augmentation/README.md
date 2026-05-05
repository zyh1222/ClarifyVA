# Clarification Query Construction Pipeline

This directory converts raw data-analysis benchmark tasks into evaluation data for clarification behavior.

Data source note: the original tasks come from the public benchmark **InfiAgent (DA-Agent)**. This project builds on top of it by generating executable code, extracting plan dimensions, and constructing three query types: explicit, ambiguous, and infeasible.

---

## 1. Script Overview

- `generate_da_code.py`
  - Input: `da-dev-questions.jsonl` + `da-dev-labels.jsonl`
  - Process: calls an LLM to generate executable Python code and checks it against ground-truth answers
  - Output: `da-dev-code.jsonl`

- `retry_failed.py`
  - Input: failed items in `da-dev-code.jsonl`
  - Process: retries code generation with error-aware feedback
  - Output: updates `da-dev-code.jsonl` in place

- `summarize_plans.py`
  - Input: `da-dev-code.jsonl`
  - Process: calls an LLM to extract `plan_steps` and `dimension_values` from code
  - Output: `da-dev-plans.jsonl`

- `fix_dimension_keys.py`
  - Input: `da-dev-plans.jsonl` / `da-dev-constructed.jsonl`
  - Process: normalizes dimension keys by removing category prefixes
  - Output: updates files in place

- `construct_queries.py`
  - Input: `da-dev-plans.jsonl` (plus labels/code as ground truth)
  - Process: constructs
    - `explicit`: keeps the original query unchanged
    - `ambiguous`: removes key decision details (so the model should clarify)
    - `infeasible`: injects contradictions or invalid constraints (so the model should detect infeasibility)
  - Output: `da-dev-constructed.jsonl`

---

## 2. Recommended Execution Order

Run from the repository root:

```bash
export OPENAI_API_KEY="your_key"
# optional
export OPENAI_BASE_URL="https://your-endpoint"
export LLM_MODEL="gpt-4o"

python benchmark/data_augmentation/generate_da_code.py
python benchmark/data_augmentation/retry_failed.py
python benchmark/data_augmentation/summarize_plans.py
python benchmark/data_augmentation/fix_dimension_keys.py
python benchmark/data_augmentation/construct_queries.py --type all
```

For resume/range runs:

```bash
python benchmark/data_augmentation/construct_queries.py --type ambiguous --start_id 0 --end_id 100
```

---

## 3. Data File Conventions

Recommended unified location:

- `benchmark/dataset/da-dev-questions.jsonl`
- `benchmark/dataset/da-dev-labels.jsonl`
- `benchmark/dataset/da-dev-code.jsonl`
- `benchmark/dataset/da-dev-plans.jsonl`
- `benchmark/dataset/da-dev-constructed.jsonl`
- `benchmark/dataset/da-dev-tables/`

---

## 4. Prompt Customization (Important for Open Source)

You can directly modify prompts based on your research goals:

- `generate_da_code.py`
  - `build_initial_prompt(...)`
  - `build_retry_prompt(...)`

- `summarize_plans.py`
  - `SYSTEM_PROMPT`

- `construct_queries.py`
  - `AMBIGUOUS_SYSTEM`
  - `INFEASIBLE_SYSTEM`

Common customizations:

- increase/decrease ambiguity strength for ambiguous queries
- constrain infeasible error injection types
- make dimension extraction more fine-grained or more compact
- restrict coding style or allowed libraries

---

## 5. Quality Checks

After each construction run, validate:

1. Record completeness: `id`, `query_type`, `strategy`, and `ground_truth_*` fields
2. Strategy balance: no severe skew toward one ambiguous/infeasible strategy
3. Consistency: `modified_dimensions` should match `constructed_query`
4. Reproducibility: pin `LLM_MODEL`, keep environment settings and run logs

---

## 6. Minimal Run Example

Generate one explicit sample (useful for path/output sanity check):

```bash
python benchmark/data_augmentation/construct_queries.py \
  --type explicit \
  --start_id 0 \
  --end_id 1 \
  --output benchmark/dataset/_path_check_constructed.jsonl
```

If the output file is created successfully, path resolution and write flow are working.
