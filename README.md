# ClarifyVA — Visual Analytics Assistant with Proactive Clarification

ClarifyVA is an uncertainty-aware proactive clarification visual analytics framework that generates Python code for data analysis and proactively identifies uncertain assumptions before execution. 
This repository currently provides the benchmark construction artifacts and curated benchmark data used to evaluate proactive clarification behavior. Code are being cleaned and will be updated as the project evolves

## Data and Artifacts
Built on top of the public data analysis benchmark InfiAgent / DA-Agent, this project generates executable code, extracts analysis plans, and constructs three types of queries—explicit, ambiguous, and infeasible—to test whether models ask clarifying questions at the right time, ask the right questions, and identify contradictions or invalid constraints in tasks.

- `benchmark/data_augmentation/`: Benchmark augmentation pipeline scripts
- `benchmark/dataset/`: Source data and generated JSONL files

### Source Dataset

- `benchmark/dataset/da-dev-questions.jsonl`: Original questions and metadata
- `benchmark/dataset/da-dev-labels.jsonl`: Ground-truth answer labels
- `benchmark/dataset/da-dev-tables/`: CSV tables for tasks

### Generated Data

The following files are produced step-by-step by the augmentation pipeline:

- `benchmark/dataset/da-dev-code.jsonl`: Generated and verified analysis code for each task
- `benchmark/dataset/da-dev-plans.jsonl`: Extracted analysis steps and decision dimensions from code
- `benchmark/dataset/da-dev-constructed.jsonl`: Constructed explicit/ambiguous/infeasible variants
- `benchmark/dataset/constructed-dataset.jsonl`: Curated 150-instance benchmark subset

The curated benchmark contains 150 instances:

| Query type | Count | Purpose |
|---|---:|---|
| `explicit` | 50 | Tests whether the assistant avoids unnecessary clarification |
| `ambiguous` | 50 | Tests whether the assistant asks useful clarification questions for underspecified requests |
| `infeasible` | 50 | Tests whether the assistant detects invalid or contradictory requests |

Each benchmark instance is derived from a unique base task. The released input and gold files are separated to avoid ground-truth leakage during evaluation:

## Data Augmentation

The augmentation pipeline runs in the following order:

1. Generate and verify analysis code: `benchmark/data_augmentation/generate_da_code.py`
2. Retry failed samples: `benchmark/data_augmentation/retry_failed.py`
3. Summarize analysis plans from code: `benchmark/data_augmentation/summarize_plans.py`
4. Normalize dimension keys: `benchmark/data_augmentation/fix_dimension_keys.py`
5. Construct explicit, ambiguous, and infeasible query variants: `benchmark/data_augmentation/construct_queries.py`

## Getting Started

The scripts require the OpenAI API. Set up environment variables first:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://your-endpoint"   # optional
export LLM_MODEL="gpt-4o"                        # optional
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the scripts in order:

```bash
python benchmark/data_augmentation/generate_da_code.py
python benchmark/data_augmentation/retry_failed.py
python benchmark/data_augmentation/summarize_plans.py
python benchmark/data_augmentation/fix_dimension_keys.py
python benchmark/data_augmentation/construct_queries.py --type all
```

To generate only a specific query type, use `--type explicit`, `--type ambiguous`, or `--type infeasible`.

## Notes

- Source tasks come from InfiAgent / DA-Agent; we recommend preserving their data distribution and label format.
- `benchmark/results/` currently has no formal evaluation scripts; focus on data augmentation and sample analysis first.
- Some scripts use legacy path constants; before reproducing, verify that data path configurations in `benchmark/data_augmentation/*.py` match your directory layout.
- `benchmark/dataset/da-dev-constructed.jsonl` is the main pipeline output; `constructed-dataset.jsonl` is a curated subset for evaluation.

## License and Data

- Code in this repository is released under the Apache License 2.0; see [LICENSE](LICENSE).
- The benchmark data is derived from InfiAgent / DA-Agent. Please ensure your usage complies with the original dataset licenses and terms.

## Citation

If you use this repository in your research, please cite the original InfiAgent / DA-Agent sources and this project (a citation entry will be added).

## Additional Documentation

For more detailed augmentation instructions, see:

- `benchmark/data_augmentation/README.md`
- `benchmark/dataset/README.md`

codes will be updated later. 