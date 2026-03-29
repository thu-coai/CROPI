# CROPI package

This directory contains the open-source CROPI code extracted from `rlselect-open/cropi` and cleaned up for standalone use.

## Modules

- `core/get_grad.py`: compute policy gradients from offline rollout data.
- `core/compute_inf_score.py`: compute train-valid influence scores from projected gradients.
- `core/select.py`: select train subsets from score files.
- `utils/split_files.py`: split JSONL files into shards.
- `utils/model_merger.py`: merge sharded checkpoints into Hugging Face format.

## Expected rollout JSONL format

Each record should look like:

```json
{
  "id": 1639,
  "prompt": "Ashley had already blown up 12 balloons...",
  "answer": "95",
  "responses": ["...", "..."],
  "rewards": [1, 0]
}
```

## Typical workflow

1. Split a large rollout file if you want parallel workers.
2. Run `cropi-get-grad` on each shard.
3. Run `cropi-compute-inf-score` to aggregate train-vs-valid influence.
4. Run `cropi-select` to emit selected parquet subsets.

The shell wrappers in `scripts/` call the same Python entry points through `uv run`.
