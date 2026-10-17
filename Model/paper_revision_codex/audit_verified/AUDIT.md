# Saved result audit

Condition: if_events; test fraction: 0.2
Verified files: 327

| Finding | Evidence |
|---|---|
| PLAN_SPLIT_MISSING | Saved plan omits test_size; dataset records 0.2. |
| SOURCE_SPLIT_DRIFT | Current source test_size=0.1; saved dataset=0.2. |
| ALL_TRAIN_PURCHASES_REMOVED | Filtering removed every training transaction. |
| SPARSITY_TABLE_MISMATCH | Table=99.9800%; unique-pair sparsity=99.989517%. |
| STALE_PROGRESS | progress.json differs from verified completion markers; do not retrain solely from progress.json. |
