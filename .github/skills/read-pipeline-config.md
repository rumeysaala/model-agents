---
name: read-pipeline-config
description: reads reports/pipeline_config.json and validates MODEL_TYPE; handles missing file with context-aware error
---

```python
import json, os

CONFIG_PATH = "reports/pipeline_config.json"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"{CONFIG_PATH} not found.\n"
        "Run Preprocess agent first; it will benchmark all 3 models and ask for your selection."
    )

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

if "MODEL_TYPE" not in cfg:
    raise KeyError(
        "MODEL_TYPE not found in pipeline_config.json.\n"
        "Benchmark is done — reply with your model choice: catboost / lightgbm / xgboost"
    )

MODEL_TYPE = cfg["MODEL_TYPE"]
assert MODEL_TYPE in {"catboost", "lightgbm", "xgboost"}, \
    f"Invalid MODEL_TYPE: '{MODEL_TYPE}'. Must be catboost / lightgbm / xgboost"

USE_GPU = bool(cfg.get("use_gpu", False))
assert cfg.get("approval_confirmed") is True, "Champion MODEL_TYPE not approved by the user (Gate 2)"
print(f"✅ MODEL_TYPE = {MODEL_TYPE}  USE_GPU = {USE_GPU}")
```
