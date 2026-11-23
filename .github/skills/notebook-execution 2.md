---
name: notebook-execution
description: write the stage .ipynb directly, run all cells with the VS Code notebook tools, read only the stage summary — no papermill / jupytext
---

# Notebook Execution Policy (all stage agents)

```
write .ipynb once  →  run all cells (VS Code)  →  read summary.json  →  append analysis cell  →  verify
```

1. **Write** `notebooks/NN_<stage>.ipynb` directly in one file write: markdown title per step,
   one responsibility per code cell, skill code pasted **verbatim**. First code cell resolves the
   project root in code (`config/pipeline_contracts.json` marker) — never rely on IDE settings.
   The last code cell calls `write_stage_summary(...)` (#file:stage-summary.md).
2. **Run** all cells in order with the VS Code notebook tools. Do not read outputs while it runs;
   the data scientist watches them in the open notebook.
3. **On error**: read only the failing cell's error, fix that cell, re-run from it. Max 3 attempts,
   then stop and report the traceback tail.
4. **Read** only `reports/summaries/<stage>_summary.json`. If a number is missing, add it to the
   summary call and re-run that cell — do not read other cell outputs.
5. **Analysis**: append a markdown cell (≤ 15 lines: what was done, key numbers, anything
   suspicious, what to check next).
6. **Verify** (no cell reading):

```bash
env/bin/python - <<'PY'
import nbformat, sys
p = "notebooks/02_preprocess.ipynb"
code = [c for c in nbformat.read(p, as_version=4).cells if c.cell_type == "code"]
executed = sum(c.get("execution_count") is not None for c in code)
errors = sum(o.get("output_type") == "error" for c in code for o in c.get("outputs", []))
print(f"code_cells={len(code)} executed={executed} errors={errors}")
sys.exit(1 if errors or executed < len(code) else 0)
PY
```

Stage succeeds only with `errors=0`, `executed == code_cells` and the summary file present.

Forbidden: reading/editing an executed `.ipynb` as text (base64 images), printing whole
DataFrames/CSVs into chat (counts and ≤ 10 rows only), reading more than 2 cell outputs per stage.
