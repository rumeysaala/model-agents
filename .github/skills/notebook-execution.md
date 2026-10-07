---
name: notebook-execution
description: write the stage notebook once, execute it once from the terminal (papermill), read only the stage summary — never run or read cells one by one
---

# Notebook Execution Policy (all stage agents)

Goal: the data scientist gets a **fully executed notebook** (every cell output embedded, watchable
while it runs) while the agent pulls as few tokens as possible into its context.

```
write source once  →  build + execute once  →  read summary.json  →  append analysis cell  →  verify
```

## 1. Author the notebook source (one file write)

Write the whole notebook as a **jupytext percent script** at `notebooks/src/NN_<stage>.py`.
Plain Python, no JSON escaping, one `editFiles` call. Never write the notebook cell by cell.

```python
# %% [markdown]
# # 02 — Preprocess
# One paragraph: what this stage does and which files it produces.

# %% tags=["parameters"]
import json
with open("reports/pipeline_config.json") as f:
    _cfg = json.load(f)
USE_GPU = bool(_cfg.get("use_gpu", False))

# %% [markdown]
# ## Imports & constants

# %%
# imports, constants from config/pipeline_contracts.json, write_stage_summary (skill: stage-summary)

# %% [markdown]
# ## <step title>

# %%
# <skill code pasted inline, verbatim — one skill per cell>

# %%
# Stage summary — always the last code cell
write_stage_summary("preprocess", ...)
```

Rules:
- Skill code goes into cells **verbatim**; do not rewrite or "improve" it.
- One responsibility per cell; a short `# %% [markdown]` title before each block.
- Keep `display(...)`, `print(...)` and plots — they are for the data scientist, not for you.
- The last code cell always calls `write_stage_summary(...)` (#file:stage-summary.md).

## 2. Build and execute (one terminal command)

```bash
mkdir -p reports/logs reports/summaries notebooks/src && \
env/bin/python -m jupytext --to ipynb --output notebooks/02_preprocess.ipynb notebooks/src/02_preprocess.py && \
env/bin/python -m papermill notebooks/02_preprocess.ipynb notebooks/02_preprocess.ipynb \
  -k python3 --cwd . --log-output --no-progress-bar > reports/logs/02_preprocess.log 2>&1; echo "EXIT=$?"
```

- Everything goes to the log file; the terminal returns only `EXIT=<code>`. Do not call
  `getTerminalOutput` afterwards.
- **Live view for the data scientist:** papermill saves the notebook after every cell, so
  `notebooks/02_preprocess.ipynb` can stay open in VS Code while it runs and outputs appear
  cell by cell. Alternatively `tail -f reports/logs/02_preprocess.log` in a second terminal.
  Tell the user this before starting a long stage.
- `-k python3` resolves to the venv's own ipykernel when run via `env/bin/python`. If it does
  not: `env/bin/python -m ipykernel install --user --name env-kernel` once, then use `-k env-kernel`.
- First time only: `env/bin/pip install papermill jupytext ipykernel nbformat`.

## 3. On failure (`EXIT≠0`) — read the tail, not the notebook

```bash
grep -n "Executing Cell\|Error\|Exception" reports/logs/02_preprocess.log | tail -n 5
tail -n 40 reports/logs/02_preprocess.log
```

Fix the failing cell **in the `.py` source**, then re-run step 2 (it rebuilds the `.ipynb`).
Max 3 attempts; then stop and report the traceback tail to the user.

## 4. Read the summary (not the cells)

```bash
cat reports/summaries/preprocess_summary.json
```

This is the only stage result you read. If a number you need is missing, add it to the
`write_stage_summary(...)` call in the source and re-run — do not read cell outputs instead.

## 5. Append the analysis cell (for the data scientist)

Write `reports/summaries/preprocess_analysis.md` — at most 15 lines: what was done, the key
numbers, anything suspicious, what to check next. Then append it without opening the notebook:

```bash
env/bin/python - <<'PY'
import nbformat
p = "notebooks/02_preprocess.ipynb"
nb = nbformat.read(p, as_version=4)
nb.cells.append(nbformat.v4.new_markdown_cell(open("reports/summaries/preprocess_analysis.md").read()))
nbformat.write(nb, p)
print("analysis cell appended")
PY
```

## 6. Verify (cheap, no cell reading)

```bash
env/bin/python - <<'PY'
import nbformat, sys
p = "notebooks/02_preprocess.ipynb"
nb = nbformat.read(p, as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
executed = sum(1 for c in code if c.get("execution_count") is not None)
errors = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "error")
print(f"code_cells={len(code)} executed={executed} errors={errors}")
sys.exit(1 if errors or executed < len(code) else 0)
PY
```

Stage is successful only when this prints `errors=0` and `executed == code_cells`, and the
summary file exists.

## Forbidden (token cost)

- `runNotebookCell` / `restartNotebookKernel` — never execute cells one by one.
- `readNotebookCellOutput` — at most **2 cells per stage**, only if the summary is insufficient.
- `readFile` / `editFiles` on an **executed** `.ipynb` — it contains base64 images. Edit the `.py` source.
- `getTerminalOutput` on the papermill log — use the `grep`/`tail` commands in §3.
- Printing whole DataFrames / CSVs into chat — show counts and top-10 rows at most.
