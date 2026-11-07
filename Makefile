PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
GIT := git

.PHONY: setup test demo eval report clean

setup:
	/opt/homebrew/bin/python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=src $(PYTEST) -v

demo:
	PYTHONPATH=src $(PYTHON) -m smriti.cli demo-persistence

eval:
	PYTHONPATH=src $(PYTHON) eval/run_eval.py --subset artifacts/eval_subset_manifest.json --output-dir artifacts/eval_results

report:
	PYTHONPATH=src $(PYTHON) eval/build_report.py --results-dir artifacts/eval_results --output EVAL.md

clean:
	rm -rf __pycache__ .pytest_cache build dist src/*.egg-info
