.PHONY: clean test

PYTHON ?= python

clean:
	$(PYTHON) -c "import pathlib, shutil; roots=(pathlib.Path('src'), pathlib.Path('tests'), pathlib.Path('scripts'), pathlib.Path('dashboard')); [shutil.rmtree(path) for root in roots for path in root.rglob('__pycache__')]; shutil.rmtree('.pytest_cache', ignore_errors=True)"

test:
	$(PYTHON) -m pytest
