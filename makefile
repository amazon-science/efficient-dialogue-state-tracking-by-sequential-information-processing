sources = src
.PHONY: test format lint unittest coverage pre-commit clean

test: format lint unittest

format:
	isort $(sources) scripts
	black $(sources) scripts
	nbqa isort notebooks
	nbqa black notebooks

lint:
	flake8 $(sources) scripts
	# mypy $(sources) scripts

unittest:
	pytest

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage
	rm -rf */lightning_logs/
	rm -rf site
	rm -f train.log
	rm -f cfg_tree.log