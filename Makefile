.PHONY: clean clean-build clean-pyc clean-out docs help
.DEFAULT_GOAL := help

help:
	@ echo
	@ echo '  Usage:'
	@ echo ''
	@ echo '    make <target> [flags...]'
	@ echo ''
	@ echo '  Targets:'
	@ echo ''
	@ awk '/^#/{ comment = substr($$0,3) } comment && /^[a-zA-Z][a-zA-Z0-9_-]+ ?:/{ print "   ", $$1, comment }' $(MAKEFILE_LIST) | column -t -s ':' | sort

## build the python virtual env for the project
venv:
	uv venv --seed --python 3.12.4
	uv pip install pip setuptools wheel poetry
	.venv/bin/poetry install

## make clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

## lint python files using black
lint:
	poetry run black -S -l 100 .

## run stock prediction
run:
	poetry run python stock_predictions/main.py -s TSLA -e 5 --v1
