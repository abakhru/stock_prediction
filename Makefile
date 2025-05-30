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
	uv sync
	uv lock -U

## make clean
clean:
	rm -rf build/ .ruff_cache/ dist/ .eggs/ .pytest_cache/ .coverage .coverage.*

## lint python files using Ruff
lint:
	uv run ruff check --fix --unsafe-fixes stock_predictions

## run stock prediction
run:
	uv run python stock_predictions/main.py -s TSLA -e 5 --v1
