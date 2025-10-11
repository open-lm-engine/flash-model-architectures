# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

install:
	git submodule update --init --recursive
	uv run pip install .

install-dev:
	git submodule update --init --recursive
	uv run pip install -e .

test:
	uv run pytest tests

test-debug:
	DEBUG_CUTOTUNE=1 TRITON_PRINT_AUTOTUNING=1 uv run pytest -s tests

update-precommit:
	uv run pre-commit autoupdate

style:
	python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) 2025, Mayank Mishra"
	uv run pre-commit run --all-files
