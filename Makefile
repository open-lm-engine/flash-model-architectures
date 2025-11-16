# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

install:
	git submodule update --init --recursive
	uv sync --extra $(accelerator)

install-dev:
	git submodule update --init --recursive
	uv sync --extra dev --extra $(accelerator)

test:
	uv run --extra dev --extra $(accelerator) pytest tests

test-debug:
	DEBUG_CUTOTUNE=1 TRITON_PRINT_AUTOTUNING=1 uv run --extra dev --extra $(accelerator) pytest -s tests

update-precommit:
	uv run --extra dev pre-commit autoupdate

style:
	uv run python tools/populate_readme.py
	uv run python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) 2025, Mayank Mishra"
	uv run --extra dev pre-commit run --all-files

cutotune-cache:
	DEBUG_CUTOTUNE=1 LOAD_CUTOTUNE_CACHE=1 TORCH_CUDA_ARCH_LIST=9.0 uv run --extra dev --extra $(accelerator) python tools/build_cutotune_cache.py
