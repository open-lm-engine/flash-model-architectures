install:
	git submodule update --init --recursive
	pip install .

install-dev:
	git submodule update --init --recursive
	pip install -e .

test:
	pytest tests

update-precommit:
	pre-commit autoupdate

style:
	pre-commit run --all-files

cutotune-cache:
	DEBUG_CUTOTUNE=1 LOAD_CUTOTUNE_CACHE=1 TORCH_CUDA_ARCH_LIST=9.0 python tools/build_cutotune_cache.py
