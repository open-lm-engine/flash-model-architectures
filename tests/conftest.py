# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os


# torch.cuda reads CUDA_VISIBLE_DEVICES once, on its first CUDA call, and ignores any later
# writes to it in the same process. conftest.py is imported before any test module, and `xma`
# touches torch.cuda at import time, so this must stay import-free (os.environ only) and run
# here rather than in a pytest_configure hook, which would already run too late.
_worker_id = os.environ.get("PYTEST_XDIST_WORKER")
_num_accelerators = int(os.environ.get("NUM_ACCELERATORS", "0"))

if _worker_id is not None and _num_accelerators > 0:
    gpu_id = int(_worker_id.replace("gw", "")) % _num_accelerators
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
