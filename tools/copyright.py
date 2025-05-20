# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os


_CPP_LIKE_EXTENSIONS = [".cu", ".h", ".c", ".cpp"]
_PYTHON_LIKE_EXTENSIONS = [
    ".py",
    ".yml",
    ".yaml",
    ".clang-format",
    "requirements-dev.txt",
    "requirements.txt",
    "setup.cfg",
    "Makefile",
]
_HTML_LIKE_EXTENSIONS = [".html", ".md"]

_BANNED_DIRECTORIES = [".git", "cutlass"]

_COPYRIGHT_HEADER = "Copyright (c) 2025, Mayank Mishra"

_CPP_HEADER = (
    f"// **************************************************\n"
    f"// {_COPYRIGHT_HEADER}\n"
    "// **************************************************\n\n"
)

_PYTHON_HEADER = (
    f"# **************************************************\n"
    f"# {_COPYRIGHT_HEADER}\n"
    "# **************************************************\n\n"
)

_HTML_HEADER = (
    f"<!-- **************************************************\n"
    f"{_COPYRIGHT_HEADER}\n"
    "************************************************** -->\n\n"
)

directory = os.path.dirname(os.path.dirname(__file__))


def _check_and_add_copyright_header(file: str, header: str) -> None:
    code = open(file, "r").read()

    if not code.startswith(header):
        code = f"{header}{code}"

    open(file, "w").writelines([code])


for root, dirs, files in os.walk(directory):
    if any([root.split(directory)[1].startswith(f"/{banned_directory}") for banned_directory in _BANNED_DIRECTORIES]):
        continue

    for file in files:
        file = os.path.join(root, file)

        if any([file.endswith(i) for i in _CPP_LIKE_EXTENSIONS]):
            _check_and_add_copyright_header(file, _CPP_HEADER)
        elif any([file.endswith(i) for i in _PYTHON_LIKE_EXTENSIONS]):
            _check_and_add_copyright_header(file, _PYTHON_HEADER)
        elif any([file.endswith(i) for i in _HTML_LIKE_EXTENSIONS]):
            _check_and_add_copyright_header(file, _HTML_HEADER)
