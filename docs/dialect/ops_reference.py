import shutil
import subprocess
import sys
import os
import re
import json
import jinja2
import mlir_graphblas
import mlir_graphblas.src

OP_HEADER_REGEX_PATTERN = re.compile(r"^### `graphblas\..*` \(::mlir::graphblas::.*\)$")


def process_markdown(markdown: str) -> str:
    markdown = markdown.replace(
        "## Operation definition", "## Operation Definitions", 1
    )
    markdown = markdown.replace(
        "# 'graphblas' Dialect", "# GraphBLAS Dialect Op Reference", 1
    )
    markdown = markdown.replace("[TOC]", "", 1)
    lines = markdown.splitlines()
    lines = map(str.rstrip, lines)
    lines = (
        " ".join(l.split()[:-1]) if OP_HEADER_REGEX_PATTERN.match(l) else l
        for l in lines
    )
    processed_markdown = "\n".join(lines)
    return processed_markdown


_SCRIPT_DIR = os.path.dirname(__file__)

IPYNB_TEMPLATE = jinja2.Template(
    """
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    {{ markdown }}
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python {{ major }} (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": {{ major }}
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "{{ major }}.{{ minor }}.{{ micro }}"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
""",
    undefined=jinja2.StrictUndefined,
)

RESULT_IPYNB_LOCATION = os.path.join(_SCRIPT_DIR, "ops_reference.ipynb")

if __name__ == "__main__":
    # build the command
    src_dir = os.path.join(*mlir_graphblas.src.__path__._path)
    includes = [
        os.path.join(src_dir, rel_dir)
        for rel_dir in (
            "include",
            "include/GraphBLAS",
            "build/include",
        )
    ]
    includes.append(os.path.join(sys.exec_prefix, "include"))
    includes = [f"-I{directory}" for directory in includes]
    command = (
        [shutil.which("mlir-tblgen"), "--gen-dialect-doc"]
        + includes
        + [os.path.join(src_dir, "include/GraphBLAS/GraphBLASOps.td")]
    )

    # run the command
    process = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=_SCRIPT_DIR
    )
    assert process.returncode == 0
    assert len(process.stderr) == 0

    # write out notebook
    markdown = process_markdown(process.stdout.decode())
    ipynb_content = IPYNB_TEMPLATE.render(
        markdown=json.dumps(markdown),
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        micro=sys.version_info.micro,
    )
    with open(RESULT_IPYNB_LOCATION, "w") as f:
        f.write(ipynb_content)
