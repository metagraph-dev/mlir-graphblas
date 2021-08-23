import jinja2
import tempfile
import os
import subprocess

from mlir_graphblas.cli import GRAPHBLAS_OPT_EXE


def test_filecheck_mlir(
    mlir_code, test_command_template, template_file, parameter_dict
):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", delete=False
    ) as temp_file:
        temp_file.write(mlir_code)
    test_command = jinja2.Template(
        test_command_template, undefined=jinja2.StrictUndefined
    ).render(graphblas_opt=GRAPHBLAS_OPT_EXE, input_file=temp_file.name)
    process = subprocess.Popen(
        ["/bin/bash"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_string, stderr_string = map(
        bytes.decode, process.communicate(test_command.encode())
    )
    assert (
        process.returncode == 0
    ), f"""
STDERR:

{stderr_string}

STDOUT:

{stdout_string}

Text Execution Command:

{test_command}

Exit Code: {process.returncode}

Template File: {template_file}

Template Parameters:

{os.linesep.join(f"    {repr(k)}: {repr(v)}" for k, v in parameter_dict.items())}
"""
    os.remove(temp_file.name)  # only remove if the test passes
