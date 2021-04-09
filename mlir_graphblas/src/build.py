import os
import sys
import site
import subprocess

from typing import Iterable, Tuple


def run_shell_commands(
    directory: str, *commands: Iterable[str], **environment_variables
) -> Tuple[str, str, str]:
    command = " && ".join(
        [f"pushd {directory}"]
        + [f"export {name}={value}" for name, value in environment_variables.items()]
        + list(commands)
        + ["popd"]
    )

    process = subprocess.Popen(
        ["/bin/bash"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_string, stderr_string = map(
        bytes.decode, process.communicate(command.encode())
    )

    if process.returncode != 0:
        error_string = (
            "\n\n"
            + f"Command Failed with exit code {process.returncode}:"
            + "\n\n"
            + command
            + "\n\n"
            + "STDOUT Messages:"
            + "\n\n"
            + stdout_string
            + "\n\n"
            + "STDERR Messages:"
            + "\n\n"
            + stderr_string
        )
        raise RuntimeError(error_string)

    return command, stdout_string, stderr_string


if __name__ == "__main__":

    script_dir = os.path.dirname(__file__)

    build_dir = os.path.join(script_dir, "build")
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    env_lib_path = os.path.join(sys.exec_prefix, "lib")
    LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", "")
    if env_lib_path not in LD_LIBRARY_PATH.split(":"):
        LD_LIBRARY_PATH = LD_LIBRARY_PATH + ":" + env_lib_path

    (PYTHONPATH,) = site.getsitepackages()

    command, stdout_string, stderr_string = run_shell_commands(
        build_dir,
        "cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit",
        "cmake --build . --target check-graphblas",
        PREFIX=sys.exec_prefix,
        BUILD_DIR=sys.exec_prefix,
        LD_LIBRARY_PATH=LD_LIBRARY_PATH,
        PYTHONPATH=PYTHONPATH,
    )

    print(
        f"""
Command:

{command}

STDOUT:

{stdout_string}

STDERR:

{stderr_string}
"""
    )
