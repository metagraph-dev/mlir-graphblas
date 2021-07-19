import os
import sys
import site
import subprocess
import shutil
import argparse

from typing import Iterable, Tuple

_SCRIPT_DIR = os.path.dirname(__file__)
_BUILD_DIR = os.path.join(_SCRIPT_DIR, "build")
GRAPHBLAS_OPT_LOCATION = os.path.join(_BUILD_DIR, "bin", "graphblas-opt")


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
            + "STDERR Messages:"
            + "\n\n"
            + stderr_string
            + "\n\n"
            + "STDOUT Messages:"
            + "\n\n"
            + stdout_string
            + "\n\n"
            + f"Command Failed with exit code {process.returncode}:"
            + "\n\n"
            + command
            + "\n\n"
        )
        raise RuntimeError(error_string)

    return command, stdout_string, stderr_string


def build_graphblas_opt(build_clean: bool) -> None:
    if build_clean and os.path.isdir(_BUILD_DIR):
        shutil.rmtree(_BUILD_DIR)

    if not os.path.isdir(_BUILD_DIR):
        os.makedirs(_BUILD_DIR)

    env_lib_path = os.path.join(sys.exec_prefix, "lib")
    LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", env_lib_path)
    if env_lib_path not in LD_LIBRARY_PATH.split(":"):
        LD_LIBRARY_PATH = LD_LIBRARY_PATH + ":" + env_lib_path

    PYTHONPATH = ":".join(site.getsitepackages())

    command, stdout_string, stderr_string = run_shell_commands(
        _BUILD_DIR,
        "cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit",  # creates the directory ./build/graphblas-opt/
        "cmake --build . --target check-graphblas --verbose",  # creates the executable ./build/bin/graphblas-opt and runs tests
        PREFIX=sys.exec_prefix,
        BUILD_DIR=sys.exec_prefix,
        LD_LIBRARY_PATH=LD_LIBRARY_PATH,
        PYTHONPATH=PYTHONPATH,
    )
    assert os.path.isfile(GRAPHBLAS_OPT_LOCATION)

    print(
        f"""
STDERR:

{stderr_string}

STDOUT:

{stdout_string}

Status: Success

Command:

{command}
"""
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tool",
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=9999
        ),
    )
    parser.add_argument(
        "-build-clean", action="store_true", help="Rebuild from scratch."
    )
    args = parser.parse_args()

    build_graphblas_opt(args.build_clean)
