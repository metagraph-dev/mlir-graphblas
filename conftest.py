import os
import distutils.core
import subprocess


def pytest_configure(config):
    distutils.core.run_setup(
        "./setup.py", script_args=["build_ext", "--inplace"], stop_after="run"
    )

    # Ensure graphblas-opt is built
    subprocess.run(["python", os.path.join("mlir_graphblas", "src", "build.py")])

    return
