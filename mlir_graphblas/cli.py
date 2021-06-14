import os
import math
import subprocess
import tempfile
from typing import List, Optional, Union
import logging

log = logging.getLogger("mlir_graphblas")


def logged_subprocess_run(*args, **kwargs):
    log.debug("RUN: %s", args[0])
    return subprocess.run(*args, **kwargs)


try:
    # when running in developer mode
    from . import src

    _SCRIPT_DIR = os.path.dirname(__file__)
    _BUILD_DIR = os.path.join(_SCRIPT_DIR, "src", "build")
    GRAPHBLAS_OPT_EXE = os.path.join(_BUILD_DIR, "bin", "graphblas-opt")
except ImportError:
    # ImportError assumes a normal install without a src directory, so graphblas-opt should
    # be available in the /bin folder of the environment
    GRAPHBLAS_OPT_EXE = "graphblas-opt"


class MlirOptError(Exception):
    pass


class MlirOptCli:
    def __init__(
        self, executable: Optional[str] = None, options: Optional[List[str]] = None
    ):
        if executable is None:
            from . import config

            executable = config.get("cli.executable", GRAPHBLAS_OPT_EXE)
        self._executable = executable
        if options is None:
            options = []
        self._options = options

    def _read_input(self, file) -> bytes:
        if isinstance(file, bytes):
            return file
        elif hasattr(file, "close"):
            return file.read()
        else:
            with open(file, "rb") as f:
                return f.read()

    def apply_passes(self, file, passes: List[str]) -> Union[str, "DebugResult"]:
        """
        Converts a file of MLIR by applying passes sequentially.
        Returns a string of the result.

        :param file: file-like object -or- path to file on disk -or- bytes of file content
        :param passes: list of mlir-opt pass options
        :return: str (if successful)
                 list of str containing transformations and eventual error (if failure)
        """
        input = self._read_input(file)
        result = logged_subprocess_run(
            [self._executable] + self._options + list(passes),
            capture_output=True,
            input=input,
        )
        if result.returncode == 0:
            return result.stdout.decode()
        err_lines = result.stderr.split(b"\n")
        err = MlirOptError("\n".join(el.decode() for el in err_lines[:3]))
        err.debug_result = self.debug_passes(input, passes) if passes else None
        raise err

    def debug_passes(self, input: bytes, passes: List[str]) -> "DebugResult":
        stages = []
        saved_passes = []
        success = True
        for p in passes:
            saved_passes.append(p.lstrip("-"))
            stages.append(input.decode())
            result = logged_subprocess_run(
                [self._executable, p], capture_output=True, input=input
            )
            if result.returncode == 0:
                input = result.stdout
            else:
                options = self._options
                debug_str = "--mlir-print-debuginfo"
                if debug_str not in self._options:
                    options = options.copy()
                    options.append(debug_str)
                result = logged_subprocess_run(
                    [self._executable] + options + [p],
                    capture_output=True,
                    input=input,
                )
                stages.append(result.stderr.decode())
                success = False
                break
        else:
            # append final output
            stages.append(result.stdout.decode())
        return DebugResult(stages, saved_passes, self, success=success)


class DebugResult:
    def __init__(self, stages, passes, cli, success):
        self.stages = stages
        self.passes = passes
        assert (
            len(self.stages) == len(self.passes) + 1
        ), "Stages must be one larger than passes"
        self._cli = cli
        self.success = success

    def __repr__(self):
        if self.success:
            ret = [self._add_banner(self.stages[-1], "Optimization successful")]
        else:
            ret = [
                self._add_banner(
                    self.stages[-1], f"Error when running {self.passes[-1]}"
                )
            ]
        for p, stage in zip(reversed(self.passes), reversed(self.stages[:-1])):
            if stage == self.stages[-2]:
                stage = self._add_row_column_numbers(stage)
            ret.append(self._add_banner(stage, f"Input to {p}"))
        return "\n\n".join(ret)

    def __getitem__(self, item):
        if type(item) is not slice:
            raise TypeError("Only slices are supported")
        if item.step is not None:
            raise TypeError("step != 1 is not supported")
        trim_stages = self.stages[item]
        if len(trim_stages) < 2:
            raise TypeError("At least two stages are required")
        if item.stop is None:
            trim_passes = self.passes[item.start :]
        else:
            trim_passes = self.passes[item.start : item.stop - 1]
        return DebugResult(trim_stages, trim_passes, self._cli, self.success)

    def _find_pass_index(self, pass_):
        for ip, p in enumerate(self.passes):
            if p == pass_:
                return ip
        else:
            raise KeyError(f"No pass found named {pass_}")

    def diff(self, pass_, diffcmd=None):
        """
        Performs a diff showing the input and output of a given `pass_`.

        diffcmd controls which diff tool to use. Default is `vimdiff`.
        """
        if diffcmd is None:
            from . import config

            diffcmd = config.get("diff.executable", "vimdiff")

        ipass = self._find_pass_index(pass_)

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "first.txt")
            file2 = os.path.join(tmpdir, "second.txt")
            with open(file1, "w") as f:
                f.write(self.stages[ipass])
            with open(file2, "w") as f:
                f.write(self.stages[ipass + 1])
            logged_subprocess_run([diffcmd, file1, file2])

    @classmethod
    def _add_banner(cls, data: str, banner_text: str, char: str = "=") -> str:
        width = len(banner_text) + 4
        return f"{char * width}\n  {banner_text}  \n{char * width}\n" + data

    @classmethod
    def _add_row_column_numbers(cls, data: str) -> str:
        splitz = data.splitlines()
        num_rows = len(splitz)
        num_cols = max(len(row) for row in splitz)

        offset = int(math.log10(num_rows)) + 1
        colheader_ones = [str(n % 10) for n in range(1, num_cols + 1)]
        colheader_tens = [" " * 9] + [
            f"{n:<10}" for n in range(1, num_cols + 1) if n % 10 == 0
        ]
        ret = [
            f'{" " * offset} {"".join(colheader_tens)}',
            f'{" " * offset} {"".join(colheader_ones)}',
            f'{" " * offset} {"-" * len(colheader_ones)}',
        ]
        for i, line in enumerate(splitz, 1):
            ret.append(f"{i:{offset}}|{line}")
        return "\n".join(ret)

    def explore(self, embed=False, initial_style=None):
        """
        Open an interactive explorer view of the passes

        :param embed: If True, embeds the explorer in the notebook output
                      If False (default), open the explorer in a new tab
        :param initial_style: dict of styles to apply for the initial view of the explorer
            line_numbers: bool
            highlight_style: str
            tab: str
            pass: str
            pass2: str (only applies if tab=="Double")
        """
        from .explorer import Explorer

        return Explorer(self).show(embed=embed, initial_style=initial_style)
