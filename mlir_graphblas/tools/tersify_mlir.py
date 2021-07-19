import fileinput
import subprocess

from ..cli import MlirOptCli

CSR64_LINES = (
    "#sparse_tensor.encoding<{ ",
    '    dimLevelType = [ "dense", "compressed" ], ',
    "    dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, ",
    "    pointerBitWidth = 64, ",
    "    indexBitWidth = 64 ",
    "}>",
)

CSC64_LINES = (
    "#sparse_tensor.encoding<{ ",
    '    dimLevelType = [ "dense", "compressed" ], ',
    "    dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, ",
    "    pointerBitWidth = 64, ",
    "    indexBitWidth = 64 ",
    "}>",
)

CSR64_PRETTY_TEXT = "\n".join(CSR64_LINES)
CSC64_PRETTY_TEXT = "\n".join(CSC64_LINES)

CSR64_EXPANDED_TEXT = " ".join(line.strip() for line in CSR64_LINES)
CSC64_EXPANDED_TEXT = " ".join(line.strip() for line in CSC64_LINES)

CLI = None


def tersify_mlir(input_string: str) -> str:
    global CLI
    if CLI is None:
        # Lazily initialize CLI to avoid circular import in MlirOptCli.__init__
        CLI = MlirOptCli()
    terse_string = CLI.apply_passes(input_string.encode(), [])
    if not isinstance(terse_string, str):
        raise terse_string
    if CSR64_EXPANDED_TEXT in terse_string:
        terse_string = terse_string.replace(CSR64_EXPANDED_TEXT, "#CSR64")
        terse_string = f"#CSR64 = {CSR64_PRETTY_TEXT}\n\n" + terse_string
    if CSC64_EXPANDED_TEXT in terse_string:
        terse_string = terse_string.replace(CSC64_EXPANDED_TEXT, "#CSC64")
        terse_string = f"#CSC64 = {CSC64_PRETTY_TEXT}\n\n" + terse_string
    return terse_string


def tersify_mlir_cli():
    input_string = "\n".join(fileinput.input())
    output_string = tersify_mlir(input_string)
    print(output_string)
    return
