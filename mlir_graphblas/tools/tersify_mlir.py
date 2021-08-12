import fileinput
import subprocess
from collections import OrderedDict

from ..cli import MlirOptCli


ALIAS_NAME_TO_LINES = OrderedDict(  # ordered for deterministic results
    [
        (
            "CSR64",
            [
                "#sparse_tensor.encoding<{ ",
                '    dimLevelType = [ "dense", "compressed" ], ',
                "    dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, ",
                "    pointerBitWidth = 64, ",
                "    indexBitWidth = 64 ",
                "}>",
            ],
        ),
        (
            "CSC64",
            [
                "#sparse_tensor.encoding<{ ",
                '    dimLevelType = [ "dense", "compressed" ], ',
                "    dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, ",
                "    pointerBitWidth = 64, ",
                "    indexBitWidth = 64 ",
                "}>",
            ],
        ),
        (
            "CSX64",
            [
                "#sparse_tensor.encoding<{ ",
                '    dimLevelType = [ "dense", "compressed" ], ',
                "    pointerBitWidth = 64, ",
                "    indexBitWidth = 64 ",
                "}>",
            ],
        ),
        (
            "SparseVec64",
            [
                "#sparse_tensor.encoding<{ ",
                '    dimLevelType = [ "compressed" ], ',
                "    pointerBitWidth = 64, ",
                "    indexBitWidth = 64 ",
                "}>",
            ],
        ),
    ]
)

CLI = None


def tersify_mlir(input_string: str) -> str:
    global CLI
    if CLI is None:
        # Lazily initialize CLI to avoid circular import in MlirOptCli.__init__
        CLI = MlirOptCli()
    terse_string = CLI.apply_passes(input_string.encode(), [])
    if not isinstance(terse_string, str):
        raise terse_string
    for alias_name, alias_lines in ALIAS_NAME_TO_LINES.items():
        pretty_text = "\n".join(alias_lines)
        expanded_text = " ".join(line.strip() for line in alias_lines)
        if expanded_text in terse_string:
            terse_string = terse_string.replace(expanded_text, "#" + alias_name)
            terse_string = f"#{alias_name} = {pretty_text}\n\n" + terse_string
    return terse_string


def tersify_mlir_cli():
    input_string = "\n".join(fileinput.input())
    output_string = tersify_mlir(input_string)
    print(output_string)
    return
