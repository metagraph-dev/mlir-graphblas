import fileinput
import subprocess
from collections import OrderedDict

from ..types import AliasMap, SparseEncodingType
from ..cli import MlirOptCli


DEFAULT_ALIASES = AliasMap()
csr64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
csc64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
csx64 = SparseEncodingType(["dense", "compressed"], None, 64, 64)
cv64 = SparseEncodingType(["compressed"], None, 64, 64)
DEFAULT_ALIASES["CSR64"] = csr64
DEFAULT_ALIASES["CSC64"] = csc64
DEFAULT_ALIASES["CSX64"] = csx64
DEFAULT_ALIASES["CV64"] = cv64

CLI = None


def tersify_mlir(input_string: str, alias_map=None) -> str:
    global CLI
    if CLI is None:
        # Lazily initialize CLI to avoid circular import in MlirOptCli.__init__
        CLI = MlirOptCli()
    terse_string = CLI.apply_passes(input_string.encode(), [])
    if not isinstance(terse_string, str):
        raise terse_string
    if alias_map is None:
        alias_map = DEFAULT_ALIASES
    for alias_name, alias_type in reversed(alias_map.items()):
        alias_text = str(alias_type)
        if alias_text in terse_string:
            # Make a pretty version of the string
            terse_string = terse_string.replace(alias_text, "#" + alias_name)
            terse_string = (
                f"#{alias_name} = {alias_type.to_pretty_string()}\n\n" + terse_string
            )
    return terse_string


def tersify_mlir_cli():
    input_string = "\n".join(fileinput.input())
    output_string = tersify_mlir(input_string)
    print(output_string)
    return
