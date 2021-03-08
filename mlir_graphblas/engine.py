import subprocess
import llvmlite.binding as llvm
from .cli import MlirOptCli, MlirOptError

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


class MlirJitEngine:
    def __init__(self, llvmlite_engine=None):
        if llvmlite_engine is None:
            # Create a target machine representing the host
            target = llvm.Target.from_default_triple()
            target_machine = target.create_target_machine()
            # And an execution engine with an empty backing module
            backing_mod = llvm.parse_assembly("")
            llvmlite_engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
        self._engine = llvmlite_engine
        self._cli = MlirOptCli()

    def add(self, mlir_text, passes, debug=False):
        if isinstance(mlir_text, str):
            mlir_text = mlir_text.encode()
        if debug:
            try:
                llvmir_text = self._cli.apply_passes(mlir_text, passes)
            except MlirOptError as e:
                return e.debug_result
        else:
            llvmir_text = self._cli.apply_passes(mlir_text, passes)
        result = subprocess.run(
            ["mlir-translate", "--mlir-to-llvmir"],
            input=llvmir_text.encode(),
            capture_output=True,
        )
        llvm_text = result.stdout.decode()
        # Create a LLVM module object from the IR
        mod = llvm.parse_assembly(llvm_text)
        mod.verify()
        # Now add the module and make sure it is ready for execution
        self._engine.add_module(mod)
        self._engine.finalize_object()
        self._engine.run_static_constructors()
        return mod

    def __getitem__(self, func_name):
        addr = self._engine.get_function_address(func_name)
        if addr == 0:  # NULL
            raise KeyError(func_name)
        return addr
