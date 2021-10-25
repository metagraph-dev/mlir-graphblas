import shutil
import subprocess
import sys
import os
import re
import pypandoc
import docutils


class OpsReference(docutils.parsers.rst.Directive):

    _op_header_regex_pattern = re.compile(
        r"^### `graphblas\..*` \(::mlir::graphblas::.*\)$"
    )

    def run(self):
        # build the command
        current_file_dir = os.path.dirname(__file__)
        src_dir = os.path.join(current_file_dir, "..", "..", "mlir_graphblas", "src")
        src_dir = os.path.abspath(src_dir)
        assert os.path.isdir(src_dir)
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
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
        assert process.returncode == 0
        assert len(process.stderr) == 0

        # process the markdown into restructured text
        markdown = process.stdout.decode()
        markdown = markdown.replace(
            "## Operation definition", "## Operation Definitions", 1
        )
        markdown = markdown.replace("# 'graphblas' Dialect", "", 1)
        markdown = markdown.replace("[TOC]", "", 1)
        lines = markdown.splitlines()
        lines = map(str.rstrip, lines)
        lines = (
            " ".join(l.split()[:-1]) if self._op_header_regex_pattern.match(l) else l
            for l in lines
        )
        markdown = "\n".join(lines)
        rst_text = pypandoc.convert_text(markdown, "rst", format="md")

        # generate nodes
        default_settings = docutils.frontend.OptionParser(
            components=(docutils.parsers.rst.Parser,)
        ).get_default_values()
        document = docutils.utils.new_document("dummy_name", default_settings)
        parser = docutils.parsers.rst.Parser()
        parser.parse(rst_text, document)

        return document.children


def setup(app):
    app.add_directive("ops_reference", OpsReference)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
