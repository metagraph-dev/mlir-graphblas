import os
import math
import subprocess
import tempfile
import string
import random
from functools import partial
from typing import List, Optional, Union
import logging

log = logging.getLogger('mlir_graphblas')



def logged_subprocess_run(*args, **kwargs):
    log.debug('RUN: %s', args[0])
    return subprocess.run(*args, **kwargs)



class MlirOptError(Exception):
    pass


class MlirOptCli:
    def __init__(self, executable: Optional[str] = None, options: Optional[List[str]] = None):
        if executable is None:
            from . import config
            executable = config.get("cli.executable", "mlir-opt")
        self._executable = executable
        if options is None:
            options = []
        self._options = options

    def _read_input(self, file) -> bytes:
        if isinstance(file, bytes):
            return file
        elif hasattr(file, 'close'):
            return file.read()
        else:
            with open(file, 'rb') as f:
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
        result = logged_subprocess_run([self._executable] + self._options + passes, capture_output=True, input=input)
        if result.returncode == 0:
            return result.stdout.decode()
        err = MlirOptError(result.stderr.split(b'\n')[0])
        err.debug_result = self.debug_passes(input, passes)
        raise err

    def debug_passes(self, input: bytes, passes: List[str]) -> "DebugResult":
        stages = []
        saved_passes = []
        for p in passes:
            saved_passes.append(p.lstrip('-'))
            stages.append(input.decode())
            result = logged_subprocess_run([self._executable, p], capture_output=True, input=input)
            if result.returncode == 0:
                input = result.stdout
            else:
                result = logged_subprocess_run([self._executable, '--mlir-print-debuginfo', p], capture_output=True, input=input)
                stages.append(result.stderr.decode())
                break
        else:
            # append final output
            stages.append(result.stdout.decode())
        return DebugResult(stages, saved_passes)


class DebugResult:
    def __init__(self, stages, passes):
        self.stages = stages
        self.passes = passes
        assert len(self.stages) == len(self.passes) + 1, "Stages must be one larger than passes"

    def __repr__(self):
        ret = [
            self._add_banner(self.stages[-1], f"Error when running {self.passes[-1][2:]}")
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
            trim_passes = self.passes[item.start:]
        else:
            trim_passes = self.passes[item.start:item.stop-1]
        return DebugResult(trim_stages, trim_passes)

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
        return (
            f"{char * width}\n" +
            f"  {banner_text}  \n" +
            f"{char * width}\n" +
            data
        )

    @classmethod
    def _add_row_column_numbers(cls, data: str) -> str:
        splitz = data.splitlines()
        num_rows = len(splitz)
        num_cols = max(len(row) for row in splitz)

        offset = int(math.log10(num_rows)) + 1
        colheader_ones = [str(n % 10) for n in range(1, num_cols + 1)]
        colheader_tens = [" " * 9] + [f"{n:<10}" for n in range(1, num_cols + 1) if n % 10 == 0]
        ret = [
            f'{" " * offset} {"".join(colheader_tens)}',
            f'{" " * offset} {"".join(colheader_ones)}',
            f'{" " * offset} {"-" * len(colheader_ones)}',
        ]
        for i, line in enumerate(splitz, 1):
            ret.append(f'{i:{offset}}|{line}')
        return "\n".join(ret)

    def explore(self, embed=False):
        import panel as pn
        pn.extension(
            js_files={
                'myCodeMirror': 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/codemirror.min.js',
                'javascript_cm': 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/mode/javascript/javascript.min.js',
            }
        )

        rndchars = ''.join(random.choice(string.ascii_letters) for _ in range(9))
        def text_replace(s, new_id):
            text = code_block.replace('{{textarea}}', s)
            text = text.replace('{{id}}', f'{rndchars}_{new_id}')
            if embed:
                text = text.replace('{{HEAD}}', '')
            else:
                text = text.replace('{{HEAD}}', non_embed_header)
            return text

        tabs = pn.Tabs()

        # Sequential
        sequential = pn.Column(
            pn.widgets.Select(name='Passes', options=self.passes),
            pn.Row(
                pn.widgets.Button(name='\u25c0', width=200, button_type='primary'),
                pn.widgets.Button(name='\u25b6', width=200, button_type='primary'),
            ),
            pn.Row(
                pn.pane.HTML(text_replace(self.stages[0], 'seq1')),
                pn.pane.HTML(text_replace(self.stages[1], 'seq2')),
                max_width=400
            ),
        )
        tabs.append(('Sequential', sequential))

        # Single
        single = pn.Column(
            pn.widgets.Select(name='Passes', options=['Initial'] + self.passes),
            pn.pane.HTML(text_replace(self.stages[0], 'single')),
        )
        tabs.append(('Single', single))

        # Double
        double = pn.Row(
            pn.Column(
                pn.widgets.Select(name='Passes', options=['Initial'] + self.passes),
                pn.pane.HTML(text_replace(self.stages[0], 'double1')),
            ),
            pn.Column(
                pn.widgets.Select(name='Passes', options=['Initial'] + self.passes, value=self.passes[0]),
                pn.pane.HTML(text_replace(self.stages[1], 'double2')),
            )
        )
        tabs.append(('Double', double))

        # Callbacks
        def code_callback(target, event, offset=0):
            if event.new == "Initial":
                new_text = self.stages[0 + offset]
            else:
                try:
                    ipass = self._find_pass_index(event.new)
                    new_text = self.stages[ipass + 1 + offset]
                except KeyError:
                    new_text = f"No pass found named {event.new}"

            a = target.object.index("<textarea id=")
            b = target.object.index(">", a) + 2
            c = target.object.index("\n</textarea", b)
            target.object = target.object[:b] + new_text + target.object[c:]

        def button_callback(target, event):
            ipass = self._find_pass_index(target.value)
            if event.obj.name == "\u25c0":
                target.value = self.passes[max(ipass-1, 0)]
            elif event.obj.name == "\u25b6":
                target.value = self.passes[min(ipass+1, len(self.passes)-1)]

        single[0].link(single[1], callbacks={'value': code_callback})
        double[0][0].link(double[0][1], callbacks={'value': code_callback})
        double[1][0].link(double[1][1], callbacks={'value': code_callback})
        sequential[0].link(sequential[2][0], callbacks={'value': partial(code_callback, offset=-1)})
        sequential[0].link(sequential[2][1], callbacks={'value': code_callback})
        sequential[1][0].link(sequential[0], callbacks={'value': button_callback})
        sequential[1][1].link(sequential[0], callbacks={'value': button_callback})

        if embed:
            return tabs
        else:
            return tabs.show("MLIR Code Pass Explorer")


code_block = '''
<html>
<head>
{{HEAD}}
</head>
<body>

<div>
<textarea id="{{id}}" style="margin: 50px">
{{textarea}}
</textarea>
</div>

<script>
var config = {
  lineNumbers: true,
  tabSize: 2,
  readOnly: true,
  mode: 'javascript',
};

try {
  var myTextArea = document.querySelector('#{{id}}');
  var cm = CodeMirror.fromTextArea(myTextArea, config);
} catch (e) {
  if (e instanceof ReferenceError) {
    // formatting will be broken
  }
}

</script>
</body>
</html>
'''

non_embed_header = '''
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/codemirror.min.css">
</link>

<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/codemirror.min.js">
</script>

<style>
.CodeMirror {
  height: auto;
}

.CodeMirror-linenumber {
  padding: 0;
  min-width: 10px;
}

.CodeMirror.cm-s-default {
  padding: 0 50px 0 0;
}
</style>
'''