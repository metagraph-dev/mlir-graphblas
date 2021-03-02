import string
import random
from pygments import highlight
from pygments.formatters import get_formatter_by_name
from pygments import token, lexer
from functools import partial


class MlirLexer(lexer.RegexLexer):
    name = 'MLIR'
    aliases = ['mlir']
    filenames = ['*.mlir']
    mimetypes = ['text/x-mlir']

    tokens = {
        "root": [
            lexer.include("whitespace"),
            (r'c?"[^"]*?"', token.Literal.String),
            (r'\^([-a-zA-Z$._][\w\-$.0-9]*)\s*', token.Name.Label),
            (r'([\w\d_$.]+)\s*=', token.Name.Label),
            lexer.include("keyword"),
            (r'->', token.Punctuation),
            (r'@([\w_][\w\d_$.]*)', token.Name.Function),
            (r'[%#][\w\d_$.]+', token.Name.Variable),
            (r'([1-9?][\d?]*\s*x)+', token.Literal.Number),
            (r'0[xX][a-fA-F0-9]+', token.Literal.Number),
            (r'-?\d+(?:[.]\d+)?(?:[eE][-+]?\d+(?:[.]\d+)?)?', token.Literal.Number),
            (r'[=<>{}\[\]()*.,!:]|x\b', token.Punctuation),
            (r'[\w\d]+', token.Text),
        ],
        "whitespace": [
            (r'(\n|\s)+', token.Text),
            (r'//.*?\n', token.Comment),
        ],
        "keyword": [
            (lexer.words(('constant', 'return')), token.Keyword.Type),
            (lexer.words(('func', 'loc', 'memref', 'tensor', 'vector')), token.Keyword.Type),
            (r'bf16|f16|f32|f64|index', token.Keyword),
            (r'i[1-9]\d*', token.Keyword),
        ],
    }


def explore(dr, embed=False):
    """
    Returns an exploration view of a DebugResult based on Panel (panel.holoviz.org)

    :param dr: DebugResult
    :param embed: bool (default False); whether to embed the explorer in a notebook
                  when False, the explorer opens in a new tab
    :return: Panel object
    """
    rndchars = ''.join(random.choice(string.ascii_letters) for _ in range(9))
    if embed:
        html_formatter = get_formatter_by_name('html', linenos='inline', noclasses=True)
    else:
        html_formatter = get_formatter_by_name('html', linenos='inline', cssclass=f'highlight_{rndchars}')
    # lexer = get_lexer_by_name('Python')
    lexer = MlirLexer()

    import panel as pn
    pn.extension(
        raw_css=[html_formatter.get_style_defs(f'.highlight_{rndchars}')]
    )

    ckbox_linenos = pn.widgets.Checkbox(name="Show Line Numbers", value=True)
    tabs = pn.Tabs()
    gspec_outer = pn.GridSpec(sizing_mode='stretch_width')
    gspec_outer[0, 0] = tabs
    # gspec_outer[0, 0] = pn.Column(ckbox_linenos, tabs)

    # Sequential
    seq_select = pn.widgets.Select(name='Passes', options=dr.passes, width=420)
    seq_btn_left = pn.widgets.Button(name='\u25c0', width=200, button_type='primary')
    seq_btn_right = pn.widgets.Button(name='\u25b6', width=200, button_type='primary')
    seq_code_left = pn.pane.HTML(highlight(dr.stages[0], lexer, html_formatter))
    seq_code_right = pn.pane.HTML(highlight(dr.stages[1], lexer, html_formatter))
    sequential = pn.GridSpec(sizing_mode='stretch_width')
    seq_code_row = pn.GridSpec(sizing_mode='stretch_width')
    seq_code_row[0, 0] = seq_code_left
    seq_code_row[0, 1] = seq_code_right
    sequential[0, 0] = pn.Column(
        seq_select,
        pn.Row(seq_btn_left, seq_btn_right),
        seq_code_row
    )
    tabs.append(('Sequential', sequential))

    # Single
    sgl_select = pn.widgets.Select(name='Passes', options=['Initial'] + dr.passes, width=420)
    sgl_code = pn.pane.HTML(highlight(dr.stages[0], lexer, html_formatter))
    single = pn.GridSpec(sizing_mode='stretch_width')
    single[0, 0] = pn.Column(
        sgl_select,
        sgl_code
    )
    tabs.append(('Single', single))

    # Double
    dbl_select_left = pn.widgets.Select(name='Passes', options=['Initial'] + dr.passes)
    dbl_select_right = pn.widgets.Select(name='Passes', options=['Initial'] + dr.passes, value=dr.passes[0])
    dbl_code_left = pn.pane.HTML(highlight(dr.stages[0], lexer, html_formatter))
    dbl_code_right = pn.pane.HTML(highlight(dr.stages[1], lexer, html_formatter))
    double = pn.GridSpec(sizing_mode='stretch_width')
    double[0, 0] = pn.Column(
        dbl_select_left,
        dbl_code_left,
    )
    double[0, 1] = pn.Column(
        dbl_select_right,
        dbl_code_right,
    )
    tabs.append(('Double', double))

    # Callbacks
    def line_number_toggle(target, event):
        print(event.obj.value)

    def code_callback(target, event, offset=0):
        if event.new == "Initial":
            new_text = dr.stages[0 + offset]
        else:
            try:
                ipass = dr._find_pass_index(event.new)
                new_text = dr.stages[ipass + 1 + offset]
            except KeyError:
                new_text = f"No pass found named {event.new}"

        new_text = highlight(new_text, lexer, html_formatter)
        target.object = new_text

    def button_callback(target, event):
        ipass = dr._find_pass_index(target.value)
        if event.obj.name == "\u25c0":
            target.value = dr.passes[max(ipass-1, 0)]
        elif event.obj.name == "\u25b6":
            target.value = dr.passes[min(ipass+1, len(dr.passes)-1)]

    ckbox_linenos.link(tabs, callbacks={'value': line_number_toggle})
    sgl_select.link(sgl_code, callbacks={'value': code_callback})
    dbl_select_left.link(dbl_code_left, callbacks={'value': code_callback})
    dbl_select_right.link(dbl_code_right, callbacks={'value': code_callback})
    seq_select.link(seq_code_left, callbacks={'value': partial(code_callback, offset=-1)})
    seq_select.link(seq_code_right, callbacks={'value': code_callback})
    seq_btn_left.link(seq_select, callbacks={'value': button_callback})
    seq_btn_right.link(seq_select, callbacks={'value': button_callback})

    if embed:
        return gspec_outer
    else:
        return gspec_outer.show("MLIR Code Pass Explorer")
