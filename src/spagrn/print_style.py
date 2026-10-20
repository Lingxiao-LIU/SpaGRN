"""Opt-in print style for SpaGRN plots (journal figure size, one text size).

Default behaviour is unchanged. Turn it on once per session:

    from spagrn import print_style
    print_style.use('print', width_mm=89, pt=7)   # 89 mm = one journal column, 183 mm = full width
    ...
    print_style.use('default')                    # back to the library's own sizes

While on, every figure made by a wrapped plotting function is, before it is shown or returned:
  * resized to `width_mm` wide (height scaled to keep the shape),
  * given one text size (`pt`) for all ticks, axis titles, legends and colour bars,
  * given thin lines (`line_width`),
  * widened step by step (width_mm -> 120 -> 150 -> 183 mm) while tick labels still overlap.
The width chosen is stored on the figure as `fig._print_width_mm`.
"""
import functools

import matplotlib
import matplotlib.pyplot as plt

MM = 1 / 25.4
MAX_WIDTH_MM = 183
_WIDTH_STEPS = (89, 120, 150, MAX_WIDTH_MM)

_state = {'mode': 'default', 'width_mm': 89, 'pt': 7, 'line_width': 0.5}
_depth = [0]          # nested wrapped calls style only once, at the outermost call


def use(mode='print', width_mm=89, pt=7, line_width=0.5):
    """Switch the style: 'print' (journal size) or 'default' (library sizes)."""
    if mode not in ('print', 'default'):
        raise ValueError("mode must be 'print' or 'default'")
    _state.update(mode=mode, width_mm=min(width_mm, MAX_WIDTH_MM), pt=pt, line_width=line_width)


def is_on():
    return _state['mode'] == 'print'


def _tick_labels_overlap(fig):
    """True if any two tick labels on the same axis overlap."""
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
            boxes = [t.get_window_extent(renderer) for t in labels if t.get_visible() and t.get_text()]
            for a, b in zip(boxes, boxes[1:]):
                if a.overlaps(b):
                    return True
    return False


def apply(fig, width_mm=None, pt=None, line_width=None):
    """Put one finished figure into print style (see module docstring). Returns the figure."""
    width_mm = width_mm or _state['width_mm']
    pt = pt or _state['pt']
    line_width = line_width or _state['line_width']
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text():
            t.set_fontsize(pt)
    for ax in fig.axes:
        for sp in ax.spines.values():
            sp.set_linewidth(min(sp.get_linewidth(), line_width))
        ax.tick_params(width=line_width, length=2)
    w, h = fig.get_size_inches()
    steps = [width_mm] + [s for s in _WIDTH_STEPS if s > width_mm]
    for step in steps:
        fig.set_size_inches(step * MM, h * step * MM / w)
        fig._print_width_mm = step
        if not _tick_labels_overlap(fig):
            break
    else:
        print(f'print_style: tick labels still overlap at {MAX_WIDTH_MM} mm; '
              'shorten labels or plot fewer items')
    return fig


def _figures_in(obj):
    """Figures held by a return value (Figure, Axes, seaborn grids, tuples of these)."""
    if isinstance(obj, matplotlib.figure.Figure):
        return [obj]
    if isinstance(obj, matplotlib.axes.Axes):
        return [obj.figure]
    if hasattr(obj, 'fig') and isinstance(obj.fig, matplotlib.figure.Figure):
        return [obj.fig]
    if isinstance(obj, (tuple, list)):
        return [f for o in obj for f in _figures_in(o)]
    return []


def printable(func):
    """Decorator: apply the print style to figures the function makes (no-op when style is off)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_on() or _depth[0]:
            return func(*args, **kwargs)
        before = set(plt.get_fignums())
        styled = set()

        def style_new():
            for n in plt.get_fignums():
                if n not in before and n not in styled:
                    apply(plt.figure(n))
                    styled.add(n)

        real_show = plt.show

        def show(*a, **k):            # style before the figure is shown (inline backends then close it)
            style_new()
            return real_show(*a, **k)

        real_savefig, real_fig_savefig = plt.savefig, matplotlib.figure.Figure.savefig

        def savefig(*a, **k):         # style before the function saves its own file (save=True)
            style_new()
            return real_savefig(*a, **k)

        def fig_savefig(self, *a, **k):
            style_new()
            return real_fig_savefig(self, *a, **k)

        plt.show, plt.savefig, matplotlib.figure.Figure.savefig = show, savefig, fig_savefig
        _depth[0] += 1
        try:
            out = func(*args, **kwargs)
        finally:
            _depth[0] -= 1
            plt.show, plt.savefig, matplotlib.figure.Figure.savefig = real_show, real_savefig, real_fig_savefig
        style_new()
        for f in _figures_in(out):
            if f.number not in styled:
                apply(f)
        return out
    return wrapper


def wrap_all(namespace, names):
    """Wrap the named plotting functions in a module's globals() with `printable`."""
    for name in names:
        if callable(namespace.get(name)):
            namespace[name] = printable(namespace[name])
