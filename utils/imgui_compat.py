# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""Compatibility shim: makes imgui-bundle look like pyimgui.

Usage:
    import utils.imgui_compat as imgui
    imgui.begin("Window", flags=imgui.WINDOW_NO_MOVE)
"""

from imgui_bundle import imgui as _imgui

# ---------------------------------------------------------------------------
# Forward anything not explicitly defined here to the underlying imgui module
# ---------------------------------------------------------------------------


def __getattr__(name):
    return getattr(_imgui, name)


# ---------------------------------------------------------------------------
# Constants  (pyimgui name -> imgui-bundle enum value)
# ---------------------------------------------------------------------------

# Condition flags
ONCE = int(_imgui.Cond_.once)
ALWAYS = int(_imgui.Cond_.always)

# Window flags
WINDOW_NO_MOVE = int(_imgui.WindowFlags_.no_move)
WINDOW_NO_RESIZE = int(_imgui.WindowFlags_.no_resize)
WINDOW_NO_BRING_TO_FRONT_ON_FOCUS = int(_imgui.WindowFlags_.no_bring_to_front_on_focus)
WINDOW_NO_TITLE_BAR = int(_imgui.WindowFlags_.no_title_bar)
WINDOW_NO_SCROLLBAR = int(_imgui.WindowFlags_.no_scrollbar)
WINDOW_ALWAYS_AUTO_RESIZE = int(_imgui.WindowFlags_.always_auto_resize)
WINDOW_NO_FOCUS_ON_APPEARING = int(_imgui.WindowFlags_.no_focus_on_appearing)
WINDOW_NO_NAV = int(_imgui.WindowFlags_.no_nav)

# Color indices  (push_style_color / pop_style_color)
COLOR_BUTTON = int(_imgui.Col_.button)
COLOR_BUTTON_HOVERED = int(_imgui.Col_.button_hovered)
COLOR_BUTTON_ACTIVE = int(_imgui.Col_.button_active)
COLOR_TEXT = int(_imgui.Col_.text)

# Input text flags
INPUT_TEXT_ENTER_RETURNS_TRUE = int(_imgui.InputTextFlags_.enter_returns_true)

# Hovered flags
HOVERED_ALLOW_WHEN_DISABLED = int(_imgui.HoveredFlags_.allow_when_disabled)

# ---------------------------------------------------------------------------
# Wrapped functions whose signature differs between pyimgui and imgui-bundle
# ---------------------------------------------------------------------------


def begin(name, closable=None, flags=0):
    """pyimgui always returns (expanded, opened)."""
    if closable is not None:
        return _imgui.begin(name, closable, flags)
    result = _imgui.begin(name, None, flags)
    # imgui-bundle returns (visible, p_open=None) when p_open is None
    if isinstance(result, tuple):
        return result
    return (result, True)


def set_next_window_position(x, y, condition=0, pivot_x=0.0, pivot_y=0.0):
    _imgui.set_next_window_pos(
        _imgui.ImVec2(x, y), condition, _imgui.ImVec2(pivot_x, pivot_y)
    )


def set_next_window_size(w, h, condition=0):
    _imgui.set_next_window_size(_imgui.ImVec2(w, h), condition)


def button(label, width=0, height=0):
    return _imgui.button(label, _imgui.ImVec2(width, height))


def get_color_u32_rgba(r, g, b, a=1.0):
    return _imgui.get_color_u32(_imgui.ImVec4(r, g, b, a))


def text_colored(text, r, g, b, a=1.0):
    """pyimgui: text first, then color.  imgui-bundle: color first."""
    _imgui.text_colored(_imgui.ImVec4(r, g, b, a), text)


def push_style_color(idx, r, g, b, a=1.0):
    _imgui.push_style_color(idx, _imgui.ImVec4(r, g, b, a))


def image(texture_id, width, height, **kwargs):
    tex_ref = _imgui.ImTextureRef(int(texture_id))
    _imgui.image(tex_ref, _imgui.ImVec2(width, height))


def input_text(label, value, buffer_length=256, flags=0):
    """pyimgui takes buffer_length; imgui-bundle does not."""
    changed, new_value = _imgui.input_text(label, value, flags)
    return changed, new_value


def calc_text_size(text, *args, **kwargs):
    return _imgui.calc_text_size(text)


def get_content_region_available():
    return _imgui.get_content_region_avail()


def get_item_rect_min():
    return _imgui.get_item_rect_min()


def get_window_position():
    return _imgui.get_window_pos()


# Re-export commonly used functions that have the same signature so they
# appear as module-level names (slightly faster lookup than __getattr__).
create_context = _imgui.create_context
get_io = _imgui.get_io
get_style = _imgui.get_style
new_frame = _imgui.new_frame
render = _imgui.render
get_draw_data = _imgui.get_draw_data
end = _imgui.end
text = _imgui.text
text_disabled = _imgui.text_disabled
spacing = _imgui.spacing
separator = _imgui.separator
same_line = _imgui.same_line
checkbox = _imgui.checkbox
slider_float = _imgui.slider_float
slider_int = _imgui.slider_int
combo = _imgui.combo
push_item_width = _imgui.push_item_width
pop_item_width = _imgui.pop_item_width
pop_style_color = _imgui.pop_style_color
tree_node = _imgui.tree_node
tree_pop = _imgui.tree_pop
is_item_hovered = _imgui.is_item_hovered
is_item_active = _imgui.is_item_active
begin_tooltip = _imgui.begin_tooltip
end_tooltip = _imgui.end_tooltip
set_keyboard_focus_here = _imgui.set_keyboard_focus_here


def set_window_font_scale(scale):
    """No-op: removed in newer Dear ImGui. Use font pushing instead."""
    pass


class _DrawListWrapper:
    """Wraps ImDrawList to accept raw floats instead of ImVec2."""

    def __init__(self, draw_list):
        self._dl = draw_list

    def add_line(self, x1, y1, x2, y2, col, thickness=1.0):
        self._dl.add_line(_imgui.ImVec2(x1, y1), _imgui.ImVec2(x2, y2), col, thickness)

    def add_rect(self, x1, y1, x2, y2, col, rounding=0.0, flags=0, thickness=1.0):
        self._dl.add_rect(
            _imgui.ImVec2(x1, y1),
            _imgui.ImVec2(x2, y2),
            col,
            rounding,
            flags,
            thickness,
        )

    def add_rect_filled(self, x1, y1, x2, y2, col, rounding=0.0, flags=0):
        self._dl.add_rect_filled(
            _imgui.ImVec2(x1, y1), _imgui.ImVec2(x2, y2), col, rounding, flags
        )

    def add_text(self, x, y, col, text):
        self._dl.add_text(_imgui.ImVec2(x, y), col, text)

    def __getattr__(self, name):
        return getattr(self._dl, name)


def get_foreground_draw_list():
    return _DrawListWrapper(_imgui.get_foreground_draw_list())


def get_window_draw_list():
    return _DrawListWrapper(_imgui.get_window_draw_list())


# ImVec2 re-export for direct use
ImVec2 = _imgui.ImVec2
ImVec4 = _imgui.ImVec4
