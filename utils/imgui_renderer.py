# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
"""ImGui renderer for moderngl-window using imgui-bundle's built-in OpenGL3 backend.

Replaces moderngl_window.integrations.imgui.ModernglWindowRenderer which
only works with the old pyimgui package.
"""

from imgui_bundle import imgui


class ModernglImguiRenderer:
    """Bridges imgui-bundle's OpenGL3 backend with moderngl-window."""

    def __init__(self, wnd):
        self.wnd = wnd
        self.ctx = wnd.ctx

        # IO setup
        io = imgui.get_io()
        io.display_size = imgui.ImVec2(float(wnd.width), float(wnd.height))
        # Let moderngl-window handle DPI; setting framebuffer_scale to (1,1)
        # avoids double-scaling on Retina displays.
        io.display_framebuffer_scale = imgui.ImVec2(1.0, 1.0)

        # Initialize imgui-bundle's OpenGL3 backend
        imgui.backends.opengl3_init("#version 330")

        # Key map for input forwarding
        self._init_key_map()

        # Track registered textures (for API compat, backend handles binding)
        self._textures = {}

    # ------------------------------------------------------------------
    # Font texture (delegated to OpenGL3 backend)
    # ------------------------------------------------------------------

    def refresh_font_texture(self):
        """Rebuild font atlas texture via the OpenGL3 backend."""
        imgui.backends.opengl3_create_device_objects()

    # ------------------------------------------------------------------
    # Texture registration (for imgui.image widgets)
    # ------------------------------------------------------------------

    def register_texture(self, texture):
        self._textures[texture.glo] = texture

    def remove_texture(self, texture):
        self._textures.pop(texture.glo, None)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, draw_data):
        if draw_data is None:
            return
        # Use imgui-bundle's built-in OpenGL3 renderer
        imgui.backends.opengl3_render_draw_data(draw_data)

    # ------------------------------------------------------------------
    # Window resize
    # ------------------------------------------------------------------

    def resize(self, width, height):
        io = imgui.get_io()
        io.display_size = imgui.ImVec2(float(width), float(height))

    # ------------------------------------------------------------------
    # Input forwarding
    # ------------------------------------------------------------------

    def _init_key_map(self):
        """Build mapping from moderngl-window key codes to imgui Key enum."""
        self._key_map = {}
        try:
            keys = self.wnd.keys
            mapping = {
                keys.TAB: imgui.Key.tab,
                keys.LEFT: imgui.Key.left_arrow,
                keys.RIGHT: imgui.Key.right_arrow,
                keys.UP: imgui.Key.up_arrow,
                keys.DOWN: imgui.Key.down_arrow,
                keys.PAGE_UP: imgui.Key.page_up,
                keys.PAGE_DOWN: imgui.Key.page_down,
                keys.HOME: imgui.Key.home,
                keys.END: imgui.Key.end,
                keys.DELETE: imgui.Key.delete,
                keys.BACKSPACE: imgui.Key.backspace,
                keys.ENTER: imgui.Key.enter,
                keys.ESCAPE: imgui.Key.escape,
                keys.SPACE: imgui.Key.space,
            }
            for wnd_key, imgui_key in mapping.items():
                if wnd_key is not None:
                    self._key_map[wnd_key] = imgui_key
        except Exception:
            pass

    def mouse_position_event(self, x, y, dx, dy):
        io = imgui.get_io()
        io.add_mouse_pos_event(x, y)

    def mouse_press_event(self, x, y, button):
        io = imgui.get_io()
        io.add_mouse_pos_event(x, y)
        # moderngl-window: 1=left, 2=right, 3=middle
        # imgui: 0=left, 1=right, 2=middle
        io.add_mouse_button_event(button - 1, True)

    def mouse_release_event(self, x, y, button):
        io = imgui.get_io()
        io.add_mouse_button_event(button - 1, False)

    def mouse_drag_event(self, x, y, dx, dy):
        io = imgui.get_io()
        io.add_mouse_pos_event(x, y)

    def mouse_scroll_event(self, x_offset, y_offset):
        io = imgui.get_io()
        io.add_mouse_wheel_event(x_offset, y_offset)

    def key_event(self, key, action, modifiers):
        io = imgui.get_io()
        imgui_key = self._key_map.get(key)
        if imgui_key is not None:
            pressed = action == self.wnd.keys.ACTION_PRESS
            io.add_key_event(imgui_key, pressed)

        # Modifier keys
        try:
            io.add_key_event(
                imgui.Key.mod_ctrl,
                bool(modifiers.ctrl) if hasattr(modifiers, "ctrl") else False,
            )
            io.add_key_event(
                imgui.Key.mod_shift,
                bool(modifiers.shift) if hasattr(modifiers, "shift") else False,
            )
            io.add_key_event(
                imgui.Key.mod_alt,
                bool(modifiers.alt) if hasattr(modifiers, "alt") else False,
            )
        except Exception:
            pass

    def unicode_char_entered(self, char):
        io = imgui.get_io()
        io.add_input_character(ord(char))
