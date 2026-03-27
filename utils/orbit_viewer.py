# pyre-ignore-all-errors
"""
Base class for 3D visualization with orbit controls and ImGui UI.

Usage:
    class MyViewer(OrbitViewer):
        def render_3d(self, time: float, frame_time: float) -> None:
            # Your 3D rendering code here
            pass

        def render_ui(self) -> None:
            # Your ImGui UI code here
            imgui.text("Hello World")

    if __name__ == "__main__":
        mglw.run_window_config(MyViewer)
"""

import platform

# --- macOS activation policy fix (MUST be done before any window code) ---
if platform.system() == "Darwin":
    try:
        from AppKit import NSApp, NSApplication, NSApplicationActivationPolicyRegular

        NSApplication.sharedApplication()
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    except ImportError:
        # Try ctypes fallback
        try:
            import ctypes
            import ctypes.util

            objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            objc.objc_msgSend.restype = ctypes.c_void_p
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            NSApplication = objc.objc_getClass(b"NSApplication")
            app = objc.objc_msgSend(
                NSApplication, objc.sel_registerName(b"sharedApplication")
            )
            objc.objc_msgSend.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_long,
            ]
            objc.objc_msgSend(app, objc.sel_registerName(b"setActivationPolicy:"), 0)
        except Exception:
            pass
    except Exception:
        pass

import imgui
import moderngl_window as mglw
import numpy as np
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from pyrr import Matrix44

scale_factor = 1
if platform.system() == "Linux":
    scale_factor = 2


class OrbitViewer(mglw.WindowConfig):
    """Base class for 3D visualization with orbit camera controls and ImGui UI."""

    title = "3D Orbit Viewer"
    window_size = (scale_factor * 1280, scale_factor * 720)
    gl_version = (3, 3)
    aspect_ratio = None
    resizable = True

    # Use GLFW backend on macOS for better focus handling (pyglet has issues)
    if platform.system() == "Darwin":
        window = "glfw"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Window configuration - override in subclass if needed

        # --- macOS window focus workaround ---
        # Track focus request attempts (done in on_render for better timing)
        self._focus_requested = False
        self._focus_attempt_frame = 0

        # On macOS, set activation policy BEFORE window is shown
        if platform.system() == "Darwin":
            self._set_macos_activation_policy()

        # --- ImGui setup ---
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)

        io = imgui.get_io()
        dpi = self.wnd.pixel_ratio
        # Increase font size on Linux for better readability
        if platform.system() == "Linux":
            io.font_global_scale = dpi * scale_factor
            # Also scale UI elements (sliders, buttons, etc.) on Linux
            style = imgui.get_style()
            # Manually scale style sizes since scale_all_sizes() may not be available
            style.window_padding = (
                style.window_padding[0] * scale_factor,
                style.window_padding[1] * scale_factor,
            )
            style.frame_padding = (
                style.frame_padding[0] * scale_factor,
                style.frame_padding[1] * scale_factor,
            )
            style.item_spacing = (
                style.item_spacing[0] * scale_factor,
                style.item_spacing[1] * scale_factor,
            )
            style.item_inner_spacing = (
                style.item_inner_spacing[0] * scale_factor,
                style.item_inner_spacing[1] * scale_factor,
            )
            style.scrollbar_size *= scale_factor
            style.grab_min_size *= scale_factor
        else:
            io.font_global_scale = dpi

        # --- Orbit camera controls ---
        self.camera_distance = 5.0
        self.camera_azimuth = 45.0  # Horizontal rotation (degrees)
        self.camera_elevation = 30.0  # Vertical rotation (degrees)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Background color: 0=White, 1=Light Grey, 2=Grey, 3=Black
        self.bg_color_index = 3
        self.bg_color_options = [
            (1.0, 1.0, 1.0),  # White
            (0.75, 0.75, 0.75),  # Light Grey
            (0.5, 0.5, 0.5),  # Grey
            (0.0, 0.0, 0.0),  # Black
        ]

        # Mouse state for orbit controls
        self.mouse_dragging = False
        self.mouse_panning = False
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.3
        self.pan_sensitivity = 0.002
        self.zoom_sensitivity = 0.05

        # Enable depth testing and backface culling
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.enable(self.ctx.CULL_FACE)  # Hide back faces

        # Call user initialization
        self.init_scene()

    def init_scene(self) -> None:
        """Override this to initialize your 3D scene (shaders, buffers, etc.)."""
        pass

    def render_3d(self, time: float, frame_time: float) -> None:
        """Override this to render your 3D scene.

        Use get_camera_matrices() to get projection, view, and mvp matrices.

        Args:
            time: Total elapsed time in seconds
            frame_time: Time since last frame in seconds
        """
        pass

    def render_ui(self) -> None:
        """Override this to render your ImGui UI.

        ImGui context is already set up. Just add your UI elements.
        """
        pass

    def get_camera_matrices(self) -> tuple[Matrix44, Matrix44, Matrix44]:
        """Get camera transformation matrices.

        Returns:
            tuple: (projection, view, mvp) matrices
        """
        # Projection matrix
        aspect_ratio = self.window_size[0] / self.window_size[1]
        projection = Matrix44.perspective_projection(45.0, aspect_ratio, 0.1, 100.0)

        # Calculate camera position from spherical coordinates (Z-up)
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)

        camera_x = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        camera_y = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        camera_z = self.camera_distance * np.sin(elevation_rad)

        camera_pos = self.camera_target + np.array([camera_x, camera_y, camera_z])

        # View matrix
        view = Matrix44.look_at(
            tuple(camera_pos),
            tuple(self.camera_target),
            (0.0, 0.0, 1.0),  # Z-up (gravity direction)
        )

        # Model matrix (identity - no transformation)
        model = Matrix44.identity()

        # Combined MVP
        mvp = projection * view * model

        return projection, view, mvp

    # -----------------------------
    # MACOS WINDOW FOCUS WORKAROUND
    # -----------------------------
    def _set_macos_activation_policy(self) -> None:
        """Set macOS activation policy to allow the app to receive focus.

        Apps launched from terminal are often 'background' apps that can't take focus.
        This sets the policy to 'regular' so the app can become frontmost.
        """
        try:
            # Try using pyobjc (most reliable)
            from AppKit import (
                NSApp,
                NSApplication,
                NSApplicationActivationPolicyRegular,
            )

            NSApplication.sharedApplication()
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
            NSApp.activateIgnoringOtherApps_(True)
            return
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: use ctypes to call Objective-C runtime directly
        try:
            import ctypes
            import ctypes.util

            # Load AppKit framework
            appkit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))
            objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))

            # Set up objc_msgSend
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            objc.objc_msgSend.restype = ctypes.c_void_p
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            # Get NSApplication class and sharedApplication
            NSApplication = objc.objc_getClass(b"NSApplication")
            sel_sharedApplication = objc.sel_registerName(b"sharedApplication")
            sel_setActivationPolicy = objc.sel_registerName(b"setActivationPolicy:")
            sel_activateIgnoringOtherApps = objc.sel_registerName(
                b"activateIgnoringOtherApps:"
            )

            # Get shared application
            app = objc.objc_msgSend(NSApplication, sel_sharedApplication)

            # Set activation policy to regular (0 = regular, 1 = accessory, 2 = prohibited)
            objc.objc_msgSend.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_long,
            ]
            objc.objc_msgSend(app, sel_setActivationPolicy, 0)

            # Activate ignoring other apps
            objc.objc_msgSend.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_bool,
            ]
            objc.objc_msgSend(app, sel_activateIgnoringOtherApps, True)
        except Exception:
            pass

    def _request_window_focus(self) -> None:
        """Request window focus on macOS to fix greyed-out title bar issue."""
        import subprocess

        # Try pyglet backend
        try:
            if hasattr(self.wnd, "_window"):
                pyglet_window = self.wnd._window
                if hasattr(pyglet_window, "activate"):
                    pyglet_window.activate()
        except Exception:
            pass

        # Try GLFW backend
        try:
            if hasattr(self.wnd, "_window") and hasattr(self.wnd._window, "focus"):
                self.wnd._window.focus()
        except Exception:
            pass

        # Always try AppleScript as well (most reliable on macOS)
        try:
            script = """
            tell application "System Events"
                set frontmost of (first process whose name contains "Python") to true
            end tell
            """
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=2)
        except Exception:
            pass

    # -----------------------------
    # RENDER LOOP
    # -----------------------------
    def on_render(self, time: float, frame_time: float):
        """Main render loop - calls user's render_3d and render_ui methods."""
        # macOS focus workaround: request focus on early frames
        if platform.system() == "Darwin" and self._focus_attempt_frame < 10:
            self._focus_attempt_frame += 1
            if self._focus_attempt_frame in [1, 5, 10]:  # Try at frames 1, 5, and 10
                self._request_window_focus()

        bg = self.bg_color_options[self.bg_color_index]
        self.ctx.clear(*bg)

        # Render 3D scene
        self.render_3d(time, frame_time)

        # Render ImGui UI
        imgui.new_frame()
        self.render_ui()
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def on_resize(self, width: int, height: int):
        """Handle window resize."""
        self.imgui.resize(width, height)

    # -----------------------------
    # MOUSE EVENT HANDLERS
    # -----------------------------
    def on_mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

        # Reset mouse state if cursor is over the UI panel
        if x < getattr(self, "ui_panel_width", 0):
            self.mouse_dragging = False
            self.mouse_panning = False
            self.last_mouse_pos = None

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)

        if x >= getattr(self, "ui_panel_width", 0):
            if self.mouse_dragging:
                # Orbit (rotate camera)
                self.camera_azimuth -= dx * self.mouse_sensitivity
                self.camera_elevation = np.clip(
                    self.camera_elevation + dy * self.mouse_sensitivity, -89.0, 89.0
                )
            elif self.mouse_panning:
                # Pan (move camera target)
                azimuth_rad = np.radians(self.camera_azimuth)
                # Right vector is tangent to the azimuth circle in XY plane (Z-up)
                right = np.array([np.sin(azimuth_rad), -np.cos(azimuth_rad), 0])
                up = np.array([0, 0, 1])  # Z-up

                # Use a minimum effective distance to prevent panning from becoming too slow when zoomed in
                effective_distance = max(self.camera_distance, 0.5)

                self.camera_target += (
                    right * dx * self.pan_sensitivity * effective_distance
                )
                self.camera_target += (
                    up * dy * self.pan_sensitivity * effective_distance
                )

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        # Manually handle imgui scroll since mouse_scroll_event may not be compatible
        io = imgui.get_io()
        if hasattr(io, "mouse_wheel"):
            io.mouse_wheel = y_offset

        # Use mouse position to decide if scroll should go to UI or 3D viewport
        mouse_x = io.mouse_pos.x if hasattr(io.mouse_pos, "x") else io.mouse_pos[0]
        if mouse_x >= getattr(self, "ui_panel_width", 0):
            # Zoom (change camera distance)
            self.camera_distance *= 1.0 + y_offset * self.zoom_sensitivity
            self.camera_distance = np.clip(self.camera_distance, 0.01, 50.0)

    def on_mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

        # Deterministic position-based check: the UI panel occupies x < ui_panel_width
        should_capture = x < getattr(self, "ui_panel_width", 0)

        if not should_capture:
            if button == 1:  # Left mouse button - pan
                self.mouse_panning = True
                self.last_mouse_pos = (x, y)
            elif button == 2:  # Right mouse button - orbit
                self.mouse_dragging = True
                self.last_mouse_pos = (x, y)
        else:
            # Explicitly clear camera state when ImGUI wants the mouse
            self.mouse_dragging = False
            self.mouse_panning = False
            self.last_mouse_pos = None

    def on_mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

        # Always clear mouse state on release regardless of ImGUI state
        if button == 1:
            self.mouse_panning = False
        elif button == 2:
            self.mouse_dragging = False

        self.last_mouse_pos = None

    def on_key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)

    def on_unicode_char_entered(self, char: str):
        self.imgui.unicode_char_entered(char)
