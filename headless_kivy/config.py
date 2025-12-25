# pyright: reportMissingImports=false
"""Implement `setup_headless_kivy`, it configures headless-kivy."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, NoReturn, NotRequired, Protocol, TypedDict

import kivy
import numpy as np
from kivy.config import Config
from kivy.metrics import dp

from headless_kivy.constants import (
    BANDWIDTH_LIMIT,
    BANDWIDTH_LIMIT_OVERHEAD,
    BANDWIDTH_LIMIT_WINDOW,
    DOUBLE_BUFFERING,
    FLIP_HORIZONTAL,
    FLIP_VERTICAL,
    HEIGHT,
    IS_DEBUG_MODE,
    REGION_SIZE,
    ROTATION,
    WIDTH,
    WINDOW_MODE,
)
from headless_kivy.logger import add_file_handler, add_stdout_handler

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy._typing import NDArray  # pyright: ignore[reportPrivateImportUsage]

kivy.require('2.1.0')  # pyright: ignore[reportAttributeAccessIssue]


class SetupHeadlessConfig(TypedDict):
    """Arguments of `setup_headless_kivy` function.

    Attributes
    ----------
    callback: `Callback`
        The callback function that will render the data to the screen.
    bandwidth_limit: `int`, optional
        Maximum bandwidth usage in pixels per second, no limit if set to 0.
    bandwidth_limit_window: `float`, optional
        Length of the time window in seconds to check the bandwidth limit.
    bandwidth_limit_overhead: `int`, optional
        The overhead of each draw command in pixels, regardless of its size.
    width: `int`, optional
        The width of the display in pixels.
    height: `int`, optional
        The height of the display in pixels.
    is_debug_mode: `bool`, optional
        If set to True, the application will consume computational resources to log
        additional debug information.
    double_buffering: `bool`, optional
        If set to True, it will let Kivy generate the next frame while sending the last
        frame to the display.
    rotation: `int`, optional
        The rotation of the display clockwise, it will be multiplied by 90.
    flip_horizontal: `bool`, optional
        Whether the screen should be flipped horizontally or not.
    flip_vertical: `bool`, optional
        Whether the screen should be flipped vertically or not.
    region_size: `int`, optional
        Approximate size of rectangles to divide the screen into and see if they need to
        be updated.
    window_mode: `str`, optional
        Control window creation behavior. Options:
        - 'auto': Default behavior, create window on available display
        - 'hidden': Create window but keep it hidden
        - 'dummy': Use dummy SDL video driver (no physical display)
        - 'offscreen': Use offscreen rendering (no window)
        - 'none': Disable window creation entirely
    display_selector: `Callable[[list[dict]], int]`, optional
        Function to select which display to use when multiple are available.
        Receives list of display info dicts with 'index', 'width', 'height', 'name'.
        Should return the index of the display to use.
        Example: lambda displays: min(
            displays, key=lambda d: d['width'] * d['height']
        )['index']

    """

    callback: Callback
    bandwidth_limit: NotRequired[int]
    bandwidth_limit_window: NotRequired[float]
    bandwidth_limit_overhead: NotRequired[int]
    width: NotRequired[int]
    height: NotRequired[int]
    is_debug_mode: NotRequired[bool]
    double_buffering: NotRequired[bool]
    rotation: NotRequired[int]
    flip_horizontal: NotRequired[bool]
    flip_vertical: NotRequired[bool]
    region_size: NotRequired[int]
    window_mode: NotRequired[str]
    display_selector: NotRequired[Callable[[list[dict]], int]]


_config: SetupHeadlessConfig | None = None


def report_uninitialized() -> NoReturn:
    """Report that the module has not been initialized."""
    msg = """You need to run `setup_headless_kivy` before importing \
`kivy.core.window` module. \
Note that it might have been imported by another module unintentionally."""
    raise RuntimeError(msg)


def setup_headless_kivy(config: SetupHeadlessConfig) -> None:
    """Configure the headless mode for the Kivy application.

    Arguments:
    ---------
    config: `SetupHeadlessConfig`

    """
    import os

    global _config  # noqa: PLW0603
    _config = config

    if is_debug_mode():
        add_stdout_handler()
        add_file_handler()

    # Configure window mode before Kivy initialization
    window_mode = config.get('window_mode', WINDOW_MODE)
    if window_mode == 'dummy':
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    elif window_mode == 'offscreen':
        os.environ['SDL_VIDEODRIVER'] = 'offscreen'
    elif window_mode == 'hidden':
        Config.set('graphics', 'window_state', 'hidden')
    elif window_mode == 'none':
        os.environ['KIVY_WINDOW'] = ''

    # Configure display selection if selector is provided
    display_selector = config.get('display_selector')
    if display_selector and window_mode == 'auto':
        try:
            import sdl2
            import sdl2.ext

            # Initialize SDL video subsystem
            sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)

            # Query available displays
            num_displays = sdl2.SDL_GetNumVideoDisplays()
            displays = []

            for i in range(num_displays):
                mode = sdl2.SDL_DisplayMode()
                sdl2.SDL_GetCurrentDisplayMode(i, mode)
                name = sdl2.SDL_GetDisplayName(i)
                displays.append({
                    'index': i,
                    'width': mode.w,
                    'height': mode.h,
                    'refresh_rate': mode.refresh_rate,
                    'name': name.decode('utf-8') if name else f'Display {i}',
                })

            # Let user select display
            selected_index = display_selector(displays)

            # Configure SDL to use selected display
            os.environ['SDL_VIDEO_FULLSCREEN_DISPLAY'] = str(selected_index)

            # Clean up SDL
            sdl2.SDL_Quit()
        except (ImportError, Exception) as e:
            # If SDL2 not available or error occurs, fall back to default behavior
            if is_debug_mode():
                from headless_kivy import logger
                logger.logger.warning(
                    f'Display selection failed: {e}, using default display',
                )

    Config.set('kivy', 'kivy_clock', 'default')
    Config.set('graphics', 'fbo', 'force-hardware')
    Config.set('graphics', 'fullscreen', '0')
    Config.set('graphics', 'resizable', '0')
    Config.set('graphics', 'width', f'{width()}')
    Config.set('graphics', 'height', f'{height()}')

    from headless_kivy.widget import HeadlessWidget

    HeadlessWidget.raw_data = np.zeros(
        (int(dp(height())), int(dp(width())), 4),
        dtype=np.uint8,
    )


def check_initialized() -> None:
    """Check if the module has been initialized."""
    if not _config:
        report_uninitialized()


class Region(TypedDict):
    """A region of the screen to be updated."""

    rectangle: tuple[int, int, int, int]
    data: NDArray[np.uint8]


class Callback(Protocol):
    """The signature of the renderer function."""

    def __call__(self: Callback, *, regions: list[Region]) -> None:
        """Render the data to the screen."""


@cache
def callback() -> Callback:
    """Return the render function, called whenever data is ready to be rendered."""
    if _config:
        return _config.get('callback', lambda **_: None)
    report_uninitialized()


@cache
def bandwidth_limit() -> int:
    """Return the bandwidth limit in pixels per second."""
    if _config:
        return _config.get('bandwidth_limit', BANDWIDTH_LIMIT)
    report_uninitialized()


@cache
def bandwidth_limit_window() -> float:
    """Return the length of the window in seconds to check the bandwidth limit."""
    if _config:
        return _config.get('bandwidth_limit_window', BANDWIDTH_LIMIT_WINDOW)
    report_uninitialized()


@cache
def bandwidth_limit_overhead() -> int:
    """Return the bandwidth overhead of each draw regardless of the region size."""
    if _config:
        return _config.get('bandwidth_limit_overhead', BANDWIDTH_LIMIT_OVERHEAD)
    report_uninitialized()


@cache
def width() -> int:
    """Return the width of the display in pixels."""
    if _config:
        return _config.get('width', WIDTH)
    report_uninitialized()


@cache
def height() -> int:
    """Return the height of the display in pixels."""
    if _config:
        return _config.get('height', HEIGHT)
    report_uninitialized()


@cache
def is_debug_mode() -> bool:
    """Return `True` if the application will consume computational resources to log."""
    if _config:
        return _config.get('is_debug_mode', IS_DEBUG_MODE)
    report_uninitialized()


@cache
def double_buffering() -> bool:
    """Generate the next frame while sending the last frame to the display."""
    if _config:
        return _config.get('double_buffering', DOUBLE_BUFFERING)
    report_uninitialized()


@cache
def rotation() -> int:
    """Return the rotation of the display."""
    if _config:
        return _config.get('rotation', ROTATION)
    report_uninitialized()


@cache
def flip_horizontal() -> bool:
    """Return `True` if the display is flipped horizontally."""
    if _config:
        return _config.get('flip_horizontal', FLIP_HORIZONTAL)
    report_uninitialized()


@cache
def flip_vertical() -> bool:
    """Return `True` if the display is flipped vertically."""
    if _config:
        return _config.get('flip_vertical', FLIP_VERTICAL)
    report_uninitialized()


@cache
def region_size() -> int:
    """Return the approximate size of rectangles to divide the screen into."""
    if _config:
        return _config.get('region_size', REGION_SIZE)
    report_uninitialized()


@cache
def window_mode() -> str:
    """Return the window mode configuration."""
    if _config:
        return _config.get('window_mode', WINDOW_MODE)
    report_uninitialized()
