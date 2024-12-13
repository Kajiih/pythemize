"""Tools for plots."""

from __future__ import annotations

from attrs import field, frozen
from coloraide import Color

DARK_BACKGROUND_COLOR = Color("#181818")
LIGHT_BACKGROUND_COLOR = Color("#CACACA")
LIGHT_BACKGROUND_LUMINANCE = 0.5


@frozen(kw_only=True)
class PlotColors:
    """
    Colors used for plots.

    Colors are supposed to be in srgb space.
    """

    background: Color = field(converter=Color)
    line: Color = field(converter=Color)

    @classmethod
    def from_background_color(cls, background: Color | str) -> PlotColors:
        """Initialize the colors basd on the background color."""
        background = Color(background)

        if background.luminance() < LIGHT_BACKGROUND_LUMINANCE:
            line = Color("white")
        else:
            line = Color("black")

        return PlotColors(background=background, line=line)
