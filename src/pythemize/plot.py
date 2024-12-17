"""Tools for plots."""

from __future__ import annotations

from attrs import field, frozen
from coloraide import Color

DARK_BACKGROUND_COLOR = Color("#181818")
LIGHT_BACKGROUND_COLOR = Color("#CACACA")
LIGHT_BACKGROUND_LUMINANCE = 0.5

type ColorTuple = tuple[float, float, float] | tuple[float, float, float, float]


@frozen(kw_only=True)
class PlotColors:
    """
    Colors used for plots.

    Colors are supposed to be in srgb space.
    """

    background: Color = field(converter=Color)
    line: Color = field(converter=Color)

    @classmethod
    def from_background_color(cls, background: Color | str | ColorTuple) -> PlotColors:
        """Initialize the colors based on the background color."""
        match background:
            case tuple():
                alpha = background[3] if len(background) == 4 else 1  # noqa: PLR2004
                background = Color("srgb", data=background[:3], alpha=alpha)
            case str():
                background = Color(background)
            case _:
                pass

        if background.luminance() < LIGHT_BACKGROUND_LUMINANCE:
            line = Color("white")
        else:
            line = Color("black")

        return PlotColors(background=background, line=line)
