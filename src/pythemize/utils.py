"""Main module."""

from collections.abc import Iterable, Mapping
from typing import Literal, NotRequired, TypedDict

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from coloraide import Color

# from coloraide.everything import ColorAll as Color
from coloraide.channels import Channel
from coloraide.everything import ColorAll
from coloraide.spaces import Space
from IPython.core.display import HTML
from IPython.core.display_functions import display


# %% === Types ===
class ThemeDict(TypedDict):
    """A VS Code theme config as a dictionary."""

    colors: Mapping[str, str]
    # tokenColors: NotRequired[Mapping[str, str]]
    # semanticTokenColors: NotRequired[Mapping[str, str]]


class ThemeColorDict(TypedDict):
    """A VS Code theme config as a dictionary whose values are Color objects."""

    colors: Mapping[str, Color]
    tokenColors: NotRequired[Mapping[str, Color]]
    semanticTokenColors: NotRequired[Mapping[str, Color]]


# %% === Exceptions ===
class ChannelNotFound(Exception):
    """Channel not found in a space channel aliases."""

    def __init__(self, channel_name: str) -> None:
        super().__init__(f"Channel {channel_name} not found.")


class ChannelAliasesNotFound(Exception):
    """None of the aliases where found in the channel aliases map."""

    def __init__(self, channel_aliases: Iterable[str]) -> None:
        super().__init__(f"None of the channel aliases {channel_aliases} were found.")


# %% === Function utils ===
def group_html(colors: list[Color], gap: int = 1) -> str:
    """Return the html representation of a list of colors."""

    def remove_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        # Find all <div> elements
        divs = soup.find_all("div")
        # Remove the last <div>, if it exists
        if divs:
            divs[-1].decompose()
        return str(soup)

    html_fragments = [remove_text(color._repr_html_()) for color in colors]  # pyright: ignore[reportPrivateUsage]

    html_output = f'<div style="display: flex; gap: {gap}px;">' + "".join(html_fragments) + "</div>"

    return html_output


def group_display(colors: list[Color], gap: int = 1) -> None:
    """Display a list of colors."""
    display(HTML(group_html(colors, gap)))


def gradient_lightness_old(
    ref_color: Color,
    space: Literal["okhsl", "oklch", "oklab", "hsluv", "hsl", "hct"],
    steps: int = 10,
) -> list[Color]:
    """Create a lightness gradient."""
    c_start = ref_color.convert(space)
    c_end = ColorAll(c_start)
    if "ok" in space or space == "hsl":
        lightness_canal = "l"
        range_max = 1
    elif space == "hsluv":
        lightness_canal = "l"
        range_max = 100
    elif space == "hct":
        lightness_canal = "t"
        range_max = 100
    else:
        raise NotImplementedError

    c_start.set(lightness_canal, 0)
    c_end.set(lightness_canal, range_max)

    return ColorAll.steps([c_start, c_end], steps=steps, space=space)


def find_channel(space_class: Space, name: str) -> Channel:
    """Return the channel with the given name."""
    channels = space_class.CHANNELS
    for channel in channels:
        if channel == name:
            return channel

    raise ChannelNotFound(name)


def find_channel_name(space_class: Space, aliases: Iterable[str]) -> str:
    """Return the first channel name found corresponding to one of the aliases."""
    channel_aliases = space_class.CHANNEL_ALIASES
    for alias in aliases:
        if alias in channel_aliases:
            return channel_aliases[alias]

    raise ChannelAliasesNotFound(aliases)


def gradient_on_channel(
    ref_color: Color,
    channel_name: str,
    space: str | None = None,
    nb_steps: int = 10,
) -> list[Color]:
    """
    Return a list of colors representing a gradient of the given channel.

    channel_name can be a channel alias.
    """
    if space is None:
        space = ref_color.space()
        ref_color = ref_color.clone()
    else:
        ref_color = ref_color.convert(space)

    space_class = ref_color.CS_MAP[space]
    channel_name = space_class.CHANNEL_ALIASES.get(channel_name, channel_name)

    channel = find_channel(space_class, channel_name)

    c_start, c_end = ref_color, ref_color.clone()
    c_start.set(channel_name, channel.low)
    c_end.set(channel_name, channel.high)

    return ColorAll.steps([c_start, c_end], steps=nb_steps, space=space)


def gradient_lightness(
    ref_color: Color,
    space: str,
    nb_steps: int = 10,
) -> list[Color]:
    """Create a lightness gradient."""
    space_class = ColorAll.CS_MAP[space]

    channel_name = find_channel_name(space_class, ["lightness"])

    return gradient_on_channel(ref_color, channel_name, space, nb_steps)


def gradient_saturation(
    ref_color: Color,
    space: str,
    nb_steps: int = 10,
) -> list[Color]:
    """Create a lightness gradient."""
    space_class = ColorAll.CS_MAP[space]

    channel_name = find_channel_name(space_class, ["saturation", "chroma"])

    return gradient_on_channel(ref_color, channel_name, space, nb_steps)


def multi_gradient_lightness(
    ref_colors: Iterable[Color],
    spaces: Iterable[str] | Literal["all"] = "all",
    nb_steps: int = 10,
) -> dict[str, dict[str, list[Color]]]:
    """Create a lightness gradient."""
    if spaces == "all":
        spaces = (
            "hsl",
            "hsluv",
            "okhsl",
            "oklab",
            "oklch",
            "hct",
        )
    return {
        str(color): {space: gradient_lightness(color, space, nb_steps) for space in spaces}
        for color in ref_colors
    }


def multi_gradient_saturations(
    ref_colors: Iterable[Color],
    spaces: Iterable[str] | Literal["all"] = "all",
    nb_steps: int = 10,
) -> dict[str, dict[str, list[Color]]]:
    """Create a lightness gradient."""
    if spaces == "all":
        spaces = (
            "hsl",
            "hsluv",
            "okhsl",
            "oklch",
            "hct",
        )
    return {
        str(color): {space: gradient_saturation(color, space, nb_steps) for space in spaces}
        for color in ref_colors
    }


def plot_coordinate(
    color_matrix: dict[str, list[Color]],
    coordinate: str | int,
    space: str = "hsl",
) -> None:
    """
    Plot the value of one particular coordinate for a matrix of gradients.

    Args:
        color_matrix: A 2D list where each row represents a gradient of colors.
        coordinate: The coordinate to plot (by channel name: e.g., "h", "s", "l"
            or channel index: 1, 2 or 3).
        space: The color space to use (default is "hsl").
    """
    for name, colors in color_matrix.items():
        # Extract the values for the specified coordinate
        values = [color.convert(space)[coordinate] for color in colors]
        # Plot the values
        plt.plot(range(len(values)), values, label=name)

    space_class = ColorAll.CS_MAP[space]
    channel_name = space_class.CHANNELS[coordinate] if isinstance(coordinate, int) else coordinate
    channel = find_channel(space_class, channel_name)
    # Add labels, legend, and title
    plt.ylim(channel.low, channel.high)  # Scale y-axis based on channel bounds
    plt.xlabel("Gradient Steps")
    plt.ylabel(f"{coordinate} Coordinate in {space.upper()} Space")
    plt.title(f"Color Coordinate {channel_name} Plot")
    plt.legend()
    plt.grid()
    plt.show()
