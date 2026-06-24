"""Component library: validated, parameterized sub-component templates."""

from morphos.library.component import ComponentLibrary, ComponentSpec
from morphos.library.channels import (
    register_channels,
    straight_channel,
    serpentine_channel,
)


def default_library() -> ComponentLibrary:
    """A :class:`ComponentLibrary` preloaded with the built-in components."""
    lib = ComponentLibrary()
    register_channels(lib)
    return lib


__all__ = [
    "ComponentLibrary",
    "ComponentSpec",
    "default_library",
    "register_channels",
    "straight_channel",
    "serpentine_channel",
]
