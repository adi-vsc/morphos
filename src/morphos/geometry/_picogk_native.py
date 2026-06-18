"""Low-level ctypes binding to the native PicoGK runtime.

PicoGK ships a native voxel-geometry library (``picogk.<ver>.dll`` / ``.dylib``)
that its C# layer drives through P/Invoke. This module reaches the same native ABI
directly from Python, so morphos can use the production Leap71 kernel without a
.NET interop layer. It is deliberately minimal: enough to load the runtime, build
a primitive, and pull the narrow-band signed-distance field into a numpy array.

Status (spike). Proven working on Windows x64 with picogk 26.2.0: the library
loads, a sphere ``Voxels`` object is created, its active bounding box and
per-Z-slice signed-distance values read back into numpy as a coherent narrow
band. Open item before this can back a real :class:`GeometryKernel`: PicoGK stores
a *narrow band*, so the deep interior is inactive and reads as the positive
background value. A correct solid mask needs an interior fill (flood fill, the
native ``Voxels_bIsInside``, or the C# BlackWhite slice mode). Until that is
wired and FD/geometry-verified, this module returns the raw band only.

The native library directory is resolved from ``$PICOGK_NATIVE_DIR`` if set, else
the conventional sibling checkout ``~/ai-agent/PicoGK/native/<platform>``.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LIB_BASENAME = "picogk.26.2"


def _platform_subdir() -> str:
    if sys.platform.startswith("win"):
        return "win-x64"
    if sys.platform == "darwin":
        return "osx-arm64"
    return "linux-x64"


def _native_dir() -> Path:
    env = os.environ.get("PICOGK_NATIVE_DIR")
    if env:
        return Path(env)
    return Path.home() / "ai-agent" / "PicoGK" / "native" / _platform_subdir()


def _library_path() -> Path:
    suffix = ".dll" if sys.platform.startswith("win") else (
        ".dylib" if sys.platform == "darwin" else ".so"
    )
    return _native_dir() / f"{_LIB_BASENAME}{suffix}"


class _Vector3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


_lib = None  # cached CDLL handle


def _load():
    """Load and bind the native library once; return the bound CDLL or None."""
    global _lib
    if _lib is not None:
        return _lib
    path = _library_path()
    if not path.exists():
        return None
    try:
        if sys.platform.startswith("win"):
            os.add_dll_directory(str(path.parent))  # so sibling deps resolve
        lib = ctypes.CDLL(str(path))
    except OSError:
        return None

    def bind(name, argtypes, restype):
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    lib.b_GetVersion = bind("Library_GetVersion", [ctypes.c_char_p], None)
    lib.b_CreateInstance = bind("Library_hCreateInstance", [ctypes.c_float], ctypes.c_int64)
    lib.b_CreateSphere = bind(
        "Voxels_hCreateSphere",
        [ctypes.c_int64, ctypes.POINTER(_Vector3), ctypes.c_float],
        ctypes.c_int64,
    )
    lib.b_GetDims = bind(
        "Voxels_GetVoxelDimensions",
        [ctypes.c_int64, ctypes.c_int64] + [ctypes.POINTER(ctypes.c_int)] * 6,
        None,
    )
    lib.b_GetZSlice = bind(
        "Voxels_GetZSlice",
        [ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)],
        None,
    )
    _lib = lib
    return _lib


def picogk_available() -> bool:
    """True if the native PicoGK runtime can be loaded on this host."""
    return _load() is not None


def version() -> str:
    lib = _load()
    if lib is None:
        raise RuntimeError("native PicoGK runtime is not available")
    buf = ctypes.create_string_buffer(256)
    lib.b_GetVersion(buf)
    return buf.value.decode(errors="replace")


@dataclass
class VoxelField:
    """A narrow-band signed-distance volume pulled from PicoGK.

    volume: float array (nz, ny, nx) of signed distance in *voxel* units, clamped
            to +/- ``background`` outside the narrow band.
    origin: (ox, oy, oz) integer voxel index of the array corner.
    voxel_mm: voxel edge length in mm (multiply ``volume`` by this for mm).
    background: narrow-band half-width in voxel units (the clamp value).
    """

    volume: np.ndarray
    origin: tuple
    voxel_mm: float
    background: float


def solidify(volume: np.ndarray, background: float) -> np.ndarray:
    """Turn a PicoGK narrow-band volume into a sign-correct solid SDF.

    PicoGK stores only the narrow band around the surface; the deep interior is
    inactive and reads as ``+background``, indistinguishable by value from the
    deep exterior. Recover the solid by flooding only the *inactive* voxels (the
    ones exactly at ``+background``) from the grid boundary: the whole active band
    (every voxel with ``|v| < background``, a shell about three voxels thick) is
    the barrier, so the flood cannot leak through the thin negative shell. Inactive
    voxels the band encloses are the deep interior and are flipped to
    ``-background``; every active-band voxel keeps its own value, which already
    carries the correct near-surface sign and distance.

    Using the full active band as the barrier (rather than the negative side only)
    makes the fill robust to single-voxel pinholes in the negative shell, which do
    leak a strictly-positive flood and which a sphere at modest resolution has.

    Parameters
    ----------
    volume:
        Narrow-band signed distance in voxel units (the ``VoxelField.volume``).
    background:
        Narrow-band half-width / clamp value in voxel units.

    Returns
    -------
    numpy.ndarray
        A copy of ``volume`` with the enclosed interior filled to ``-background``.
    """
    from scipy import ndimage

    bg = float(background)
    inactive = volume >= bg - 1e-9   # at the clamp value; graded band is strictly less
    structure = ndimage.generate_binary_structure(volume.ndim, 1)  # face connectivity
    labels, _ = ndimage.label(inactive, structure=structure)

    border_labels = set()
    for axis in range(volume.ndim):
        for face in (0, -1):
            sl = [slice(None)] * volume.ndim
            sl[axis] = face
            border_labels.update(np.unique(labels[tuple(sl)]).tolist())
    border_labels.discard(0)  # 0 labels the active band, never the exterior bulk

    exterior = np.isin(labels, list(border_labels))
    trapped = inactive & ~exterior  # inactive voxels enclosed by the active band

    out = volume.copy()
    out[trapped] = -bg
    return out


def sphere_sdf_volume(center_mm, radius_mm: float, voxel_mm: float = 0.5) -> VoxelField:
    """Build a sphere in PicoGK and return its narrow-band SDF as a VoxelField."""
    lib = _load()
    if lib is None:
        raise RuntimeError("native PicoGK runtime is not available")

    h_lib = lib.b_CreateInstance(ctypes.c_float(voxel_mm))
    center = _Vector3(*[float(c) for c in center_mm])
    h_vox = lib.b_CreateSphere(h_lib, ctypes.byref(center), ctypes.c_float(radius_mm))

    ints = [ctypes.c_int() for _ in range(6)]
    lib.b_GetDims(h_lib, h_vox, *[ctypes.byref(v) for v in ints])
    ox, oy, oz, nx, ny, nz = (v.value for v in ints)

    volume = np.zeros((nz, ny, nx), dtype=np.float32)
    background = ctypes.c_float(0.0)
    for z in range(nz):
        # The Z index is field-local (0..nz-1); the active box absolute origin is
        # oz, applied separately when the block is placed in world space. PicoGK
        # fills the whole slice buffer; stride is x + y*nx (C# ImageGrayScale).
        slice_buf = np.zeros(ny * nx, dtype=np.float32)
        lib.b_GetZSlice(
            h_lib, h_vox, z, slice_buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(background)
        )
        volume[z] = slice_buf.reshape(ny, nx)

    return VoxelField(
        volume=volume,
        origin=(ox, oy, oz),
        voxel_mm=float(voxel_mm),
        background=float(background.value),
    )
