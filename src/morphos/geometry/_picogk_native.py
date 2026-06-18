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
the vendored kernel at ``morphos/vendor/PicoGK/native/<platform>``.
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
    # Vendored kernel: morphos/vendor/PicoGK/native/<platform>, resolved relative
    # to this module (src/morphos/geometry/_picogk_native.py -> repo root at [3]).
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "vendor" / "PicoGK" / "native" / _platform_subdir()


def _library_path() -> Path:
    suffix = ".dll" if sys.platform.startswith("win") else (
        ".dylib" if sys.platform == "darwin" else ".so"
    )
    return _native_dir() / f"{_LIB_BASENAME}{suffix}"


class _Vector3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class _Triangle(ctypes.Structure):
    # vendor/PicoGK/Internals/Types.cs: Triangle { int A, B, C }, Pack = 1.
    _fields_ = [("a", ctypes.c_int), ("b", ctypes.c_int), ("c", ctypes.c_int)]


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
    # Box/cylinder have no native voxel primitive (only sphere and capsule do);
    # the real PicoGK C# API builds them as a Mesh and rasterizes it with
    # Voxels_RenderMesh. These are the same DllImports vendor/PicoGK/Internals/
    # Interop.cs declares for that path (Mesh class ~line 131, Voxels class
    # ~line 300/376), bound directly so morphos can do the same from Python.
    lib.b_MeshCreate = bind("Mesh_hCreate", [ctypes.c_int64], ctypes.c_int64)
    lib.b_MeshAddVertex = bind(
        "Mesh_nAddVertex",
        [ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(_Vector3)],
        ctypes.c_int,
    )
    lib.b_MeshAddTriangle = bind(
        "Mesh_nAddTriangle",
        [ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(_Triangle)],
        ctypes.c_int,
    )
    lib.b_VoxelsCreate = bind("Voxels_hCreate", [ctypes.c_int64], ctypes.c_int64)
    lib.b_RenderMesh = bind(
        "Voxels_RenderMesh", [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64], None
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


def box_mesh(center, size):
    """Axis-aligned box as a closed triangle mesh: 8 vertices, 12 triangles.

    Mirrors the vertex layout and winding of the real PicoGK C# helper
    ``Utils.mshCreateCube`` (vendor/PicoGK/Utils/Utils.cs:233), which is not
    itself exported to the native ABI -- only the ``Mesh_*`` primitive calls
    are, so morphos rebuilds the same cube here in plain Python.

    Parameters
    ----------
    center: (cx, cy, cz) box center, mm.
    size: (sx, sy, sz) full edge lengths (not half-extents), mm.

    Returns
    -------
    (vertices, triangles): ``vertices`` is a list of 8 ``(x, y, z)`` tuples,
    ``triangles`` is a list of 12 ``(a, b, c)`` vertex-index tuples, wound so
    the surface normal points outward.
    """
    cx, cy, cz = (float(c) for c in center)
    sx, sy, sz = (float(s) for s in size)
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    # Same ordering as Utils.mshCreateCube: X is the slow-varying bit, then Y,
    # then Z, i.e. vertex i has X = (i>>2), Y = (i>>1)&1, Z = i&1 (in sign terms).
    vertices = [
        (cx - hx, cy - hy, cz - hz),  # 0
        (cx - hx, cy - hy, cz + hz),  # 1
        (cx - hx, cy + hy, cz - hz),  # 2
        (cx - hx, cy + hy, cz + hz),  # 3
        (cx + hx, cy - hy, cz - hz),  # 4
        (cx + hx, cy - hy, cz + hz),  # 5
        (cx + hx, cy + hy, cz - hz),  # 6
        (cx + hx, cy + hy, cz + hz),  # 7
    ]

    triangles = [
        # -X face
        (0, 1, 3), (0, 3, 2),
        # +X face
        (4, 6, 7), (4, 7, 5),
        # -Y face
        (0, 2, 6), (0, 6, 4),
        # +Y face
        (1, 5, 7), (1, 7, 3),
        # -Z face
        (2, 3, 7), (2, 7, 6),
        # +Z face
        (0, 4, 5), (0, 5, 1),
    ]
    return vertices, triangles


def _orthonormal_basis(axis):
    """Return (u, v, w) unit vectors with w along ``axis``, u/v spanning its
    perpendicular plane. Used to build a prism cap around an arbitrary axis.
    """
    w = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        raise ValueError("cylinder axis must be non-zero")
    w = w / norm
    # pick a helper vector not parallel to w
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, w)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v, w


def cylinder_mesh(center, axis, radius: float, height: float, segments: int = 32):
    """N-sided prism approximating a cylinder, as a closed triangle mesh.

    PicoGK has no native cylinder primitive (only sphere and capsule), so this
    approximates one with a capped regular prism: ``segments`` vertices on the
    bottom rim, ``segments`` on the top rim, plus one center vertex per cap.

    Parameters
    ----------
    center: (cx, cy, cz) mm, the midpoint of the cylinder's axis.
    axis: direction vector of the cylinder's axis (need not be unit length).
    radius: prism circumradius, mm.
    height: full height along the axis, mm.
    segments: number of sides of the prism (>= 3). 32 is a reasonable default.

    Returns
    -------
    (vertices, triangles), same convention as :func:`box_mesh`.
    """
    if segments < 3:
        raise ValueError("cylinder_mesh needs at least 3 segments")

    c = np.asarray(center, dtype=float)
    u, v, w = _orthonormal_basis(axis)
    half_h = float(height) / 2.0
    r = float(radius)

    angles = [2.0 * np.pi * i / segments for i in range(segments)]
    bottom_rim = [c - half_h * w + r * (np.cos(a) * u + np.sin(a) * v) for a in angles]
    top_rim = [c + half_h * w + r * (np.cos(a) * u + np.sin(a) * v) for a in angles]
    bottom_center = c - half_h * w
    top_center = c + half_h * w

    vertices = []
    vertices.extend(bottom_rim)                       # 0 .. segments-1
    vertices.extend(top_rim)                           # segments .. 2*segments-1
    bottom_center_idx = len(vertices)
    vertices.append(bottom_center)
    top_center_idx = len(vertices)
    vertices.append(top_center)

    triangles = []
    for i in range(segments):
        i_next = (i + 1) % segments
        b0, b1 = i, i_next
        t0, t1 = segments + i, segments + i_next
        # side wall, outward-facing: two triangles per quad
        triangles.append((b0, b1, t1))
        triangles.append((b0, t1, t0))
        # bottom cap (normal points -w, i.e. outward/down): fan from center
        triangles.append((bottom_center_idx, b1, b0))
        # top cap (normal points +w, outward/up): fan from center
        triangles.append((top_center_idx, t0, t1))

    vertices = [tuple(float(x) for x in p) for p in vertices]
    return vertices, triangles


def _read_voxel_field(lib, h_lib, h_vox, voxel_mm: float) -> VoxelField:
    """Pull a ``Voxels`` handle's narrow-band SDF into a :class:`VoxelField`.

    Shared by every primitive builder (sphere, mesh-rasterized box/cylinder):
    once a ``Voxels`` handle exists, reading it back is identical regardless
    of how it was built.
    """
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


def sphere_sdf_volume(center_mm, radius_mm: float, voxel_mm: float = 0.5) -> VoxelField:
    """Build a sphere in PicoGK and return its narrow-band SDF as a VoxelField."""
    lib = _load()
    if lib is None:
        raise RuntimeError("native PicoGK runtime is not available")

    h_lib = lib.b_CreateInstance(ctypes.c_float(voxel_mm))
    center = _Vector3(*[float(c) for c in center_mm])
    h_vox = lib.b_CreateSphere(h_lib, ctypes.byref(center), ctypes.c_float(radius_mm))

    return _read_voxel_field(lib, h_lib, h_vox, voxel_mm)


def mesh_sdf_volume(vertices, triangles, voxel_mm: float = 0.5) -> VoxelField:
    """Rasterize a closed triangle mesh in PicoGK and return its narrow-band SDF.

    This is the path the real PicoGK C# API uses for primitives with no native
    voxel constructor (box, cylinder, arbitrary meshes): build a ``Mesh`` by
    adding vertices and triangles one at a time, then ``Voxels_RenderMesh`` it
    into a fresh ``Voxels`` handle. Read-back is identical to the sphere path.

    Parameters
    ----------
    vertices: sequence of (x, y, z) mm tuples.
    triangles: sequence of (a, b, c) vertex-index tuples into ``vertices``.
    voxel_mm: voxel edge length in mm.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("native PicoGK runtime is not available")

    h_lib = lib.b_CreateInstance(ctypes.c_float(voxel_mm))
    h_msh = lib.b_MeshCreate(h_lib)

    for vx, vy, vz in vertices:
        v = _Vector3(float(vx), float(vy), float(vz))
        lib.b_MeshAddVertex(h_lib, h_msh, ctypes.byref(v))

    for a, b, c in triangles:
        t = _Triangle(int(a), int(b), int(c))
        lib.b_MeshAddTriangle(h_lib, h_msh, ctypes.byref(t))

    h_vox = lib.b_VoxelsCreate(h_lib)
    lib.b_RenderMesh(h_lib, h_vox, h_msh)

    return _read_voxel_field(lib, h_lib, h_vox, voxel_mm)
