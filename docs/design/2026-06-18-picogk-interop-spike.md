# PicoGK interop spike

Date: 2026-06-18

## Question

Can morphos drive the production PicoGK (Leap71) geometry kernel from Python, and
read its voxel signed-distance field into the numpy grid the engine speaks?

## Answer: yes, via ctypes to the native runtime (no .NET needed)

PicoGK ships a native library (`native/<platform>/picogk.26.2.{dll,dylib}`) plus
sibling deps (blosc, lz4, tbb12, z, zstd on win-x64). The C# layer is a P/Invoke
wrapper over this library. Python reaches the same C ABI directly with `ctypes`,
so no .NET / pythonnet bridge is required. Proven on Windows x64, picogk 26.2.0.

Working binding: `morphos/geometry/_picogk_native.py`, exercised by
`tests/test_picogk_native.py` (skip-guarded when the runtime is absent).

### ABI used (all `cdecl`, handles are `int64` by value)

| native entry point            | signature (ctypes view)                                   |
|-------------------------------|-----------------------------------------------------------|
| `Library_GetVersion`          | `(char* buf) -> void`                                      |
| `Library_hCreateInstance`     | `(float voxel_mm) -> int64 libHandle`                     |
| `Voxels_hCreateSphere`        | `(int64 lib, Vector3* center, float radius) -> int64`     |
| `Voxels_GetVoxelDimensions`   | `(int64 lib, int64 vox, int* ox,oy,oz, int* nx,ny,nz)`   |
| `Voxels_GetZSlice`            | `(int64 lib, int64 vox, int z, float* buf, float* bg)`   |

`Vector3` is three packed floats; a 12-byte struct is passed by pointer under the
x64 ABI either way. A Z slice fills a `nx*ny` float buffer, stride `x + y*nx`
(matches the C# `ImageGrayScale`), value = narrow-band signed distance in *voxel*
units, clamped to `+/- background` (`background = 1.5` voxels for the sphere).

The Z index passed to `Voxels_GetZSlice` is **field-local** (`0 .. nz-1`), not
an absolute voxel coordinate. The active box origin `oz` from
`Voxels_GetVoxelDimensions` is applied separately when the block is placed in
world space. Passing `oz + z` (the original spike code) silently shifts the
read window by `oz` and truncates the band on one Z end, which reads back as a
lopsided half-sphere. The X and Y axes are unaffected because they are read
within each slice, so the symptom is a clean Z-only offset.

## The gotcha: narrow band, not a filled solid

PicoGK is OpenVDB-style: it stores only a *narrow band* (about +/-1.5 voxels)
around the surface. Voxels deeper than the band are inactive and `GetZSlice`
returns them at the **positive** background value, not a negative interior value.
Consequences measured on a radius-5mm sphere at 0.5mm voxels:

- surface band reads correctly (mean surface radius ~5.0mm), but
- reading raw SD sign as solid/void agrees with the analytic sphere only **74.6%**
  of voxels; the filled interior is undercounted (3374 vs 4139 inside voxels).

So the raw band cannot be used directly as a solid signed-distance Field.

## Resolved: the band is now filled and `PicoGKKernel` is wired

`morphos.geometry._picogk_native.solidify` turns the band into a sign-correct
solid SDF. It floods only the *inactive* voxels (those exactly at
`+background`) from the grid boundary, using the whole active band
(`|v| < background`, a shell about three voxels thick) as the barrier. This is
robust to single-voxel pinholes in the thin negative shell, which DO leak a
naive strictly-positive flood (that variant topped out near 92% sign-correct).
Inactive voxels the band encloses are the deep interior and are flipped to
`-background`; active-band voxels keep their own values, so near-surface
distances stay physical.

`PicoGKKernel.build` (in `morphos/geometry/picogk.py`) builds the native
sphere, solidifies it, and drops the active block onto the kernel grid at its
voxel origin. `tests/test_picogk_kernel.py` verifies the result against an
off-center analytic sphere: the solid mask agrees with the analytic sign on
>99.9% of voxels away from the surface band (vs the raw band's ~75%), and an
axis-swap or origin-offset bug cannot hide behind the sphere's symmetry.
`tests/test_picogk_solidify.py` verifies the fill with no native dependency.

## Box, cylinder, and resampling

PicoGK has no native box or cylinder voxel primitive -- only sphere
(`Voxels_hCreateSphere`) and capsule (`Voxels_hCreateCapsule`). The real
PicoGK C# API builds these via a `Mesh` (vertices + triangles) rasterized
with `Voxels_RenderMesh`; morphos does the same. `box_mesh` and
`cylinder_mesh` (in `_picogk_native.py`) build the vertex/triangle data in
plain Python -- a cube cross-checked against the C# `Utils.mshCreateCube`
layout, and an N-sided capped prism (default 32 segments) for the cylinder.
`mesh_sdf_volume` binds `Mesh_hCreate`/`Mesh_nAddVertex`/`Mesh_nAddTriangle`,
`Voxels_hCreate`, and `Voxels_RenderMesh`, then reads the result back through
the same `Voxels_GetVoxelDimensions`/`Voxels_GetZSlice` path as the sphere
(refactored into a shared `_read_voxel_field` helper), so the same
`solidify()` + grid-placement pipeline applies unchanged. `PicoGKKernel.build`
dispatches `"box"` (center + size) and `"cylinder"` (center + axis + radius +
height [+ segments]) alongside `"sphere"`.

`PicoGKKernel.build` also no longer requires PicoGK's build voxel size to
match the kernel grid's spacing exactly. An optional `picogk_voxel_mm`
argument lets PicoGK build at its own (still isotropic) voxel size;
`resample_volume` (trilinear, `scipy.ndimage.zoom`) and `resample_origin`
rescale PicoGK's solid block -- both the array density and the
signed-distance values themselves, by the same zoom factor -- onto the
kernel's requested grid before placement. Defaults to the kernel's own
spacing (a no-op) so existing callers are unaffected.

Verified in `tests/test_picogk_mesh.py` (mesh geometry, no native
dependency: closed, correctly wound, correct analytic volume),
`tests/test_picogk_resample.py` (resampling math, no native dependency:
shape scaling, analytic-sphere agreement, volume conservation), and
skip-guarded additions to `tests/test_picogk_native.py` and
`tests/test_picogk_kernel.py` for the live-runtime path.

## Note on scope

The .NET path (build `PicoGK.sln`, drive via pythonnet) also works in principle
but is heavier (a build step + CLR boot per process) and buys nothing the native
ctypes path lacks for the geometry read. Prefer the native path.
