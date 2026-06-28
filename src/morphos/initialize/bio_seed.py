"""Bio-inspired density seeding without PDEs.

These are fast, targeted alternatives to reaction-diffusion seeding
(:mod:`morphos.initialize.reaction_diffusion`):

- :func:`murray_network` lays down a Murray's-law branching tube network
  (the topology that biological vascular and respiratory systems converge
  on to minimize flow resistance for a given material budget) and rasterizes
  it into a smooth density field.
- :func:`uniform_noise` is a minimal symmetry-breaking perturbation of the
  classic uniform-gray initial guess.
- :func:`from_image` seeds a design from an external (e.g. biological)
  binary image, such as a micrograph of trabecular bone.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.ndimage import distance_transform_edt

from morphos.field import Field


def _rasterize_segment(tube_mask: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> None:
    """Mark grid points along the line segment p0->p1 as tube centers.

    Walks the segment in small steps (sub-voxel) and sets the nearest voxel
    to True at each step, so the line is rasterized without gaps regardless
    of its length or the grid's dimensionality.
    """
    length = float(np.linalg.norm(p1 - p0))
    n_steps = max(2, int(np.ceil(length * 2)) + 1)
    shape = tube_mask.shape
    for t in np.linspace(0.0, 1.0, n_steps):
        p = p0 + t * (p1 - p0)
        idx = tuple(
            int(np.clip(round(p[d]), 0, shape[d] - 1)) for d in range(len(shape))
        )
        tube_mask[idx] = True


def _rasterize_branches(
    tube_mask: np.ndarray,
    node: np.ndarray,
    radius: float,
    depth: int,
    max_depth: int,
    n_branches: int,
    rng: np.random.Generator,
    ndim: int,
    direction: np.ndarray = None,
) -> None:
    """Recursively grow a Murray's-law branching tree and rasterize it.

    At every node, ``n_branches`` children are spawned at angles spread
    around the incoming direction (120 degrees apart for the canonical
    3-way split), each a distance ``radius`` away. The branch radius (and
    hence, implicitly, flow capacity) shrinks at each generation following
    Murray's law r_child = r_parent / n_branches**(1/3); here we encode that
    via shrinking step length, since the rasterized tube radius is applied
    uniformly afterwards based on distance-from-centerline.
    """
    if depth >= max_depth or radius < 0.5:
        return

    if direction is None:
        # Root: branch in random directions over the full sphere/circle.
        for _ in range(n_branches):
            vec = rng.normal(size=ndim)
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = np.eye(ndim)[0]
            else:
                vec = vec / norm
            child = node + vec * radius
            _rasterize_segment(tube_mask, node, child)
            _rasterize_branches(
                tube_mask,
                child,
                radius / (n_branches ** (1.0 / 3.0)),
                depth + 1,
                max_depth,
                n_branches,
                rng,
                ndim,
                direction=vec,
            )
        return

    # Non-root: spread children around the incoming direction.
    base_angle = np.arctan2(direction[1], direction[0]) if ndim >= 2 else 0.0
    for b in range(n_branches):
        angle_offset = (2.0 * np.pi / n_branches) * b + rng.uniform(-0.2, 0.2)
        angle = base_angle + angle_offset
        if ndim == 2:
            vec = np.array([np.cos(angle), np.sin(angle)])
        else:
            # 3D: rotate within a plane perturbed by a random tilt so
            # branches spread in 3-space rather than collapsing to 2D.
            tilt = rng.uniform(-0.5, 0.5)
            vec = np.array(
                [np.cos(angle), np.sin(angle), tilt] + [0.0] * (ndim - 3)
            )
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        child = node + vec * radius
        _rasterize_segment(tube_mask, node, child)
        _rasterize_branches(
            tube_mask,
            child,
            radius / (n_branches ** (1.0 / 3.0)),
            depth + 1,
            max_depth,
            n_branches,
            rng,
            ndim,
            direction=vec,
        )


def murray_network(
    shape,
    volume_fraction: float,
    spacing=1.0,
    n_branches: int = 3,
    seed: int = 42,
) -> Field:
    """Generate a Murray's-law branching network density field.

    Builds a recursive branching tree rooted at the domain center,
    rasterizes the tube centerlines, then converts the distance-to-nearest-
    tube field into a smooth density via a sigmoid, and rescales to hit
    ``volume_fraction``.
    """
    shape = tuple(int(s) for s in shape)
    rng = np.random.default_rng(seed)
    ndim = len(shape)

    center = np.array([s / 2.0 for s in shape], dtype=float)
    tube_mask = np.zeros(shape, dtype=bool)

    root_radius = max(shape) * 0.4
    _rasterize_branches(
        tube_mask,
        center,
        radius=root_radius,
        depth=0,
        max_depth=3,
        n_branches=max(2, int(n_branches)),
        rng=rng,
        ndim=ndim,
    )

    # Ensure at least the root voxel is marked, in case of pathological
    # tiny grids where every branch step rounds to the same voxel.
    root_idx = tuple(int(np.clip(round(c), 0, shape[d] - 1)) for d, c in enumerate(center))
    tube_mask[root_idx] = True

    dist = distance_transform_edt(~tube_mask)
    tube_radius_voxels = max(1.0, min(shape) * 0.05)

    rho = 1.0 / (1.0 + np.exp(5.0 * (dist - tube_radius_voxels)))

    rho = rho / (rho.mean() + 1e-10) * volume_fraction
    rho = np.clip(rho, 1e-3, 1.0)

    return Field(rho, spacing=spacing)


def uniform_noise(
    shape,
    volume_fraction: float,
    noise_scale: float = 0.1,
    spacing=1.0,
    seed: int = 42,
) -> Field:
    """Uniform density perturbed by small Gaussian noise to break symmetry."""
    shape = tuple(int(s) for s in shape)
    rng = np.random.default_rng(seed)
    rho = volume_fraction + noise_scale * rng.standard_normal(shape)
    rho = np.clip(rho, 1e-3, 1.0)
    return Field(rho, spacing=spacing)


def from_image(
    path_or_array: Union[str, "np.ndarray"],
    shape,
    volume_fraction: float,
    spacing=1.0,
) -> Field:
    """Seed a density field from a binary (or grayscale) image.

    ``path_or_array`` may be a path to an image file (requires PIL or
    scikit-image, imported lazily here so the dependency stays optional) or
    a numpy array already in memory. The image is resized to ``shape`` with
    :func:`scipy.ndimage.zoom`, thresholded at 0.5, lightly smoothed, and
    rescaled to hit ``volume_fraction``.
    """
    shape = tuple(int(s) for s in shape)

    if isinstance(path_or_array, str):
        array = None
        try:
            from PIL import Image

            array = np.asarray(Image.open(path_or_array).convert("L"), dtype=float)
        except ImportError:
            try:
                from skimage.io import imread
                from skimage.color import rgb2gray

                img = imread(path_or_array)
                array = np.asarray(img, dtype=float)
                if array.ndim == 3:
                    array = rgb2gray(img)
            except ImportError as exc:
                raise ImportError(
                    "from_image requires PIL (pip install pillow) or "
                    "scikit-image (pip install scikit-image) to load an "
                    "image from a path."
                ) from exc
    else:
        array = np.asarray(path_or_array, dtype=float)

    if array.max() > 1.0:
        array = array / array.max()

    from scipy.ndimage import zoom, gaussian_filter

    zoom_factors = tuple(t / s for t, s in zip(shape, array.shape))
    resized = zoom(array, zoom_factors, order=1)

    # Defensive: zoom can occasionally land one voxel off due to rounding.
    if resized.shape != shape:
        resized = resized[tuple(slice(0, s) for s in shape)]
        if resized.shape != shape:
            padded = np.zeros(shape, dtype=float)
            slices = tuple(slice(0, min(s, r)) for s, r in zip(shape, resized.shape))
            padded[slices] = resized[slices]
            resized = padded

    binary = (resized > 0.5).astype(float)
    smoothed = gaussian_filter(binary, sigma=1.0)

    rho = smoothed / (smoothed.mean() + 1e-10) * volume_fraction
    rho = np.clip(rho, 1e-3, 1.0)

    return Field(rho, spacing=spacing)
