"""Initial conditions for a barred or unbarred spiral galaxy disk, plus the
NFW halo enclosed-mass formula. Built for seeding N-body galaxy simulations.
"""

from dataclasses import dataclass

import numpy as np

STAR = 0
GAS = 1

__all__ = ["STAR", "GAS", "GalaxyDisk", "build_galaxy_disk", "nfw_enclosed_mass"]


def nfw_enclosed_mass(r, total_mass, r_vir, concentration):
    """Mass enclosed within radius r for a Navarro-Frenk-White (1996) halo
    profile, normalized so that nfw_enclosed_mass(r_vir, total_mass, r_vir,
    concentration) == total_mass exactly, for any concentration.

    This describes a dark matter halo's mass distribution. It has nothing to
    do with build_galaxy_disk's particle positions below, which follow a
    standard bulge/bar/exponential-disk morphology, not an NFW profile. The
    two are separate, complementary pieces of a galaxy IC pipeline: this
    function is what you'd use to compute the halo's contribution to
    gravitational acceleration or a rotation curve; build_galaxy_disk is
    what you'd use to place the visible stars and gas.

    r_s = r_vir / concentration is the NFW scale radius. The closed-form
    enclosed-mass integral of the NFW density profile is:
        m(r) = total_mass * f(r / r_s) / f(concentration)
        f(x) = ln(1 + x) - x / (1 + x)
    """
    r_s = r_vir / concentration
    x = r / r_s
    f_x = np.log(1.0 + x) - x / (1.0 + x)
    f_c = np.log(1.0 + concentration) - concentration / (1.0 + concentration)
    return total_mass * (f_x / f_c)


@dataclass
class GalaxyDisk:
    positions: np.ndarray
    """(n, 3) float32 cartesian positions, centered on the origin."""
    velocity_directions: np.ndarray
    """(n, 3) float32 unit vectors: the direction of circular orbital
    motion at each particle's position. Confined to the z=0 plane (the
    z-component is always 0) regardless of a particle's own height above
    or below the disk plane: this is a thin-disk kinematic model, not a
    full 3D velocity-dispersion model."""
    radii: np.ndarray
    """(n,) float32 cylindrical radius from the center for each particle."""
    particle_types: np.ndarray
    """(n,) int32, STAR for the first n_stars entries, GAS for the rest."""


def build_galaxy_disk(n_stars, n_gas, r_vir, pitch_angle_deg, num_arms, is_barred, rng=None):
    """Builds a galaxy disk's particle positions, in-plane orbital-direction
    unit vectors, cylindrical radii, and star/gas type tags.

    Does not compute velocity magnitudes: circular-orbit speed depends on
    the true gravitational potential (self-gravity, any companion galaxy,
    a halo mass profile like nfw_enclosed_mass above), which this function
    has no way to know. Only the direction of circular motion at each
    particle's position is purely geometric, and that's what this computes.

    Particles are drawn from three structural components, matching a
    standard spiral galaxy morphology (not derived from an NFW profile):
    a central bulge (15% of particles, Gaussian), an optional bar (up to
    another 20%, only if is_barred), and a disk (the remainder, exponential
    radial profile). With num_arms > 0 the disk traces a log-spiral pattern
    banded into that many discrete arms; with num_arms == 0 the disk is
    azimuthally uniform instead (a smooth, armless exponential disk, e.g.
    an S0 lenticular galaxy), rather than raising an error.

    Fully vectorized: draws every particle's structural-component choice in
    one batched call, then draws each component's remaining parameters for
    all of its particles in one batched call each, in this fixed order:
    branch selector, then bulge, then bar, then disk. This is roughly two
    orders of magnitude faster than a per-particle Python loop at realistic
    particle counts (tens of thousands), but it means the exact sequence of
    values drawn from a given seed is not the same as drawing one particle
    at a time; only the same generator, given the same seed, is guaranteed
    to reproduce the same output across calls to this version of the
    function.

    rng: an optional numpy.random.Generator (e.g. np.random.default_rng(seed)).
    Pass one explicitly for reproducible output; omit it for a fresh,
    unseeded generator (nondeterministic output) each call.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_total = n_stars + n_gas
    r = np.empty(n_total, dtype=np.float64)
    theta = np.empty(n_total, dtype=np.float64)
    z = np.empty(n_total, dtype=np.float64)

    r_max = r_vir * 0.4
    r_bar = r_max * 0.2 if is_barred else 0.0
    pitch = np.radians(pitch_angle_deg)
    b = np.tan(pitch)

    branch = rng.random(n_total)
    bulge_mask = branch < 0.15
    bar_mask = is_barred & (branch >= 0.15) & (branch < 0.35)
    disk_mask = ~bulge_mask & ~bar_mask

    n_bulge = int(bulge_mask.sum())
    if n_bulge > 0:
        r[bulge_mask] = np.abs(rng.normal(0, r_max * 0.05, size=n_bulge))
        theta[bulge_mask] = rng.uniform(0, 2 * np.pi, size=n_bulge)
        z[bulge_mask] = rng.normal(0, r_max * 0.05, size=n_bulge)

    n_bar = int(bar_mask.sum())
    if n_bar > 0:
        x_bar = rng.uniform(-r_bar, r_bar, size=n_bar)
        y_bar = rng.normal(0, r_max * 0.02, size=n_bar)
        r[bar_mask] = np.sqrt(x_bar**2 + y_bar**2)
        theta[bar_mask] = np.arctan2(y_bar, x_bar)
        z[bar_mask] = rng.normal(0, 0.5, size=n_bar)

    n_disk = int(disk_mask.sum())
    if n_disk > 0:
        r_disk = rng.exponential(r_max * 0.3, size=n_disk)
        r_disk = np.clip(r_disk, r_bar, r_max)
        r[disk_mask] = r_disk
        if num_arms > 0:
            arm_offset = rng.integers(0, num_arms, size=n_disk) * (2 * np.pi / num_arms)
            theta_spiral = np.log(r_disk / (r_bar + 1.0)) / (b + 1e-3)
            theta[disk_mask] = theta_spiral + arm_offset + rng.normal(0, 0.2, size=n_disk)
        else:
            theta[disk_mask] = rng.uniform(0, 2 * np.pi, size=n_disk)
        z[disk_mask] = rng.normal(0, 1.0, size=n_disk)

    p = np.empty((n_total, 3), dtype=np.float32)
    p[:, 0] = r * np.cos(theta)
    p[:, 1] = r * np.sin(theta)
    p[:, 2] = z

    v_dir = np.empty((n_total, 3), dtype=np.float32)
    v_dir[:, 0] = -np.sin(theta)
    v_dir[:, 1] = np.cos(theta)
    v_dir[:, 2] = 0.0

    t = np.where(np.arange(n_total) >= n_stars, GAS, STAR).astype(np.int32)

    return GalaxyDisk(
        positions=p,
        velocity_directions=v_dir,
        radii=r.astype(np.float32),
        particle_types=t,
    )
