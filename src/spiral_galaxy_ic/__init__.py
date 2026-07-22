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
    another 20%, only if is_barred), and a logarithmic spiral disk (the
    remainder, exponential radial profile).

    rng: an optional numpy.random.Generator (e.g. np.random.default_rng(seed)).
    Pass one explicitly for reproducible output; omit it for a fresh,
    unseeded generator (nondeterministic output) each call.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_total = n_stars + n_gas
    p = np.zeros((n_total, 3), dtype=np.float32)
    v_dir = np.zeros((n_total, 3), dtype=np.float32)
    r_arr = np.zeros(n_total, dtype=np.float32)
    t = np.zeros(n_total, dtype=np.int32)

    r_max = r_vir * 0.4
    r_bar = r_max * 0.2 if is_barred else 0.0
    pitch = np.radians(pitch_angle_deg)
    b = np.tan(pitch)

    for i in range(n_total):
        rand = rng.random()
        if rand < 0.15:
            r = np.abs(rng.normal(0, r_max * 0.05))
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.normal(0, r_max * 0.05)
        elif is_barred and rand < 0.35:
            x_bar = rng.uniform(-r_bar, r_bar)
            y_bar = rng.normal(0, r_max * 0.02)
            r = np.sqrt(x_bar**2 + y_bar**2)
            theta = np.arctan2(y_bar, x_bar)
            z = rng.normal(0, 0.5)
        else:
            r = rng.exponential(r_max * 0.3)
            r = np.clip(r, r_bar, r_max)
            arm_offset = rng.integers(0, num_arms) * (2 * np.pi / num_arms)
            theta_spiral = np.log(r / (r_bar + 1.0)) / (b + 1e-3)
            theta = theta_spiral + arm_offset + rng.normal(0, 0.2)
            z = rng.normal(0, 1.0)

        p[i] = [r * np.cos(theta), r * np.sin(theta), z]
        r_arr[i] = r
        v_dir[i] = [-np.sin(theta), np.cos(theta), 0.0]
        t[i] = GAS if i >= n_stars else STAR

    return GalaxyDisk(positions=p, velocity_directions=v_dir, radii=r_arr, particle_types=t)
