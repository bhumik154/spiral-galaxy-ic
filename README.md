# spiral-galaxy-ic

Initial conditions for a barred or unbarred spiral galaxy disk (positions, orbital directions, star/gas type tags), plus the NFW halo enclosed-mass formula, for seeding N-body galaxy simulations.

## The problem

Setting up a galaxy for an N-body simulation means answering two separate questions that get conflated a lot: where do the stars and gas actually sit (a bulge, a bar, spiral arms, an exponential disk), and how does the dark matter halo's mass pull on everything (usually an NFW profile). These are genuinely different pieces of physics. This package keeps them as two small, independent functions instead of one that quietly does both, so you can use either on its own.

`build_galaxy_disk` places particles using a standard bulge/bar/exponential-disk/log-spiral-arm recipe. It does not touch velocity magnitudes: how fast something needs to orbit depends on the true gravitational potential (self-gravity, a companion galaxy, the halo), which this function has no way to know. What it can compute, geometrically, is the direction of circular motion at each particle's position, and that's what it returns.

`nfw_enclosed_mass` is the separate, closed-form Navarro-Frenk-White (1996) enclosed-mass formula, for whenever you do need the halo's contribution to gravity or a rotation curve.

`build_galaxy_disk` is fully vectorized: every particle's structural-component choice is drawn in one batched call, then each component's remaining parameters are drawn for all of its particles in one batched call each (branch selector, then bulge, then bar, then disk, in that fixed order). 60,000 particles takes about 8ms; an earlier per-particle Python loop took about 300ms for the same input. `num_arms=0` (a smooth, armless exponential disk, e.g. an S0 lenticular galaxy) is a real, supported case, not an edge case that happens to work: the disk is azimuthally uniform instead of tracing a spiral with the arm count forced to zero.

## Tested against exact output, not just shape

[`tests/test_spiral_galaxy_ic.py`](tests/test_spiral_galaxy_ic.py) has 16 cases. The core one locks in real numbers, not just array shapes:

```python
def test_exact_output_for_a_fixed_seed():
    rng = np.random.default_rng(42)
    disk = build_galaxy_disk(n_stars=3, n_gas=2, r_vir=100.0, pitch_angle_deg=18.0,
                              num_arms=4, is_barred=True, rng=rng)

    np.testing.assert_allclose(disk.radii, [8.0, 12.55873, 8.0, 13.068283, 2.604359], rtol=1e-5)
    np.testing.assert_array_equal(disk.particle_types, [STAR, STAR, STAR, GAS, GAS])
```

That's computed directly by running the function once and locking the result in, not derived by hand. Other things it checks:

| Scenario | Why it's there |
|---|---|
| Same seed, twice | Must produce byte-identical output |
| Different seeds | Must actually diverge |
| No seed passed at all | Must still work, nondeterministically, like before this function was made seedable |
| Exact output for a fixed seed | Real regression protection, not just "it ran" |
| Particle count matches n_stars + n_gas | Basic shape contract |
| Type ordering (stars first, then gas) | Matches the documented convention exactly |
| Velocity direction is always a unit vector, always z=0 | This is a thin-disk kinematic model; that assumption is enforced, not just described |
| Radii are never negative | Basic physical sanity |
| Spiral-arm particles stay within [r_bar, r_max] | The one radius bound this function actually guarantees by construction (bulge and bar are unclipped Gaussians, not bounded) |
| `num_arms=0` doesn't crash | The naive spiral-arm formula divides by `num_arms` and calls `rng.integers(0, num_arms)`, both of which blow up at 0 |
| `num_arms=0` gives an azimuthally uniform disk | Not just "doesn't crash": confirms it's a real armless disk, not a spiral with the arm count silently forced to 0 |
| Barred vs. unbarred diverge under the same seed | The bar branch is only reachable when `is_barred=True` |
| NFW enclosed mass equals total mass at the virial radius | True for any concentration, by construction of the formula |
| NFW enclosed mass is zero at the center | Basic edge case |
| NFW enclosed mass increases monotonically with radius | Basic physical sanity |
| A known reference value | Computed by calling the function, not hand-derived (a wrong hand calculation was caught and fixed while writing this test) |

## Usage

```python
import numpy as np
from spiral_galaxy_ic import build_galaxy_disk, nfw_enclosed_mass

rng = np.random.default_rng(42)
disk = build_galaxy_disk(
    n_stars=50_000,
    n_gas=10_000,
    r_vir=200.0,
    pitch_angle_deg=18.0,
    num_arms=4,
    is_barred=True,
    rng=rng,
)

disk.positions            # (60000, 3) float32
disk.velocity_directions   # (60000, 3) float32, unit vectors, in-plane
disk.radii                 # (60000,) float32
disk.particle_types        # (60000,) int32, STAR or GAS

halo_mass_within_50kpc = nfw_enclosed_mass(r=50.0, total_mass=1e12, r_vir=200.0, concentration=8.0)
```

Omit `rng` for the previous nondeterministic behavior (a fresh, unseeded draw every call). Pass a seeded `np.random.default_rng(seed)` for reproducible output.

## Install

```bash
pip install spiral-galaxy-ic
```

Depends on NumPy only.

## What this is not

- Not a gravity solver. It doesn't compute forces, integrate orbits, or give you final velocity magnitudes.
- Not a full N-body IC generator like GalIC, DICE, or MAGI. Those solve for genuine dynamical equilibrium (distribution functions, Jeans equations); this gives you a plausible morphological starting point and leaves equilibration to your own simulation's gravity solve.
- Not derived from real galaxy survey data. The bulge/bar/spiral-arm fractions and profile shapes are a standard parametric recipe, not a fit to any specific observed galaxy.

## Where this came from

Extracted from the galaxy-pair collision simulator I've been building ([galaxy-collision](https://github.com/bhumik154/galaxy-collision), private, GPU N-body with a real Poisson solve for self-gravity). This is the part of it that places stars and gas before the simulation starts, made seedable and pulled out on its own because the underlying math has nothing to do with the rest of that project.

## License

MIT
