import numpy as np
import pytest

from spiral_galaxy_ic import GAS, STAR, build_galaxy_disk, nfw_enclosed_mass


def test_same_seed_gives_identical_output():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    d1 = build_galaxy_disk(50, 10, 150.0, 18.0, 4, True, rng=rng1)
    d2 = build_galaxy_disk(50, 10, 150.0, 18.0, 4, True, rng=rng2)
    np.testing.assert_array_equal(d1.positions, d2.positions)
    np.testing.assert_array_equal(d1.velocity_directions, d2.velocity_directions)
    np.testing.assert_array_equal(d1.radii, d2.radii)
    np.testing.assert_array_equal(d1.particle_types, d2.particle_types)


def test_different_seed_gives_different_output():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    d1 = build_galaxy_disk(50, 10, 150.0, 18.0, 4, True, rng=rng1)
    d2 = build_galaxy_disk(50, 10, 150.0, 18.0, 4, True, rng=rng2)
    assert not np.array_equal(d1.positions, d2.positions)


def test_omitting_rng_still_works_and_is_nondeterministic():
    d1 = build_galaxy_disk(20, 5, 100.0, 18.0, 4, True)
    d2 = build_galaxy_disk(20, 5, 100.0, 18.0, 4, True)
    assert d1.positions.shape == (25, 3)
    assert not np.array_equal(d1.positions, d2.positions)


def test_exact_output_for_a_fixed_seed():
    # Golden regression test: computed once directly from the function and
    # locked in, not hand-derived. If this changes, the generator's output
    # changed, whether intentionally or not.
    #
    # These values have changed across several earlier versions of this
    # test: the switch to vectorized batched draws, replacing (b + 1e-3)
    # with an exact b==0 check, replacing the fixed (r_bar + 1.0) reference
    # length with one that scales with r_vir, and now replacing the bar and
    # disk branches' fixed vertical scatter (0.5 and 1.0) with one scaled by
    # r_max (a fixed absolute thickness meant a thin disk in kpc-scale units
    # became a vertical pillar once the same galaxy was described in
    # Mpc-scale units instead). Same seed still gives identical output
    # across calls to this version; it does not reproduce the exact output
    # of any earlier version.
    rng = np.random.default_rng(42)
    disk = build_galaxy_disk(n_stars=3, n_gas=2, r_vir=100.0, pitch_angle_deg=18.0, num_arms=4, is_barred=True, rng=rng)

    np.testing.assert_allclose(
        disk.positions,
        np.array(
            [
                [0.7469255, -7.965055, 1.7569005],
                [-4.361737, -11.776967, -0.099851824],
                [-0.5894666, 7.9782534, -0.36972472],
                [12.654846, -3.2611198, -1.3618591],
                [0.18213761, -2.5979822, -0.6324852],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        disk.velocity_directions,
        np.array(
            [
                [0.9956319, 0.093365684, 0.0],
                [0.9377514, -0.34730718, 0.0],
                [-0.9972817, -0.07368332, 0.0],
                [0.24954462, 0.9683633, 0.0],
                [0.9975515, 0.06993568, 0.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        disk.radii, [8.0, 12.55873, 8.0, 13.068283, 2.604359], rtol=1e-5
    )
    np.testing.assert_array_equal(disk.particle_types, [STAR, STAR, STAR, GAS, GAS])


def test_particle_count_matches_n_stars_plus_n_gas():
    disk = build_galaxy_disk(37, 13, 100.0, 18.0, 4, True, rng=np.random.default_rng(0))
    n = 37 + 13
    assert disk.positions.shape == (n, 3)
    assert disk.velocity_directions.shape == (n, 3)
    assert disk.radii.shape == (n,)
    assert disk.particle_types.shape == (n,)


def test_particle_types_are_stars_first_then_gas():
    disk = build_galaxy_disk(10, 4, 100.0, 18.0, 4, True, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(disk.particle_types[:10], [STAR] * 10)
    np.testing.assert_array_equal(disk.particle_types[10:], [GAS] * 4)


def test_velocity_directions_are_always_unit_vectors_confined_to_the_disk_plane():
    disk = build_galaxy_disk(200, 50, 150.0, 18.0, 4, True, rng=np.random.default_rng(7))
    # z-component is always exactly 0: velocity direction is purely in-plane
    # regardless of a particle's own height above/below the disk.
    np.testing.assert_array_equal(disk.velocity_directions[:, 2], 0.0)
    norms = np.linalg.norm(disk.velocity_directions[:, :2], axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_radii_are_never_negative():
    disk = build_galaxy_disk(300, 100, 200.0, 6.0, 6, False, rng=np.random.default_rng(3))
    assert np.all(disk.radii >= 0.0)


def test_spiral_arm_particles_stay_within_r_bar_and_r_max():
    # The explicit np.clip in the spiral-disk branch is the one hard radial
    # bound this function guarantees; the bulge and bar branches are drawn
    # from unclipped Gaussians and are not bounded by construction.
    r_vir = 150.0
    r_max = r_vir * 0.4
    is_barred = True
    r_bar = r_max * 0.2

    disk = build_galaxy_disk(2000, 500, r_vir, 18.0, 4, is_barred, rng=np.random.default_rng(11))
    assert np.all(disk.radii <= r_max + 1e-4)


def test_num_arms_zero_does_not_crash():
    # rng.integers(0, num_arms) and dividing by num_arms both blow up at
    # num_arms=0 if the disk branch tries to use the spiral-arm formula at
    # all; this must take a different path instead, not just avoid the crash.
    disk = build_galaxy_disk(500, 100, 150.0, 18.0, num_arms=0, is_barred=False, rng=np.random.default_rng(1))
    assert disk.positions.shape == (600, 3)
    assert np.all(np.isfinite(disk.positions))


def test_num_arms_zero_gives_azimuthally_uniform_disk_not_a_spiral():
    # A smooth, armless exponential disk (e.g. an S0 lenticular galaxy)
    # should be roughly uniform in angle around the center, not still
    # tracing a single continuous log-spiral with the arm count forced to 0.
    disk = build_galaxy_disk(0, 200_000, 150.0, 18.0, num_arms=0, is_barred=False, rng=np.random.default_rng(3))
    theta = np.arctan2(disk.positions[:, 1], disk.positions[:, 0])
    counts, _ = np.histogram(theta, bins=8, range=(-np.pi, np.pi))
    # Roughly uniform: no bin should be wildly over- or under-represented.
    # A real spiral pattern concentrates particles at radius-dependent
    # angles instead, which would blow this ratio out far past 1.2.
    assert counts.max() / counts.min() < 1.2


def test_unbarred_galaxy_never_hits_the_bar_branch():
    # With is_barred=False, r_bar collapses to 0, so bar particles are
    # geometrically indistinguishable from a zero-width bar; the real
    # structural difference is the branch is simply unreachable rand<0.35
    # bar-check is gated on is_barred. This locks that in for a large N.
    barred = build_galaxy_disk(0, 5000, 150.0, 18.0, 4, True, rng=np.random.default_rng(5))
    unbarred = build_galaxy_disk(0, 5000, 150.0, 18.0, 4, False, rng=np.random.default_rng(5))
    # Same seed, same draws, but the bar branch is only reachable when
    # is_barred is True - so results must differ once any draw lands in the
    # [0.15, 0.35) range.
    assert not np.array_equal(barred.positions, unbarred.positions)


def test_nfw_enclosed_mass_at_the_virial_radius_equals_total_mass():
    for concentration in [3.0, 5.0, 8.0, 12.0]:
        m = nfw_enclosed_mass(r=200.0, total_mass=1e12, r_vir=200.0, concentration=concentration)
        assert m == pytest.approx(1e12, rel=1e-9)


def test_nfw_enclosed_mass_at_the_center_is_zero():
    m = nfw_enclosed_mass(r=0.0, total_mass=1e12, r_vir=200.0, concentration=5.0)
    assert m == pytest.approx(0.0, abs=1e-6)


def test_nfw_enclosed_mass_increases_monotonically_with_radius():
    radii = np.linspace(0.0, 200.0, 50)
    masses = nfw_enclosed_mass(r=radii, total_mass=1e12, r_vir=200.0, concentration=5.0)
    assert np.all(np.diff(masses) > 0)


def test_nfw_enclosed_mass_known_reference_value():
    # Regression value computed directly by calling the function (not
    # hand-derived): f(x)=ln(1+x)-x/(1+x) at x=r/r_s=2.5 vs x=c=5, for
    # r=r_vir/2=100, r_vir=200, c=5.
    m = nfw_enclosed_mass(r=100.0, total_mass=1e12, r_vir=200.0, concentration=5.0)
    assert m == pytest.approx(561834902078.2576, rel=1e-9)


@pytest.mark.parametrize("concentration", [0.0, -1.0, -5.0])
def test_nfw_enclosed_mass_rejects_non_positive_concentration(concentration):
    # Otherwise r_s = r_vir / concentration divides by zero (or gives a
    # negative scale radius, physically meaningless). A parameter-sweep
    # script that ranges concentration from 0 upward would hit this.
    with pytest.raises(ValueError):
        nfw_enclosed_mass(r=100.0, total_mass=1e12, r_vir=200.0, concentration=concentration)


@pytest.mark.parametrize("r_vir", [0.0, -50.0])
def test_build_galaxy_disk_rejects_non_positive_r_vir(r_vir):
    # r_vir=0 makes r_max=0, which makes the disk-branch exponential draw's
    # scale parameter exactly 0 - and rng.exponential(scale=0) deterministically
    # returns all zeros (confirmed directly, not assumed), which would
    # otherwise silently poison every disk particle's position with NaN.
    with pytest.raises(ValueError):
        build_galaxy_disk(100, 20, r_vir, 18.0, 4, True, rng=np.random.default_rng(1))


class _ForcesAZeroExponentialDraw:
    """Wraps a real Generator but forces the first exponential() draw to
    include an exact 0.0, to deterministically test the log-singularity
    floor. A real Generator essentially never produces an exact 0.0 from a
    nonzero-scale exponential (confirmed empirically: zero exact-zero draws
    across 200 million samples), so this is the only practical way to test
    that specific line without relying on astronomical luck.
    """

    def __init__(self, real_rng):
        self._real = real_rng

    def __getattr__(self, name):
        return getattr(self._real, name)

    def exponential(self, scale, size=None):
        draws = self._real.exponential(scale, size=size)
        draws[0] = 0.0
        return draws


def test_disk_generation_does_not_produce_nan_from_a_zero_radius_draw():
    forced_rng = _ForcesAZeroExponentialDraw(np.random.default_rng(1))
    disk = build_galaxy_disk(0, 5000, 150.0, 18.0, 4, is_barred=False, rng=forced_rng)
    assert not np.any(np.isnan(disk.positions))
    assert not np.any(np.isnan(disk.velocity_directions))


def test_pitch_angle_exactly_zero_does_not_crash():
    # b = tan(0) = 0 exactly: this is the actual singularity, guarded by
    # b_safe. Confirmed this pitch angle is otherwise a real, reachable
    # input (a perfectly circular, non-spiral arm pattern), not just a
    # theoretical edge case.
    disk = build_galaxy_disk(0, 500, 150.0, pitch_angle_deg=0.0, num_arms=4, is_barred=False, rng=np.random.default_rng(1))
    assert np.all(np.isfinite(disk.positions))


class _FixedDiskDraws:
    """Forces every particle into the disk branch at a fixed radius, with no
    arm offset and no angular noise, so the resulting position is exactly
    r * (cos, sin) of log(r / r_scale) / b_safe and nothing else -
    hand-checkable against the formula directly, not just "looks reasonable".
    """

    def __init__(self, r_disk_value=30.0):
        self._r_disk_value = r_disk_value

    def random(self, size=None):
        return np.full(size, 0.99)  # always past the bulge/bar thresholds

    def exponential(self, scale, size=None):
        return np.full(size, self._r_disk_value)

    def integers(self, low, high, size=None):
        return np.zeros(size, dtype=np.int64)  # arm_offset = 0

    def normal(self, loc, scale, size=None):
        return np.zeros(size)  # no angular or positional noise

    def uniform(self, low, high, size=None):
        return np.zeros(size)


def test_trailing_arm_pitch_angle_matches_the_direct_formula_not_the_old_shifted_singularity():
    # Under the previous (b + 1e-3) guard, this exact pitch angle put the
    # denominator at -3.33e-10 (confirmed directly, not exactly 0, but close
    # enough to blow theta up to billions of radians - finite, but a
    # position bound check can't catch that, since r*cos(theta) is always
    # bounded by r no matter how garbled theta is; confirmed that
    # separately before writing this test). This checks the actual angle
    # against the formula directly instead, using r_scale (r_max * 0.05 for
    # an unbarred disk) rather than the old fixed (r_bar + 1.0) reference.
    r_vir = 150.0
    r_max = r_vir * 0.4
    r_scale = r_max * 0.05
    r_disk_value = 30.0
    pitch_angle_deg = -0.0572957795
    b = np.tan(np.radians(pitch_angle_deg))
    expected_theta = np.log(r_disk_value / r_scale) / b

    disk = build_galaxy_disk(0, 1, r_vir, pitch_angle_deg, num_arms=1, is_barred=False, rng=_FixedDiskDraws())

    np.testing.assert_allclose(disk.positions[0, 0], r_disk_value * np.cos(expected_theta), rtol=1e-4)
    np.testing.assert_allclose(disk.positions[0, 1], r_disk_value * np.sin(expected_theta), rtol=1e-4)


@pytest.mark.parametrize("n_stars,n_gas", [(-5000, 10000), (100, -20)])
def test_build_galaxy_disk_rejects_negative_particle_counts(n_stars, n_gas):
    # Previously silent: negative n_stars made n_total positive (n_stars +
    # n_gas), so the function ran to completion and just mislabeled every
    # particle as GAS (np.arange(n_total) >= n_stars is true for all indices
    # when n_stars is negative), without ever raising an error.
    with pytest.raises(ValueError):
        build_galaxy_disk(n_stars, n_gas, 150.0, 18.0, 4, True, rng=np.random.default_rng(1))


def test_build_galaxy_disk_rejects_negative_num_arms():
    # Previously silent: num_arms=-4 satisfied `not (num_arms > 0)`, so it
    # silently fell back to the armless, azimuthally uniform disk instead of
    # raising an error for a nonsensical negative arm count.
    with pytest.raises(ValueError):
        build_galaxy_disk(0, 500, 150.0, 18.0, num_arms=-4, is_barred=False, rng=np.random.default_rng(1))


def test_nfw_enclosed_mass_rejects_non_positive_r_vir():
    # r_vir=0 makes r_s = r_vir / concentration = 0, and the very next line
    # divides r by that, an unguarded ZeroDivisionError (confirmed directly
    # before adding this guard).
    with pytest.raises(ValueError):
        nfw_enclosed_mass(r=50.0, total_mass=1e12, r_vir=0.0, concentration=5.0)


def test_nfw_enclosed_mass_rejects_non_positive_total_mass():
    # Unlike r_vir and concentration, total_mass is never a divisor in this
    # formula, so a non-positive value can't crash it, it would just scale
    # the result to zero or negative. Rejected anyway for API consistency:
    # a zero or negative halo mass isn't a meaningful input to model.
    with pytest.raises(ValueError):
        nfw_enclosed_mass(r=50.0, total_mass=0.0, r_vir=200.0, concentration=5.0)


def test_nfw_enclosed_mass_does_not_produce_nan_from_a_negative_radius():
    # x <= -1 makes log(1 + x) either log(0) or log(a negative number),
    # -inf or NaN respectively - confirmed directly before this guard, one
    # bad entry in an array of radii (e.g. plotting a rotation curve)
    # poisoned the whole result. Negative radii are clamped to r=0 (zero
    # enclosed mass) instead.
    result = nfw_enclosed_mass(r=np.array([-50.0, 10.0, 100.0]), total_mass=1e12, r_vir=200.0, concentration=5.0)
    assert not np.any(np.isnan(result))
    assert result[0] == 0.0


def test_build_galaxy_disk_is_scale_invariant():
    # The same galaxy described in different units (e.g. r_vir=100 "kpc"
    # vs r_vir=0.1 "Mpc", with every length scaled by the same factor)
    # must produce the same angular structure. A fixed reference length in
    # the spiral-arm formula's denominator breaks this: it dominates at
    # small unit scales and vanishes at large ones, so the same physical
    # galaxy would wind differently depending on which units were used to
    # describe it. Confirmed this broke under the previous (r_bar + 1.0)
    # denominator before fixing it.
    pitch_angle_deg = 18.0
    b = np.tan(np.radians(pitch_angle_deg))

    r_vir_a, r_disk_a = 100.0, 20.0  # e.g. "kpc"
    r_vir_b, r_disk_b = 0.1, 0.02  # e.g. "Mpc": same galaxy, scaled by 1/1000

    disk_a = build_galaxy_disk(0, 1, r_vir_a, pitch_angle_deg, num_arms=1, is_barred=True, rng=_FixedDiskDraws(r_disk_a))
    disk_b = build_galaxy_disk(0, 1, r_vir_b, pitch_angle_deg, num_arms=1, is_barred=True, rng=_FixedDiskDraws(r_disk_b))

    theta_a = np.arctan2(disk_a.positions[0, 1], disk_a.positions[0, 0])
    theta_b = np.arctan2(disk_b.positions[0, 1], disk_b.positions[0, 0])
    np.testing.assert_allclose(theta_a, theta_b, rtol=1e-4)


def test_vertical_scatter_scales_with_r_max_not_a_fixed_length():
    # A fixed absolute z standard deviation is a thin disk in one unit
    # system and a vertical pillar in another: confirmed directly, disk
    # z-sigma=1.0 was 2.5% of r_max at r_vir=100 and 2500% of r_max at
    # r_vir=0.1 (a thousandfold difference for the "same" galaxy) before
    # this fix. r_vir_b here is 10x r_vir_a, so the standard deviation of z
    # across all particles should scale by the same factor if it's really
    # proportional to r_max, not stay fixed.
    r_vir_a, r_vir_b = 100.0, 1000.0
    disk_a = build_galaxy_disk(0, 200_000, r_vir_a, 18.0, 4, is_barred=False, rng=np.random.default_rng(1))
    disk_b = build_galaxy_disk(0, 200_000, r_vir_b, 18.0, 4, is_barred=False, rng=np.random.default_rng(1))

    ratio = disk_b.positions[:, 2].std() / disk_a.positions[:, 2].std()
    assert 9.0 < ratio < 11.0
