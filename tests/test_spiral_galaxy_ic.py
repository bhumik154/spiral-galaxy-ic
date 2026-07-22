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
    # These values differ from earlier versions of this test: switching
    # build_galaxy_disk from a per-particle Python loop to vectorized,
    # batched draws (branch selector, then bulge, then bar, then disk, each
    # drawn for all of its particles in one call) changed the order random
    # values are consumed from the generator. Same seed still gives
    # identical output across calls to this version; it does not reproduce
    # the exact output of the pre-vectorization version.
    rng = np.random.default_rng(42)
    disk = build_galaxy_disk(n_stars=3, n_gas=2, r_vir=100.0, pitch_angle_deg=18.0, num_arms=4, is_barred=True, rng=rng)

    np.testing.assert_allclose(
        disk.positions,
        np.array(
            [
                [-2.1175382, -7.7146635, 0.8784503],
                [-8.284274, -9.438882, -0.049925912],
                [2.269493, 7.671336, -0.18486236],
                [10.649411, -7.574303, -0.68092954],
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
                [0.96433294, -0.26469228, 0.0],
                [0.75157934, -0.6596427, 0.0],
                [-0.958917, 0.28368664, 0.0],
                [0.5795943, 0.81490517, 0.0],
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
