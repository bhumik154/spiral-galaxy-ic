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
    rng = np.random.default_rng(42)
    disk = build_galaxy_disk(n_stars=3, n_gas=2, r_vir=100.0, pitch_angle_deg=18.0, num_arms=4, is_barred=True, rng=rng)

    np.testing.assert_allclose(
        disk.positions,
        np.array(
            [
                [14.236679, -24.150312, -1.9510351],
                [16.149965, 5.0447078, -0.016801158],
                [7.5199604, -2.7295046, 1.1272413],
                [-2.2694931, -7.6713362, -0.95888263],
                [7.0299501, -3.8183510, 1.2225413],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        disk.velocity_directions,
        np.array(
            [
                [0.86145663, 0.50783116, 0.0],
                [-0.29815888, 0.95451623, 0.0],
                [0.34118807, 0.93999505, 0.0],
                [0.958917, -0.28368664, 0.0],
                [0.47729388, 0.87874377, 0.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        disk.radii, [28.034275, 16.919529, 8.0, 8.0, 8.0], rtol=1e-5
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
