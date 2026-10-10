"""Einheiten des EKF/UKF-Kerns gegen Handrechnung auf der Mini-Instanz und Randfälle."""

import numpy as np
import pytest

import ekf_algorithm as A
import ekf_constants as C
import ekf_scenario as S


def test_wrap_maps_angles_into_minus_pi_pi():
    assert A.wrap(2 * np.pi + 0.1) == pytest.approx(0.1)
    assert A.wrap(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    assert A.wrap(0.0) == 0.0


def test_h_polar_and_inverse_on_hand_values(mini_state):
    x, _ = mini_state
    z = A.h_polar(x)
    assert z == pytest.approx([5.0, np.arctan2(4.0, 3.0)])
    assert A.polar_to_position(z) == pytest.approx([3.0, 4.0])


def test_jacobian_hand_values(mini_state):
    x, _ = mini_state
    H = A.jacobian_polar(x)
    assert H[0] == pytest.approx([0.6, 0.8, 0.0, 0.0])
    assert H[1] == pytest.approx([-4 / 25, 3 / 25, 0.0, 0.0])


def test_cv_model_2d_has_no_coupling_between_axes():
    F, Q = A.cv_model_2d(1.0, 1.0)
    assert Q[0, 0] == pytest.approx(1 / 3) and Q[0, 2] == pytest.approx(1 / 2) and Q[2, 2] == pytest.approx(1.0)
    assert Q[0, 1] == 0.0 and Q[0, 3] == 0.0 and Q[1, 2] == 0.0
    assert F[0, 2] == 1.0 and F[1, 3] == 1.0


def test_ekf_update_hand_example(mini_state):
    x, P = mini_state
    R = np.diag([1.0, 0.04])
    x1, P1 = A.ekf_update(x, P, A.h_polar(x), R)
    assert x1 == pytest.approx(x)                       # Messung gleich Vorhersage: kein Sprung
    assert P1[0, 0] == pytest.approx(0.5) and P1[1, 1] == pytest.approx(0.5) and P1[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert P1[2, 2] == pytest.approx(1.0) and P1[3, 3] == pytest.approx(1.0)


def test_bearing_innovation_is_wrapped_at_plus_minus_pi():
    x = np.array([-50.0, 0.5, 0.0, 0.0])               # Peilung knapp unter +pi
    z = np.array([np.hypot(50.0, 0.5), -np.pi + 0.01])  # Messung knapp über -pi: Unterschied 0,03 rad, nicht 2 pi
    for update in (A.ekf_update, A.ukf_update):
        x1, _ = update(x, np.diag([4.0, 4.0, 1.0, 1.0]), z, np.diag([1.0, 1e-4]))
        assert np.linalg.norm(x1[:2] - x[:2]) < 5.0


def test_sigma_points_reproduce_mean_and_covariance():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 4))
    P = M @ M.T + np.eye(4)
    x = np.array([1.0, -2.0, 0.5, 3.0])
    pts, wm, wc = A.sigma_points(x, P)
    assert len(pts) == 9
    assert wm @ pts == pytest.approx(x)
    assert wm.sum() == pytest.approx(1.0)
    cov = (wm[:, None] * (pts - x)).T @ (pts - x)
    assert cov == pytest.approx(P)


def test_nees_hand_example():
    assert A.nees(np.array([2.0, 1.0, 0.0, 0.0]), np.diag([4.0, 1.0, 1.0, 1.0])) == pytest.approx(2.0, rel=1e-6)


def test_init_from_polar_reproduces_position():
    z = A.h_polar(np.array([-120.0, 35.0, 0.0, 0.0]))
    x, P = A.init_from_polar(z, 5.0, 0.02, 10.0)
    assert x[:2] == pytest.approx([-120.0, 35.0])
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_simulation_is_reproducible_and_regler_scale_the_same_randomness():
    a = S.simulate(7, 5.0, 2.0, 40.0, 0.2)
    b = S.simulate(7, 5.0, 2.0, 40.0, 0.2)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.Z, b.Z)
    c = S.simulate(7, 10.0, 4.0, 40.0, 0.2)
    assert np.array_equal(a.X, c.X)                        # Sensorrauschen ändert die Wahrheit nicht
    noise_a = a.Z - np.array([A.h_polar(x) for x in a.X])
    noise_c = c.Z - np.array([A.h_polar(x) for x in c.X])
    assert np.allclose(noise_c[:, 0], 2 * noise_a[:, 0])


def test_vehicle_passes_the_beacon_at_about_the_minimum_distance():
    tr = S.simulate(1, 1.0, 0.5, 40.0, 0.05)
    r = np.hypot(tr.X[:, 0], tr.X[:, 1])
    assert r.min() == pytest.approx(40.0, abs=8.0)
    assert tr.X[0, 0] == C.START_X


def test_constants_are_consistent():
    assert C.SIGMA_B_MIN <= min(C.SIGMA_B_SWEEP) and max(C.SIGMA_B_SWEEP) <= C.SIGMA_B_MAX
    assert C.D_MIN <= min(C.NEES_GRID_D) and max(C.NEES_GRID_D) <= C.D_MAX
    assert C.NEES_SPIKE > C.NEES_BOUND_95 > C.STATE_DIM
