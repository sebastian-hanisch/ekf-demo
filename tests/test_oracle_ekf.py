"""Orakel-Test ekf-demo: EKF und UKF gegen ANDERE Rechenwege.

1. Jacobi-Matrix gegen zentrale finite Differenzen,
2. Unscented-Transformation gegen Monte-Carlo-Momente (und: sie trifft den Mittelwert besser als die Linearisierung),
3. Gitter-Posterior (numerische Integration) gegen Importance-Sampling mit anderem Zufallsstrom,
4. beide Filter gegen den exakten Posterior über viele Ziehungen: nahe dran bei mäßiger Nichtlinearität, weit weg beim Ring,
5. Konsistenz: mittlere NEES über viele Fahrten bei mäßiger Nichtlinearität gleich 4,
6. die angezeigten Kennzahlen gegen unabhängige Neuberechnung.
"""

import numpy as np
import pytest

import ekf_algorithm as A
import ekf_constants as C
import ekf_evaluation as ev
import ekf_scenario as S


def test_jacobian_matches_central_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = np.array([rng.uniform(-200, 200), rng.uniform(-200, 200), 0.0, 0.0])
        if np.hypot(x[0], x[1]) < 5:
            continue
        H = A.jacobian_polar(x)
        eps = 1e-5
        H_num = np.zeros((2, 4))
        for j in range(2):
            dx = np.zeros(4)
            dx[j] = eps
            hp, hm = A.h_polar(x + dx), A.h_polar(x - dx)
            d = hp - hm
            d[1] = A.wrap(d[1])
            H_num[:, j] = d / (2 * eps)
        assert H == pytest.approx(H_num, rel=1e-5, abs=1e-8)


def _mc_moments(x, P, n=300_000, seed=1):
    smp = np.random.default_rng(seed).multivariate_normal(x, P, n)
    Z = np.c_[np.hypot(smp[:, 0], smp[:, 1]), np.arctan2(smp[:, 1], smp[:, 0])]
    return Z.mean(0), np.cov(Z.T)


def test_unscented_transform_matches_monte_carlo_moments_and_beats_linearisation_for_the_mean():
    x = np.array([30.0, 20.0, 0.0, 0.0])
    P = np.diag([100.0, 100.0, 1.0, 1.0])
    mc_m, mc_c = _mc_moments(x, P)
    pts, wm, wc = A.sigma_points(x, P)
    _, zm, _, S_ = A.unscented_transform(pts, wm, wc)
    assert zm == pytest.approx(mc_m, rel=0.01, abs=0.02)
    assert np.sqrt(S_[0, 0]) == pytest.approx(np.sqrt(mc_c[0, 0]), rel=0.05)
    assert abs(zm[0] - mc_m[0]) < abs(A.h_polar(x)[0] - mc_m[0])      # Entfernungsmittel: Linearisierung ist verzerrt


def _importance_posterior_mean(x, s, z, R, n=600_000, seed=5):
    smp = np.random.default_rng(seed).normal(0, 1, (n, 2)) * s + x[:2]
    r, b = np.hypot(smp[:, 0], smp[:, 1]), np.arctan2(smp[:, 1], smp[:, 0])
    w = np.exp(-0.5 * ((z[0] - r) ** 2 / R[0, 0] + A.wrap(z[1] - b) ** 2 / R[1, 1]))
    return (w[:, None] * smp).sum(0) / w.sum()


def test_grid_posterior_matches_importance_sampling():
    R = np.diag([25.0, np.deg2rad(2.0) ** 2])
    for xy, s in (((100.0, 40.0), 10.0), ((30.0, 20.0), 10.0), ((15.0, 5.0), 8.0)):
        t = np.array([xy[0] + 2.0, xy[1] - 3.0, 0.0, 0.0])
        z = A.h_polar(t)
        grid = ev.exact_posterior_mean(xy, s, z, R)
        mc = _importance_posterior_mean(np.array([*xy, 0, 0]), s, z, R)
        assert grid == pytest.approx(mc, abs=0.35)


def test_filters_against_the_exact_posterior_over_many_draws():
    res = {r["name"]: r for r in ev.sweep_exact_update()}
    mild = res["Weit weg, mäßig unsicher"]
    assert mild["ekf"] == pytest.approx(mild["exact"], rel=0.06) and mild["ukf"] == pytest.approx(mild["exact"], rel=0.06)
    ring = res["Sehr unsicher, auf der Achse"]
    assert ring["ekf"] > 2.0 * ring["exact"] and ring["ukf"] > 2.0 * ring["exact"]
    for r in res.values():
        assert r["exact"] <= min(r["ekf"], r["ukf"]) + 0.15      # der exakte Posterior-Mittelwert ist der bestmögliche Schätzer


def test_mean_nees_is_state_dimension_for_moderate_nonlinearity():
    for kind in ("ekf", "ukf"):
        vals = []
        for seed in range(150):
            tr = S.simulate(900 + seed, 5.0, 2.0, 80.0, 0.2)
            run = A.run_filter(tr.Z, kind, C.DT, 0.2, 5.0, np.deg2rad(2.0), C.V_PRIOR_STD)
            vals.append(np.mean([A.nees(tr.X[k] - run.x[k], run.P[k]) for k in range(C.N_BURN, C.T_STEPS)]))
        assert np.mean(vals) == pytest.approx(4.0, abs=0.5), kind


def test_displayed_numbers_match_independent_recomputation():
    for seed in (3, 11, 40):
        s = ev.Settings(seed=seed)
        a = ev.analyse(s)
        tr = S.simulate(seed, s.sigma_r, s.sigma_b, s.d_min, s.q)
        raw = np.c_[tr.Z[:, 0] * np.cos(tr.Z[:, 1]), tr.Z[:, 0] * np.sin(tr.Z[:, 1])]
        sl = slice(C.N_BURN, None)
        assert a.rmse_raw == pytest.approx(np.sqrt(np.mean(np.sum((raw[sl] - tr.X[sl, :2]) ** 2, axis=1))))
        assert a.rmse_ekf == pytest.approx(np.sqrt(np.mean(np.sum((a.ekf.x[sl, :2] - tr.X[sl, :2]) ** 2, axis=1))))
        n = [float((tr.X[k] - a.ukf.x[k]) @ np.linalg.inv(a.ukf.P[k]) @ (tr.X[k] - a.ukf.x[k])) for k in range(C.T_STEPS)]
        assert a.nees_ukf == pytest.approx(np.mean(n[C.N_BURN:]), rel=1e-6)
        assert a.peak_ukf == pytest.approx(max(n[C.N_BURN:]), rel=1e-6)
