"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (feste Fahrten).

Plattformrobust: Bänder statt exakter Werte (CI installiert neueste numpy/scipy)."""

import numpy as np
import pytest

import ekf_constants as C
import ekf_evaluation as ev


@pytest.fixture(scope="module")
def bearing_rows():
    return {r["sigma_b"]: r for r in ev.sweep_bearing(ev.Settings())}


@pytest.fixture(scope="module")
def grid_cells():
    return {(c["d"], c["sigma_b"]): c for c in ev.sweep_nees_grid(ev.Settings())}


@pytest.fixture(scope="module")
def singularity():
    return ev.sweep_singularity()


@pytest.fixture(scope="module")
def exact_rows():
    return {r["name"]: r for r in ev.sweep_exact_update()}


def test_standard_track_numbers():
    a = ev.analyse(ev.Settings())
    assert a.rmse_raw == pytest.approx(6.9, abs=0.6)
    assert a.rmse_ekf == pytest.approx(3.85, abs=0.4) and a.rmse_ukf == pytest.approx(3.87, abs=0.4)
    assert a.nees_ekf == pytest.approx(3.85, abs=0.5)


def test_filters_halve_the_error_of_the_converted_position_and_are_equal(bearing_rows):
    r = bearing_rows[2.0]
    assert r["raw"] == pytest.approx(7.6, abs=0.8) and r["ekf"] == pytest.approx(3.9, abs=0.4)
    assert r["ekf"] < 0.6 * r["raw"]
    for row in bearing_rows.values():
        assert abs(row["ekf"] - row["ukf"]) < 0.5
        assert row["lost_ekf"] == 0.0 and row["lost_ukf"] == 0.0


def test_the_typical_run_is_consistent_but_ekf_has_rare_spikes_near_and_sharp(grid_cells):
    for c in grid_cells.values():
        if c["sigma_b"] < 10.0:
            assert 3.4 < c["median_ekf"] < 4.6 and 3.4 < c["median_ukf"] < 4.6
    near = grid_cells[(10.0, 0.5)]
    assert near["spike_ekf"] == pytest.approx(0.10, abs=0.06) and near["spike_ukf"] <= 0.02
    assert grid_cells[(20.0, 0.5)]["spike_ekf"] == pytest.approx(0.08, abs=0.06)
    for sb in (0.5, 2.0, 5.0):
        assert grid_cells[(80.0, sb)]["spike_ekf"] <= 0.02


def test_rough_bearing_inflates_the_ekf_median_more_than_the_ukf(grid_cells):
    for d in C.NEES_GRID_D:
        c = grid_cells[(d, 10.0)]
        assert c["median_ekf"] == pytest.approx(5.0, abs=0.5) and c["median_ukf"] == pytest.approx(4.3, abs=0.3)
        assert c["median_ekf"] > c["median_ukf"] + 0.3


def test_ekf_fails_at_the_singularity_and_the_ukf_does_not(singularity):
    assert singularity["ekf"]["ok"] == 0.0 and singularity["ekf"]["median"] > 100.0
    assert singularity["ukf"]["ok"] == pytest.approx(0.95, abs=0.08) and singularity["ukf"]["median"] < 10.0


def test_exact_posterior_comparison_numbers(exact_rows):
    mild = exact_rows["Weit weg, mäßig unsicher"]
    assert (mild["ekf"], mild["ukf"], mild["exact"]) == pytest.approx((6.2, 6.2, 6.2), abs=0.5)
    near = exact_rows["Nah, mäßig unsicher"]
    assert near["ekf"] == pytest.approx(6.1, abs=0.6) and near["exact"] == pytest.approx(4.7, abs=0.5)
    ring = exact_rows["Sehr unsicher, auf der Achse"]
    assert ring["ekf"] == pytest.approx(14.0, abs=2.0) and ring["ukf"] == pytest.approx(15.0, abs=2.5) and ring["exact"] == pytest.approx(5.5, abs=0.8)
    assert ring["ukf"] >= ring["ekf"] - 0.5      # der UKF ist hier nicht besser
