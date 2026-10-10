"""Szenario: ein Fahrzeug fährt geradeaus an einer Bake vorbei, die Entfernung und Peilung zu ihm misst."""

from dataclasses import dataclass

import numpy as np

import ekf_algorithm as A
import ekf_constants as C


@dataclass(frozen=True)
class Track:
    t: np.ndarray        # Zeit in s
    X: np.ndarray        # (T, 4) wahrer Zustand (x, y, vx, vy)
    Z: np.ndarray        # (T, 2) Messung (Entfernung, Peilung in Radiant)


def simulate(seed, sigma_r, sigma_b_deg, d_min, q):
    """Wahre Fahrt plus Messungen. Alle Zufallszahlen kommen in fester Reihenfolge aus einem Generator; die Regler skalieren sie nur."""
    rng = np.random.default_rng(seed)
    n_proc = rng.standard_normal((C.T_STEPS, 4))
    n_meas = rng.standard_normal((C.T_STEPS, 2))
    F, Q = A.cv_model_2d(C.DT, q)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    X = np.zeros((C.T_STEPS, 4))
    x = np.array([C.START_X, d_min, C.SPEED, 0.0])
    for k in range(C.T_STEPS):
        X[k] = x
        x = F @ x + L @ n_proc[k]
    sb = np.deg2rad(sigma_b_deg)
    Z = np.array([A.h_polar(X[k]) for k in range(C.T_STEPS)])
    Z[:, 0] += sigma_r * n_meas[:, 0]
    Z[:, 1] = A.wrap(Z[:, 1] + sb * n_meas[:, 1])
    return Track(np.arange(C.T_STEPS) * C.DT, X, Z)
