"""Erweiterter und Unscented Kalman-Filter für Entfernungs-/Peilmessungen, numpy-only. Jede Funktion ist eine Einheit.

Zustand x = (x, y, vx, vy) in der Ebene, konstante Geschwindigkeit mit zufälliger Beschleunigung (q in m/s^2).
Eine Bake im Ursprung (oder an `sensor`) misst Entfernung r und Peilung b; das Messmodell h(x) = (r, b) ist nichtlinear.
"""

from dataclasses import dataclass

import numpy as np

ORIGIN = np.zeros(2)


def wrap(a):
    """Winkel auf (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def cv_model_2d(dt, q):
    """F und Q für Konstante-Geschwindigkeit-Modell in der Ebene, Zustandsreihenfolge (x, y, vx, vy)."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    Q = np.zeros((4, 4))
    for a in (0, 1):
        Q[a, a] = q ** 2 * dt ** 3 / 3.0
        Q[a, a + 2] = Q[a + 2, a] = q ** 2 * dt ** 2 / 2.0
        Q[a + 2, a + 2] = q ** 2 * dt
    return F, Q


def predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q


def h_polar(x, sensor=ORIGIN):
    """Messmodell: Entfernung und Peilung (Radiant) der Position relativ zur Bake."""
    d = x[:2] - sensor
    return np.array([np.hypot(d[0], d[1]), np.arctan2(d[1], d[0])])


def jacobian_polar(x, sensor=ORIGIN):
    """Jacobi-Matrix von h an der Stelle x (2 x 4). Singulär, wenn die Position auf der Bake liegt."""
    d = x[:2] - sensor
    r2 = d[0] ** 2 + d[1] ** 2
    r = np.sqrt(r2)
    H = np.zeros((2, 4))
    H[0, 0], H[0, 1] = d[0] / r, d[1] / r
    H[1, 0], H[1, 1] = -d[1] / r2, d[0] / r2
    return H


def polar_to_position(z, sensor=ORIGIN):
    """Gemessene (Entfernung, Peilung) in eine Position umgerechnet: der „Rohwert“."""
    return sensor + z[0] * np.array([np.cos(z[1]), np.sin(z[1])])


def init_from_polar(z, sigma_r, sigma_b, v_std, sensor=ORIGIN):
    """Startzustand aus der ersten Messung: Position umgerechnet, Kovarianz über die Jacobi-Matrix der Umrechnung, Geschwindigkeit unbekannt."""
    r, b = z
    x = np.array([*polar_to_position(z, sensor), 0.0, 0.0])
    J = np.array([[np.cos(b), -r * np.sin(b)], [np.sin(b), r * np.cos(b)]])
    P = np.zeros((4, 4))
    P[:2, :2] = J @ np.diag([sigma_r ** 2, sigma_b ** 2]) @ J.T
    P[2, 2] = P[3, 3] = v_std ** 2
    return x, P


def ekf_update(x, P, z, R, sensor=ORIGIN):
    """EKF-Korrektur: h wird an der VORHERSAGE linearisiert (Jacobi-Matrix), Joseph-Form."""
    nu = z - h_polar(x, sensor)
    nu[1] = wrap(nu[1])
    H = jacobian_polar(x, sensor)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(len(x)) - K @ H
    return x + K @ nu, I_KH @ P @ I_KH.T + K @ R @ K.T


def sigma_points(x, P, alpha=1.0, beta=2.0, kappa=1.0):
    """Sigma-Punkte und Gewichte (Julier/Uhlmann): 2n+1 Punkte, deren gewichteter Mittelwert/Kovarianz x und P ergeben."""
    n = len(x)
    lam = alpha ** 2 * (n + kappa) - n
    L = np.linalg.cholesky((n + lam) * P + 1e-9 * np.eye(n))
    pts = np.vstack([x, x + L.T, x - L.T])
    wm = np.full(2 * n + 1, 0.5 / (n + lam))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + 1 - alpha ** 2 + beta
    return pts, wm, wc


def unscented_transform(pts, wm, wc, sensor=ORIGIN):
    """Sigma-Punkte durch h schicken; Mittelwert und Kovarianz der Bilder (Winkel relativ zum Bild des Mittelpunkts, damit ±pi nicht stört)."""
    Z = np.array([h_polar(p, sensor) for p in pts])
    Z[:, 1] = Z[0, 1] + wrap(Z[:, 1] - Z[0, 1])
    zm = wm @ Z
    dz = Z - zm
    S = (wc[:, None] * dz).T @ dz
    return Z, zm, dz, S


def ukf_update(x, P, z, R, sensor=ORIGIN, alpha=1.0, beta=2.0, kappa=1.0):
    """UKF-Korrektur: die Wahrscheinlichkeitsverteilung wird durch Sigma-Punkte statt durch eine Linearisierung weitergereicht."""
    pts, wm, wc = sigma_points(x, P, alpha, beta, kappa)
    Z, zm, dz, S = unscented_transform(pts, wm, wc, sensor)
    S = S + R
    Pxz = ((pts - x).T * wc) @ dz
    K = Pxz @ np.linalg.inv(S)
    nu = z - zm
    nu[1] = wrap(nu[1])
    return x + K @ nu, P - K @ S @ K.T


def nees(err, P):
    """Normiertes Fehlerquadrat err' P^-1 err; Erwartung bei ehrlichem Filter: Zustandsdimension (hier 4)."""
    return float(err @ np.linalg.solve(P + 1e-12 * np.eye(len(err)), err))


@dataclass(frozen=True)
class FilterRun:
    x: np.ndarray    # (T, 4)
    P: np.ndarray    # (T, 4, 4)


def run_filter(Z, kind, dt, q, sigma_r, sigma_b, v_std, sensor=ORIGIN):
    """Filter über eine ganze Spur. kind: 'ekf' oder 'ukf'. Z: (T, 2) Entfernung und Peilung; sigma_b in Radiant."""
    F, Q = cv_model_2d(dt, q)
    R = np.diag([sigma_r ** 2, sigma_b ** 2])
    update = ekf_update if kind == "ekf" else ukf_update
    T = len(Z)
    x, P = init_from_polar(Z[0], sigma_r, sigma_b, v_std, sensor)
    xs, Ps = np.zeros((T, 4)), np.zeros((T, 4, 4))
    xs[0], Ps[0] = x, P
    for t in range(1, T):
        x, P = predict(x, P, F, Q)
        x, P = update(x, P, Z[t], R, sensor)
        xs[t], Ps[t] = x, P
    return FilterRun(xs, Ps)
