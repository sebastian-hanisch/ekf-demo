"""Kennzahlen, Verdikte und Sweeps über die festen Instanzen (SWEEP_SEEDS)."""

from dataclasses import dataclass, replace

import numpy as np

import ekf_algorithm as A
import ekf_constants as C
import ekf_scenario as S


@dataclass(frozen=True)
class Settings:
    sigma_r: float = C.DEFAULT_SIGMA_R      # Entfernungsrauschen in m
    sigma_b: float = C.DEFAULT_SIGMA_B      # Peilrauschen in Grad
    d_min: float = C.DEFAULT_D              # Mindestabstand der Straße zur Bake in m
    q: float = C.DEFAULT_Q                  # Beschleunigungsrauschen (Wahrheit UND Annahme des Filters)
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    track: S.Track
    raw: np.ndarray            # (T, 2) aus der Messung umgerechnete Position
    ekf: A.FilterRun
    ukf: A.FilterRun
    err_raw: np.ndarray        # Positionsfehler je Schritt (Betrag)
    err_ekf: np.ndarray
    err_ukf: np.ndarray
    nees_ekf_t: np.ndarray     # NEES je Schritt
    nees_ukf_t: np.ndarray
    rmse_raw: float
    rmse_ekf: float
    rmse_ukf: float
    nees_ekf: float            # Mittel ab N_BURN (Soll 4)
    nees_ukf: float
    share_ok_ekf: float        # Anteil der Schritte mit NEES unter der 95-%-Grenze
    share_ok_ukf: float
    peak_ekf: float            # größte NEES einer Fahrt ab N_BURN (seltene Einbrüche der Konsistenz)
    peak_ukf: float


def _nees_series(run, X):
    return np.array([A.nees(X[k] - run.x[k], run.P[k]) for k in range(len(run.x))])


def analyse(settings):
    tr = S.simulate(settings.seed, settings.sigma_r, settings.sigma_b, settings.d_min, settings.q)
    sb = np.deg2rad(settings.sigma_b)
    ekf = A.run_filter(tr.Z, "ekf", C.DT, settings.q, settings.sigma_r, sb, C.V_PRIOR_STD)
    ukf = A.run_filter(tr.Z, "ukf", C.DT, settings.q, settings.sigma_r, sb, C.V_PRIOR_STD)
    raw = np.array([A.polar_to_position(z) for z in tr.Z])
    pos = tr.X[:, :2]
    e_raw = np.linalg.norm(raw - pos, axis=1)
    e_ekf = np.linalg.norm(ekf.x[:, :2] - pos, axis=1)
    e_ukf = np.linalg.norm(ukf.x[:, :2] - pos, axis=1)
    n_ekf, n_ukf = _nees_series(ekf, tr.X), _nees_series(ukf, tr.X)
    sl = slice(C.N_BURN, None)
    rm = lambda e: float(np.sqrt(np.mean(e[sl] ** 2)))
    return Analysis(
        settings, tr, raw, ekf, ukf, e_raw, e_ekf, e_ukf, n_ekf, n_ukf,
        rm(e_raw), rm(e_ekf), rm(e_ukf), float(np.mean(n_ekf[sl])), float(np.mean(n_ukf[sl])),
        float(np.mean(n_ekf[sl] <= C.NEES_BOUND_95)), float(np.mean(n_ukf[sl] <= C.NEES_BOUND_95)),
        float(np.max(n_ekf[sl])), float(np.max(n_ukf[sl])),
    )


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------

def _over_seeds(settings, fn):
    return [fn(analyse(replace(settings, seed=s))) for s in C.SWEEP_SEEDS]


def sweep_bearing(settings):
    """Peilrauschen variieren: Positionsfehler Rohwert, EKF, UKF (Median) und Anteil verlorener Fahrten."""
    rows = []
    for sb in C.SIGMA_B_SWEEP:
        res = _over_seeds(replace(settings, sigma_b=sb), lambda a: (a.rmse_raw, a.rmse_ekf, a.rmse_ukf))
        arr = np.array(res)
        rows.append({"sigma_b": sb, "raw": float(np.median(arr[:, 0])), "ekf": float(np.median(arr[:, 1])), "ukf": float(np.median(arr[:, 2])),
                     "lost_ekf": float(np.mean(arr[:, 1] > C.FAIL_RMSE)), "lost_ukf": float(np.mean(arr[:, 2] > C.FAIL_RMSE))})
    return rows


def sweep_nees_grid(settings):
    """Konsistenz je Mindestabstand x Peilrauschen über GRID_SEEDS: typische NEES (Median) und Anteil der Fahrten mit NEES-Spitze.

    Der Mittelwert wäre irreführend: ein einziger Einbruch (NEES in Tausendern) verdirbt ihn, obwohl die meisten Fahrten ehrlich sind.
    """
    cells = []
    for d in C.NEES_GRID_D:
        for sb in C.NEES_GRID_SIGMA_B:
            res = np.array([(a.nees_ekf, a.nees_ukf, a.peak_ekf, a.peak_ukf)
                            for a in (analyse(replace(settings, d_min=d, sigma_b=sb, seed=s)) for s in C.GRID_SEEDS)])
            cells.append({"d": d, "sigma_b": sb, "median_ekf": float(np.median(res[:, 0])), "median_ukf": float(np.median(res[:, 1])),
                          "spike_ekf": float(np.mean(res[:, 2] > C.NEES_SPIKE)), "spike_ukf": float(np.mean(res[:, 3] > C.NEES_SPIKE))})
    return cells


# --- Sonderfall: nur Entfernung, Anfangsschätzung auf der Singularität -----------------------------------------------------------------------

def _range_only_models(use_second):
    s2 = np.array(C.SECOND_BEACON)

    def h(x, t):
        out = [np.hypot(x[0], x[1])]
        if use_second and t >= C.SECOND_BEACON_FROM:
            out.append(np.hypot(x[0] - s2[0], x[1] - s2[1]))
        return np.array(out)

    def H(x, t):
        d = x[:2]
        rows = [np.r_[d / np.hypot(*d), 0, 0]]
        if use_second and t >= C.SECOND_BEACON_FROM:
            d2 = x[:2] - s2
            rows.append(np.r_[d2 / np.hypot(*d2), 0, 0])
        return np.array(rows)

    return h, H


def range_only_run(seed, kind, use_second=True):
    """Eine Fahrt mit reiner Entfernungsmessung von einer Bake (ab Schritt SECOND_BEACON_FROM eine zweite).

    Der Anfangsschätzwert liegt exakt auf der Achse y = 0, auf der die Jacobi-Matrix der Entfernung in y-Richtung null ist,
    während die Wahrheit bei y = 60 fährt. Gibt (Endfehler der Position, Endschätzung) zurück.
    """
    rng = np.random.default_rng(6000 + seed)
    F, Q = A.cv_model_2d(C.DT, 0.05)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    X = np.zeros((C.RANGE_ONLY_STEPS, 4))
    x = np.array([C.RANGE_ONLY_START[0], C.RANGE_ONLY_START[1], C.RANGE_ONLY_SPEED, 0.0])
    for k in range(C.RANGE_ONLY_STEPS):
        X[k] = x
        x = F @ x + L @ rng.standard_normal(4)
    h, H = _range_only_models(use_second)
    sr2 = C.RANGE_ONLY_SIGMA ** 2
    x = np.array([C.RANGE_ONLY_START[0], 0.0, C.RANGE_ONLY_SPEED, 0.0])
    P = np.diag([5.0 ** 2, C.RANGE_ONLY_Y_PRIOR_STD ** 2, 1.0, 1.0])
    n = 4
    for t in range(1, C.RANGE_ONLY_STEPS):
        z = h(X[t], t) + C.RANGE_ONLY_SIGMA * rng.standard_normal(len(h(X[t], t)))
        x, P = A.predict(x, P, F, Q)
        m = len(z)
        Rm = np.eye(m) * sr2
        if kind == "ekf":
            Hm = H(x, t)
            S_ = Hm @ P @ Hm.T + Rm
            K = P @ Hm.T @ np.linalg.inv(S_)
            x = x + K @ (z - h(x, t))
            P = (np.eye(n) - K @ Hm) @ P
        else:
            pts, wm, wc = A.sigma_points(x, P)
            Zp = np.array([h(p, t) for p in pts])
            zm = wm @ Zp
            dz = Zp - zm
            S_ = (wc[:, None] * dz).T @ dz + Rm
            Pxz = ((pts - x).T * wc) @ dz
            K = Pxz @ np.linalg.inv(S_)
            x = x + K @ (z - zm)
            P = P - K @ S_ @ K.T
    return float(np.linalg.norm(x[:2] - X[-1, :2])), x


def sweep_singularity():
    """Anteil der Fahrten mit Endfehler unter RANGE_ONLY_OK_M: EKF gegen UKF, mit zweiter Bake (löst die Mehrdeutigkeit auf)."""
    out = {}
    for kind in ("ekf", "ukf"):
        errs = np.array([range_only_run(s, kind)[0] for s in C.SWEEP_SEEDS])
        out[kind] = {"ok": float(np.mean(errs < C.RANGE_ONLY_OK_M)), "median": float(np.median(errs))}
    return out


# --- Experiment 4: eine Korrektur gegen den exakten Posterior --------------------------------------------------------------------------------

def exact_posterior_mean(x_prior, s, z, R, n=C.EXACT_GRID, span=5.5):
    """Exakter Posterior-Mittelwert der Position bei Gauß-Prior (isotrop, Streuung s) und Messung z = (r, b): numerische Integration auf einem Gitter."""
    gx = np.linspace(x_prior[0] - span * s, x_prior[0] + span * s, n)
    gy = np.linspace(x_prior[1] - span * s, x_prior[1] + span * s, n)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    prior = np.exp(-0.5 * (((GX - x_prior[0]) / s) ** 2 + ((GY - x_prior[1]) / s) ** 2))
    r, b = np.hypot(GX, GY), np.arctan2(GY, GX)
    lik = np.exp(-0.5 * ((z[0] - r) ** 2 / R[0, 0] + A.wrap(z[1] - b) ** 2 / R[1, 1]))
    w = prior * lik
    w = w / w.sum()
    return np.array([(w * GX).sum(), (w * GY).sum()])


def sweep_exact_update():
    """Eine einzige Korrektur, gemittelt über Ziehungen aus dem Modell (Wahrheit ~ Prior, Messung ~ h(Wahrheit) + Rauschen).

    Gibt je Fall den RMSE der Positionsschätzung von EKF, UKF und exaktem Posterior-Mittel zurück (das exakte ist das bestmögliche).
    """
    out = []
    for i, (name, xy, s, sb_deg) in enumerate(C.EXACT_CASES):
        rng = np.random.default_rng(7000 + i)
        sb = np.deg2rad(sb_deg)
        R = np.diag([C.EXACT_SIGMA_R ** 2, sb ** 2])
        x = np.array([xy[0], xy[1], 0.0, 0.0])
        P = np.diag([s ** 2, s ** 2, 1.0, 1.0])
        se = {"ekf": 0.0, "ukf": 0.0, "exact": 0.0}
        for _ in range(C.EXACT_DRAWS):
            t = np.array([xy[0] + s * rng.standard_normal(), xy[1] + s * rng.standard_normal(), 0.0, 0.0])
            z = A.h_polar(t) + np.array([C.EXACT_SIGMA_R * rng.standard_normal(), sb * rng.standard_normal()])
            est = {"ekf": A.ekf_update(x, P, z, R)[0][:2], "ukf": A.ukf_update(x, P, z, R)[0][:2], "exact": exact_posterior_mean(xy, s, z, R)}
            for k, v in est.items():
                se[k] += float(np.sum((v - t[:2]) ** 2))
        out.append({"name": name, "xy": xy, "s": s, "sigma_b": sb_deg,
                    **{k: float(np.sqrt(v / C.EXACT_DRAWS)) for k, v in se.items()}})
    return out
