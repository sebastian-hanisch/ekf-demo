"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import ekf_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_map(a):
    """Draufsicht: wahre Bahn, aus den Messungen umgerechnete Rohpositionen, EKF- und UKF-Schätzung, Bake im Ursprung (Achsen nicht gleich skaliert)."""
    X = a.track.X
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.raw[:, 0], y=a.raw[:, 1], mode="markers", marker=dict(color=C.COLOR_RAW, size=5), name="aus Entfernung+Peilung umgerechnet",
                             hovertemplate="x %{x:.0f} m, y %{y:.0f} m<extra>Rohwert</extra>"))
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), name="Wahrheit"))
    fig.add_trace(go.Scatter(x=a.ekf.x[:, 0], y=a.ekf.x[:, 1], mode="lines", line=dict(color=C.COLOR_EKF, width=3), name="EKF"))
    fig.add_trace(go.Scatter(x=a.ukf.x[:, 0], y=a.ukf.x[:, 1], mode="lines", line=dict(color=C.COLOR_UKF, width=2, dash="dash"), name="UKF"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color=C.COLOR_BEACON, size=13, symbol="star"), name="Bake"))
    fig.update_xaxes(title_text="x entlang der Straße (m)")
    fig.update_yaxes(title_text="y quer dazu (m)")
    return _base(fig, 380)


def build_error_chart(a):
    t = a.track.t
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.err_raw, mode="markers", marker=dict(color=C.COLOR_RAW, size=5), name="Rohwert"))
    fig.add_trace(go.Scatter(x=t, y=a.err_ekf, mode="lines", line=dict(color=C.COLOR_EKF, width=3), name="EKF"))
    fig.add_trace(go.Scatter(x=t, y=a.err_ukf, mode="lines", line=dict(color=C.COLOR_UKF, width=2, dash="dash"), name="UKF"))
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Positionsfehler (m)")
    return _base(fig, 300)


def build_nees_chart(a):
    t = a.track.t
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[C.N_BURN:], y=np.maximum(a.nees_ekf_t[C.N_BURN:], 1e-2), mode="lines", line=dict(color=C.COLOR_EKF, width=3), name="EKF"))
    fig.add_trace(go.Scatter(x=t[C.N_BURN:], y=np.maximum(a.nees_ukf_t[C.N_BURN:], 1e-2), mode="lines", line=dict(color=C.COLOR_UKF, width=2, dash="dash"), name="UKF"))
    fig.add_hline(y=C.STATE_DIM, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Soll = 4", annotation_position="bottom right")
    fig.add_hline(y=C.NEES_BOUND_95, line=dict(color=C.COLOR_BEACON, dash="dot"), annotation_text="95-%-Grenze", annotation_position="top right")
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="NEES (logarithmisch)", type="log")
    return _base(fig, 300)


def build_bearing_sweep(rows):
    xs = [r["sigma_b"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["raw"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_RAW, width=3), name="Rohwert"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ekf"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_EKF, width=3), name="EKF"))
    fig.add_trace(go.Scatter(x=xs, y=[r["ukf"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_UKF, width=2, dash="dash"), name="UKF"))
    fig.update_xaxes(title_text="Peilrauschen (Grad)")
    fig.update_yaxes(title_text="Positionsfehler, RMSE (m)")
    return _base(fig)
