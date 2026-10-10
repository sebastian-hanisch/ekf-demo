"""Erweiterter Kalman-Filter: Entfernung und Peilung statt Position - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dieselbe Fahrt wie im Kalman-Filter-Stück, aber der Sensor misst nicht mehr die Position, sondern Entfernung und Peilung zu einer Bake: ein
nichtlineares Messmodell. Der Erweiterte Kalman-Filter (EKF) linearisiert es; der Unscented Kalman-Filter (UKF) dient als Vergleich.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import ekf_constants as C
import ekf_evaluation as ev
from ekf_evaluation import Settings, analyse
from ekf_pdf_export import build_pdf
from ekf_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ekf_visualization import build_bearing_sweep, build_error_chart, build_map, build_nees_chart

st.set_page_config(page_title="Erweiterter Kalman-Filter – Sebastian Hanisch", layout="wide")


def de(x, nd=1):
    """Dezimalkomma statt Punkt."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x):
    return f"{100 * x:.0f} %"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_bearing(settings):
    return ev.sweep_bearing(settings)


@st.cache_data(show_spinner=False)
def _sweep_grid(settings):
    return ev.sweep_nees_grid(settings)


@st.cache_data(show_spinner=False)
def _sweep_singularity(_key):
    return ev.sweep_singularity()


@st.cache_data(show_spinner=False)
def _sweep_exact(_key):
    return ev.sweep_exact_update()


def run_experiment(key, fn, settings, label):
    """Teure Sweeps nur auf Knopfdruck: das Ergebnis bleibt sichtbar, solange die Einstellungen gleich bleiben."""
    if st.button(label, key=f"run_{key}"):
        st.session_state[f"done_{key}"] = settings
    if st.session_state.get(f"done_{key}") != settings:
        st.info("Noch nicht gerechnet für diese Einstellungen. Der Knopf startet die Rechnung (wenige Sekunden); danach bleibt das Ergebnis stehen.")
        return None
    with st.spinner("Rechne … (je nach Rechner bis etwa 20 Sekunden)"):
        return fn(settings)


st.title("🧭 Erweiterter Kalman-Filter: Entfernung und Peilung statt Position")
st.markdown(
    """
Im ersten Stück maß das GPS die Position selbst; der Kalman-Filter war dafür **linear** und damit exakt. Hier misst ein Sensor an der Straße (eine **Bake**) nur
**Entfernung und Peilung** zum Fahrzeug. Die Position folgt daraus über Wurzeln und Winkelfunktionen: das Messmodell ist **nichtlinear**, die
Gauß-Verteilungen des Filters bleiben es nach der Messung nicht mehr. Der **Erweiterte Kalman-Filter (EKF)** hilft sich mit einer **Linearisierung** an der Vorhersage;
zum Vergleich läuft der **Unscented Kalman-Filter (UKF)** mit, der die Verteilung stattdessen durch ausgewählte Punkte schickt.
Die Demo ist das dritte Stück der Linie **Zustandsschätzung und Filter**. Im Unterschied zu den Fall-Demos zeigt sie ein einziges Verfahren an einem wachsenden Beispiel.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Erweiterte Kalman-Filter", expanded=True):
    st.markdown(
        """
1. **Vorhersage** wie gehabt: das Fahrzeug fährt mit gleicher Geschwindigkeit weiter, die Unsicherheit wächst.
2. **Linearisieren:** Die Messfunktion (Position → Entfernung, Peilung) wird am vorhergesagten Punkt durch ihre **Tangente** ersetzt (die Jacobi-Matrix).
3. **Korrigieren** mit dieser Tangente wie im linearen Filter.

**Wo es trägt:** Solange die Unsicherheit klein gegenüber der Krümmung der Messfunktion ist, liegt die Tangente nah an der Kurve. Die Demo zeigt, wo das nicht mehr stimmt.
Der **UKF** braucht keine Tangente: er schickt 9 Punkte um die Vorhersage durch die echte Funktion und fasst die Bilder wieder zu Mittelwert und Kovarianz zusammen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Sensor**")
    sigma_r = st.slider("Entfernungsrauschen (m)", *bounds("sr_slider"), key="sr_slider", step=0.5,
                        help="Streuung der gemessenen Entfernung (Standardabweichung).")
    sigma_b = st.slider("Peilrauschen (Grad)", *bounds("sb_slider"), key="sb_slider", step=0.5,
                        help="Streuung der gemessenen Richtung (Standardabweichung, in Grad).")
    st.markdown("**Die Fahrt**")
    d_min = st.slider("Mindestabstand zur Bake (m)", *bounds("d_slider"), key="d_slider", step=5.0,
                      help="Wie nah die Straße an der Bake vorbeiführt. Nah heißt: die Peilung ändert sich im Vorbeifahren schnell.")
    q = st.slider("Fahrunruhe (m/s²)", *bounds("q_slider"), key="q_slider", step=0.05,
                  help="Beschleunigungsrauschen der Fahrt; der Filter kennt den wahren Wert.")
    seed = st.number_input("Zufalls-Seed der Fahrt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Fahrt generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"sr_slider": float(sigma_r), "sb_slider": float(sigma_b), "d_slider": float(d_min), "q_slider": float(q), "seed_input": int(seed)})

settings = Settings(float(sigma_r), float(sigma_b), float(d_min), float(q), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Positionsfehler EKF", f"{de(a.rmse_ekf)} m", delta=f"{a.rmse_ekf - a.rmse_raw:+.1f} m gegen Rohwert".replace(".", ","), delta_color="inverse",
          help="Wurzel des mittleren quadratischen Fehlers (RMSE) ab Sekunde 5.")
m2.metric("Positionsfehler UKF", f"{de(a.rmse_ukf)} m", delta=f"{a.rmse_ukf - a.rmse_ekf:+.1f} m gegen EKF".replace(".", ","), delta_color="inverse")
m3.metric("Positionsfehler Rohwert", f"{de(a.rmse_raw)} m", delta="Referenz (aus der Messung umgerechnet)", delta_color="off")
m4.metric("Größte NEES (Soll 4)", f"{a.peak_ekf:.0f}".replace(",", "."), delta=f"UKF: {a.peak_ukf:.0f}", delta_color="off",
          help="Höchster Wert des normierten Fehlerquadrats dieser Fahrt. Ein ehrlicher Filter bleibt meist unter 9,5.")

st.markdown("**Draufsicht** – grau: aus der Messung umgerechnete Rohpositionen, gestrichelt grün: Wahrheit, blau: EKF, orange gestrichelt: UKF, Stern: Bake. Die Achsen sind nicht gleich skaliert.")
st.plotly_chart(build_map(a), width="stretch", key="map_chart")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Positionsfehler über die Zeit**")
    st.plotly_chart(build_error_chart(a), width="stretch", key="error_chart")
with c2:
    st.markdown("**NEES über die Zeit** – ist der Filter ehrlich? (Soll 4)")
    st.plotly_chart(build_nees_chart(a), width="stretch", key="nees_chart")

st.download_button("📄 Auswertung als PDF", data=build_pdf(a), file_name="ekf_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann trägt die Linearisierung nicht mehr?")
if a.peak_ekf > C.NEES_SPIKE:
    st.warning(
        f"**Einbruch der Konsistenz in dieser Fahrt.** Die NEES des EKF springt auf {a.peak_ekf:.0f} (Soll 4, UKF: {a.peak_ukf:.0f}): "
        "der Filter hielt sich für sehr genau, der wahre Fehler lag weit außerhalb seiner angegebenen Unsicherheit. Der mittlere Positionsfehler sieht dabei oft unauffällig aus."
    )
elif a.nees_ekf - a.nees_ukf > 0.5:
    st.info(f"Der EKF ist etwas überselbstsicher (mittlere NEES {de(a.nees_ekf, 2)} gegen {de(a.nees_ukf, 2)} beim UKF; Soll 4). Der Positionsfehler ist praktisch gleich.")
else:
    st.success(f"Beide Filter sind hier konsistent (mittlere NEES EKF {de(a.nees_ekf, 2)}, UKF {de(a.nees_ukf, 2)}; Soll 4). Die Linearisierung trägt bei diesen Einstellungen.")
st.caption("Eine einzelne Fahrt zeigt nur, ob es hier passiert ist; wie oft, zeigt Experiment 2 über 100 feste Fahrten je Einstellung.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie viel Fehler bringt die Linearisierung?")
st.caption(f"Median über {len(C.SWEEP_SEEDS)} feste Fahrten; Abstand, Entfernungsrauschen und Fahrunruhe wie in der Seitenleiste, das Peilrauschen läuft von 0,5 bis 10 Grad.")
rows = run_experiment("bearing", _sweep_bearing, settings, f"▶ Experiment 1 rechnen ({len(C.SIGMA_B_SWEEP) * len(C.SWEEP_SEEDS)} Fahrten)")
if rows is not None:
    st.plotly_chart(build_bearing_sweep(rows), width="stretch", key="bearing_chart")
    worst = max(abs(r["ekf"] - r["ukf"]) for r in rows)
    lost = max(r["lost_ekf"] for r in rows)
    st.info(
        f"Der größte Unterschied im Positionsfehler zwischen EKF und UKF beträgt in diesem Sweep **{de(worst, 2)} m**; beide halbieren etwa den Fehler der umgerechneten Rohposition "
        f"(bei 2 Grad: Rohwert {de(next(r for r in rows if r['sigma_b'] == 2.0)['raw'])} m, EKF {de(next(r for r in rows if r['sigma_b'] == 2.0)['ekf'])} m). "
        f"Verlorene Fahrten (Fehler über {C.FAIL_RMSE:.0f} m): {'keine' if lost == 0 else 'höchstens ' + pct(lost)}. **Beim Positionsfehler zahlt sich der UKF hier nicht aus.**"
    )

st.subheader("🔬 Experiment 2: Wann wird der EKF überselbstsicher?")
st.caption(
    f"Je Zelle {len(C.GRID_SEEDS)} feste Fahrten. Zahlen: typische NEES (Median der Fahrt-Mittel) und Anteil der Fahrten mit einer NEES-Spitze über {C.NEES_SPIKE:.0f} "
    "(Soll der NEES: 4). Der Mittelwert wäre irreführend, weil einzelne Einbrüche ihn verdrehen."
)
cells = run_experiment("grid", _sweep_grid, settings, f"▶ Experiment 2 rechnen ({len(C.NEES_GRID_D) * len(C.NEES_GRID_SIGMA_B) * len(C.GRID_SEEDS)} Fahrten)")
if cells is not None:
    head = "| Abstand \\ Peilrauschen | " + " | ".join(f"{de(s)} Grad" for s in C.NEES_GRID_SIGMA_B) + " |\n|---|" + "---|" * len(C.NEES_GRID_SIGMA_B) + "\n"
    body = ""
    for d in C.NEES_GRID_D:
        row = [c for c in cells if c["d"] == d]
        body += f"| **{d:.0f} m** | " + " | ".join(
            f"EKF {de(c['median_ekf'])} / {pct(c['spike_ekf'])}<br>UKF {de(c['median_ukf'])} / {pct(c['spike_ukf'])}" for c in row) + " |\n"
    st.markdown("Zelle: **Median-NEES / Anteil der Fahrten mit Spitze**\n\n" + head + body)
    worst_spike = max(cells, key=lambda c: c["spike_ekf"])
    far = [c for c in cells if c["d"] == max(C.NEES_GRID_D) and c["sigma_b"] <= 5.0]
    st.info(
        f"Typisch bleibt der EKF konsistent (Median-NEES meist 4 bis 5). In **{pct(worst_spike['spike_ekf'])}** der Fahrten bei {worst_spike['d']:.0f} m Abstand und "
        f"{de(worst_spike['sigma_b'])} Grad Peilrauschen springt seine NEES aber über {C.NEES_SPIKE:.0f}; der UKF in {pct(worst_spike['spike_ukf'])}. "
        f"Bei {max(C.NEES_GRID_D):.0f} m Abstand und höchstens 5 Grad Peilrauschen kommt es kaum vor ({pct(max(c['spike_ekf'] for c in far))})."
    )

st.subheader("🔬 Experiment 3: Wenn die Tangente nichts sagt – nur Entfernung, Start genau auf der Achse")
st.caption(
    f"Feste Anordnung, {len(C.SWEEP_SEEDS)} feste Fahrten: nur die Entfernung zu einer Bake wird gemessen, die Anfangsschätzung liegt auf der Achse y = 0, die Wahrheit fährt bei y = "
    f"{C.RANGE_ONLY_START[1]:.0f} m. Auf dieser Achse ist die Entfernung in y-Richtung stationär, die Jacobi-Matrix hat dort keine y-Komponente; ab Sekunde "
    f"{C.SECOND_BEACON_FROM} kommt eine zweite Bake dazu, die die Mehrdeutigkeit auflöst."
)
sing = run_experiment("sing", lambda s: _sweep_singularity(0), settings, f"▶ Experiment 3 rechnen (2 × {len(C.SWEEP_SEEDS)} Fahrten)")
if sing is not None:
    s1, s2 = st.columns(2)
    s1.metric("EKF: Endfehler unter 15 m", pct(sing["ekf"]["ok"]), delta=f"Median {de(sing['ekf']['median'], 0)} m", delta_color="off")
    s2.metric("UKF: Endfehler unter 15 m", pct(sing["ukf"]["ok"]), delta=f"Median {de(sing['ukf']['median'], 0)} m", delta_color="off")
    st.info(
        "Die Tangente an der Singularität sagt dem EKF nichts über die y-Richtung, und er findet auch mit der zweiten Bake nicht zurück; "
        "die Sigma-Punkte des UKF liegen links und rechts der Achse und sehen den Unterschied. Das ist ein **konstruierter** Extremfall, kein typischer Fahrtverlauf."
    )

st.subheader("🔬 Experiment 4: Wie weit liegt die Gauß-Näherung vom exakten Ergebnis?")
st.caption(
    f"Eine einzelne Korrektur, gemittelt über {C.EXACT_DRAWS} Ziehungen aus dem Modell (Wahrheit aus dem Prior, Messung mit Rauschen). Referenz: der **exakte** Posterior-Mittelwert, "
    f"auf einem {C.EXACT_GRID} × {C.EXACT_GRID}-Gitter numerisch berechnet; er ist der bestmögliche Schätzer. Entfernungsrauschen {C.EXACT_SIGMA_R:.0f} m."
)
ex = run_experiment("exact", lambda s: _sweep_exact(0), settings, f"▶ Experiment 4 rechnen ({len(C.EXACT_CASES)} Fälle × {C.EXACT_DRAWS} Ziehungen)")
if ex is not None:
    tbl = "| Fall | Prior-Mittel, Streuung | Peilrauschen | EKF | UKF | exakt |\n|---|---|---|---|---|---|\n"
    for r in ex:
        tbl += (f"| {r['name']} | ({r['xy'][0]:.0f}; {r['xy'][1]:.0f}) m, {r['s']:.0f} m | {de(r['sigma_b'])} Grad | {de(r['ekf'], 2)} m | "
                f"{de(r['ukf'], 2)} m | **{de(r['exact'], 2)} m** |\n")
    st.markdown("Positionsfehler (RMSE) der Schätzung nach **einer** Messung:\n\n" + tbl)
    wide = ex[-1]
    near = ex[1]
    st.info(
        f"EKF und UKF liegen praktisch gleichauf: im ersten Fall ist der Unterschied zum exakten Ergebnis klein ({de(ex[0]['ekf'], 2)} gegen {de(ex[0]['exact'], 2)} m). Je näher die Bake und je schärfer die Peilung, "
        f"desto größer wird die Lücke ({de(near['ekf'], 2)} gegen {de(near['exact'], 2)} m), und bei sehr unsicherem Prior ({de(wide['ekf'], 1)} und {de(wide['ukf'], 1)} gegen "
        f"{de(wide['exact'], 1)} m) fehlt eine Gauß-Näherung die ringförmige Verteilung ganz. **Der UKF behebt das nicht**; er ist ein anderer Weg zur Gauß-Näherung, nicht zur exakten Lösung."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Messfunktion ist über die Unsicherheit fast linear** | Bei kleinem Abstand und kleinem Peilrauschen, oder bei grobem Peilrauschen, wird die Tangente schlecht; der EKF wird selten, aber stark überselbstsicher (Experiment 2). | UKF, Partikelfilter |
| **Die Verteilung bleibt gaußsch** | Eine Messung kann mehrere Orte zugleich erklären (Spiegelbild, zwei Straßen). Gauß-Filter, auch der UKF, mitteln dann zwischen den Möglichkeiten. | Partikelfilter, Hidden-Markov-Modelle |
| **Der Startwert ist brauchbar** | Ein Startwert auf der Singularität oder weit daneben führt den EKF in die Irre (Experiment 3). | UKF, mehrere Starthypothesen |
| **Rauschen gaußsch und bekannt, keine Ausreißer** | Ein Peilsprung wird wie jede Messung eingearbeitet. | Robuste Filter |
| **Nur die Vergangenheit** | Auch hier ohne Zukunftsblick. | Glätter |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Zustand $x = (x, y, v_x, v_y)^\top$, $x_{k+1} = F x_k + w_k$ wie im ersten Stück (zwei Achsen). Messung der Bake im Ursprung:
$$z_k = h(x_k) + \varepsilon_k,\quad h(x) = \begin{pmatrix}\sqrt{x^2 + y^2}\\ \operatorname{atan2}(y, x)\end{pmatrix},\quad \varepsilon_k \sim \mathcal N(0, R),\; R = \operatorname{diag}(\sigma_r^2, \sigma_b^2).$$

**EKF.** Linearisierung an der Vorhersage $\hat x^-$: $H = \partial h/\partial x\,|_{\hat x^-}$ mit
$$H = \begin{pmatrix} x/r & y/r & 0 & 0\\ -y/r^2 & x/r^2 & 0 & 0\end{pmatrix},\quad r = \sqrt{x^2+y^2}.$$
Dann wie im linearen Filter: $S = HP^-H^\top + R$, $K = P^-H^\top S^{-1}$, $\hat x = \hat x^- + K\,(z - h(\hat x^-))$ (Winkeldifferenz auf $(-\pi, \pi]$ gebracht), $P = (I-KH)P^-(I-KH)^\top + KRK^\top$.

**UKF.** Sigma-Punkte $\mathcal X_i$ aus $\hat x^-$ und $P^-$ (Cholesky), durch $h$ geschickt: $\mathcal Z_i = h(\mathcal X_i)$; Mittelwert $\hat z = \sum_i w^m_i \mathcal Z_i$,
$S = \sum_i w^c_i(\mathcal Z_i-\hat z)(\mathcal Z_i-\hat z)^\top + R$, Kreuzkovarianz $P_{xz} = \sum_i w^c_i(\mathcal X_i-\hat x^-)(\mathcal Z_i-\hat z)^\top$, $K = P_{xz}S^{-1}$.
Die Parameter $\alpha = 1$, $\beta = 2$, $\kappa = 1$ sind eine übliche Wahl der skalierten Unscented-Transformation; es gibt keine Feinabstimmung.

**Konsistenz.** $\mathrm{NEES} = (x - \hat x)^\top P^{-1}(x - \hat x)$ hat bei ehrlichem Filter den Erwartungswert 4 (Zustandsdimension).

**Literatur.** S. J. Julier, J. K. Uhlmann, H. F. Durrant-Whyte (2000), *A New Method for the Nonlinear Transformation of Means and Covariances in Filters and Estimators*,
IEEE Transactions on Automatic Control 45(3), 477–482.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
