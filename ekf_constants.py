"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Zeitachse: ein Schritt = eine Sekunde, 60 Schritte
DT = 1.0
T_STEPS = 60
N_BURN = 5                        # Einschwingzeit: Kennzahlen zählen erst ab diesem Schritt

# Fahrt: gerade Straße (x-Richtung), die Bake steht im Ursprung, die Straße läuft in Mindestabstand d an ihr vorbei
START_X = -300.0                  # Startposition entlang der Straße in m
SPEED = 10.0                      # Sollgeschwindigkeit in m/s
V_PRIOR_STD = 10.0                # Vorwissen des Filters über die Geschwindigkeit je Achse (weit gefasst)
FAIL_RMSE = 50.0                  # Fahrt gilt als „verloren“, wenn der Positionsfehler (RMSE) darüber liegt
NEES_BOUND_95 = 9.4877            # 95-%-Quantil der Chi-Quadrat-Verteilung mit 4 Freiheitsgraden
STATE_DIM = 4                     # Zustand (x, y, vx, vy): Soll-Wert der mittleren NEES

# Reglergrenzen
SIGMA_R_MIN, SIGMA_R_MAX, DEFAULT_SIGMA_R = 1.0, 15.0, 5.0
SIGMA_B_MIN, SIGMA_B_MAX, DEFAULT_SIGMA_B = 0.5, 10.0, 2.0      # Peilrauschen in Grad
D_MIN, D_MAX, DEFAULT_D = 5.0, 100.0, 40.0
Q_MIN, Q_MAX, DEFAULT_Q = 0.05, 0.8, 0.2
SEED_MAX, DEFAULT_SEED = 999999, 35

# feste Instanzen für alle Sweeps: nie ein einzelner Seed als Befund
SWEEP_SEEDS = tuple(range(100000, 100040))
GRID_SEEDS = tuple(range(200000, 200100))   # für die Konsistenz-Tabelle: seltene Einbrüche brauchen mehr Fahrten
NEES_SPIKE = 50.0                           # eine Fahrt zählt als „Einbruch“, wenn ihre NEES einmal darüber liegt (Soll 4)
SIGMA_B_SWEEP = (0.5, 1.0, 2.0, 5.0, 10.0)
NEES_GRID_D = (10.0, 20.0, 40.0, 80.0)
NEES_GRID_SIGMA_B = (0.5, 2.0, 5.0, 10.0)

# Sonderfall „nur Entfernung“ (Experiment 3), feste Anordnung
RANGE_ONLY_SIGMA = 3.0
RANGE_ONLY_START = (-100.0, 60.0)
RANGE_ONLY_SPEED = 5.0
RANGE_ONLY_STEPS = 70
RANGE_ONLY_Y_PRIOR_STD = 60.0
SECOND_BEACON = (0.0, 30.0)
SECOND_BEACON_FROM = 30
RANGE_ONLY_OK_M = 15.0

COLOR_TRUTH = "#2F6B65"
COLOR_RAW = "rgba(150,150,150,0.8)"
COLOR_EKF = "#4c78a8"
COLOR_UKF = "#c97b2a"
COLOR_BEACON = "#b04a4a"

PRESETS = {
    "Standard (Voreinstellung)": {"sr": DEFAULT_SIGMA_R, "sb": DEFAULT_SIGMA_B, "d": DEFAULT_D, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Scharfer Sensor, naher Vorbeiflug": {"sr": DEFAULT_SIGMA_R, "sb": 0.5, "d": 20.0, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Grobe Peilung": {"sr": DEFAULT_SIGMA_R, "sb": 10.0, "d": DEFAULT_D, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
    "Weit entfernt vorbei": {"sr": DEFAULT_SIGMA_R, "sb": DEFAULT_SIGMA_B, "d": 100.0, "q": DEFAULT_Q, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard (Voreinstellung)": "Entfernung auf 5 m, Peilung auf 2 Grad genau, die Straße läuft in 40 m Abstand an der Bake vorbei.",
    "Scharfer Sensor, naher Vorbeiflug": "Peilung auf 0,5 Grad genau, nur 20 m Abstand: hier ist die Messung am stärksten gekrümmt gegenüber ihrer Genauigkeit.",
    "Grobe Peilung": "Peilung nur auf 10 Grad genau: das Messrauschen überdeckt die Krümmung.",
    "Weit entfernt vorbei": "100 m Abstand: die Messung ist über den Bereich der Unsicherheit fast linear.",
}

# Experiment 4: eine einzelne Korrektur gegen den exakten Posterior (Gitter), Ziehungen aus dem Modell
EXACT_DRAWS = 100
EXACT_GRID = 400
EXACT_SIGMA_R = 5.0
EXACT_CASES = (
    ("Weit weg, mäßig unsicher", (100.0, 40.0), 10.0, 2.0),
    ("Nah, mäßig unsicher", (30.0, 20.0), 10.0, 2.0),
    ("Sehr nah, scharfe Peilung", (15.0, 5.0), 8.0, 1.0),
    ("Sehr unsicher, auf der Achse", (80.0, 0.0), 30.0, 1.0),
)
