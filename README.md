# Erweiterter Kalman-Filter: Entfernung und Peilung statt Position – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-ekf-demo.streamlit.app/)**

Drittes Stück der Linie **Zustandsschätzung und Filter** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt [Kalman-Filter](https://github.com/sebastian-hanisch/kalman-filter-demo) fort und behebt dessen benennbare Schwäche **lineares Modell**: hier misst ein Sensor an der Straße
nicht die Position, sondern **Entfernung und Peilung** (nichtlinear). Der Erweiterte Kalman-Filter (EKF) linearisiert; der Unscented Kalman-Filter (UKF) läuft als Vergleich mit.
Im Unterschied zu den Fall-Demos zeigt die Demo ein einziges Verfahren an einem wachsenden Beispiel.

## Warum dieses Problem

Entfernungs- und Winkelmessungen sind der Normalfall (Funk, Radar, Sichtpeilung); die Position folgt daraus nichtlinear. Die Kernfrage: **Wann trägt die Linearisierung nicht
mehr?** Und: lohnt sich der Aufwand des UKF? Die Demo beantwortet beides mit Messungen, nicht mit Lehrbuchaussagen.

## Modell

Zustand $(x, y, v_x, v_y)$, Konstante-Geschwindigkeit-Modell (zwei Achsen), eine Bake im Ursprung misst $h(x) = (\sqrt{x^2+y^2},\, \mathrm{atan2}(y, x))$. EKF mit Jacobi-Matrix
an der Vorhersage und Joseph-Form; UKF mit 9 Sigma-Punkten ($\alpha = 1$, $\beta = 2$, $\kappa = 1$). Formeln im 📐-Expander der App, Kern in `ekf_algorithm.py`.
Die Wahrheit wird mit dem Modell des Filters erzeugt; der Filter kennt die Rauschwerte.

## Methodik

Vier Experimente auf Knopfdruck (feste Fahrten, `SWEEP_SEEDS`/`GRID_SEEDS`):

1. Positionsfehler gegen das Peilrauschen: Rohwert, EKF, UKF.
2. Konsistenz (NEES) über Mindestabstand × Peilrauschen, 100 Fahrten je Zelle; typische NEES und Anteil der Fahrten mit Spitze.
3. Sonderfall nur Entfernung, Start auf der Singularität der Jacobi-Matrix.
4. Eine einzelne Korrektur gegen den **exakten** Posterior (numerische Integration auf dem Gitter).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Fahrten der festen Instanzen, Entfernungsrauschen 5 m, Fahrunruhe 0,2 m/s²; `tests/test_claims.py` rechnet jede nach.

| Frage | Befund |
|---|---|
| Was bringt das Filtern gegen die umgerechnete Rohposition? | Bei 2 Grad Peilrauschen Rohwert **7,6 m**, EKF **3,9 m**; bei 0,5 Grad **5,0 m** gegen **2,5 m**; bei 10 Grad **30 m** gegen **12 m**. |
| Ist der UKF beim Positionsfehler besser? | **Nein**: EKF und UKF liegen in allen fünf Peilrauschen-Stufen um weniger als 0,5 m auseinander, es gibt keine verlorenen Fahrten (Fehler über 50 m). |
| Ist der EKF ehrlich? | Typisch ja: die Median-NEES liegt für Peilrauschen bis 5 Grad bei 3,4 bis 4,6 (Soll 4). |
| Aber in Einzelfällen? | Bei 10 m Mindestabstand und 0,5 Grad Peilrauschen springt die NEES des EKF in **etwa 10 %** der Fahrten über 50, bei 20 m und 0,5 Grad in etwa 8 %; der UKF in höchstens 2 %. Bei 80 m Abstand und höchstens 5 Grad praktisch nie. |
| Grobe Peilung (10 Grad) | Die Median-NEES des EKF liegt bei etwa **5,0**, die des UKF bei etwa **4,3**: der EKF wird leicht überselbstsicher. |
| Anfangsschätzung auf der Singularität (nur Entfernung) | EKF: **0 %** der Fahrten enden unter 15 m Fehler (Median 439 m); UKF: **95 %** (Median 4,5 m). Konstruierter Extremfall. |
| Wie weit ist die Gauß-Näherung vom exakten Ergebnis? | Nach einer Messung: weit weg 6,2 m gegen 6,2 m exakt (praktisch gleich); nah 6,1 gegen 4,7 m; sehr unsicher auf der Achse EKF **14 m**, UKF **15 m**, exakt **5,5 m**. Der UKF ist dabei nicht näher am exakten Ergebnis. |

## Ehrliche Grenzen

- **Die Wahrheit ist das Modell.** Fahrt und Filter teilen dasselbe Modell und bekannte Rauschwerte; es ist kein echter Sensordatensatz im Spiel.
- **Seltene Ereignisse brauchen viele Fahrten.** Die Einbrüche der NEES betreffen wenige Prozent der Fahrten; deshalb 100 Fahrten je Zelle und der Anteil statt des Mittelwerts. Bei 100 Fahrten sind einzelne Prozentpunkte Zufall.
- **Die Spitzen schlagen im Positionsfehler kaum durch.** Der RMSE bleibt unauffällig; die Gefahr ist die falsche Selbstsicherheit.
- **Die Parameter des UKF sind eine übliche Wahl, nicht abgestimmt** ($\alpha = 1$, $\kappa = 1$). Die Gegenprobe über andere Werte änderte die Aussage nicht, aber sie ist nicht Teil der Demo.
- **Experiment 3 ist konstruiert.** Die Anfangsschätzung liegt absichtlich auf der Achse, auf der die Jacobi-Matrix keine y-Komponente hat.
- **Nur Gauß-Näherungen.** Beide Filter bleiben bei einer Gauß-Verteilung; bei mehrgipfligen Verteilungen (Experiment 4, letzter Fall) fehlt ihnen die Form.

## Verifikation

- **Kern:** Winkelumbruch, Messmodell und Jacobi-Matrix gegen Handrechnung, Sigma-Punkte reproduzieren Mittelwert und Kovarianz, Korrektur gegen ein Handbeispiel (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_ekf.py`, anderer Rechenweg): Jacobi-Matrix gegen finite Differenzen; Unscented-Transformation gegen Monte-Carlo-Momente; **Gitter-Posterior gegen Importance-Sampling**; beide Filter gegen den exakten Posterior über viele Ziehungen; mittlere NEES über 150 Fahrten bei mäßiger Nichtlinearität gleich 4; angezeigte Kennzahlen gegen unabhängige Neuberechnung.
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, Experiment-Knöpfe) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `ekf_algorithm.py` | Kern: Modell, Jacobi-Matrix, EKF- und UKF-Korrektur, Sigma-Punkte, NEES, Filterlauf |
| `ekf_scenario.py` | Fahrt und Messungen (ein Generator, Regler skalieren denselben Zufall) |
| `ekf_evaluation.py` | Kennzahlen, Sweeps über feste Fahrten, Singularität, exakter Posterior |
| `ekf_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `ekf_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `ekf_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ekf_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Eigenes UKF-Stück:** Die Vorab-Messung zeigte keinen Unterschied im Positionsfehler; der UKF läuft deshalb als Vergleich in dieser Demo mit.
- **Iterierter EKF und Gauß-Summen:** nicht umgesetzt.
- **Echte Sensordaten** und ein Straßennetz: kein Datensatz eingebunden.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
