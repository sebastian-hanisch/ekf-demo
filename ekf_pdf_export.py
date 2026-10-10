"""PDF-Export der aktuellen Auswertung (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF


def _num(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def build_pdf(a):
    """Einseitiger Bericht: Einstellungen, Kennzahlen, Verdikt. Gibt die Bytes der PDF-Datei zurück."""
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Erweiterter Kalman-Filter: Entfernung und Peilung", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    rows = [
        ("Entfernungsrauschen (wahr)", f"{_num(s.sigma_r, 1)} m"),
        ("Peilrauschen (wahr)", f"{_num(s.sigma_b, 1)} Grad"),
        ("Mindestabstand zur Bake", f"{_num(s.d_min, 0)} m"),
        ("Beschleunigungsrauschen", f"{_num(s.q, 2)} m/s²"),
        ("Zufalls-Seed", str(s.seed)),
    ]
    for k, v in rows:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Kennzahlen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    metrics = [
        ("Positionsfehler Rohwert (RMSE)", f"{_num(a.rmse_raw)} m"),
        ("Positionsfehler EKF (RMSE)", f"{_num(a.rmse_ekf)} m"),
        ("Positionsfehler UKF (RMSE)", f"{_num(a.rmse_ukf)} m"),
        ("Mittlere NEES EKF / UKF (Soll 4)", f"{_num(a.nees_ekf)} / {_num(a.nees_ukf)}"),
        ("Größte NEES EKF / UKF", f"{_num(a.peak_ekf, 1)} / {_num(a.peak_ukf, 1)}"),
    ]
    for k, v in metrics:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Synthetische Fahrt mit bekanntem Modell; die Zahlen gelten für dieses Szenario und diese Einstellungen.")
    return bytes(pdf.output())
