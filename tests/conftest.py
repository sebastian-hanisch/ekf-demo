"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_state():
    """Position (3, 4) m von der Bake, Geschwindigkeit 0, Kovarianz Einheitsmatrix.

    Handrechnung: Entfernung 5 m, Peilung atan2(4, 3); Jacobi-Matrix Zeile 1 (0,6; 0,8; 0; 0), Zeile 2 (-4/25; 3/25; 0; 0).
    Korrektur mit Messung exakt gleich der Vorhersage und R = diag(1; 0,04): S = diag(2; 0,08), P_xx = P_yy = 0,5 (siehe Test).
    """
    return np.array([3.0, 4.0, 0.0, 0.0]), np.eye(4)
