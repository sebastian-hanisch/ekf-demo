"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt."""

import ekf_constants as C
import ekf_evaluation as ev
import ekf_presets as P


def _settings_from(p):
    return ev.Settings(sigma_r=p["sr"], sigma_b=p["sb"], d_min=p["d"], q=p["q"], seed=p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, p in C.PRESETS.items():
        assert C.SIGMA_R_MIN <= p["sr"] <= C.SIGMA_R_MAX and C.SIGMA_B_MIN <= p["sb"] <= C.SIGMA_B_MAX, name
        assert C.D_MIN <= p["d"] <= C.D_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert _settings_from(C.PRESETS["Standard (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_clamps_numbers():
    d = P.SETTING_SPECS["d_slider"]
    assert P.clamp_value(d, 1e6) == C.D_MAX and P.clamp_value(d, -1.0) == C.D_MIN


def test_the_presets_show_what_their_help_text_claims():
    base = C.PRESETS["Standard (Voreinstellung)"]
    assert C.PRESETS["Scharfer Sensor, naher Vorbeiflug"]["sb"] < base["sb"] and C.PRESETS["Scharfer Sensor, naher Vorbeiflug"]["d"] < base["d"]
    assert C.PRESETS["Grobe Peilung"]["sb"] > base["sb"]
    assert C.PRESETS["Weit entfernt vorbei"]["d"] > base["d"]
