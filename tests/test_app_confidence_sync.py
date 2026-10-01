"""
app.py's hardcoded REAL_CORPUS_TAU must match what
eval/run_confidence_real_eval.py actually computes.

app.py is a Streamlit script, not an importable module -- it calls
st.set_page_config() and renders UI at module scope, so plain `import app`
outside a Streamlit runtime is unsafe (this is exactly why app.py itself is
tested via streamlit.testing.v1.AppTest elsewhere, not by import). This file
follows test_paper_consistency.py's approach instead: read app.py as text and
check the hardcoded constant against the freshly recomputed one, the same way
that file guards paper.tex's numbers against the eval scripts. A cached
threshold that silently drifts from its own calibration script is a
regression this project has already been burned by once (the toy-corpus gate
misapplied to the real corpus); this test exists so the real-corpus gate
cannot repeat it.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")


@pytest.fixture(scope="module")
def app_source():
    if not os.path.exists(APP):
        pytest.skip("app.py not present in this checkout")
    with open(APP) as f:
        return f.read()


def test_real_corpus_tau_matches_the_eval_script(app_source, confidence_real):
    m = re.search(r"REAL_CORPUS_TAU\s*=\s*([0-9.]+)", app_source)
    assert m, "app.py must define REAL_CORPUS_TAU"
    app_tau = float(m.group(1))
    assert app_tau == pytest.approx(confidence_real["tau"], abs=0.001), (
        f"app.py's REAL_CORPUS_TAU ({app_tau}) has drifted from "
        f"eval/run_confidence_real_eval.py's computed tau "
        f"({confidence_real['tau']}) -- update app.py's constant to match.")


def test_app_gates_tab6_rather_than_skipping(app_source):
    # Regression guard for the earlier state: Tab 6 used to skip gating
    # entirely and say so on screen. Confirms that caveat text is gone and
    # a real gate call is wired into the button handler.
    assert "get_confidence_gate_real()" in app_source
    assert "No confidence gate is applied on this tab" not in app_source
