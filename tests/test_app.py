import pathlib

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")


def test_app_demo_flow_runs_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.sidebar.selectbox[0].select("Demo (offline, sample only)").run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Trust score"] == "67%" and metrics["Contradicted"] == "1"
    assert any("Fleming" in m.value for m in at.markdown)


def test_app_reports_setup_problem_instead_of_crashing():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.selectbox[0].select("Groq").run()   # no model, no key
    at.text_area[0].input("Paris is in France.").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Could not run the check" in e.value for e in at.error)
