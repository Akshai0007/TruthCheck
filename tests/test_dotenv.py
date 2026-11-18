import os

from truthcheck import env


def test_dotenv_loads_values_but_never_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("TRUTHCHECK_NO_DOTENV", raising=False)
    monkeypatch.setenv("TRUTHCHECK_MODEL", "from-shell")
    f = tmp_path / ".env"
    f.write_text('# comment\nTRUTHCHECK_PROVIDER=groq\nexport TRUTHCHECK_API_KEY="abc 123"\n'
                 "TRUTHCHECK_MODEL=from-file\nEMPTY=\n\nnot a line\n", encoding="utf-8")
    monkeypatch.delenv("TRUTHCHECK_PROVIDER", raising=False)
    monkeypatch.delenv("TRUTHCHECK_API_KEY", raising=False)
    assert env.load_dotenv(f) == 2
    assert os.environ["TRUTHCHECK_PROVIDER"] == "groq"
    assert os.environ["TRUTHCHECK_API_KEY"] == "abc 123"        # quotes stripped
    assert os.environ["TRUTHCHECK_MODEL"] == "from-shell"        # real env wins
    assert "EMPTY" not in os.environ
    for k in ("TRUTHCHECK_PROVIDER", "TRUTHCHECK_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_missing_file_and_opt_out_are_harmless(tmp_path, monkeypatch):
    monkeypatch.delenv("TRUTHCHECK_NO_DOTENV", raising=False)
    assert env.load_dotenv(tmp_path / "nope.env") == 0
    (tmp_path / ".env").write_text("TRUTHCHECK_PROVIDER=groq", encoding="utf-8")
    monkeypatch.setenv("TRUTHCHECK_NO_DOTENV", "1")
    assert env.load_dotenv(tmp_path / ".env") == 0
