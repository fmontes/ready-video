from ready_video.agent import select_agent_backend


def test_auto_agent_checks_claude_codex_opencode_then_none(monkeypatch):
    checked = []

    def fake_which(name):
        checked.append(name)
        return None

    monkeypatch.setattr("shutil.which", fake_which)

    selection = select_agent_backend("auto")

    assert selection.name == "none"
    assert checked == ["claude", "codex", "opencode"]


def test_auto_agent_selects_first_available(monkeypatch):
    def fake_which(name):
        return f"/bin/{name}" if name == "codex" else None

    monkeypatch.setattr("shutil.which", fake_which)

    selection = select_agent_backend("auto")

    assert selection.name == "codex"
    assert selection.path == "/bin/codex"
