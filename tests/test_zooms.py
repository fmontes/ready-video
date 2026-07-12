import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ready_video.agent import AgentSelection
from ready_video.config import Config
from ready_video.errors import ZoomValidationError
from ready_video.transcript import Transcript, TranscriptSegment, Word
from ready_video.zooms import CliBackend, Zoom, ZoomPlan, generate_zooms, plan_zooms, validate_agent_response, validate_zoom_plan, validate_zooms


def test_backend_none_writes_empty_zooms(tmp_path):
    selection = AgentSelection("none", None, True, "disabled")
    output = tmp_path / "zooms.json"

    payload = generate_zooms(selection, {"segments": []}, output)

    assert payload["zooms"] == []
    assert output.exists()


def test_validate_zooms_normalizes_scale_alias():
    payload = validate_zooms(
        {"zooms": [{"start": 1, "end": 2, "zoom": 1.4}]},
        media_duration=3,
    )

    assert payload["zooms"] == [{"start": 1.0, "end": 2.0, "scale": 1.4}]


def test_validate_zooms_rejects_out_of_bounds():
    with pytest.raises(ZoomValidationError, match="ends after media duration"):
        validate_zooms({"zooms": [{"start": 1, "end": 5, "scale": 1.2}]}, media_duration=4)


def test_validate_agent_response_parses_fenced_json_with_inclusive_end_and_clamps_intensity():
    config = _config(min_duration_s=0.1, min_gap_s=0)
    transcript = _transcript([(0, 0.4), (0.5, 1.2), (1.3, 1.9)])

    plan = validate_agent_response(
        """```json
{"zooms": [{"rank": 1, "start_word_index": 0, "end_word_index": 1, "intensity": 2.0, "reason": "beat"}]}
```""",
        transcript,
        config,
        backend="codex",
    )

    assert plan.backend == "codex"
    assert len(plan.zooms) == 1
    assert plan.zooms[0].start == 0.0
    assert plan.zooms[0].end == 1.2
    assert plan.zooms[0].end_word_index == 1
    assert plan.zooms[0].intensity == 1.15


def test_validate_agent_response_rejects_duplicate_ranks_and_invalid_indices():
    config = _config(min_duration_s=0.1, min_gap_s=0)
    transcript = _transcript([(0, 0.4), (0.5, 1.0)])

    plan = validate_agent_response(
        _agent_response(
            zooms=[
                {"rank": 1, "start_word_index": 0, "end_word_index": 0, "intensity": 1.12, "reason": "ok"},
                {"rank": 1, "start_word_index": 1, "end_word_index": 1, "intensity": 1.12, "reason": "duplicate"},
                {"rank": 2, "start_word_index": -1, "end_word_index": 0, "intensity": 1.12, "reason": "bad"},
                {"rank": 3, "start_word_index": 0, "end_word_index": 7, "intensity": 1.12, "reason": "bad"},
                {"rank": 4, "start_word_index": 1, "end_word_index": 0, "intensity": 1.12, "reason": "bad"},
            ]
        ),
        transcript,
        config,
    )

    assert [zoom.rank for zoom in plan.zooms] == [1]
    assert [item["reason"] for item in plan.rejected] == [
        "duplicate_or_nonpositive_rank",
        "invalid_word_indices",
        "invalid_word_indices",
        "invalid_word_indices",
    ]


def test_validate_agent_response_truncates_long_candidates_to_max_duration():
    config = _config(min_duration_s=0.1, max_duration_s=2.5, min_gap_s=0)
    transcript = _transcript([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])

    plan = validate_agent_response(
        _agent_response(
            zooms=[
                {"rank": 1, "start_word_index": 0, "end_word_index": 4, "intensity": 1.12, "reason": "long"},
            ]
        ),
        transcript,
        config,
    )

    assert len(plan.zooms) == 1
    assert plan.zooms[0].start == 0.0
    assert plan.zooms[0].end == 2.0
    assert plan.zooms[0].end_word_index == 1


def test_validate_agent_response_extends_short_candidates_to_min_duration():
    config = _config(min_duration_s=1.0, max_duration_s=3.0, min_gap_s=0)
    transcript = _transcript([(0, 0.4), (0.5, 0.9), (1.0, 1.4)])

    plan = validate_agent_response(
        _agent_response(
            zooms=[
                {"rank": 1, "start_word_index": 1, "end_word_index": 1, "intensity": 1.02, "reason": "short"},
            ]
        ),
        transcript,
        config,
    )

    assert len(plan.zooms) == 1
    assert plan.zooms[0].start == 0.0
    assert plan.zooms[0].end == 1.4
    assert plan.zooms[0].start_word_index == 0
    assert plan.zooms[0].end_word_index == 2
    assert plan.zooms[0].intensity == 1.08


def test_validate_agent_response_enforces_overlaps_and_boundary_gaps():
    config = _config(min_duration_s=0.1, max_duration_s=2.0, min_gap_s=1.0)
    transcript = _transcript([(0, 0.5), (0.75, 1.25), (1.4, 1.8), (2.0, 2.5), (3.5, 4.0)])

    plan = validate_agent_response(
        _agent_response(
            zooms=[
                {"rank": 1, "start_word_index": 0, "end_word_index": 0, "intensity": 1.12, "reason": "first"},
                {"rank": 2, "start_word_index": 1, "end_word_index": 1, "intensity": 1.12, "reason": "gap too small"},
                {"rank": 3, "start_word_index": 0, "end_word_index": 2, "intensity": 1.12, "reason": "overlap"},
                {"rank": 4, "start_word_index": 4, "end_word_index": 4, "intensity": 1.12, "reason": "far enough"},
            ]
        ),
        transcript,
        config,
    )

    assert [zoom.rank for zoom in plan.zooms] == [1, 4]
    assert [item["reason"] for item in plan.rejected] == ["overlap_or_gap", "overlap_or_gap"]


def test_validate_agent_response_raises_zoom_validation_error_for_malformed_json():
    config = _config()
    transcript = _transcript([(0, 0.5)])

    with pytest.raises(ZoomValidationError, match="malformed agent zoom response"):
        validate_agent_response("not json", transcript, config)


def test_validate_saved_zoom_plan_accepts_exact_word_boundaries():
    config = _config(min_duration_s=0.1, max_duration_s=2.0, min_gap_s=0)
    transcript = _transcript([(0, 0.5), (0.6, 1.2)])
    plan = ZoomPlan(
        backend="manual",
        zooms=[Zoom(start=0.0, end=0.5, intensity=1.12, rank=1, start_word_index=0, end_word_index=0)],
    )

    validate_zoom_plan(plan, transcript, config, edited_duration=1.2)


def test_validate_saved_zoom_plan_rejects_manual_edits_off_word_boundaries():
    config = _config(min_duration_s=0.1, max_duration_s=2.0, min_gap_s=0)
    transcript = _transcript([(0, 0.5), (0.6, 1.2)])
    plan = ZoomPlan(
        backend="manual",
        zooms=[Zoom(start=0.1, end=0.5, intensity=1.12, rank=1, start_word_index=0, end_word_index=0)],
    )

    with pytest.raises(ZoomValidationError) as error:
        validate_zoom_plan(plan, transcript, config, edited_duration=1.2)

    assert "not_on_word_boundaries" in (error.value.details or "")


def test_validate_saved_zoom_plan_rejects_spacing_and_interpolated_boundaries():
    config = _config(min_duration_s=0.1, max_duration_s=2.0, min_gap_s=1.0)
    transcript = _transcript([(0, 0.5), (0.25, 1.2), (1.5, 2.0)])
    transcript.words[0].timing = "interpolated"
    transcript.words[1].timing = "aligned"
    plan = ZoomPlan(
        backend="manual",
        zooms=[
            Zoom(start=0.0, end=0.5, intensity=1.12, rank=1, start_word_index=0, end_word_index=0),
            Zoom(start=0.25, end=1.2, intensity=1.12, rank=2, start_word_index=1, end_word_index=1),
        ],
    )

    with pytest.raises(ZoomValidationError) as error:
        validate_zoom_plan(plan, transcript, config, edited_duration=2.0)

    details = error.value.details or ""
    assert "interpolated_boundary" in details
    assert "overlap_or_gap" in details


def test_plan_zooms_falls_back_with_warning_for_malformed_agent_response(tmp_path, monkeypatch):
    config = _config()
    transcript = _transcript([(0, 0.5)])
    backend = _MalformedBackend()
    monkeypatch.setattr("ready_video.zooms.select_backend", lambda config: backend)

    plan = plan_zooms(transcript, tmp_path / "zooms.json", config)

    assert plan.backend == "codex"
    assert plan.zooms == []
    assert plan.warnings[0]["code"] == "ZOOM_PLANNING_FAILED"
    assert "malformed agent zoom response" in plan.warnings[0]["message"]


def test_plan_zooms_announces_cli_and_prompts_for_schema(tmp_path, monkeypatch, capsys):
    config = _config(min_duration_s=0.1, min_gap_s=0)
    transcript = _transcript([(0, 0.5)])
    backend = _RecordingBackend()
    monkeypatch.setattr("ready_video.zooms.select_backend", lambda config: backend)

    plan = plan_zooms(transcript, tmp_path / "zooms.json", config)

    assert plan.backend == "codex"
    assert "selected local agent CLI 'codex'" in capsys.readouterr().err
    assert backend.prompt_text is not None
    assert "Return strict JSON only" in backend.prompt_text
    assert '"required": ["zooms"]' in backend.prompt_text


@pytest.mark.parametrize(
    ("backend_name", "expected_schema_flag"),
    [("claude", "--json-schema"), ("codex", "--output-schema"), ("opencode", None)],
)
def test_cli_backend_invocation_contracts(tmp_path, monkeypatch, backend_name, expected_schema_flag):
    prompt_path = tmp_path / "prompt.txt"
    schema_path = tmp_path / "schema.json"
    prompt_path.write_text("Segment 0: 0:secret-word")
    schema_path.write_text("{}")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        assert "secret-word" not in " ".join(map(str, args))
        if backend_name in {"claude", "codex"}:
            assert kwargs["input"] == prompt_path.read_text()
        if backend_name == "codex":
            response_path = Path(args[args.index("--output-last-message") + 1])
            response_path.write_text(_agent_response(zooms=[]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout=_agent_response(zooms=[]), stderr="")

    monkeypatch.setattr("ready_video.zooms.subprocess.run", fake_run)
    backend = CliBackend(backend_name)
    backend.executable = f"/bin/{backend_name}"

    response = backend.plan_zooms(prompt_path, schema_path, 10)

    args = calls[0][0]
    assert args[0] == f"/bin/{backend_name}"
    if expected_schema_flag:
        assert args[args.index(expected_schema_flag) + 1] == str(schema_path)
    if backend_name == "opencode":
        assert args[args.index("--file") + 1] == str(prompt_path)
    assert json.loads(response) == {"zooms": []}


def test_plan_zooms_retries_once_with_validation_summary(tmp_path, monkeypatch):
    config = _config(min_duration_s=0.1, min_gap_s=0)
    transcript = _transcript([(0, 0.5)])
    backend = _RetryBackend()
    monkeypatch.setattr("ready_video.zooms.select_backend", lambda config: backend)

    plan = plan_zooms(transcript, tmp_path / "zooms.json", config)

    assert [call.name for call in backend.calls] == ["agent-prompt.txt", "agent-prompt-retry.txt"]
    assert "previous response could not be parsed" in backend.calls[1].read_text()
    assert plan.backend == "codex"
    assert len(plan.zooms) == 1
    assert plan.warnings[0]["code"] == "ZOOM_PLANNING_RETRIED"
    assert (tmp_path / "agent-response.txt").exists()
    assert (tmp_path / "agent-response-retry.txt").exists()


class _MalformedBackend:
    name = "codex"

    def plan_zooms(self, prompt_path, schema_path, timeout_s):
        return "not json"


class _RecordingBackend:
    name = "codex"

    def __init__(self):
        self.prompt_text = None

    def plan_zooms(self, prompt_path, schema_path, timeout_s):
        self.prompt_text = prompt_path.read_text()
        return _agent_response(zooms=[{"rank": 1, "start_word_index": 0, "end_word_index": 0, "intensity": 1.12, "reason": "beat"}])


class _RetryBackend:
    name = "codex"

    def __init__(self):
        self.calls = []

    def plan_zooms(self, prompt_path, schema_path, timeout_s):
        self.calls.append(prompt_path)
        if len(self.calls) == 1:
            return "not json"
        return _agent_response(zooms=[{"rank": 1, "start_word_index": 0, "end_word_index": 0, "intensity": 1.12, "reason": "retry"}])


def _config(**zoom_overrides):
    config = Config().resolved()
    for key, value in zoom_overrides.items():
        setattr(config.zooms, key, value)
    return config


def _agent_response(**payload):
    return json.dumps(payload)


def _transcript(spans):
    words = [Word(i=i, w=f"w{i}", start=start, end=end) for i, (start, end) in enumerate(spans)]
    return Transcript(
        duration=words[-1].end if words else 0,
        words=words,
        segments=[
            TranscriptSegment(
                id=0,
                text=" ".join(word.w for word in words),
                start=words[0].start if words else 0,
                end=words[-1].end if words else 0,
                word_range=(0, len(words) - 1),
            )
        ]
        if words
        else [],
    )
