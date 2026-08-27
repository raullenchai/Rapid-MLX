# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the issue #2222 weather-routing eval coverage.

We cannot start a model here (CI has no resident model), so these tests exercise
the pure, decision-relevant helpers of ``evals/run_eval.py`` — the tool-subset
resolver and the forbidden-tool rejection logic — and the scenario's own
configuration. They prove the NEGATIVE paths that guard the #2222 contract
(weather selected, never web_search):
  * a response that calls weather AND web_search must be rejected,
  * a final completion that calls web_search must be rejected even when its text
    carries a result marker,
  * a final completion that calls ANY tool (even weather) is not a final answer,
  * an empty string argument must not match an expected argument,
  * a weather-only response must pass,
  * a malformed tools config must fail fast.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EVAL = _REPO_ROOT / "evals" / "run_eval.py"
_PROMPTS = _REPO_ROOT / "evals" / "prompts" / "tool_calling.json"


def _load():
    spec = importlib.util.spec_from_file_location("eval_run_eval", _EVAL)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def re():
    return _load()


def _tc():
    data = json.loads(_PROMPTS.read_text())
    return next(c for c in data if c["id"] == "tc31-weather-explicit")


def _call(name):
    return {
        "id": "c",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


class TestForbiddenToolRejection:
    def test_weather_plus_web_search_first_turn_rejected(self, re):
        # issue #2222: "use weather, not web search" — calling web_search at all,
        # even after a correct weather call, violates the contract.
        sc = _tc()
        assert re._forbidden_tool_names(
            [_call("weather"), _call("web_search")], sc["forbid_tools"]
        ) == ["web_search"]

    def test_weather_only_ok(self, re):
        sc = _tc()
        assert re._forbidden_tool_names([_call("weather")], sc["forbid_tools"]) == []

    def test_unrelated_extra_tool_not_forbidden(self, re):
        sc = _tc()
        # forbid_tools only bans web_search; an unrelated extra is a separate concern.
        assert (
            re._forbidden_tool_names(
                [_call("weather"), _call("exec")], sc["forbid_tools"]
            )
            == []
        )

    def test_empty_or_none_calls_no_forbidden(self, re):
        sc = _tc()
        assert re._forbidden_tool_names([], sc["forbid_tools"]) == []
        assert re._forbidden_tool_names(None, sc["forbid_tools"]) == []

    def test_final_completion_calling_forbidden_tool_rejected(self, re):
        # The exact round-5 shape: final text carries a result marker AND the final
        # completion calls web_search. Must be rejected.
        sc = _tc()
        final_calls = [_call("weather"), _call("web_search")]
        assert re._forbidden_tool_names(final_calls, sc["forbid_tools"]) == [
            "web_search"
        ]

    def test_scenario_forbids_web_search(self, re):
        sc = _tc()
        assert sc["forbid_tools"] == ["web_search"]
        assert sc["verify_final_text"] is True
        assert sc["first_call_stream"] is False


class TestResolveTools:
    def test_tc31_resolves_both_desktop_schemas(self, re):
        sc = _tc()
        names = [t["function"]["name"] for t in re._resolve_tools(sc)]
        assert names == ["weather", "web_search"]
        web = next(
            t for t in re._resolve_tools(sc) if t["function"]["name"] == "web_search"
        )
        # The Desktop-authentic web_search carries the weather cross-reference.
        assert "Do not use it for current weather" in web["function"]["description"]

    def test_absent_tools_defaults_to_shared_list(self, re):
        tools = re._resolve_tools({})
        assert len(tools) == len(re.TOOLS)
        assert all(t["function"]["name"] != "weather" for t in tools)

    def test_unknown_tool_name_fails_fast(self, re):
        with pytest.raises(ValueError, match="unknown tool name"):
            re._resolve_tools({"id": "x", "tools": ["weather", "web_seach"]})

    @pytest.mark.parametrize("bad", [[], {"name": "weather"}, 123, "weather"])
    def test_malformed_tools_fails_fast(self, re, bad):
        with pytest.raises(ValueError, match="malformed tools"):
            re._resolve_tools({"id": "x", "tools": bad})

    def test_malformed_schema_dict_fails_fast(self, re):
        # A dict that is not a well-formed OpenAI tool schema (no function.name)
        # must fail here, not silently reach the request path.
        with pytest.raises(ValueError, match="function.name"):
            re._resolve_tools(
                {
                    "id": "x",
                    "tools": [{"function": {"name": "weather"}}, {"name": "oops"}],
                }
            )
        with pytest.raises(ValueError, match="function.name"):
            re._resolve_tools({"id": "x", "tools": [{"type": "function"}]})

    def test_wellformed_schema_dict_accepted(self, re):
        # A real OpenAI function-tool schema (type + name + parameters) passes
        # through verbatim.
        schema = {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
        out = re._resolve_tools({"id": "x", "tools": [schema]})
        assert out[0]["function"]["name"] == "weather"

    @pytest.mark.parametrize(
        "partial",
        [
            {"function": {"name": "weather"}},  # no type == function
            {"type": "function", "function": {"name": "weather"}},  # no parameters
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {}},
            },  # non-string params.type
        ],
    )
    def test_partial_schema_dict_fails_fast(self, re, partial):
        with pytest.raises(ValueError, match="function-tool dict"):
            re._resolve_tools({"id": "x", "tools": [partial]})

    def test_explicit_null_tools_fails_fast(self, re):
        # "tools": null is an explicit override, not an absent key — broadening
        # to the shared list silently would violate the fail-fast policy.
        with pytest.raises(ValueError, match="explicit null"):
            re._resolve_tools({"id": "x", "tools": None})


class TestFuzzyArgMatching:
    def test_nonempty_substring_matches(self, re):
        # A real, usable value still matches the expected location.
        assert (
            re.fuzzy_match_args({"location": "tokyo"}, {"location": "Tokyo, Japan"})
            == 1.0
        )

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_empty_actual_does_not_match(self, re, empty):
        # `"" in "tokyo"` is true, so an empty location would score a perfect
        # match and let tc31 pass without a usable weather request (the real
        # tool rejects an empty location). Fail closed on unset/blank.
        assert (
            re.fuzzy_match_args({"location": "tokyo"}, {"location": empty or ""}) == 0.0
        )

    def test_missing_key_does_not_match(self, re):
        assert re.fuzzy_match_args({"location": "tokyo"}, {}) == 0.0


def _tool_call(name, args='{"location": "Tokyo"}'):
    return {
        "id": "call_eval",
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _nons(a, *names):
    """A non-streaming chat_request response: text ``a`` plus the named tools."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": a,
                    "tool_calls": [_tool_call(n) for n in names] or None,
                }
            }
        ]
    }


@pytest.fixture
def suite_with_mock(re, monkeypatch, tmp_path):
    """Drive the REAL run_tool_calling_suite over a single tc31-shaped scenario,
    with chat_request stubbed so no model/network is needed. The suite reads
    PROMPTS_DIR/tool_calling.json at call time, so we point it at a temp prompts
    dir containing only that one scenario; the returned details[0] corresponds to
    it 1:1.
    """
    import copy

    sc = copy.deepcopy(_tc())
    scenario_file = tmp_path / "tool_calling.json"
    scenario_file.write_text(json.dumps([sc]))

    calls = []

    def make_fake(first_names, first_text, final_names, final_text):
        def fake(host, port, messages, **kwargs):
            calls.append(kwargs)
            # verify_final_text runs a second completion after the tool result.
            if any(m.get("role") == "tool" for m in messages):
                return _nons(final_text, *final_names)
            return _nons(first_text, *first_names)

        return fake

    monkeypatch.setattr(re, "PROMPTS_DIR", tmp_path)
    # default fake: correct weather, then a clean final report.
    monkeypatch.setattr(re, "chat_request", make_fake(["weather"], "", [], None))
    return {
        "sc": sc,
        "calls": calls,
        "set_fake": lambda fn, ft, gn, gt: monkeypatch.setattr(
            re, "chat_request", make_fake(fn, ft, gn, gt)
        ),
        "run": lambda: re.run_tool_calling_suite("localhost", 1),
    }


class TestSuiteEndToEnd:
    def test_successful_weather_only_flow_passes(self, suite_with_mock):
        # correct first-turn weather; final report reflects the tool result and
        # calls no forbidden tool -> fully_correct True.
        suite_with_mock["set_fake"](
            ["weather"], "", [], "Partly cloudy, 18°C, humidity 62%"
        )
        out = suite_with_mock["run"]()
        d = out["details"][0]
        assert out["total"] == 1 and out["passed"] == 1
        assert d["fully_correct"] is True

    def test_first_turn_weather_plus_web_search_fails(self, suite_with_mock):
        # _check_tool_call passes (weather is tool_calls[0]) but the forbidden
        # web_search in the SAME turn must still fail the scenario.
        suite_with_mock["set_fake"](
            ["weather", "web_search"], "", [], "Partly cloudy, 18°C, humidity 62%"
        )
        out = suite_with_mock["run"]()
        d = out["details"][0]
        assert out["passed"] == 0
        assert d["fully_correct"] is False
        assert d["forbidden_tool_called"] == ["web_search"]

    def test_final_completion_weather_plus_web_search_fails(self, suite_with_mock):
        # round-5 shape: first turn is a correct weather call, but the final
        # completion ALSO calls web_search while still reporting the result.
        suite_with_mock["set_fake"](
            ["weather"],
            "",
            ["weather", "web_search"],
            "Partly cloudy, 18°C, humidity 62%",
        )
        out = suite_with_mock["run"]()
        d = out["details"][0]
        assert out["passed"] == 0
        assert d["fully_correct"] is False
        assert d["forbidden_tool_called"] == ["web_search"]

    def test_first_turn_no_tool_fails(self, suite_with_mock):
        # no tool at all -> nothing to route; must not pass via absence of forbid.
        suite_with_mock["set_fake"]([], "", [], "no tool here")
        out = suite_with_mock["run"]()
        assert out["passed"] == 0
        assert out["details"][0]["fully_correct"] is False

    def test_verifies_final_terms_are_required(self, suite_with_mock):
        # clean final text that does NOT reflect the tool result must fail.
        suite_with_mock["set_fake"](["weather"], "", [], "I cannot help with that.")
        out = suite_with_mock["run"]()
        d = out["details"][0]
        assert out["passed"] == 0
        assert d["fully_correct"] is False
        assert "reflect the supplied tool result" in d.get("final_text_error", "")

    def test_final_completion_calling_any_tool_fails(self, suite_with_mock):
        # round-9 finding: a final completion that calls WEATHER again (a
        # non-forbidden tool) while its text carries the result marker is still
        # NOT a final answer — it signs a looping/retry model. `forbid_tools`
        # only bans web_search, so the forbid check alone would let it pass;
        # the final turn must call NO tool.
        suite_with_mock["set_fake"](
            ["weather"], "", ["weather"], "Partly cloudy, 18°C, humidity 62%"
        )
        out = suite_with_mock["run"]()
        d = out["details"][0]
        assert out["passed"] == 0
        assert d["fully_correct"] is False
        assert "called a tool rather than answering" in d.get("final_text_error", "")
        # weather is not forbidden, so no forbidden-tool marker — the replay
        # fails on the any-tool final-turn rule, not on the forbid list.
        assert (
            "forbidden_tool_called" not in d or d.get("forbidden_tool_called") is None
        )

    @pytest.mark.parametrize(
        "denial_text",
        [
            # Each carries a required result term AND a weather-unavailable denial
            # that lives outside the original short phrase list, so a pure
            # substring check on the required terms alone would pass it. The
            # denial must make the final turn fail.
            "18°C, but the weather tool is unavailable, so I searched instead",
            "Partly cloudy here, but Weather is unavailable for Tokyo",
            "The temperature is 18°C, however weather is unavailable",
            "I can't use the weather function, though it is 62% humidity",
            "62% humidity, but there's no access to weather",
        ],
    )
    def test_contradictory_denials_fail(self, suite_with_mock, denial_text):
        # round-11 pr_validate finding: a final text that reflects the result
        # WHILE denying the just-used tool must fail, across realistic phrasing.
        suite_with_mock["set_fake"](["weather"], "", [], denial_text)
        out = suite_with_mock["run"]()
        d = out["details"][0]
        assert out["passed"] == 0
        assert d["fully_correct"] is False
        assert "denies" in d.get("final_text_error", "")
