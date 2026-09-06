# SPDX-License-Identifier: Apache-2.0
"""Tests for the CLI: login URL, the startup banner, and argument handling."""

from __future__ import annotations

import pytest

from rmlx_web import cli
from rmlx_web.connectors import ConnectorStore


class TestLoginURL:
    def test_the_token_rides_in_the_fragment(self):
        url = cli._login_url("127.0.0.1", 7788, "secret-token")

        # A fragment is never sent to the server, so it cannot land in an
        # access log, a proxy log, or a tunnel provider's request
        # history — all of which a query parameter would reach.
        assert "#token=secret-token" in url
        assert "?token=" not in url

    def test_the_token_is_percent_encoded(self):
        url = cli._login_url("127.0.0.1", 7788, "a+b/c=d&e")

        # `&` would otherwise start a second fragment parameter and
        # truncate the token; `/` and `+` are ambiguous in a URL.
        assert "a%2Bb%2Fc%3Dd%26e" in url
        assert "&e" not in url.split("#token=")[1]

    def test_a_generated_token_round_trips(self):
        from rmlx_web import auth

        token = auth.generate_token()
        url = cli._login_url("127.0.0.1", 7788, token)

        from urllib.parse import unquote

        assert unquote(url.split("#token=")[1]) == token


class TestBanner:
    def test_prints_the_url_and_token(self, capsys):
        cli._print_banner(host="127.0.0.1", port=7788, token="tok", loopback=True)
        out = capsys.readouterr().out

        assert "http://127.0.0.1:7788/" in out
        assert "tok" in out

    def test_prints_the_sign_in_link(self, capsys):
        # The link carries the token in its fragment, so pasting it is what
        # saves retyping 43 characters.
        cli._print_banner(host="127.0.0.1", port=7788, token="tok", loopback=True)
        out = capsys.readouterr().out

        assert "#token=tok" in out

    def test_prints_no_qr_code(self, capsys):
        # A 25-row block of blocks pushed the token off a short terminal
        # window, which is the one thing the user cannot proceed without.
        cli._print_banner(host="127.0.0.1", port=7788, token="tok", loopback=True)
        out = capsys.readouterr().out

        assert "Scan" not in out
        assert "qr" not in out.lower()
        # Whatever else changes, the banner stays short enough to read.
        assert len(out.splitlines()) < 15

    def test_a_non_loopback_bind_is_still_token_protected(self, capsys, monkeypatch):
        monkeypatch.setattr(cli, "_display_host", lambda host: "192.168.1.5")
        cli._print_banner(host="0.0.0.0", port=7788, token="tok", loopback=False)
        out = capsys.readouterr().out
        assert "Token: tok" in out
        assert "WARNING" not in out

    def test_no_warning_on_loopback(self, capsys, monkeypatch):
        cli._print_banner(host="127.0.0.1", port=7788, token="tok", loopback=True)
        assert "WARNING" not in capsys.readouterr().out


class TestLoopbackDetection:
    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "127.0.0.53", "localhost", "::1", "0:0:0:0:0:0:0:1"]
    )
    def test_loopback_addresses(self, host):
        # The whole 127/8 block is loopback, not just 127.0.0.1.
        assert cli._is_loopback(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::", "example.com"])
    def test_non_loopback_addresses(self, host):
        # Unparseable names must be treated as non-loopback: getting this
        # wrong in the permissive direction silently skips the exposure
        # warning.
        assert cli._is_loopback(host) is False


class TestDisplayHost:
    def test_a_wildcard_bind_is_not_echoed_back(self, monkeypatch):
        # "0.0.0.0" is not a reachable address, so printing it produces a
        # URL that does not work.
        shown = cli._display_host("0.0.0.0")
        assert shown != "0.0.0.0"

    def test_a_concrete_host_is_passed_through(self):
        assert cli._display_host("192.168.1.5") == "192.168.1.5"
        assert cli._display_host("127.0.0.1") == "127.0.0.1"


class TestTokenDecision:
    def test_main_loads_a_token_without_an_auth_flag(self, monkeypatch):
        captured = {}

        def load_or_create_token(*, override, rotate):
            captured.update(override=override, rotate=rotate)
            return "generated-token"

        monkeypatch.setattr(cli.auth, "load_or_create_token", load_or_create_token)
        monkeypatch.setattr(cli, "ConnectorStore", lambda: object())
        monkeypatch.setattr(
            cli,
            "_resolve_engine",
            lambda *args, **kwargs: (object(), None, None),
        )
        monkeypatch.setattr(cli, "create_app", lambda config: config)
        monkeypatch.setattr(cli, "_print_banner", lambda **kwargs: None)
        monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

        assert cli.main([]) == 0
        assert captured == {"override": None, "rotate": False}

    def test_explicit_and_rotated_tokens_reach_the_store(self, monkeypatch):
        captured = {}

        def load_or_create_token(*, override, rotate):
            captured.update(override=override, rotate=rotate)
            return override or "rotated-token"

        monkeypatch.setattr(cli.auth, "load_or_create_token", load_or_create_token)
        monkeypatch.setattr(cli, "ConnectorStore", lambda: object())
        monkeypatch.setattr(
            cli,
            "_resolve_engine",
            lambda *args, **kwargs: (object(), None, None),
        )
        monkeypatch.setattr(cli, "create_app", lambda config: config)
        monkeypatch.setattr(cli, "_print_banner", lambda **kwargs: None)
        monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

        assert cli.main(["--token", "chosen", "--new-token"]) == 0
        assert captured == {"override": "chosen", "rotate": True}


class TestOptionalModelArgument:
    """The model alias is optional; the page's picker is the other way in.

    This used to be a hard `SystemExit`, so the tests below are the thing
    stopping it from being reintroduced as an "obviously required"
    argument.
    """

    def _args(self, argv: list[str]):
        return cli.build_parser().parse_args(argv)

    def _connectors(self, tmp_path):
        # Never the real ~/.config: this constructor reads a file other tools
        # on this Mac use, and a test must not depend on what is in it.
        return ConnectorStore(
            config_path=tmp_path / "mcp.json",
            settings_path=tmp_path / "rmlx-web.json",
        )

    def test_starting_with_no_model_is_allowed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "find_rapid_mlx_binary", lambda explicit: "/bin/true")

        engine, catalog, downloads = cli._resolve_engine(
            self._args([]),
            downloads_enabled=True,
            connectors=self._connectors(tmp_path),
        )

        # A supervisor, not an attached engine — it owns the child it will
        # later spawn, which is what makes the picker able to switch.
        assert engine.can_switch is True
        assert engine.status().model is None
        # The catalog is what the picker lists, so it must exist even
        # though nothing is loaded yet.
        assert catalog is not None
        assert downloads is not None

    def test_an_alias_is_still_honoured(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "find_rapid_mlx_binary", lambda explicit: "/bin/true")

        engine, catalog, _ = cli._resolve_engine(
            self._args(["some-alias"]),
            downloads_enabled=False,
            connectors=self._connectors(tmp_path),
        )

        assert engine.can_switch is True
        assert catalog is not None

    def test_attach_still_refuses_a_model(self, tmp_path):
        # --attach targets a server this process does not own, so the
        # model is not ours to choose.
        with pytest.raises(SystemExit):
            cli._resolve_engine(
                self._args(["--attach", "http://x", "alias"]),
                downloads_enabled=False,
                connectors=self._connectors(tmp_path),
            )

    def test_the_help_text_does_not_name_a_specific_model(self):
        # A concrete alias in the help reads as a default and goes stale
        # as the catalog moves.
        help_text = cli.build_parser().format_help()
        assert "qwen" not in help_text.lower()
