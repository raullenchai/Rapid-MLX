# SPDX-License-Identifier: Apache-2.0
"""Tests for the bearer token and the browser-origin gate.

These are the two controls standing between a tunnelled port and the
public internet, so they are tested for the ways they could be *too
permissive*, not just for the happy path.
"""

from __future__ import annotations

import stat

import pytest

from rmlx_web import auth


class TestTokenFile:
    def test_creates_a_token_with_owner_only_permissions(self, tmp_path):
        path = tmp_path / "web-token"
        token = auth.load_or_create_token(path)

        assert token
        assert path.exists()
        # Anything looser leaks the secret to every other local user for
        # the whole session.
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_reuses_the_stored_token_across_runs(self, tmp_path):
        path = tmp_path / "web-token"
        first = auth.load_or_create_token(path)
        second = auth.load_or_create_token(path)

        # Rotating per launch would log every phone out on each restart.
        assert first == second

    def test_rotate_replaces_the_stored_token(self, tmp_path):
        path = tmp_path / "web-token"
        first = auth.load_or_create_token(path)
        second = auth.load_or_create_token(path, rotate=True)

        assert first != second
        assert path.read_text().strip() == second

    def test_override_wins_and_does_not_touch_the_file(self, tmp_path):
        path = tmp_path / "web-token"
        assert auth.load_or_create_token(path, override="explicit") == "explicit"
        assert not path.exists()

    def test_tightens_permissions_on_a_preexisting_loose_file(self, tmp_path):
        path = tmp_path / "web-token"
        path.write_text("preexisting")
        path.chmod(0o644)

        assert auth.load_or_create_token(path) == "preexisting"
        # A file restored from a backup or copied by hand is easily 0644;
        # reading it as-is would leave the secret world-readable.
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_empty_file_is_replaced_rather_than_accepted(self, tmp_path):
        path = tmp_path / "web-token"
        path.write_text("   \n")

        token = auth.load_or_create_token(path)

        # An empty token would otherwise become a credential that any
        # request satisfies by sending "Bearer ".
        assert token.strip()
        assert len(token) > 20


class TestExtractBearer:
    @pytest.mark.parametrize(
        "header,expected",
        [
            ("Bearer abc123", "abc123"),
            ("bearer abc123", "abc123"),
            ("BEARER abc123", "abc123"),
            ("Bearer   abc123  ", "abc123"),
        ],
    )
    def test_accepts_any_scheme_casing(self, header, expected):
        assert auth.extract_bearer(header) == expected

    @pytest.mark.parametrize(
        "header",
        [None, "", "abc123", "Basic abc123", "Bearer", "Bearer   "],
    )
    def test_rejects_everything_else(self, header):
        assert auth.extract_bearer(header) is None


class TestTokenMatches:
    def test_matching_token_is_accepted(self):
        assert auth.token_matches("secret", "secret")

    @pytest.mark.parametrize("presented", [None, "", "secre", "secrets", "SECRET"])
    def test_non_matching_token_is_rejected(self, presented):
        assert not auth.token_matches("secret", presented)


class TestOriginGate:
    def test_absent_origin_is_allowed(self):
        # curl, a phone shortcut, a script — none send Origin, and none
        # are the confused deputy this guard is about.
        assert auth.origin_is_allowed(None, "example.com", None)

    def test_same_origin_fetch_metadata_is_allowed(self):
        assert auth.origin_is_allowed(
            "https://evil.example", "tunnel.example", "same-origin"
        )

    @pytest.mark.parametrize("site", ["cross-site", "same-site"])
    def test_cross_site_fetch_metadata_is_refused(self, site):
        # The browser's own verdict, which page JS cannot forge. It wins
        # over the Origin/Host comparison below.
        assert not auth.origin_is_allowed(
            "https://tunnel.example", "tunnel.example", site
        )

    def test_matching_origin_and_host_is_allowed_without_fetch_metadata(self):
        # Under a tunnel both carry the tunnel's hostname, so this holds
        # without the user pre-registering an external hostname they
        # cannot know in advance.
        assert auth.origin_is_allowed("https://tunnel.example", "tunnel.example", None)

    def test_mismatched_origin_is_refused(self):
        assert not auth.origin_is_allowed(
            "https://evil.example", "tunnel.example", None
        )

    def test_default_ports_do_not_cause_a_spurious_mismatch(self):
        assert auth.origin_is_allowed(
            "https://tunnel.example", "tunnel.example:443", None
        )
        assert auth.origin_is_allowed(
            "http://tunnel.example:80", "tunnel.example", None
        )

    def test_non_default_port_must_still_match(self):
        assert auth.origin_is_allowed("http://127.0.0.1:7788", "127.0.0.1:7788", None)
        assert not auth.origin_is_allowed(
            "http://127.0.0.1:9999", "127.0.0.1:7788", None
        )

    def test_null_origin_is_refused(self):
        # Sandboxed iframe or a file:// page. Not something a legitimate
        # client produces.
        assert not auth.origin_is_allowed("null", "tunnel.example", None)

    def test_origin_without_host_header_is_refused(self):
        assert not auth.origin_is_allowed("https://tunnel.example", None, None)


class TestContentTypeGate:
    def test_json_is_accepted_with_and_without_parameters(self):
        assert auth.content_type_is_json("application/json")
        assert auth.content_type_is_json("application/json; charset=utf-8")
        assert auth.content_type_is_json("APPLICATION/JSON")

    @pytest.mark.parametrize(
        "content_type",
        [
            None,
            "",
            # The three CORS "simple" content types: a cross-origin page
            # can send these with NO preflight, so accepting any of them
            # would defeat the CSRF control entirely.
            "text/plain",
            "application/x-www-form-urlencoded",
            "multipart/form-data; boundary=x",
        ],
    )
    def test_simple_content_types_are_refused(self, content_type):
        assert not auth.content_type_is_json(content_type)

    def test_multipart_requires_a_boundary(self):
        assert auth.content_type_is_multipart("multipart/form-data; boundary=abc")
        assert not auth.content_type_is_multipart("multipart/form-data")
        assert not auth.content_type_is_multipart("application/json")
