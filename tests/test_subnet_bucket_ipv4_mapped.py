"""Regression test for #589: IPv4-mapped IPv6 rate-limit bucketing.

Before the fix, all IPv4-mapped IPv6 addresses (::ffff:a.b.c.d) mapped
to the same ``::`` bucket, so every IPv4 client behind a dual-stack
server shared a single rate-limit quota.
"""

from __future__ import annotations

from vllm_mlx.middleware.auth import _subnet_bucket


class TestSubnetBucketIPv4Mapped:
    """IPv4-mapped IPv6 addresses must bucket by their underlying /24."""

    def test_ipv4_mapped_same_subnet(self):
        """Two IPv4-mapped addresses in the same /24 share a bucket."""
        assert _subnet_bucket("::ffff:192.0.2.1") == "192.0.2.0"
        assert _subnet_bucket("::ffff:192.0.2.99") == "192.0.2.0"

    def test_ipv4_mapped_matches_plain_ipv4(self):
        """IPv4-mapped address buckets the same as the plain IPv4."""
        assert _subnet_bucket("::ffff:192.0.2.1") == _subnet_bucket("192.0.2.1")

    def test_ipv4_mapped_different_subnets(self):
        """Different /24s get different buckets (not all collapsed to ::)."""
        assert _subnet_bucket("::ffff:10.0.0.1") == "10.0.0.0"
        assert _subnet_bucket("::ffff:172.16.0.1") == "172.16.0.0"
        assert _subnet_bucket("::ffff:10.0.0.1") != _subnet_bucket("::ffff:172.16.0.1")

    def test_plain_ipv4_unchanged(self):
        """Plain IPv4 bucketing is unaffected by the fix."""
        assert _subnet_bucket("192.168.1.100") == "192.168.1.0"
        assert _subnet_bucket("10.0.0.5") == "10.0.0.0"

    def test_plain_ipv6_unchanged(self):
        """Pure IPv6 bucketing is unaffected by the fix."""
        assert _subnet_bucket("2001:db8::1") == "2001:db8::"
        assert _subnet_bucket("2001:db8:abcd:1234::1") == "2001:db8:abcd:1234::"

    def test_invalid_host_passthrough(self):
        """Non-IP host strings pass through unchanged."""
        assert _subnet_bucket("localhost") == "localhost"
        assert _subnet_bucket("not-an-ip") == "not-an-ip"
