# SPDX-License-Identifier: Apache-2.0
"""Verified TLS for the provider API and relay, including Python on macOS."""

from __future__ import annotations

import ssl
import urllib.error


def client_context() -> ssl.SSLContext:
    # Preserve system/private roots and SSL_CERT_FILE/SSL_CERT_DIR overrides.
    try:
        context = ssl.create_default_context()
        import certifi

        context.load_verify_locations(cafile=certifi.where())
        return context
    except OSError:
        # Trust-store setup failures must use the same terminal path as an
        # untrusted peer, including relay and heartbeat shutdown.
        raise ssl.SSLCertVerificationError(CERTIFICATE_HINT) from None


def certificate_error(exc: BaseException | None) -> bool:
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, ssl.SSLCertVerificationError)
    return isinstance(exc, ssl.SSLCertVerificationError)


CERTIFICATE_HINT = (
    "TLS certificate verification failed. Update your Python CA certificates "
    "and certifi, or set SSL_CERT_FILE to your trusted CA bundle. "
    "Certificate verification remains enabled; this error will not be retried."
)
