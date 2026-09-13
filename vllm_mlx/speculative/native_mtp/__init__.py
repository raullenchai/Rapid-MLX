# SPDX-License-Identifier: Apache-2.0
"""Qualified serial native-MTP backend."""

from .eligibility import NativeMTPUnavailableError, resolve_native_mtp_pair

__all__ = ["NativeMTPUnavailableError", "resolve_native_mtp_pair"]
