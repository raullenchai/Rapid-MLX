# SPDX-License-Identifier: Apache-2.0
"""
Route modules for the Rapid-MLX server.

Routes are organized by API domain. Each module creates an APIRouter
that server.py includes via app.include_router().

Global server state (engine, config, parsers) is accessed through the
server module's globals — routes import what they need from server.py.
"""
