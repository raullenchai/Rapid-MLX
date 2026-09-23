"""Vendored model-foundation modules (upstream ``mlx_vlm.models`` @ 0.7.1).

VENDOR-DEVIATION(subset-exports): upstream ``models/__init__.py`` is empty
and the vendored package holds only the foundations the speculative core
needs (``base.py``, ``linear.py``). The cache primitives live at the
package root (step 2a layout), so ``models.cache`` resolves as ``..cache``
via documented redirects.
"""
