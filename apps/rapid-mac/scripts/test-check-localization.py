#!/usr/bin/env python3
"""Regression fixtures for the custom Swift String and SF Symbol boundaries."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "localization", Path(__file__).with_name("check-localization.py")
)
localization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(localization)


class LocalizationScannerTests(unittest.TestCase):
    def scan(self, source, name="AudioView.swift"):
        with tempfile.TemporaryDirectory(prefix="rapid-localization-fixture-") as directory:
            path = Path(directory) / name
            path.write_text(source)
            return localization.scan(path)

    def test_string_helpers_require_explicit_localization(self):
        for helper, parameters in localization.STRING_UI_HELPERS.items():
            for parameter in parameters:
                prefix = f"{parameter}: " if parameter else ""
                with self.subTest(helper=helper, parameter=parameter):
                    self.assertTrue(self.scan(f'{helper}({prefix}"English copy")'))
                    self.assertFalse(self.scan(
                        f'{helper}({prefix}String(localized: "English copy"))'
                    ))
                    self.assertTrue(self.scan(
                        f'let title = "English copy"; {helper}({prefix}title)'
                    ))

    def test_dynamic_copy_and_nested_arguments(self):
        self.assertFalse(self.scan('SectionHeader(model.name)'))
        self.assertFalse(self.scan('SectionHeader("Claude Desktop")'))
        self.assertFalse(self.scan('QuietIconButton(symbol: "arrow.up", label: String(localized: "Generate"))'))
        self.assertFalse(self.scan('RapidSegmentedControl(options: [.init(value: "wire-id", title: model.name)], accessibilityLabel: String(localized: "Mode"))'))
        self.assertTrue(self.scan('QuietIconButton(symbol: "play", label: playing ? "Stop" : "Play")'))
        self.assertTrue(self.scan('var voicePlaceholder: String { "Choose a voice" }; popupControlLabel(voicePlaceholder)'))
        self.assertFalse(self.scan('// SectionHeader("Comment")\nText("SwiftUI localizes literals")'))

    def test_symbol_must_handle_its_own_accessibility(self):
        self.assertTrue(self.scan('Image(systemName: "film.stack")'))
        self.assertTrue(self.scan('Image(systemName: "film.stack").accessibilityHidden(false)'))
        self.assertTrue(self.scan('Image(systemName: "film.stack").accessibilityLabel("")'))
        self.assertTrue(self.scan('Image(systemName: "film.stack"); Text("Title").accessibilityLabel("Title")'))
        self.assertFalse(self.scan('Image(systemName: "film.stack").font(.system(size: 40)).accessibilityHidden(true)'))
        self.assertFalse(self.scan('Image(systemName: done ? "checkmark.circle.fill" : "circle.dashed").accessibilityLabel(done ? String(localized: "Completed") : String(localized: "Not completed"))'))
        self.assertFalse(self.scan('// Image(systemName: "comment")'))

    def test_about_and_update_decorations_require_accessibility_handling(self):
        for name, symbol in [("AboutPanel.swift", "bolt.fill"),
                             ("AboutPanel.swift", "exclamationmark.triangle.fill"),
                             ("WhatsNewBanner.swift", "sparkles")]:
            with self.subTest(name=name, symbol=symbol):
                source = f'Image(systemName: "{symbol}").font(.title); Text("Title")'
                self.assertTrue(self.scan(source, name))
                self.assertFalse(self.scan(source.replace('.font(.title)',
                    '.font(.title).accessibilityHidden(true)'), name))

    def test_symbol_audit_scope_is_explicit(self):
        for name in localization.SYMBOL_AUDIT_FILES:
            self.assertTrue(self.scan('Image(systemName: "film.stack")', name))
        self.assertFalse(self.scan('Image(systemName: "film.stack")', "OtherView.swift"))


if __name__ == "__main__":
    unittest.main()
