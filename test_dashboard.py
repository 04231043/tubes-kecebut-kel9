"""Exercise Streamlit navigation, both inference routes, and CSV validation."""
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from inference import DEFAULT, forecast, sample_input

APP_PATH = str(Path(__file__).with_name("app.py"))
FORECAST_PAGE = "Prakiraan CO"
EVALUATION_PAGE = "Evaluasi Model"


class DashboardTests(unittest.TestCase):
    def test_pages_and_manual_forecast(self):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertEqual(app.sidebar.radio[0].options, [
            "Ringkasan", "Eksplorasi Data", "Seleksi Fitur", "Optimasi PSO",
            "Jaringan Saraf", "Fuzzy Logic", FORECAST_PAGE, EVALUATION_PAGE,
        ])
        for label in app.sidebar.radio[0].options:
            with self.subTest(page=label):
                app.sidebar.radio[0].set_value(label).run()
                self.assertFalse(app.exception)

        app.sidebar.radio[0].set_value(FORECAST_PAGE).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "1.492")
        self.assertEqual(app.metric[1].value, "24.24 / 100")
        self.assertIn("Terpantau", app.success[0].value)

        app.selectbox[0].set_value("pso").run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        expected = forecast(sample_input(DEFAULT, "pso"), DEFAULT, "pso").iloc[0].CO_mg_m3
        self.assertEqual(app.metric[0].value, f"{expected:.3f}")

    def test_overview_navigation_buttons(self):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        for label, destination in [("Coba prakiraan", FORECAST_PAGE), ("Bandingkan model", EVALUATION_PAGE)]:
            with self.subTest(button=label):
                app.sidebar.radio[0].set_value("Ringkasan").run()
                next(button for button in app.button if button.label == label).click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.sidebar.radio[0].value, destination)

    def test_uploaded_csv_validation(self):
        sample = (DEFAULT / "contoh_sensor.csv").read_bytes()
        header, *rows = sample.splitlines()
        with_metadata = b"\n".join([header + b",nama_sensor", *(row + b",Sensor A" for row in rows)])
        with_duplicate = b"\n".join([header + b",T", *(row + b",7.0" for row in rows)])
        cases = [
            ("example", sample, None),
            ("metadata", with_metadata, None),
            ("missing column", b"timestamp,T\n2005-02-05 02:00:00,14\n", "Kolom wajib"),
            ("empty file", b"", "CSV tidak memuat baris observasi"),
            ("duplicate column", with_duplicate, "Nama kolom harus unik"),
        ]
        for name, contents, expected_error in cases:
            with self.subTest(csv=name), patch.object(
                st, "file_uploader", side_effect=lambda label, **kw:
                BytesIO(contents) if label == "Riwayat per jam" else None,
            ):
                app = AppTest.from_file(APP_PATH, default_timeout=30).run()
                app.sidebar.radio[0].set_value(FORECAST_PAGE).run()
                self.assertFalse(app.exception)
                if expected_error:
                    self.assertTrue(any(expected_error in error.value for error in app.error))
                    continue
                self.assertFalse(app.error)
                values = app.dataframe[-1].value
                self.assertEqual(values.iloc[0].prioritas, "Terpantau")
                self.assertAlmostEqual(values.iloc[0].CO_mg_m3, 1.492385, places=3)
                self.assertIn("kualitas_input", values)


if __name__ == "__main__":
    unittest.main()
