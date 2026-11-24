"""Exercise Streamlit pages, both inference routes, and a bad CSV through AppTest."""
from io import BytesIO
from unittest.mock import patch
from pathlib import Path
import unittest

import streamlit as st
from streamlit.testing.v1 import AppTest
from inference import DEFAULT


class DashboardTests(unittest.TestCase):
    def test_pages_and_manual_forecast(self):
        a = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
        for label in a.sidebar.radio[0].options:
            a.sidebar.radio[0].set_value(label).run()
            self.assertFalse(a.exception, label)
            if label == "Prediksi dan simulasi":
                a.button[0].click().run()
                self.assertFalse(a.exception)
                self.assertEqual(a.metric[0].value, "1.492")
                self.assertEqual(a.metric[1].value, "24.24 / 100")
                self.assertIn("Terpantau", a.success[0].value)

    def test_uploaded_csv_and_missing_column(self):
        sample = (DEFAULT / "contoh_sensor.csv").read_bytes()
        for contents, valid in [(sample, True), (b"timestamp,T\n2005-02-05 02:00:00,14\n", False)]:
            with patch.object(st, "file_uploader", side_effect=lambda label, **kw:
                              BytesIO(contents) if label == "Riwayat per jam" else None):
                a = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
                a.sidebar.radio[0].set_value("Prediksi dan simulasi").run()
                self.assertFalse(a.exception)
                if valid:
                    values = a.dataframe[-1].value
                    self.assertEqual(values.iloc[0].prioritas, "Terpantau")
                    self.assertAlmostEqual(values.iloc[0].CO_mg_m3, 1.492385, places=5)
                else:
                    self.assertTrue(any("Kolom wajib" in e.value for e in a.error))


if __name__ == "__main__":
    unittest.main()
