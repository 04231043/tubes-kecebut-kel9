"""Exercise Streamlit navigation, both inference routes, and CSV validation."""
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import streamlit as st
from streamlit.testing.v1 import AppTest

from air_data import TARGET
from fuzzy_system import fuzzy_detail
from inference import DEFAULT, forecast, load_run, sample_input

APP_PATH = str(Path(__file__).with_name("app.py"))
FORECAST_PAGE = "Prakiraan CO"
EVALUATION_PAGE = "Evaluasi Model"
CHART_COUNTS = {
    "Ringkasan": 1,
    "Eksplorasi Data": 5,
    "Seleksi Fitur": 1,
    "Optimasi PSO": 1,
    "Jaringan Saraf": 4,
    "Fuzzy Logic": 2,
    FORECAST_PAGE: 0,
    EVALUATION_PAGE: 2,
}


def chart_tables(chart):
    proto = chart.proto
    buffers = [dataset.data.data for dataset in proto.datasets]
    if proto.data.data:
        buffers.append(proto.data.data)
    return [pa.ipc.open_stream(buffer).read_all().to_pandas() for buffer in buffers]


def find_table(tables, *columns):
    return next(table for table in tables if set(columns).issubset(table.columns))


class DashboardTests(unittest.TestCase):
    def test_pages_and_manual_forecast(self):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        experiment = load_run(DEFAULT)
        self.assertEqual(app.sidebar.radio[0].options, list(CHART_COUNTS))
        for label, expected_charts in CHART_COUNTS.items():
            with self.subTest(page=label):
                app.sidebar.radio[0].set_value(label).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                charts = app.get("arrow_vega_lite_chart")
                self.assertEqual(len(charts), expected_charts)
                tables = []
                for chart in charts:
                    datasets = chart_tables(chart)
                    self.assertTrue(any(not dataset.empty for dataset in datasets))
                    tables.extend(datasets)
                self.assert_chart_values(label, tables, app, experiment)

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

    def assert_chart_values(self, page, tables, app, experiment):
        if page == "Ringkasan":
            self.assertIn("Arsitektur model", [heading.value for heading in app.subheader])
            self.assertTrue(any("<ol" in html.proto.body for html in app.get("html")))
            first = find_table(tables, "Aktual", "Prakiraan hibrida").iloc[0]
            self.assertEqual(first["Waktu target"], pd.Timestamp("2005-02-05 03:00", tz="UTC"))
            self.assertAlmostEqual(first["Aktual"], 1.6)
            self.assertAlmostEqual(first["Prakiraan hibrida"], 1.4923856)
        elif page == "Eksplorasi Data":
            source = next(row for row in experiment["data"]["summary"] if row["variabel"] == TARGET)
            missing = find_table(tables, "Variabel", "hilang (%)").set_index("Variabel").loc[TARGET]
            self.assertEqual(missing["missing"], source["missing"])
            histogram = find_table(tables, "Awal", "Akhir", "Jumlah")
            self.assertEqual(histogram["Jumlah"].sum(), source["observed"])
        elif page in ("Seleksi Fitur", "Optimasi PSO"):
            key = "ga" if page == "Seleksi Fitur" else "pso_ga"
            series = find_table(tables, "Langkah", "Seri", "Nilai")
            self.assertEqual(len(series), sum(len(log["history"]) for log in experiment[key]))
            self.assertAlmostEqual(series.iloc[0]["Nilai"], experiment[key][0]["history"][0])
        elif page == "Jaringan Saraf":
            checkpoint = find_table(tables, "Tanda", "Keterangan")
            self.assertEqual(checkpoint.iloc[0]["Tanda"], experiment["models"]["hybrid"]["best_epoch"])
            scatter = find_table(tables, "Waktu target", "Aktual", "Prakiraan")
            self.assertEqual(len(scatter), experiment["data"]["splits"]["test"]["n"])
        elif page == "Fuzzy Logic":
            markers = [table.iloc[0]["Tanda"] for table in tables if "Tanda" in table]
            self.assertEqual(len(markers), 2)
            self.assertAlmostEqual(markers[0], 3.0)
            self.assertAlmostEqual(markers[1], fuzzy_detail(3.0)[0])
        elif page == EVALUATION_PAGE:
            bars = find_table(tables, "Model", "Metrik", "Nilai").set_index(["Model", "Metrik"])
            self.assertAlmostEqual(bars.loc[("GA + PSO", "MAE uji"), "Nilai"],
                                   experiment["models"]["hybrid"]["test"]["mae"])

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

    def test_multi_hour_csv_displays_a_forecast_chart(self):
        sample = sample_input(DEFAULT, "hybrid")
        next_hour = sample.iloc[-1:].copy()
        next_hour["timestamp"] = (pd.to_datetime(next_hour["timestamp"]) + pd.Timedelta(hours=1)).astype(str)
        readings = pd.concat([sample, next_hour], ignore_index=True)
        upload = BytesIO(readings.to_csv(index=False).encode())
        expected = forecast(readings)

        with patch.object(st, "file_uploader", side_effect=lambda label, **kw:
                          upload if label == "Riwayat per jam" else None):
            app = AppTest.from_file(APP_PATH, default_timeout=30).run()
            app.sidebar.radio[0].set_value(FORECAST_PAGE).run()

        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        charts = app.get("arrow_vega_lite_chart")
        self.assertEqual(len(charts), 1)
        self.assertTrue(any(len(table) == len(expected) == 2 for table in chart_tables(charts[0])))
        self.assertAlmostEqual(app.dataframe[-1].value.iloc[-1].CO_mg_m3, expected.iloc[-1].CO_mg_m3, places=4)


if __name__ == "__main__":
    unittest.main()
