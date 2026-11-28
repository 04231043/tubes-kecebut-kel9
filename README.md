# Air Quality Lab — Forecasting Kualitas Udara

**TE2514031 Kecerdasan Buatan · Teknik Elektro ITK · Kelas B · Gasal 2026/2027**

Prakiraan konsentrasi karbon monoksida (CO) pada jam berikutnya (*t+1*) dari riwayat sensor kimia dan cuaca. Sistem GA–PSO–RNN–Fuzzy menggunakan data historis [UCI Air Quality](https://doi.org/10.24432/C59K5F). Tidak menggunakan feed sensor live atau nilai acak sebagai hasil evaluasi.

## 1. Jalankan aplikasi

Lingkungan pengujian: Python 3.12, Streamlit 1.51.0, Altair 5.5.0, NumPy 2.3.5, Pandas 2.2.3 dan Matplotlib 3.10.8. Paket deployment menggunakan nama dependensi tanpa pin agar resolver Streamlit Cloud dapat memilih versi yang kompatibel. Buka terminal pada folder yang berisi app.py.

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Dashboard membawa empat checkpoint. Setelah dependensi terpasang, prediksi bekerja luring tanpa mengunduh dataset atau melatih ulang.

Dashboard menggunakan layout responsif, sidebar berkelompok, kartu KPI, grafik Altair interaktif dengan tooltip, dan grafik analisis Matplotlib. Menu mencakup delapan area sesuai rubrik: Ringkasan, Eksplorasi data, Seleksi fitur GA, Optimasi PSO, Jaringan saraf, Aturan fuzzy, Prediksi dan simulasi, dan Perbandingan model.

## 2. Demo prediksi

1. Pilih **Prediksi dan simulasi** dan model **GA + PSO + RNN**.
2. Klik **Hitung prakiraan** pada contoh historis. Untuk target 5 Februari 2005 pukul 03.00, prakiraan 1,492 mg/m³, skor 24,24/100, prioritas Terpantau, dan CO aktual 1,6 mg/m³.
3. Ubah riwayat sensor atau pilih model lain. Jumlah jam dan kolom wajib mengikuti konfigurasi terpilih.
4. Di tab Unggah CSV, unduh contoh, tambahkan jam berurutan dan unggah kembali. CSV dapat memuat semua delapan fitur, meskipun model GA hanya membutuhkan subsetnya.
5. Dashboard menunjukkan jumlah imputasi, atau menolak jendela jika suatu kanal wajib sama sekali tidak teramati pada empat jam terakhir. Input di luar rentang pengamatan latih ditandai. Timestamp adalah jam penuh lokal tanpa zona waktu.
6. Ekspor CSV berisi prakiraan, kualitas input, keputusan fuzzy dan rentang referensi galat.

Nilai referensi bawah/atas adalah prakiraan ± persentil ke-90 galat absolut validasi (batas bawah tidak negatif). Rentang tersebut **bukan interval kepercayaan terkalibrasi**. Fuzzy menetapkan prioritas pemeriksaan sensor pada skala internal, **bukan AQI/ISPU atau penilaian kesehatan**.

## 3. Data, metode dan hasil terukur

Kode memeriksa SHA-256 CSV UCI, membaca pemisah titik koma dan koma desimal, mengabaikan baris kosong, dan menganggap −200 sebagai data hilang. Split kronologis 70/15/15 berdasarkan jam target; median dan standardisasi dipelajari dari latih. Riwayat sensor berakhir satu jam sebelum target. Agar waktu target sama, evaluasi mensyaratkan sedikitnya delapan jam riwayat, setidaknya satu pembacaan setiap kandidat kanal dalam empat jam terakhir, serta label CO valid.

| Konfigurasi | Fitur | MAE (mg/m³) | RMSE (mg/m³) | R² |
|---|---:|---:|---:|---:|
| RNN dasar | 8 | 0,4721 | 0,6404 | 0,7419 |
| GA + RNN | 4 | 0,4569 | 0,6518 | 0,7326 |
| PSO + RNN | 8 | 0,4506 | 0,6326 | 0,7482 |
| GA + PSO + RNN | 4 | 0,4462 | 0,6363 | 0,7452 |
| Persistensi (CO terakhir) | 1 nilai CO | 0,5060 | 0,7764 | 0,6206 |

Keempat RNN dilatih terpisah dan diuji pada 1.302 waktu target identik. Model hibrida unggul pada MAE dengan empat sensor, sedangkan PSO+RNN unggul pada RMSE dan R². Fuzzy bukan model regresi tambahan. Semua angka dihitung dari prediksi checkpoint, bukan faktor pengali satu model.

GA/PSO menggunakan seed 7, 19, dan 42. GA memakai 6 individu × 5 generasi/seed; PSO memakai 5 partikel × 5 iterasi/seed pada dua ruang fitur. RNN pencarian memakai 1.024 jendela latih, 5 epoch dan seed 271. Pelatihan akhir memakai 4.957 jendela latih, seed 42, maksimum 32 epoch, patience 7. Set validasi (1.107 target) memilih hyperparameter dan checkpoint; set uji hanya untuk evaluasi akhir. Bukti disimpan dalam bundle.

## 4. Reproduksi eksperimen & pengujian

```sh
python train.py --out runs/eksperimen-1
python -m unittest -v test_pipeline test_dashboard
```

Pilih folder keluaran baru setiap mengulang pelatihan. Jika CSV resmi sudah tersedia, gunakan `python train.py --csv AirQualityUCI.csv --out runs/eksperimen-2`. Untuk Colab gunakan notebook mandiri `Forecasting_CO_Kelompok_9.ipynb`. Waktu komputasi bergantung pada perangkat/thread. Dashboard bisa memakai hasil eksperimen sesi melalui Pelatihan ulang. Untuk deployment tetap, jalankan offline dan tempatkan bundle terverifikasi pada `artifacts/uci-v1/`. Jangan pindahkan hanya bobot tanpa preprocessing, log dan prediksi.

## 5. Isi repositori

| Berkas | Fungsi |
|---|---|
| app.py | Delapan halaman Streamlit, prediksi manual/CSV semua model, training opsional |
| air_data.py | Pembacaan CSV, timestamp, data hilang, split dan standardisasi |
| optimize.py | GA: selection, crossover, mutation, elitism; PSO: pbest, gbest, velocity |
| recurrent.py | RNN Elman, Adam, BPTT, dropout dan checkpoint validasi |
| train.py | Empat eksperimen, persistensi, statistik, manifest |
| inference.py, fuzzy_system.py | Inferensi model terpilih, kualitas input dan Mamdani |
| evaluation.py | Tabel perbandingan dan analisis residual per rentang |
| artifacts/uci-v1/ | Model NPZ, parameter, statistik, validasi/uji, contoh, manifest SHA-256 |
| Forecasting_CO_Kelompok_9.ipynb | Notebook eksperimen mandiri |
| test_pipeline.py, test_dashboard.py | Pengujian checkpoint, metrik, split, kualitas input, fuzzy dan UI |
| docs/VALIDASI_DAN_DEMO_UAS.md | Checklist rubrik, protokol demo dan batasan |

Empat checkpoint menggunakan NPZ tanpa pickle.

## 6. Aplikasi online dan source code

- [Dashboard Streamlit](https://tubes-kecebut-kel9.streamlit.app/)
- [Repository Kelompok 9](https://github.com/04231043/tubes-kecebut-kel9)

Entry point: app.py. Seluruh checkpoint, preprocessing, ringkasan EDA, log, prediksi validasi/uji dan manifest harus tetap bersama di artifacts/uci-v1. Inference berjalan tanpa mengunduh dataset atau melatih model ulang.
Versi layout ini memakai kembali checkpoint dan nilai metrik dari eksperimen sebelumnya (tidak mengubah bobot, prediksi uji, split, algoritma GA/PSO, atau aturan fuzzy). Tambahan artefak validasi digunakan untuk visualisasi dan pemeriksaan galat.

Arsitektur UI:
- dashboard_ui.py: komponen visual (hero, sidebar, kartu, notifikasi, gaya chart).
- assets/dashboard.css: style responsif dan tata letak.
- .streamlit/config.toml: tema terang dengan sidebar gelap.
- app.py: halaman, navigasi, grafik, dan integrasi inference.

Jalankan test_pipeline dan test_dashboard sebelum perubahan kode. Riwayat Git menyimpan versi terdahulu.
