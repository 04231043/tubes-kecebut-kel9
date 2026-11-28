# Validasi teknis dan persiapan demo UAS

Proyek: Forecasting Air Quality, Teknik Elektro ITK, Kelas B Kelompok 9
Revisi: 28 November 2026

## 1. Perbaikan dari versi online sebelumnya

- Gunakan observasi asli UCI Air Quality; tidak mengganti sumber dengan data acak.
- Evaluasi forecast satu jam ke depan, bukan estimasi target pada jam masukan.
- Median dan scaler dihitung hanya dari train. GA dan PSO memakai validation.
- GA melakukan seleksi kromosom biner. PSO memakai velocity, pbest, dan gbest, bukan pencarian acak semata.
- Empat model RNN berlatih dan menyimpan checkpoint secara terpisah.
- Fuzzy Mamdani beroperasi setelah RNN; skor internal verifikasi sensor bukan ISPU resmi.
- Dashboard menyediakan 8 area wajib plus prediksi manual dan CSV untuk semua konfigurasi, validasi timestamp, pengecekan kelengkapan dan rentang sensor, log unduh, grafik residual, dan referensi galat validasi.
- Manifest mencocokkan SHA-256 semua artefak yang dipakai. Deployment tidak membutuhkan raw CSV atau pelatihan tiap klik.

## 2. Rekap eksperimen aktual

Dataset: sumber UCI, 9.357 jam historis. Empat eksperimen menggunakan 4.957 target latih, 1.107 validasi dan 1.302 uji yang sama.

| Model | Fitur | Jendela | Unit | LR | Dropout | MAE uji | RMSE uji | R² uji |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RNN dasar | 8 | 6 | 12 | 0.00300 | 0.10 | 0.4721 | 0.6404 | 0.7419 |
| GA + RNN | 4 | 6 | 12 | 0.00300 | 0.10 | 0.4569 | 0.6518 | 0.7326 |
| PSO + RNN | 8 | 6 | 20 | 0.00980 | 0.22 | 0.4506 | 0.6326 | 0.7482 |
| GA + PSO + RNN | 4 | 6 | 20 | 0.00893 | 0.14 | 0.4462 | 0.6363 | 0.7452 |

Fitur GA terpilih: PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), T.

MAE hibrida terendah dengan empat kanal; PSO + RNN memiliki RMSE dan R² terbaik. Ranking bukan klaim bahwa hybrid menang pada semua metrik. Angka galat dalam mg/m³. Hasil ini sama dengan eksperimen laporan progres yang dikerjakan sebelumnya. Durasi komputasi berubah mengikuti perangkat.

### Referensi besarnya galat

| Model | Q90 absolut validasi (mg/m³) | Proporsi uji dengan galat ≤ Q90 |
|---|---:|---:|
| RNN dasar | 1.4746 | 96.2% |
| GA + RNN | 1.4909 | 95.0% |
| PSO + RNN | 1.4835 | 95.9% |
| GA + PSO + RNN | 1.4394 | 95.0% |

Q90 dihitung dari validasi setelah pemilihan checkpoint. Rentang perkiraan ± Q90 bertujuan menunjukkan besarnya galat historis; tidak menjamin probabilitas cakupan tertentu untuk input baru. Set uji tidak dipakai untuk tuning.

## 3. Skenario demo

1. Di **Ringkasan**, jelaskan horizon +1 jam, artefak eksperimen, dan alur GA–PSO–RNN–FIS.
2. Di **Eksplorasi Data**, bedakan 9.357 observasi per jam, nilai CO teramati, data hilang, dan jumlah target valid tiap bagian.
3. Di **Seleksi Fitur**, tunjukkan kromosom, fitness, seed, konvergensi, dan perbandingan RNN dasar dengan GA + RNN.
4. Di **Optimasi PSO**, demonstrasikan ruang pencarian seluruh sensor dan hasil GA, gbest serta parameter terpilih.
5. Di **Jaringan Saraf**, pilih keempat model untuk memperlihatkan checkpoint, kurva pelatihan dan validasi, scatter, dan residual.
6. Di **Fuzzy Logic**, ubah CO contoh dari 0 ke 3 lalu 15 mg/m³. Jelaskan min–max, centroid, dan skor internal.
7. Di **Prakiraan CO**, klik **Hitung prakiraan**. Di tab **Unggah CSV**, unduh contoh riwayat sensor lalu unggah kembali. Pilih model PSO + RNN dan hitung ulang. Hilangkan satu nilai sensor untuk menunjukkan imputasi. Jika satu kanal kosong sepanjang empat jam terakhir, sistem menolak prakiraan.
8. Di **Evaluasi Model**, jelaskan trade-off fitur, MAE, RMSE, R², biaya komputasi dan kecenderungan galat saat CO tinggi.

### Keluaran demo historis

Semua model di bawah memprakirakan 5 Februari 2005 pukul 03.00. CO aktual 1,6 mg/m³ (tidak diberikan kepada model).

| Model | Prakiraan (mg/m³) | Skor fuzzy | Prioritas |
|---|---:|---:|---|
| RNN dasar | 1.3536 | 23.77 | Terpantau |
| GA + RNN | 1.4138 | 23.99 | Terpantau |
| PSO + RNN | 1.1674 | 22.96 | Terpantau |
| GA + PSO + RNN | 1.4924 | 24.24 | Terpantau |

## 4. Hasil QA lokal

- 14 tes otomatis lulus (`test_pipeline` dan `test_dashboard`). Cakupan mencakup empat checkpoint, metrik uji dan validasi, split kronologis, transformasi data latih, konvergensi, aturan fuzzy, integritas bundle, delapan halaman, navigasi ringkasan, serta prakiraan manual dan validasi CSV.
- Browser lokal: ringkasan, EDA, input manual, CSV, perbandingan dan unduhan hasil tersedia. Tidak ada exception halaman pada uji terakhir.
- Notebook mandiri: semua sel kode berhasil dieksekusi menggunakan CSV UCI asli yang telah diunduh. Empat checkpoint, prediksi, dan ZIP artefak berhasil dibuat.
- Data baru/CSV alternatif perlu divalidasi sebelum membuat klaim kinerja lapangan.

## 5. Checklist menurut panduan dosen

| Minggu | Bukti teknis dalam paket | Status |
|---|---|---|
| 12 | EDA, kamus fitur, statistik missing, pembagian train/val/test, baseline | Selesai lokal |
| 13 | GA, fitness, 3 seed, log konvergensi, fitur dan baseline vs GA | Selesai lokal |
| 14 | PSO, training RNN, parameter, empat model, checkpoint dan kurva | Selesai lokal |
| 15 | FIS, inferensi end-to-end, 8 halaman, manual/CSV, requirements, QA | Tersedia di dashboard |
| 16 | Panduan demo, source, config, seed, logs, pengujian | Siap demo lokal; presentasi/video, kontribusi aktual, dan submit akhir mengikuti kelompok |

## 6. Batasan dan tanggung jawab

Model memakai satu lokasi/dataset historis, satu split temporal, satu seed pelatihan akhir. Hasil untuk sensor/lokasi/tahun lain tidak dijamin; perlu kalibrasi dan validasi. Fuzzy bukan sistem penilaian paparan atau rekomendasi keselamatan. Catat kontribusi anggota berdasarkan pekerjaan yang benar-benar dilakukan, jangan mengarang bukti.

[Dashboard Streamlit Kelompok 9](https://tubes-kecebut-kel9.streamlit.app/)
