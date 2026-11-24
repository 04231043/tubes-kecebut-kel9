# Prakiraan CO satu jam ke depan

Proyek Kecerdasan Buatan TE2514031, Teknik Elektro ITK, kelas B, kelompok 9. Aplikasi memprakirakan konsentrasi karbon monoksida pada jam berikutnya dari riwayat respons sensor dan cuaca. Keluaran fuzzy menunjukkan prioritas verifikasi sensor pada skala internal proyek.

## Jalankan dashboard

Gunakan Python 3.11 atau 3.12. Jalankan perintah berikut dari folder yang berisi app.py. Setelah dependensi terpasang, dashboard dan contoh prediksi dapat digunakan luring.

```sh
python -m venv .venv
# Linux atau macOS
source .venv/bin/activate
# Di Windows PowerShell, gunakan .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Pilih **Prediksi dan simulasi** lalu **Hitung prakiraan** untuk menggunakan enam baris contoh. Berkas yang sama ada di `artifacts/uci-v1/contoh_sensor.csv`. Untuk 5 Februari 2005 pukul 03.00, hasil contoh adalah 1,492 mg/m³, skor 24,24 dan prioritas Terpantau. CSV yang lebih panjang menghasilkan satu prediksi per jendela. Tanggal harus teratur setiap jam. Masukan −200 dan sel kosong menandai pembacaan hilang. Jika suatu kanal tidak memiliki pembacaan dalam empat jam terakhir, baris keluaran ditandai “Data kurang”.

## Eksperimen

Sumbernya [Air Quality, UCI, DOI 10.24432/C59K5F](https://doi.org/10.24432/C59K5F). Arsip asli diambil sekali saat pelatihan. Kode memeriksa SHA-256 CSV, membaca pemisah titik koma dan koma desimal, serta mengabaikan baris dan kolom kosong. Eksplorasi harian yang ditampilkan aplikasi dan seluruh prediksi uji disimpan bersama checkpoint.

```sh
python train.py --out runs/eksperimen-1
python -m unittest -v test_pipeline test_dashboard
```

Pilih folder keluaran baru setiap kali mengulang pelatihan. Jika CSV resmi sudah tersedia, gunakan `python train.py --csv AirQualityUCI.csv --out runs/eksperimen-2`. Di dashboard, tab **Pelatihan ulang** juga mendukung CSV latih dengan delapan kandidat fitur dan CO(GT). Notebook mandiri `Forecasting_CO_Kelompok_9.ipynb` dapat dijalankan sel demi sel di Colab. Ia menyertakan kode eksperimen dan mengambil sendiri data UCI. Ketika melatih kembali, waktu komputasi dapat berbeda menurut perangkat dan jumlah thread BLAS.

Tabel memakai 1.302 target uji yang sama. MAE dan RMSE menggunakan mg/m³.

| Konfigurasi | Fitur | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| RNN dasar | 8 | 0,4721 | 0,6404 | 0,7419 |
| GA + RNN | 4 | 0,4569 | 0,6518 | 0,7326 |
| PSO + RNN | 8 | 0,4506 | 0,6326 | 0,7482 |
| GA + PSO + RNN | 4 | 0,4462 | 0,6363 | 0,7452 |
| Persistensi (CO terakhir) | 1 nilai CO | 0,5060 | 0,7764 | 0,6206 |

Split kronologis 70/15/15 berdasarkan jam target. Median dan standardisasi dipelajari pada bagian latih. Label CO hilang dikeluarkan. Fitur berakhir satu jam sebelum target. Untuk menyamakan waktu target pada semua konfigurasi, evaluasi mensyaratkan delapan jam riwayat dan sedikitnya satu pembacaan setiap kanal selama empat jam sebelum label. Seed GA/PSO 7, 19, 42. RNN pencarian memakai 1.024 jendela latih, lima epoch dan inisialisasi 271. RNN akhir memakai seluruh 4.957 jendela latih dengan inisialisasi 42. Empat model dibekukan memakai validasi sebelum evaluasi uji.

## Isi proyek

| Berkas | Fungsi |
|---|---|
| app.py | Delapan halaman Streamlit dan pelatihan ulang opsional per sesi |
| air_data.py | Pembacaan CSV UCI, data hilang, pembagian temporal dan standardisasi |
| optimize.py | GA turnamen, crossover, mutasi, elitisme, dan PSO dengan pbest/gbest |
| recurrent.py | RNN Elman, Adam, backpropagation through time dan checkpoint validasi |
| train.py | Empat eksperimen dan pembanding persistensi |
| inference.py, fuzzy_system.py | Inferensi bersama untuk input manual maupun CSV dan fuzzy Mamdani |
| artifacts/uci-v1/ | Empat bobot NPZ, preprocessing, riwayat semua kandidat, kurva, statistik dan prediksi |
| test_pipeline.py, test_dashboard.py | Verifikasi split, checkpoint dan metrik, fuzzy, navigasi, serta masukan manual/CSV |

Aplikasi memuat hanya bundle di `artifacts/uci-v1/` saat mulai. Nama `models/` pada riwayat repositori merujuk ke eksperimen terdahulu. Seluruh perhitungan regresi memakai RNN yang sama dalam pelatihan dan inferensi.

## Integrasi ke repositori kelompok

Versi dasar yang digunakan adalah commit `68e2af89c6972a87e8c697ac080cd55ea0cc33f4` pada [repositori kelompok](https://github.com/vandrialmas/air-quality-streamlit-kelompok9). Arsip penyerahan dapat langsung diekstrak dan dijalankan. Untuk memasukkan perubahan ke checkout Git dari commit dasar, terapkan patch yang disertakan dari akar checkout yang bersih.

```sh
git switch -c perbaikan/eksperimen-co
git apply --check /path/ke/perubahan-git.patch
git apply /path/ke/perubahan-git.patch
```

Pada Streamlit Community Cloud, pilih cabang yang sudah memuat perbaikan, Python 3.12 dan entry point `app.py`. Versi ini diuji pada server lokal menggunakan model tersimpan.
