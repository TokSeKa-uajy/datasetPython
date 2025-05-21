# Laporan Proyek Machine Learning - Tok Se Ka

## Domain Proyek

Segmentasi pelanggan menjadi strategi penting dalam pemasaran modern untuk memahami perilaku konsumen dan meningkatkan efektivitas promosi. Data pelanggan dari sistem keanggotaan, seperti usia, pendapatan, dan spending score, dapat dianalisis untuk mengelompokkan konsumen ke dalam segmen yang lebih spesifik.

Masalah yang diangkat adalah kurangnya pemahaman terhadap kelompok pelanggan potensial, yang dapat menyebabkan promosi tidak efektif. Pendekatan seperti clustering memungkinkan bisnis menyusun strategi yang lebih personal dan efisien. Wedel & Kamakura (2000) menekankan bahwa segmentasi berbasis data mampu meningkatkan hasil pemasaran secara signifikan. Selain itu, Xu et al. (2016) menunjukkan bahwa integrasi analitik data besar mendukung keberhasilan produk baru dalam pasar yang kompetitif.

Refrensi:
- Wedel, M., & Kamakura, W. A. (2000). Market Segmentation: Conceptual and Methodological Foundations. Springer Science & Business Media. [Google books](https://books.google.co.id/books?hl=en&lr=&id=R4fq4IOm82YC&oi=fnd&pg=PA1&dq=Wedel,+M.,+%26+Kamakura,+W.+A.+(2000).+Market+Segmentation:+Conceptual+and+Methodological+Foundations.+Springer+Science+%26+Business+Media.&ots=ed9eicISxO&sig=NGlUlSbKLdLammswstTRSZXYGIY&redir_esc=y#v=onepage&q=Wedel%2C%20M.%2C%20%26%20Kamakura%2C%20W.%20A.%20(2000).%20Market%20Segmentation%3A%20Conceptual%20and%20Methodological%20Foundations.%20Springer%20Science%20%26%20Business%20Media.&f=false)
- Xu, Z., Frankwick, G. L., & Ramirez, E. (2016). Effects of big data analytics and traditional marketing analytics on new product success. Journal of Business Research, 69(5), 1562–1566. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0148296315004403?via%3Dihub)

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements
- Pernyataan Masalah 1: Strategi pemasaran toko saat ini belum tersegmentasi dengan baik, sehingga kurang efektif dalam menjangkau pelanggan dengan pendekatan yang sesuai.
- Pernyataan Masalah 2: Tidak diketahui segmen pelanggan mana yang paling potensial untuk ditargetkan guna meningkatkan loyalitas dan pendapatan toko.

### Goals
- Jawaban Masalah 1: Mengelompokkan pelanggan ke dalam segmen-segmen berdasarkan perilaku dan demografi mereka, guna mendukung strategi pemasaran yang lebih terfokus.
- Jawaban Masalah 2: Menentukan segmen pelanggan bernilai tinggi yang berpotensi memberikan kontribusi besar terhadap profit toko.

### Solution statements
- Solusi 1: Menggunakan algoritma unsupervised learning seperti K-Means Clustering untuk membagi pelanggan ke dalam beberapa kelompok berdasarkan variabel seperti Annual Income, Spending Score, dan Age.
- Solusi 2: Melakukan evaluasi performa model clustering menggunakan metrik seperti Silhouette Score dan Davies-Bouldin Index untuk memilih model yang memberikan segmentasi paling optimal.

## Data Understanding
Dataset yang digunakan dalam proyek ini berjudul Shop Customer Data, yang tersedia secara publik di platform Kaggle melalui tautan berikut: [Kaggle](https://www.kaggle.com/datasets/datascientistanna/customers-dataset).

Dataset ini berisi informasi pelanggan dari sebuah toko imajinatif yang menggunakan sistem membership untuk mengumpulkan data. Data ini bertujuan untuk membantu toko memahami profil dan perilaku konsumennya secara lebih mendalam, terutama dalam konteks segmentasi pelanggan.

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Dataset Overview**: 
- Jumlah data: 2.000 baris (pelanggan).
- Jumlah fitur: 8 kolom.

**Variabel-variabel dalam dataset**: 
- Customer ID: Nomor unik yang mengidentifikasi setiap pelanggan.
- Gender: Jenis kelamin pelanggan (Male/Female).
- Age: Usia pelanggan (dalam tahun).
- Annual Income: Pendapatan tahunan pelanggan (dalam satuan ribu, misal: 60 berarti 60.000).
- Spending Score: Skor yang diberikan toko berdasarkan perilaku dan kebiasaan belanja pelanggan, umumnya dalam skala 1–100.
- Profession: Pekerjaan pelanggan (misalnya: Engineer, Doctor, Lawyer, dll.).
- Work Experience: Lama pengalaman kerja pelanggan (dalam tahun).
- Family Size: Jumlah anggota keluarga pelanggan.

### Beberapa Contoh untuk Analisis Distribusi dan Korelasi
[https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/gender.png]
[https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/profesi.png]
[https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/umur.png]

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

