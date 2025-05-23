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
- Solusi 1: Menggunakan pendekatan supervised learning berupa klasifikasi untuk memprediksi kelompok segmentasi pelanggan berdasarkan fitur-fitur seperti Annual Income, Spending Score, Family Size, Gender, dan Work Experience. Label target dapat diperoleh dari hasil clustering sebelumnya (misalnya hasil KMeans), sehingga klasifikasi bertujuan meniru pola segmentasi tersebut pada data baru.
- Solusi 2: Melatih model klasifikasi seperti K-Nearest Neighbors lalu melakukan evaluasi performa menggunakan metrik seperti accuracy, precision, recall, dan F1-score untuk menilai seberapa baik model memetakan data pelanggan ke segmen yang sesuai.

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
![alt text](https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/gender.png)
![alt text](https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/profesi.png)
![alt text](https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/umur.png)

## Modeling
Tahapan ini menjelaskan proses pembangunan model machine learning untuk menyelesaikan permasalahan segmentasi pelanggan berdasarkan dataset Shop Customer Data.
### Algoritma yang Digunakan
Dalam proyek ini, algoritma utama yang digunakan adalah K-Means Clustering, dengan proses tambahan evaluasi dan optimasi melalui:
- Principal Component Analysis (PCA) untuk reduksi dimensi
- Silhouette Score sebagai metrik evaluasi kualitas cluster
### Tahapan Pemodelan
1. Pra-pemrosesan Data
- Normalisasi fitur numerik menggunakan MinMaxScaler.
- One-hot encoding terhadap variabel kategorikal Gender dan Profession.
- Penghapusan fitur CustomerID, Age, dan Profession karena dinilai tidak valid atau tidak relevan.
2. Penentuan Jumlah Klaster (k)
- Menggunakan metode Elbow dengan bantuan KElbowVisualizer untuk menentukan nilai optimal k.
- Nilai awal optimal yang diperoleh adalah k = 4.
3. Pelatihan Model KMeans
- Model pertama dilatih pada seluruh fitur terpilih dengan k = 4.
- Hasil clustering awal menunjukkan nilai Silhouette Score sebesar 0.19, yang mengindikasikan segmentasi masih kurang optimal.
4. Improvement: Feature Selection & PCA
Fitur dengan variansi sangat rendah dihapus menggunakan VarianceThreshold.
PCA diterapkan untuk mereduksi dimensi menjadi 2 komponen utama.
Model KMeans dilatih ulang pada data PCA dengan berbagai nilai k (2–9).
Ditemukan nilai optimal baru k = 4 dengan Silhouette Score meningkat drastis menjadi 0.9125
### Kelebihan dan Kekurangan KMeans
Kelebihan:
- Cepat dan efisien dalam membagi data ke dalam klaster.
- Mudah diinterpretasi ketika data memiliki bentuk yang jelas (bulat dan seimbang).
- Dapat digunakan dalam kombinasi dengan PCA untuk peningkatan performa.
Kekurangan:
- Sensitif terhadap skala dan outlier.
- Perlu menentukan jumlah klaster di awal.
- Tidak optimal jika bentuk cluster bukan sferis atau distribusi data tidak seimbang.

## Evaluation
Pada proyek ini, karena kasusnya adalah clustering (unsupervised learning), maka metrik evaluasi yang digunakan adalah Silhouette Score.

Silhouette Score mengukur seberapa mirip sebuah data dengan cluster-nya sendiri dibandingkan dengan cluster lain. Nilai skor berada dalam rentang -1 hingga 1:
- Skor mendekati 1 menunjukkan bahwa data sangat cocok dengan cluster-nya dan sangat berbeda dengan cluster lain.
- Skor mendekati 0 menunjukkan bahwa data berada di antara dua cluster.
- Skor negatif (< 0) menunjukkan bahwa data mungkin salah ditempatkan.
Rumus Silhouette Score:
![alt text](https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/rumus.png).

di mana:
- a(i) adalah jarak rata-rata antara titik i dan semua titik lain dalam cluster yang sama.
- b(i) adalah jarak rata-rata dari titik i ke titik-titik di cluster terdekat lainnya.
#### Hasil Evaluasi
- Model awal (KMeans tanpa seleksi fitur dan PCA) menghasilkan Silhouette Score sebesar 0.24. Ini menandakan bahwa hasil klasterisasi awal belum optimal.
- Setelah dilakukan Feature Selection dan PCA (reduksi dimensi ke 2D), Silhouette Score meningkat menjadi 0.673.
- Tahap akhir dilakukan tuning jumlah cluster k dari 2 sampai 9, dan ditemukan bahwa nilai optimal adalah k=4, dengan Silhouette Score tertinggi sebesar 0.9125.
- Visualisasi hasil clustering menunjukkan pemisahan klaster yang jelas dan logis berdasarkan profil pelanggan.
#### Kesimpulan
Penerapan metrik Silhouette Score memungkinkan evaluasi objektif terhadap kualitas segmentasi yang dilakukan. Peningkatan skor dari 0.24 ke 0.9125 menunjukkan bahwa pendekatan kombinasi KMeans + PCA berhasil memperbaiki pemisahan klaster secara signifikan. Hal ini menunjukkan bahwa fitur dan struktur cluster yang digunakan sudah cukup representatif untuk mengelompokkan pelanggan berdasarkan perilaku belanja dan demografi mereka.
