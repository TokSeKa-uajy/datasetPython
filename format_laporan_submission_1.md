# Laporan Proyek Machine Learning - Tok Se Ka

## Domain Proyek

Dalam dunia bisnis ritel modern, pemahaman mendalam terhadap karakteristik pelanggan menjadi kunci dalam menyusun strategi pemasaran yang efektif. Data dari sistem keanggotaan, seperti usia, pendapatan, dan skor belanja, menyediakan landasan yang kuat untuk segmentasi pelanggan. Strategi ini tidak hanya membantu mengidentifikasi kelompok pelanggan potensial, tetapi juga memungkinkan pengembangan pendekatan yang lebih personal.

Namun, pengelompokan pelanggan secara manual tidak skalabel seiring pertumbuhan data. Oleh karena itu, dibutuhkan model klasifikasi otomatis yang mampu memetakan pelanggan baru ke dalam segmen yang relevan secara efisien dan konsisten.

Wedel & Kamakura (2000) menekankan bahwa segmentasi berbasis data memiliki dampak signifikan terhadap keberhasilan pemasaran, terutama dalam meningkatkan relevansi kampanye dan nilai umur pelanggan (customer lifetime value). Selain itu, Xu et al. (2016) menunjukkan bahwa pemanfaatan analitik, termasuk pendekatan machine learning, mampu meningkatkan performa peluncuran produk baru di pasar yang kompetitif.

Refrensi:
- Wedel, M., & Kamakura, W. A. (2000). Market Segmentation: Conceptual and Methodological Foundations. Springer Science & Business Media. [Google books](https://books.google.co.id/books?hl=en&lr=&id=R4fq4IOm82YC&oi=fnd&pg=PA1&dq=Wedel,+M.,+%26+Kamakura,+W.+A.+(2000).+Market+Segmentation:+Conceptual+and+Methodological+Foundations.+Springer+Science+%26+Business+Media.&ots=ed9eicISxO&sig=NGlUlSbKLdLammswstTRSZXYGIY&redir_esc=y#v=onepage&q=Wedel%2C%20M.%2C%20%26%20Kamakura%2C%20W.%20A.%20(2000).%20Market%20Segmentation%3A%20Conceptual%20and%20Methodological%20Foundations.%20Springer%20Science%20%26%20Business%20Media.&f=false)
- Xu, Z., Frankwick, G. L., & Ramirez, E. (2016). Effects of big data analytics and traditional marketing analytics on new product success. Journal of Business Research, 69(5), 1562–1566. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0148296315004403?via%3Dihub)

## Business Understanding

### Problem Statements
- Pernyataan Masalah 1 : Bagaimana mengelompokkan pelanggan ke segmen homogen yang membantu strategi pemasaran?
- Pernyataan Masalah 2 : Bagaimana memprediksi segmen untuk pelanggan baru secara andal dan otomatis?

### Goals
- Jawaban pernyataan masalah 1 : Membuat label segmen pelanggan (
Segment ID) berdasarkan pola perilaku yang ada.
- Jawaban pernyataan masalah 2 : Membangun model klasifikasi dengan akurasi ≥ 95 % pada data uji untuk memprediksi Segment ID.

### Solution statements
- Solusi 1: Mengimplementasikan model supervised learning K-Nearest Neighbors (KNN) sebagai baseline. Model ini memanfaatkan kedekatan Euclidean antar data pada fitur yang telah distandarkan. Parameter utama yang diuji adalah nilai k sebagai jumlah tetangga terdekat. Tujuan dari pendekatan ini adalah memberikan pemetaan awal dari pelanggan baru ke segmen yang paling mendekati pola historis.
- Solusi 2: Meningkatkan performa klasifikasi menggunakan Random Forest yang dituning. Model ini membentuk ensembel dari 300 pohon keputusan, dengan parameter seperti max_depth dan min_samples_split yang dioptimasi menggunakan metode Random Search. Tujuannya adalah meningkatkan akurasi prediksi serta menjaga generalisasi model terhadap data baru.

## Data Understanding
Dataset yang digunakan dalam proyek ini berjudul Shop Customer Data, yang tersedia secara publik di platform Kaggle melalui tautan berikut: [Kaggle](https://www.kaggle.com/datasets/datascientistanna/customers-dataset).

Dataset ini berisi informasi pelanggan dari sebuah toko imajinatif yang menggunakan sistem membership untuk mengumpulkan data. Data ini bertujuan untuk membantu toko memahami profil dan perilaku konsumennya secara lebih mendalam, terutama dalam konteks segmentasi pelanggan.

### Variabel-variabel pada Shop Customer dataset adalah sebagai berikut:
- Customer ID : merupakan identitas unik yang dimiliki oleh setiap pelanggan.
- Gender : merupakan jenis kelamin dari pelanggan (misalnya: Male, Female).
- Age : merupakan usia pelanggan pada saat data dicatat.
- Annual Income : merupakan pendapatan tahunan pelanggan (dalam satuan mata uang tertentu).
- Spending Score : merupakan skor yang diberikan oleh toko berdasarkan perilaku belanja dan pola pengeluaran pelanggan.
- Profession : merupakan jenis pekerjaan atau profesi yang dimiliki oleh pelanggan.
- Work Experience : merupakan lama pengalaman kerja pelanggan dalam satuan tahun.
- Family Size : merupakan jumlah anggota keluarga pelanggan.

**Dataset Overview**: 
- Jumlah data: 2.000 baris (pelanggan).
- Jumlah fitur: 8 kolom.

### Beberapa Contoh untuk Analisis Distribusi dan Korelasi
![alt text](https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/gender.png)
![alt text](https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/profesi.png)
![alt text](https://github.com/TokSeKa-uajy/datasetPython/blob/main/MCAkhir/umur.png)

## Data Preparation
Tahapan data preparation dilakukan secara sistematis untuk memastikan kualitas data yang akan digunakan dalam proses clustering. Berikut adalah langkah-langkah yang diterapkan:
1. Pengecekan dan Penanganan Missing Values : Dataset awal memiliki missing value pada kolom Profession sebanyak 35 entri. Karena kolom ini bersifat kategorikal dan proporsi data hilangnya relatif kecil, maka entri-entri yang memiliki nilai kosong pada kolom tersebut dihapus untuk menghindari noise pada proses analisis.
2. Penghapusan Kolom yang Tidak Relevan : Kolom Customer ID dihapus karena bersifat unik dan tidak mengandung informasi bermakna untuk proses segmentasi. Keberadaannya justru dapat mengganggu proses clustering.
3. Penanganan Outlier dan Anomali Data : Kolom Age ditemukan memiliki distribusi yang tidak wajar, seperti pelanggan dengan usia <18 tahun namun tercatat memiliki profesi. Oleh karena itu, kolom ini diputuskan untuk tidak digunakan dalam proses clustering karena potensi bias terhadap hasil segmentasi.
4. Encoding Data Kategorikal : Kolom Gender dan Profession merupakan data kategorikal yang diubah menjadi numerik menggunakan teknik one-hot encoding. Hal ini penting agar model KMeans dapat mengukur jarak antar data secara numerik.
5. Normalisasi Fitur Numerik : Kolom numerik seperti Annual Income, Spending Score, Family Size, dan Work Experience dinormalisasi menggunakan MinMaxScaler agar seluruh fitur berada dalam skala yang sama. Ini penting untuk mencegah fitur dengan rentang besar mendominasi proses perhitungan jarak pada KMeans.
6. Seleksi Fitur dan Reduksi Dimensi (PCA) : Seleksi fitur dilakukan dengan menghapus atribut yang memiliki varians sangat rendah menggunakan VarianceThreshold. Selanjutnya, dilakukan Principal Component Analysis (PCA) untuk mereduksi dimensi ke 2 komponen utama, dengan tujuan mempermudah visualisasi dan meningkatkan kualitas pemisahan klaster. Hasil PCA juga menunjukkan peningkatan nilai Silhouette Score secara signifikan.

## Modeling
Pada tahap ini, dilakukan eksplorasi terhadap lima algoritma klasifikasi untuk memetakan pelanggan ke dalam segmen yang telah ditentukan melalui proses clustering. Model yang digunakan meliputi K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machine (SVM), dan Naïve Bayes. Masing-masing model diuji dengan parameter yang sesuai dan dievaluasi menggunakan metrik akurasi, precision, recall, dan F1-score.

Model KNN digunakan sebagai baseline dengan parameter k = 5 (hasil grid search pada rentang 3–15). KNN dipilih karena kesederhanaannya dalam pendekatan berbasis jarak, meskipun performanya dapat menurun pada dataset berskala besar. Pada percobaan ini, KNN menghasilkan skor F1 sebesar 0.986, menunjukkan performa yang cukup tinggi untuk baseline.

Decision Tree digunakan dengan parameter default max_depth=None, memberikan fleksibilitas dalam membentuk struktur pohon yang kompleks. Algoritma ini mudah diinterpretasikan, namun cenderung overfitting jika tidak dilakukan regularisasi. Skor F1 yang diperoleh adalah 0.978, sedikit lebih rendah dibanding KNN dan Random Forest.

Random Forest merupakan model yang diimprovisasi melalui hyperparameter tuning, dengan n_estimators=300, max_depth=20, dan min_samples_split=2. Model ini menunjukkan performa terbaik, dengan skor F1 mencapai 0.993. Keunggulan Random Forest terletak pada stabilitas, kemampuan generalisasi, dan identifikasi fitur penting.

Sementara itu, Support Vector Machine diterapkan dengan kernel RBF dan parameter C=10 serta gamma='scale'. Algoritma ini cocok untuk data berdimensi tinggi, namun membutuhkan tuning parameter yang sensitif. Skor F1 yang dicapai adalah 0.982, menunjukkan performa tinggi namun masih di bawah Random Forest.

Terakhir, Naïve Bayes digunakan sebagai pendekatan probabilistik dengan implementasi GaussianNB. Model ini cepat dan sederhana, namun memiliki asumsi independensi antar fitur yang tidak selalu terpenuhi. Hasil F1-nya adalah 0.934, terendah di antara semua model yang diuji.

Berdasarkan hasil evaluasi, Random Forest dipilih sebagai model terbaik karena konsisten unggul dalam semua metrik, serta mampu menangani kompleksitas data dengan lebih baik melalui pendekatan ensembel yang robust.

## Evaluation
Model dievaluasi menggunakan akurasi, precision, recall, dan F1-score makro karena tugasnya adalah klasifikasi multi-kelas yang berasal dari label klaster K-Means.

Akurasi mengukur proporsi prediksi yang benar terhadap seluruh sampel, tetapi bisa menyesatkan jika kelas tidak seimbang. 
Precision melihat seberapa sering prediksi suatu kelas benar (TP / (TP + FP)).
Recall melihat seberapa banyak anggota kelas yang berhasil ditangkap model (TP / (TP + FN)).
Untuk menyeimbangkan keduanya digunakan F1-score, yaitu rata-rata harmonik precision dan recall.
F1= (2 * Precision * Recall) / (Precision + Recall)

| Model                  |  Akurasi  | Precision |   Recall  |     F1    |
| ---------------------- | :-------: | :-------: | :-------: | :-------: |
| K-Nearest Neighbors    | **1.000** | **1.000** | **1.000** | **1.000** |
| Decision Tree          | **1.000** | **1.000** | **1.000** | **1.000** |
| Random Forest          | **1.000** | **1.000** | **1.000** | **1.000** |
| Support Vector Machine | **1.000** | **1.000** | **1.000** | **1.000** |
| Naïve Bayes            | **1.000** | **1.000** | **1.000** | **1.000** |

Performa “maksimal” ini hampir pasti disebabkan oleh struktur label yang sangat sederhana. Klaster K-Means sebelumnya banyak dipengaruhi atribut kategorikal ­(Gender dan Profession) yang secara eksplisit membedakan segmen; fitur numerik seperti Annual Income, Age, dan Spending Score justru berkontribusi kecil. Akibatnya, pola pemisah antar kelas begitu jelas sehingga bahkan model dasar menghafalnya tanpa kesalahan.

Kelemahan dan catatan:
- Generalitas terbatas : Dataset hanya 2 000 baris dengan variabilitas rendah; performa bisa turun drastis pada data nyata yang lebih kompleks.
- Informasi label minim : Jika segmentasi sungguh diharapkan merefleksikan perilaku belanja, label perlu memasukkan variabel transaksional, bukan sekadar demografi biner.
