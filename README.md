# 📘 Machine Learning Project -- Playstore App Review Classification

## 🔍 Overview

Proyek ini merupakan implementasi lengkap proses machine learning untuk
melakukan **klasifikasi sentimen pada ulasan aplikasi Playstore**.
Notebook mencakup seluruh tahap mulai dari pengambilan data (scraping),
pembersihan data, pelabelan, eksplorasi tiga pendekatan model, hingga
evaluasi dan perbandingan performa.

Tujuan utama proyek ini adalah: - Mengembangkan model klasifikasi teks
berbasis ulasan Playstore. - Menganalisis perbedaan performa dari tiga
pendekatan utama: **Klasik**, **Deep Learning**, dan **Transfer
Learning**. - Memberikan pipeline end-to-end yang dapat direplikasi dan
dikembangkan lebih lanjut.

------------------------------------------------------------------------

## 📂 1. Scraping Data dari Playstore

Bagian pertama notebook melakukan scraping menggunakan tools tertentu
(misalnya API/selenium/requests). Tahap ini menghasilkan dataset mentah
berupa: - Nama aplikasi - Rating - Review pengguna - Informasi tambahan
yang relevan

Data kemudian dibersihkan untuk menghilangkan duplikasi, noise, HTML
tags, dan karakter tidak penting lainnya.

------------------------------------------------------------------------

## 🏷️ 2. Labelling Dataset

Dataset kemudian diberi label berdasarkan: - Polaritas review (positif,
negatif, netral) - Aturan tertentu (misal rating \> 3 dianggap
positif) - Atau hasil klasifikasi awal yang kemudian direvisi manual

Tahap ini penting karena kualitas label sangat mempengaruhi performa
model.

------------------------------------------------------------------------

## 🧠 3. Pendekatan Klasik (Traditional ML)

Menggunakan teknik NLP konvensional seperti: - Text preprocessing (case
folding, stopwords, stemming/lemmatization) - Ekstraksi fitur
menggunakan TF-IDF / Bag-of-Words - Model ML: - Logistic Regression -
Support Vector Machine (SVM) - Naive Bayes

Kelebihan: - Cepat dan ringan - Mudah di-train dan dipahami

Kekurangan: - Tidak memahami konteks dalam teks panjang - Representasi
kata bersifat statis

------------------------------------------------------------------------

## 🧠 4. Pendekatan Deep Learning

Pada tahap ini, notebook menggunakan arsitektur neural network modern
seperti: - Embedding Layer - LSTM / GRU / BiLSTM - Dense Layer untuk
klasifikasi

Kelebihan: - Mampu belajar representasi kata - Lebih baik dalam memahami
urutan kata dan konteks lokal

Kekurangan: - Training lebih lama - Membutuhkan GPU untuk performa
optimal

------------------------------------------------------------------------

## 🧠 5. Pendekatan Transfer Learning

Bagian ini menggunakan model pra-latih berbasis Transformer seperti: -
BERT - IndoBERT - DistilBERT (lebih kecil dan cepat)

Kelebihan: - Mampu memahami konteks kalimat secara menyeluruh - Performa
jauh lebih baik pada tugas NLP modern - Tidak butuh dataset sangat besar
karena telah dilatih sebelumnya

Kekurangan: - Konsumsi resource lebih besar - Training membutuhkan waktu
lebih lama

------------------------------------------------------------------------

## 📊 6. Evaluasi dan Perbandingan Model

Notebook menampilkan: - Akurasi - Precision, recall, dan F1-score -
Grafik training vs validation loss/accuracy - Tabel perbandingan
performa

Hasil umum: - **Transfer Learning** memberikan akurasi tertinggi -
**Deep Learning** lebih baik dari Klasik, terutama pada data kompleks -
**Model Klasik** menjadi baseline cepat dan efisien

------------------------------------------------------------------------

## 📁 Struktur Notebook

-   Scraping Data dari Playstore
-   Labelling Dataset
-   Pendekatan Klasik
-   Pendekatan Deep Learning
-   Pendekatan Transfer Learning
-   Perbandingan 3 Pendekatan

------------------------------------------------------------------------

## 🚀 Cara Menjalankan Proyek

1.  Install dependensi:

```{=html}
<!-- -->
```
    pip install tensorflow scikit-learn transformers pandas numpy pypandoc

2.  Jalankan notebook secara berurutan.
3.  Pastikan dataset scraping tersedia atau lakukan scraping ulang.
4.  Jalankan sel training untuk ketiga model.
5.  Bandingkan hasil evaluasi.

------------------------------------------------------------------------

## 📝 Kesimpulan

Proyek ini memberikan workflow lengkap dan komprehensif untuk
klasifikasi teks berbasis review Playstore. Melalui perbandingan
pendekatan klasik, deep learning, dan transfer learning, pengguna dapat
memahami kekuatan masing-masing metode dan memilih model yang paling
sesuai untuk kebutuhan produksi atau riset.

------------------------------------------------------------------------

## 💡 Pengembangan Lanjutan

-   Tambahkan analisis sentimen berbasis aspek (Aspect-Based Sentiment
    Analysis)
-   Gunakan model LLM modern seperti GPT atau LLaMA untuk zero-shot
    classification
-   Integrasi ke aplikasi web atau API untuk deployment

------------------------------------------------------------------------

Created automatically from notebook content.
