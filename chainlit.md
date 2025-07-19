# Sistem Tanya-Jawab Peraturan Teknologi Informasi Indonesia

## Pendahuluan

Proyek ini adalah **Sistem Tanya-Jawab Peraturan Teknologi Informasi Indonesia berbasis Graph-RAG** yang diimplementasikan sebagai aplikasi Chainlit. Sistem ini menggabungkan kemampuan **Large Language Models (LLMs)** dengan **graf pengetahuan yang terstruktur berisi regulasi Teknologi Informasi di Indonesia** (mencakup **63 peraturan**, dengan pengambilan data terakhir dilakukan pada **Februari 2025**) untuk menghasilkan jawaban hukum yang relevan, akurat, dan kontekstual.

Berbeda dengan metode Retrieval-Augmented Generation (RAG) standar yang hanya mengandalkan pencarian berbasis teks atau semantik, **Graph-RAG memanfaatkan hubungan antar entitas dalam graf pengetahuan**, seperti keterkaitan antara peraturan, pasal, dan konsep hukum. Pendekatan ini memungkinkan sistem untuk:

- Menavigasi hubungan antar entitas hukum dengan lebih efisien.
- Menghasilkan jawaban yang lebih rinci dan terstruktur.
- Membantu pengguna mengakses informasi hukum meskipun tidak mengetahui judul atau detail spesifik peraturan.

Dengan mengintegrasikan pendekatan tersebut, sistem ini bertujuan untuk **mempermudah akses masyarakat umum terhadap informasi hukum di bidang Teknologi Informasi**, khususnya bagi mereka yang tidak memiliki latar belakang hukum, serta **mengurangi risiko halusinasi jawaban dari LLM** dengan mengambil informasi dari sumber data hukum resmi.

> [!NOTE]
> - Sistem ini dioptimalkan untuk bahasa Indonesia, dan performanya mungkin berbeda jika digunakan dalam bahasa lain.
> - Baca bagian **"Ketentuan Penggunaan & Disclaimer**" setelah bagian **"Apa yang Bisa Ditanyakan?"** berikut untuk informasi lebih lanjut mengenai ketentuan dan batasan penggunaan proyek ini!

## Apa yang Bisa Ditanyakan?

Anda dapat menanyakan hal-hal seperti:

- **Definisi umum**  
  Contoh: *"Apa itu Penyelenggara Sistem Elektronik?"*
- **Hak dan kewajiban**  
  Contoh: *"Apa kewajiban penyedia layanan telekomunikasi?"*
- **Sanksi atau larangan**  
  Contoh: *"Apa sanksi jika melanggar UU Nomor 11 Tahun 2008?"*
- **Prosedur atau persyaratan**  
  Contoh: *"Bagaimana cara mendapatkan izin penyiaran satelit?"*
- **Isi suatu pasal peraturan**  
  Contoh: *"Apa isi Pasal 28 UU Nomor 11 Tahun 2008?"*
- **Daftar pasal yang sudah tidak berlaku**  
  Contoh: *"Apa saja pasal yang sudah tidak berlaku di UU Nomor 11 Tahun 2008?"*
- **Isi pasal hasil perubahan atau amandemen**  
  Contoh: *"Apa isi terbaru (amandemen) dari Pasal 28 UU Nomor 11 Tahun 2008?"*
- **Hubungan antarpasal atau antarperaturan**  
  Contoh: *"Apa saja pasal yang merujuk ke Pasal 28 UU Nomor 11 Tahun 2008?"*

Sistem ini akan menganalisis pertanyaan Anda, mengambil data hukum yang relevan dari basis data peraturan, dan menyajikan ringkasan jawabannya secara terstruktur.

## Ketentuan Penggunaan & Disclaimer

⚠️ **Disclaimer:**

Proyek ini dan seluruh keluarannya ditujukan hanya untuk **keperluan informasi dan edukasi**. Sistem ini **bukan dibuat untuk menggantikan nasihat hukum resmi**.

- Selalu konsultasikan dengan profesional hukum yang berwenang atau rujuk pada sumber resmi pemerintah sebelum mengambil keputusan atau penafsiran berdasarkan informasi yang diberikan oleh sistem ini.
- Pengembang dan kontributor proyek ini **tidak bertanggung jawab atas tindakan, keputusan, atau kerugian apa pun** yang timbul akibat penggunaan atau penyalahgunaan informasi yang dihasilkan oleh sistem tanya-jawab.
- Sistem ini menggunakan peraturan perundang-undangan yang tersedia untuk umum dan diambil dari Jaringan Dokumentasi dan Informasi Hukum (JDIH) sebagai referensi. Namun, **kelengkapan dan keakuratan jawaban tidak dijamin**, karena sistem dan topik pengembangannya masih memerlukan peningkatan dan evaluasi yang lebih menyeluruh.
- **Ruang lingkup data:** Sistem ini mencakup **63 peraturan Teknologi Informasi Indonesia**, dengan **pengambilan data terakhir dilakukan pada Februari 2025**. Oleh karena itu, mungkin terdapat peraturan yang telah diubah atau sudah tidak berlaku lagi pada saat Anda menggunakan sistem ini.
- Dengan menggunakan aplikasi ini, Anda dianggap telah memahami dan menyetujui seluruh ketentuan ini.

## Daftar Peraturan yang Tercakup

Sistem tanya-jawab ini mencakup **63 peraturan** di bidang peraturan teknologi informasi Indonesia, yaitu:

- **Undang-Undang (UU):**
  - UU Nomor 36 Tahun 1999 tentang Telekomunikasi
  - UU Nomor 40 Tahun 1999 tentang Pers
  - UU Nomor 11 Tahun 2008 tentang Informasi dan Transaksi Elektronik
  - UU Nomor 19 Tahun 2016 tentang Perubahan atas UU ITE
  - UU Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi
  - UU Nomor 1 Tahun 2024 tentang Perubahan Kedua atas UU ITE

- **Peraturan Pemerintah (PP):**
  - PP Nomor 52 Tahun 2000 tentang Penyelenggaraan Telekomunikasi
  - PP Nomor 71 Tahun 2019 tentang Penyelenggaraan Sistem dan Transaksi Elektronik
  - PP Nomor 80 Tahun 2019 tentang Perdagangan Melalui Sistem Elektronik
  - PP Nomor 46 Tahun 2021 tentang Pos, Telekomunikasi, dan Penyiaran

- **Peraturan Menteri Kominfo:**
  - PERMENKOMINFO Nomor 11/PER/M.KOMINFO/4/2007 Tahun 2007 tentang Penyediaan Kewajiban Pelayanan Universal Telekomunikasi
  - PERMENKOMINFO Nomor 26/PER/M.KOMINFO/5/2007 Tahun 2007 tentang Pengamanan Pemanfaatan Jaringan Telekomunikasi Berbasis Protokol Internet
  - PERMENKOMINFO Nomor 38/PER/M.KOMINFO/9/2007 Tahun 2007 tentang Perubahan atas Peraturan Menteri Komunikasi dan Informatika Nomor 11/PER/M.KOMINFO/04/2007 tentang Penyediaan Kewajiban Pelayanan Universal Telekomunikasi
  - PERMENKOMINFO Nomor 41/PER/M.KOMINFO/10/2009 Tahun 2009 tentang Tata Cara Penilaian Pencapaian Tingkat Komponen Dalam Negeri pada Penyelenggaraan Telekomunikasi
  - PERMENKOMINFO Nomor 42/PER/M.KOMINFO/10/2009 Tahun 2009 tentang Tata Cara Memperoleh Izin bagi Lembaga Penyiaran Asing yang Melakukan Kegiatan Peliputan di Indonesia
  - PERMENKOMINFO Nomor 1/PER/M.KOMINFO/1/2010 Tahun 2010 tentang Penyelenggaraan Jaringan Telekomunikasi
  - PERMENKOMINFO Nomor 16/PER/M.KOMINFO/10/2010 Tahun 2010 tentang Perubahan atas Peraturan Menteri Komunikasi dan Informatika Nomor 26/PER/M.KOMINFO/5/2007 tentang Pengamanan Pemanfaatan Jaringan Telekomunikasi Berbasis Protokol Internet
  - PERMENKOMINFO Nomor 29/PER/M.KOMINFO/12/2010 Tahun 2010 tentang Perubahan Kedua atas Peraturan Menteri Komunikasi dan Informatika Nomor : 26/PER/M.KOMINFO/5/2007 tentang Pengamanan Pemanfaatan Jaringan Telekomunikasi Berbasis Protokol Internet
  - PERMENKOMINFO Nomor 2/PER/M.KOMINFO/3/2011 Tahun 2011 tentang Sertifikasi Radio Elektronika dan Operator Radio
  - PERMENKOMINFO Nomor 24/PER/M.KOMINFO/12/2011 Tahun 2011 tentang Perubahan Ketiga atas Peraturan Menteri Komunikasi dan Informatika Nomor 26/Per/M.Kominfo/5/2007 tentang Pengamanan Pemanfaatan Jaringan Telekomunikasi Berbasis Protokol Internet
  - PERMENKOMINFO Nomor 41 Tahun 2012 tentang Penyelenggaraan Penyiaran Lembaga Penyiaran Berlangganan Melalui Satelit, Kabel, dan Terestrial
  - PERMENKOMINFO Nomor 20 Tahun 2013 tentang Jaringan Dokumentasi dan Informasi Hukum Kementerian Komunikasi dan Informatika
  - PERMENKOMINFO Nomor 23 Tahun 2013 tentang Pengelolaan Nama Domain
  - PERMENKOMINFO Nomor 24 Tahun 2013 tentang Layanan Jelajah (Roaming) Internasional
  - PERMENKOMINFO Nomor 14 Tahun 2014 tentang Kampanye Pemilihan Umum melalui Penggunaan Jasa Telekomunikasi
  - PERMENKOMINFO Nomor 38 Tahun 2014 tentang Perubahan atas Peraturan Menteri Komunikasi dan Informatika Nomor 1/PER/M.KOMINFO/1/2010 Tentang Penyelenggaraan Jaringan Telekomunikasi
  - PERMENKOMINFO Nomor 7 Tahun 2015 tentang Perubahan Kedua Atas Peraturan Menteri Komunikasi Dan Informatika Nomor : 01/Per/M.Kominfo/01/2010 Tentang Penyelenggaraan Jaringan Telekomunikasi
  - PERMENKOMINFO Nomor 17 Tahun 2015 tentang Persyaratan Teknis Pembaca Kartu Cerdas Nirkontak (Contactless Smart Card Reader)
  - PERMENKOMINFO Nomor 26 Tahun 2015 tentang Pelaksanaan Penutupan Konten dan/atau Hak Akses Pengguna Pelanggaran Hak Cipta dan/atau Hak Terkait Dalam Sistem Elektronik
  - PERMENKOMINFO Nomor 32 Tahun 2015 tentang Pengelolaan Nomor Protokol Internet
  - PERMENKOMINFO Nomor 35 Tahun 2015 tentang Persyaratan Teknis Alat dan Perangkat Telekomunikasi Jarak Dekat
  - PERMENKOMINFO Nomor 4 Tahun 2016 tentang Sistem Manajemen Pengamanan Informasi
  - PERMENKOMINFO Nomor 5 Tahun 2016 tentang Uji Coba Teknologi Telekomunikasi, Informatika, Dan Penyiaran
  - PERMENKOMINFO Nomor 10 Tahun 2016 tentang Layanan Nomor Tunggal Panggilan Darurat
  - PERMENKOMINFO Nomor 13 Tahun 2016 tentang Hasil Pemetaan Urusan Pemerintahan Daerah di Bidang Komunikasi dan Informatika
  - PERMENKOMINFO Nomor 14 Tahun 2016 tentang Pedoman Nomenklatur Perangkat Daerah Bidang Komunikasi dan Informatika
  - PERMENKOMINFO Nomor 20 Tahun 2016 tentang Perlindungan Data Pribadi Dalam Sistem Elektronik
  - PERMENKOMINFO Nomor 5 Tahun 2017 tentang Perubahan Keempat atas Peraturan Menteri Komunikasi dan Informatika Nomor 26/PER/M.KOMINFO/5/2007 tentang Pengamanan Pemanfaatan Jaringan Telekomunikasi Berbasis Protokol Internet
  - PERMENKOMINFO Nomor 6 Tahun 2017 tentang Penyelenggaraan Layanan Televisi Protokol Internet (Internet Protocol Television)
  - PERMENKOMINFO Nomor 1 Tahun 2018 tentang Akreditasi Lembaga Pelatihan Teknis Bidang Teknologi Informasi dan Komunikasi
  - PERMENKOMINFO Nomor 3 Tahun 2018 tentang Organisasi Dan Tata Kerja Badan Aksesibilitas Telekomunikasi dan Informasi
  - PERMENKOMINFO Nomor 10 Tahun 2018 tentang Pelaksanaan Kewajiban Pelayanan Universal Telekomunikasi dan Informatika
  - PERMENKOMINFO Nomor 11 Tahun 2018 tentang Penyelenggaraan Sertifikasi Elektronik
  - PERMENKOMINFO Nomor 12 Tahun 2018 tentang Penyelenggaraan Telekomunikasi Khusus Untuk Keperluan Instansi Pemerintah atau Badan Hukum
  - PERMENKOMINFO Nomor 16 Tahun 2018 tentang Ketentuan Operasional Sertifikasi Alat dan/atau Perangkat Telekomunikasi
  - PERMENKOMINFO Nomor 4 Tahun 2019 tentang Persyaratan Teknis Alat dan/atau Perangkat Telekomunikasi untuk Keperluan Penyelenggaraan Televisi Siaran dan Radio Siaran
  - PERMENKOMINFO Nomor 5 Tahun 2019 tentang Optimalisasi Penggunaan Spektrum Frekuensi Radio
  - PERMENKOMINFO Nomor 13 Tahun 2019 tentang Penyelenggaraan Jasa Telekomunikasi
  - PERMENKOMINFO Nomor 1 Tahun 2020 tentang Pengendalian Alat Dan/Atau Perangkat Telekomunikasi Yang Tersambung Ke Jaringan Bergerak Seluler Melalui Identifikasi International Mobile Equipment Identity
  - PERMENKOMINFO Nomor 2 Tahun 2020 tentang Perubahan Atas Peraturan Menteri Komunikasi Dan Informatika Nomor 13 Tahun 2019 Tentang Penyelenggaraan Jasa Telekomunikasi
  - PERMENKOMINFO Nomor 5 Tahun 2020 tentang Penyelenggara Sistem Elektronik Lingkup Privat
  - PERMENKOMINFO Nomor 1 Tahun 2021 tentang Perubahan Kedua atas Peraturan Menteri Komunikasi dan Informatika Nomor 13 Tahun 2019 tentang Penyelenggaraan Jasa Telekomunikasi
  - PERMENKOMINFO Nomor 5 Tahun 2021 tentang Penyelenggaraan Telekomunikasi
  - PERMENKOMINFO Nomor 6 Tahun 2021 tentang Penyelenggaraan Penyiaran
  - PERMENKOMINFO Nomor 7 Tahun 2021 tentang Penggunaan Spektrum Frekuensi Radio
  - PERMENKOMINFO Nomor 10 Tahun 2021 tentang Perubahan atas Peraturan Menteri Komunikasi dan Informatika Nomor 5 Tahun 2020 tentang Penyelenggara Sistem Elektronik Lingkup Privat
  - PERMENKOMINFO Nomor 11 Tahun 2021 tentang Perubahan atas Peraturan Menteri Komunikasi dan Informatika Nomor 6 Tahun 2021 tentang Penyelenggaraan Penyiaran
  - PERMENKOMINFO Nomor 13 Tahun 2021 tentang Standar Teknis Alat Telekomunikasi dan/atau Perangkat Telekomunikasi Bergerak Seluler Berbasis Standar Teknologi Long Term Evolution dan Standar Teknologi International Mobile Telecommunication-2020
  - PERMENKOMINFO Nomor 14 Tahun 2021 tentang Perubahan Ketiga atas Peraturan Menteri Komunikasi dan Informatika Nomor 13 Tahun 2019 tentang Penyelenggaraan Jasa Telekomunikasi
  - PERMENKOMINFO Nomor 16 Tahun 2022 tentang Kebijakan Umum Penyelenggaraan Audit Teknologi Informasi dan Komunikasi
  - PERMENKOMINFO Nomor 3 Tahun 2024 tentang Sertifikasi Alat Telekomunikasi dan/atau Perangkat Telekomunikasi
  - PERMENKOMINFO Nomor 4 Tahun 2024 tentang PENYELENGGARAAN URUSAN PEMERINTAHAN KONKUREN BIDANG KOMUNIKASI DAN INFORMATIKA
  - PERMENKOMINFO Nomor 6 Tahun 2024 tentang Tata Cara Seleksi Pengguna Pita Frekuensi Radio
