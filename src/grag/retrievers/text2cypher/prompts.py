"""Text2Cypher retriever prompts"""

from langchain_core.prompts import PromptTemplate


FEW_SHOT_PREFIX_TEMPLATE = """### Contoh:

Berikut adalah beberapa contoh relevan yang menunjukkan bagaimana pertanyaan pengguna 
dapat diterjemahkan menjadi kueri Cypher Neo4j yang sesuai. Gunakan contoh-contoh ini 
sebagai panduan saat Anda membuat kueri untuk 'Pertanyaan Pengguna Saat Ini'.

**Informasi tambahan:** Jika Anda memutuskan bahwa untuk mnenjawab 'Pertanyaan Pengguna 
Saat Ini' diperlukan proses analisis graf, Pengembang basis data graf telah menyediakan 
Graph Projection dari Graph Data Science (GDS) Cypher Neo4j dengan nama `graph`. 
Silahkan menggunakan Graph Projection bernama `graph` tersebut untuk melakukan analisis 
graf.


Tujuan: mendapatkan semua teks Pasal yang isinya mengandung topik tertentu. Gunakan cara 
        ini jika `Pertanyaan Pengguna Saat Ini` bersifat umum / tidak spesifik ingin 
        mendapatkan data apa. Cukup ambil teks Pasal yang mungkin sesuai dengan 
        `Pertanyaan Pengguna Saat Ini`.
Contoh: apakah informasi elektronik itu bersifat privat?
Cypher Neo4j: ```cypher
MATCH (a:Article)
WHERE 
    lower(a.text) CONTAINS 'informasi elektronik'
    OR 
    lower(a.text) CONTAINS 'privat'
// Catatan: Sesuaikan jumlah kata kunci di klausa WHERE tersebut mengikuti pertanyaan pengguna
RETURN a.text AS text
LIMIT 20
```
"""

CYPHER_GENERATION_TEMPLATE = """
## Tugas: Generasi Kueri Cypher Neo4j

**Tujuan Anda** adalah menghasilkan satu (1) pernyataan kueri Cypher Neo4j yang akurat 
dan valid berdasarkan 'Pertanyaan Pengguna Saat Ini' dan 'Skema Database' yang disediakan. 
Kueri ini akan digunakan untuk mengambil informasi dari database graf hukum.

### Instruksi Utama:

1.  **Pahami Pertanyaan Pengguna:** Analisis 'Pertanyaan Pengguna Saat Ini' untuk 
    mengidentifikasi informasi apa yang dibutuhkan.
2.  **Patuhi Skema Database:** Hanya gunakan skema yang secara eksplisit tercantum dalam 
    bagian 'Skema Database' di bawah. Dan JANGAN PERNAH menggunakan elemen skema lain yang 
    tidak terdaftar.
3.  **Hanya Kueri BACA (READ Only):** Kueri yang dihasilkan **HARUS HANYA** berupa operasi 
    BACA (READ), seperti `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `UNWIND`. 
    **JANGAN PERNAH** menghasilkan kueri yang memodifikasi data (WRITE), seperti `CREATE`, 
    `MERGE`, `SET`, `DELETE`, `REMOVE`, `DETACH DELETE`.
5.  **Jika pertanyaan pengguna tidak secara eksplisit menyebutkan ingin mengambil bagian 
    tertentu dari suatu entitas**, maka cukup ambil bagian `.title`-nya. Jika pertanyaan 
    mengisyaratkan isi konten suatu entitas, maka ambil bagian `.text`-nya.
6.  **Pertimbangkan Analisis Graf (Jika Relevan):** Jika pertanyaan pengguna menyiratkan 
    kebutuhan akan analisis graf (seperti mencari pasal paling berpengaruh, entitas yang 
    berperan sebagai penjembatan, rekomendasi pasal, deteksi komunitas tertentu, jalur 
    atau hubungan terpendek antar entitas hukum, dan analisis graf lainnya.) dan skema 
    mendukungnya, rumuskan kueri Cypher yang sesuai menggunakan algoritma GDS (Graph Data 
    Science).
    *   **Informasi tambahan:** Jika Anda memutuskan bahwa diperlukan proses analisis graf, 
        Pengembang basis data graf telah menyediakan Graph Projection dari Graph Data Science 
        (GDS) Cypher Neo4j dengan nama `"graph"`. Silahkan gunakan Graph Projection bernama 
        `"graph"` tersebut untuk melakukan analisis graf.
7.  Dalam klausa RETURN dari kode Cypher Neo4j yang dibuat, **wajib gunakan nama alias berupa 
    nama yang sama dengan nama properti yang diambil**. Contoh yang benar: `RETURN a.title AS 
    title`, `RETURN a.number AS number`, `RETURN a.name AS name`, `RETURN a.text AS text`, 
    dan sebagainya.

### Glosarium Terminologi:

Berikut adalah pemetaan istilah hukum umum yang mungkin muncul dalam pertanyaan pengguna ke 
elemen yang sesuai dalam Skema Database. Gunakan ini sebagai panduan:

*   **Pasal:** Node dengan label `Article`
*   **Definisi:** Node dengan label `Definition`
*   **Bagian Menimbang:** Node dengan label `Consideration`
*   **Bagian Mengingat:** Node dengan label `Observation`
*   **Peraturan:** Node dengan label `Regulation`
*   **Ayat:** Tidak ada node terpisah untuk Ayat. Ayat merupakan bagian dari properti `.text` 
    di dalam node `Article`. Jika pertanyaan pengguna merujuk pada 'Ayat', kueri Anda harus 
    menargetkan dan mengembalikan node `Article` induk yang memuat Ayat tersebut, BUKAN mencoba 
    mengembalikan node Ayat (karena tidak ada).

### Jenis Peraturan yang Tersedia:

Saat mengueri node `Regulation`, perhatikan pemetaan berikut untuk properti jenis peraturan 
(properti `type` pada node `Regulation`):

*   **Peraturan Menteri Komunikasi dan Informatika:** disingkat "PERMENKOMINFO".
*   **Undang-Undang:** disingkat "UU".
*   **Peraturan Pemerintah:** disingkat "PP".

Gunakan singkatan ini dalam klausa `WHERE` atau kondisi lain saat memfilter berdasarkan jenis 
peraturan.

### Skema Database:

Berikut adalah skema lengkap database graf Neo4j yang tersedia.

{schema}

{example}

### Format Output:

*   Output Anda HARUS berupa satu (1) pernyataan kueri Cypher Neo4j yang valid.
*   Kueri Cypher Neo4j yang Anda hasilkan **HARUS** dibungkus dalam tiga backticks (```) 
    diikuti dengan `cypher` untuk penyorotan sintaks.
*   JANGAN sertakan teks lain apa pun (penjelasan, pengantar, penutup, permintaan maaf, dll.) 
    sebelum atau sesudah kueri.
*   Pastikan kueri Cypher Neo4j yang Anda hasilkan siap dieksekusi.
*   Jika 'Pertanyaan Pengguna Saat Ini' tidak dapat dipetakan ke kueri Cypher yang valid 
    menggunakan elemen skema yang diberikan, output Anda harus KOSONG (tidak ada teks sama 
    sekali).

### Pertanyaan Pengguna Saat Ini:

{question}

### Kueri Cypher yang Sesuai:
""".strip()

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "example", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

CYPHER_FIX_TEMPLATE = """
## Tugas: Perbaikan Kueri Cypher Neo4j

**Tujuan Anda** adalah bertindak sebagai ahli kueri Cypher Neo4j untuk memperbaiki 
'Kueri Cypher yang Dihasilkan Sebelumnya` yang salah. Perbaikan harus didasarkan pada 
'Pesan Error Cypher' yang diberikan dan mematuhi 'Skema Database' yang tersedia.

### Informasi yang Diberikan:

*   **Kueri Cypher yang Dihasilkan Sebelumnya:**
    ```    
    {cypher_query}
    ```

*   **Pesan Error Cypher:**
    ```
    {cypher_error}
    ```
    *(Pesan ini menjelaskan mengapa kueri di atas tidak dapat dieksekusi.)*

*   **Skema Database Neo4j:**
    
    {schema}
    
    *(Gunakan skema ini sebagai referensi tunggal untuk label node, tipe relasi, dan 
    properti yang valid.)*

### Instruksi Perbaikan:

1.  **Analisis Pesan Error:** Baca dan pahami akar masalah yang dijelaskan dalam 
    'Pesan Error Cypher'.
2.  **Periksa Kueri Asli:** Identifikasi bagian mana dari 'Kueri Cypher yang Dihasilkan' 
    yang kemungkinan menyebabkan error, dengan merujuk pada pesan error.
3.  **Patuhi Skema Database:** Hanya gunakan skema yang secara eksplisit tercantum dalam 
    bagian 'Skema Database' di atas. Dan JANGAN PERNAH menggunakan elemen skema lain yang 
    tidak terdaftar.
4.  **Hanya Kueri BACA (READ Only):** Pastikan kueri yang diperbaiki **HARUS tetap** hanya 
    menggunakan operasi BACA (seperti `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, 
    `UNWIND`). **JANGAN PERNAH** menghasilkan kueri yang memodifikasi data (operasi TULIS: 
    `CREATE`, `MERGE`, `SET`, `DELETE`, `REMOVE`, `DETACH DELETE`).

### Format Output:

*   Output Anda **HARUS** berupa satu (1) pernyataan kueri Cypher Neo4j yang sudah diperbaiki.
*   **JANGAN** sertakan teks lain apa pun (penjelasan, pengantar, penutup, permintaan maaf, 
    dll.) sebelum atau sesudah kueri.
*   Kueri yang diperbaiki **HARUS** dibungkus dalam tiga backticks (```) diikuti dengan 
    `cypher` untuk penyorotan sintaks.

### Kueri Cypher yang Diperbaiki:
""".strip()

CYPHER_FIX_PROMPT = PromptTemplate(
    input_variables=["schema", "cypher_query", "cypher_error"],
    template=CYPHER_FIX_TEMPLATE,
)

CYPHER_QA_TEMPLATE = """
## Tugas: Merumuskan Jawaban Berdasarkan Data Hukum

**Tujuan Anda** adalah bertindak sebagai Asisten Hukum Cerdas untuk merumuskan jawaban yang 
jelas, akurat, dan mudah dipahami oleh pengguna. Jawaban Anda **HARUS** sepenuhnya didasarkan 
pada 'Data yang Disediakan' di bawah, sebagai respons langsung terhadap 'Pertanyaan Pengguna'.

### Instruksi Utama:

1.  **Gunakan Data yang Disediakan:**
    *   Data di bagian 'Data yang Disediakan' adalah sumber informasi **tunggal dan otoritatif** 
        Anda.
    *   **JANGAN PERNAH** menggunakan pengetahuan internal Anda atau mencoba memvalidasi
        atau mengoreksi data yang disediakan.
    *   **JANGAN** menambahkan informasi apa pun yang tidak ada dalam 'Data yang Disediakan'.
2.  **Jawab Pertanyaan Pengguna:** Sintesis informasi dari 'Data yang Disediakan' untuk merumuskan 
    jawaban yang langsung dan relevan dengan 'Pertanyaan Pengguna'.
3.  **Format Jawaban:**
    *   Susun jawaban dalam Bahasa Indonesia yang jelas dan mudah dipahami.
    *   Gunakan format Markdown untuk keterbacaan (misalnya, heading, bullet points, bold).
4.  **Sertakan Referensi:** Jika data yang disediakan mencakup informasi referensi (seperti nama 
    peraturan, nomor, tahun, pasal, ayat, sumber), **WAJIB** sertakan referensi tersebut dalam 
    jawaban Anda untuk mendukung klaim yang dibuat.
5.  **Hindari Meta-Komentar:** JANGAN sebutkan bahwa jawaban Anda didasarkan pada 'Data yang 
    Disediakan' atau hasil kueri database. Langsung berikan jawaban seolah-olah Anda mengetahui 
    informasi tersebut.

### Penanganan Data Kosong:

*   Jika bagian 'Data yang Disediakan' kosong atau tidak mengandung informasi relevan yang cukup 
    untuk merumuskan jawaban yang berarti untuk 'Pertanyaan Pengguna', berikan respons tunggal 
    berikut: `Tidak dapat menemukan data yang sesuai dengan permintaan.`

### Informasi yang Diberikan:

**Pertanyaan Pengguna:**
{question}

**Data yang Disediakan:**
{context}

*(Data ini adalah hasil dari eksekusi kueri Cypher yang dibuat berdasarkan 'Pertanyaan Pengguna'. 
Karena itu, data ini kemungkinan besar sudah sangat relevan dan **bisa saja langsung menjadi jawaban 
yang sesuai** untuk pertanyaan tersebut.)*

### Jawaban Akhir:
""".strip()

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)
