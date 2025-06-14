"""Graph visualizer prompts"""

from langchain_core.prompts import PromptTemplate


CYPHER_GENERATION_TEMPLATE = """
## Tugas: Pastikan Kueri Cypher Mengembalikan Semua Variabel (`RETURN *`)

**Tujuan Anda** adalah memodifikasi 'Kueri Cypher Asli' yang diberikan sehingga kueri hasil akhir **selalu** diakhiri dengan klausa `RETURN *`. Kueri yang dimodifikasi ini akan digunakan untuk memvisualisasikan semua node dan relasi yang ditemukan dalam klausa `MATCH` atau `OPTIONAL MATCH`-nya.

### Instruksi:

1.  **Analisis Kueri Asli:** Baca dan pahami 'Kueri Cypher Asli' yang diberikan.
2.  **Modifikasi Klausa RETURN:**
    *   Temukan klausa `RETURN` di akhir kueri (jika ada).
    *   Ganti seluruh klausa `RETURN` di akhir kueri tersebut dengan `RETURN *`.
    *   Jika kueri tidak memiliki klausa `RETURN` di akhir, tambahkan `RETURN *` di akhir kueri.
3.  **Pertahankan Struktur Kueri Lain:**
    *   Kueri yang diperbaiki **HARUS hanya** menggunakan operasi BACA (seperti `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `UNWIND`). JANGAN PERNAH menghasilkan kueri yang memodifikasi data (operasi WRITE).
    *   **JANGAN PERNAH** memodifikasi klausa lain selain `RETURN`. Ini termasuk `MATCH`, `OPTIONAL MATCH`, `WHERE`, `WITH`, `UNWIND`, `ORDER BY`, `LIMIT`, `SKIP`, atau klausa lainnya.
    *   Secara khusus, **JANGAN PERNAH** memodifikasi arah, tipe, atau properti relasi/edge yang terdapat dalam klausa `MATCH` atau `OPTIONAL MATCH`.
    *   **JANGAN PERNAH** menambahkan atau menghapus node atau relasi dari klausa `MATCH` atau `OPTIONAL MATCH`.
4.  **Penanganan Kueri yang Sudah Sesuai:** Jika 'Kueri Cypher Asli' sudah diakhiri dengan `RETURN *`, kembalikan kueri tersebut apa adanya tanpa perubahan.

### Format Output Hasil Modifikasi:

*   Output Anda **HARUS** berupa satu (1) pernyataan kueri Neo4j Cypher yang sudah dimodifikasi.
*   **JANGAN** sertakan teks lain apa pun (penjelasan, pengantar, penutup, permintaan maaf, dll.) sebelum atau sesudah kueri.
*   Kueri yang dimodifikasi **HARUS** dibungkus dalam tiga backticks (```) diikuti dengan cypher untuk penyorotan sintaks.
*   Contoh format output:

    ```cypher
    MATCH (n:NodeContoh)-[r:RELASI_CONTOH]->(m:NodeLain)
    WHERE n.properti = 'nilai'
    RETURN *
    ```

### Kueri Cypher Asli:

Berikut ini adalah kueri Cypher yang harus Anda analisis dan modifikasi:

```cypher
{question}
```

### Kueri Cypher yang Dimodifikasi:
""".strip()

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_FIX_TEMPLATE = """
## Tugas: Perbaikan Kueri Cypher yang Dimodifikasi Berdasarkan Error

**Tujuan Anda** adalah bertindak sebagai ahli kueri Neo4j Cypher untuk menganalisis 'Pesan Error Cypher' dan memperbaiki 'Kueri Cypher yang Dimodifikasi' yang menyebabkannya. Perbaikan harus dilakukan dengan hati-hati berdasarkan informasi yang tersedia, sambil tetap memastikan kueri hasil akhir diakhiri dengan `RETURN *`.

### Informasi yang Diberikan:

*   **Kueri Cypher yang Dimodifikasi (Menyebabkan Error):**
    ```cypher
    {cypher_query}
    ```
    *(Ini adalah kueri yang sebelumnya dimodifikasi untuk menambahkan `RETURN *` dan menghasilkan error saat dieksekusi.)*

*   **Pesan Error Cypher:**
    ```
    {cypher_error}
    ```
    *(Pesan ini menjelaskan mengapa 'Kueri Cypher yang Dimodifikasi' tidak dapat dieksekusi.)*

### Instruksi Perbaikan:

1.  **Analisis Error dan Kueri:** Baca dan pahami 'Pesan Error Cypher'. Analisis 'Kueri Cypher yang Dimodifikasi' untuk mengidentifikasi bagian mana yang kemungkinan menyebabkan error, dengan merujuk pada pesan error.
2.  **Lakukan Perbaikan:** Modifikasi 'Kueri Cypher yang Dimodifikasi' untuk memperbaiki error yang teridentifikasi.
3.  **Pertahankan Aturan Modifikasi (`RETURN *`):** Pastikan kueri yang diperbaiki **HARUS tetap** diakhiri dengan klausa `RETURN *`. JANGAN ubah klausa `RETURN *` ini menjadi hal lain.
4.  **Hanya Kueri BACA (READ Only):** Kueri yang diperbaiki **HARUS hanya** menggunakan operasi BACA (seperti `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `WITH`, `UNWIND`). JANGAN PERNAH menghasilkan kueri yang memodifikasi data (operasi WRITE).
5.  **Hasilkan Kueri yang Diperbaiki:** Buat kueri Cypher yang sudah benar, valid, dan siap dieksekusi, yang memperbaiki error asli sambil tetap mempertahankan struktur kueri asli (kecuali klausa `RETURN` yang harus `RETURN *`).

### Format Output Hasil Perbaikan:

*   Output Anda **HARUS** berupa satu (1) pernyataan kueri Neo4j Cypher yang sudah diperbaiki.
*   **JANGAN** sertakan teks lain apa pun (penjelasan, pengantar, penutup, permintaan maaf, dll.) sebelum atau sesudah kueri.
*   Kueri yang diperbaiki **HARUS** dibungkus dalam tiga backticks (```) diikuti dengan `cypher` untuk penyorotan sintaks.
*   Contoh format output:

    ```cypher
    MATCH (n:NodeContoh)-[r:RELASI_CONTOH]->(m:NodeLain)
    WHERE n.properti = 'nilai'
    RETURN *
    ```

### Kueri Cypher yang Diperbaiki:
""".strip()

CYPHER_FIX_PROMPT = PromptTemplate(
    input_variables=["cypher_query", "cypher_error"], template=CYPHER_FIX_TEMPLATE
)
