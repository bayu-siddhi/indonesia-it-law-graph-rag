"""Constants used for scraping regulation data, including selectors and regex patterns."""

ALPHABET = "abcdefghijklmnopqrstuvwxyz"

BPK_SELECTORS = {
    "active_regulation": {
        "pagination_box": "ul.pagination.justify-content-center",
        "reg_box": "/html/body/div/div/div[2]/div[2]/div[2]",
        "reg_items": "div.row.mb-8[class='row mb-8']",
        "reg_number": "div.col-lg-8.fw-semibold.fs-5.text-gray-600",
        "reg_title": "div.col-lg-10.fs-2.fw-bold.pe-4",
        "reg_subject": "span.badge.badge-light-primary.mb-2",
        "reg_href": "div.col-lg-10.fs-2.fw-bold.pe-4 a",
    },
    "regulation_metadata": {
        "metadata_box": "/html/body/div/div/div[2]/div[2]/div/div[1]/div[2]/div",
        "metadata_inner_box": "div.container.fs-6",
        "download_box": "/html/body/div/div/div[2]/div[2]/div/div[2]/div[1]",
        "status_box": "/html/body/div/div/div[2]/div[2]/div/div[2]/div[2]",
        "status_inner_box": "div.container.fs-6",
    },
}

BPK_REGEX_PATTERNS = {
    "active_regulation": {
        "page_number": r"p=(\d+)",
        "ineffective_reg": r"dicabut dengan",
        "reg_new_number": r"\b(?:nomor|no\.)\s+(\d+)",
        "reg_old_number": r"(\d+)\/",
        "reg_year": r"tahun\s+(\d+)",
    },
    "regulation_metadata": {
        "reg_title": r"judul\s(.*)",
        "reg_short_title": r"((?=(?:No|No.|Nomor) \d).*Tahun \d{4})",
        "reg_about": r"tentang (.*)",
        "reg_amendment_number_1": r"^perubahan atas",
        "reg_amendment_number_2": r"^perubahan (.+) atas",
        "reg_number": r"nomor\s(\d+)",
        "reg_type": r"bentuk\s(.*)",
        "reg_short_type": r"bentuk singkat\s(.*)",
        "reg_year": r"tahun\s(.*)",
        "reg_issue_place": r"tempat penetapan\s(.*)",
        "reg_issue_date": r"tanggal penetapan\s(.*)",
        "reg_effective_date": r"tanggal berlaku\s(.*)",
        "reg_subject": r"subjek\s(.*)",
        "reg_status": r"status\s(.*)",
        "reg_institution": r"lokasi\s(.*)",
        "status_type": r"(dicabut dengan :|diubah dengan :|mengubah :|mencabut :)",
    },
}

KOMDIGI_SELECTORS = {
    "regulation_product_content": {"regulation_box": "div#produk-content"}
}

KOMDIGI_REGEX_PATTERNS = {
    "_check_ol_tag": {
        "html_tag": r"<\s*([a-zA-Z0-9]+)([^>]*)>",
        "ol_type": r"\b(lower-alpha|decimal|a)\b",
    },
    "_process_parent_element_text": {
        "special_token_pattern_1": (
            r"(^bab \w+$)|(^pasal \w+$)|(^bagian \w+$)|(^paragraf \w+$)|"
            r"(^menimbang$)|(^mengingat$)|(^memutuskan$)|(^menetapkan$)|"
            r"(^memperhatikan$)"
        ),
        "special_token_pattern_2": (
            r"(^dengan rahmat Tuhan Yang Maha Esa)|(^dengan persetujuan)|"
            r"(^agar setiap orang mengetahuinya)|(^ditetapkan di)|"
            r"(^disahkan di)|(^diundangkan di)"
        ),
    },
}
