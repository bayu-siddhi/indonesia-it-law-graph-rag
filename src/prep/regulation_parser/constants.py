"""Regular expression patterns used for parsing regulation text."""

PARSING_REGEX_PATTERNS = {
    "document": {
        # Format: Type, year, and regulation number
        "metadata": r"^(\w+)_(\w+)_(\w+)"
    },

    # Main structure in base regulation
    "main": {
        # Menimbang
        "consideration": r"(?<=## menimbang)([\S\s]*?)(?=## mengingat)",
        # Mengingat
        "observation": (
            r"(?<=## mengingat)([\S\s]*?)"
            r"(?=(?:dengan persetujuan|## memperhatikan|## memutuskan))"
        ),
        # Check if it is an amendment regulation
        "amendment_to": r"^perubahan",
        # List of Chapters
        "chapter": (
            r"(## BAB[\S\s]*?)"
            r"(?=\n+(?:## BAB|agar setiap orang mengetahuinya|ditetapkan di))"
        )
    },

    # For every Chapters (BAB)
    "chapter": {
        # Chapter name
        "about": r"## (BAB [^#]+)##",
        # List of Parts
        "part": r"(## Bagian [\S\s]*?)(?=\n+(?:## Bagian|$))",
        # List of Paragraphs
        "paragraph": r"(## Paragraf [\S\s]*?)(?=\n+(?:## Paragraf|$))",
        # List of Articles
        "article": r"(## Pasal \w+[\S\s]*?)(?=(?:##|ditetapkan di|$))"
    },

    # For every Parts (Bagian)
    "part": {
        # Part name
        "about": r"## (Bagian [^#]+)##",
        # Part number
        "number": r"Bagian (\w+) -"
    },

    # For every Paragraphs (Paragraf)
    "paragraph": {
        # Paragraph name
        "about": r"## (Paragraf [^#]+)##",
        # Paragraph number
        "number": r"Paragraf (\w+) -"
    },

    # For every Articles (Pasal)
    "article": {
        # Article number
        "number": r"## Pasal (\d+\w*)",
        # Article content
        "text": r"## Pasal \w+\n*([\S\s]*)",
        # Article marked with **NO_REF**
        "no_ref": r"\*{2}NO_REF\*{2}",
        # Article reference type 1
        "reference_1": r"Pasal (\d+\w*)",
        # Article reference type 2
        "reference_2": r"Pasal (\d+\w*) sampai dengan Pasal (\d+\w*)",
        # Check if Article 1 contains definitions
        "check_definition": r"^dalam (?:undang-undang|peraturan)",
        # List of definitions in Article 1
        "definition": (
            r"\(\d+[a-z]?\) ((.*?) (?=(?:adalah|yang selanjutnya|selanjutnya)).*)"
        )
    },

    # For amendment regulation
    "amendment_to": {
        # Amendment point type 1
        "amendment_point_1": r"(## \d+\.[\S\s]*?)(?=\n+(?:## \d+\.|## Pasal II))",
        # Amendment point type 2
        "amendment_point_2": r"(?<=## Pasal I)([\s\S]*?)(?=## Pasal II)",
        # List of Chapters
        "chapter": (
            r"(## BAB[\S\s]*?)(?=\n+(?:## BAB|agar setiap orang mengetahuinya|ditetapkan di))"
        ),
        # List of Parts
        "part": r"(## Bagian [\S\s]*?)(?=\n+(?:## Bagian|$))",
        # List of Paragraphs
        "paragraph": r"(## Paragraf [\S\s]*?)(?=\n+(?:## Paragraf|$))",
    }
}
