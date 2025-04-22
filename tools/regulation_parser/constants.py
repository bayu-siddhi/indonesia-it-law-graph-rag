PARSER_REGEX_PATTERNS = {
    "document": {
        "metadata": r"^(\w+)_(\w+)_(\w+)"  # Format: Type, year, and regulation number
    },

    # Main structure in base regulation
    "main": {
        "consideration": r"(?<=## menimbang)([\S\s]*?)(?=## mengingat)",  # (Menimbang)
        "observation": r"(?<=## mengingat)([\S\s]*?)(?=(?:dengan persetujuan|## memperhatikan|## memutuskan))",  # (Mengingat)
        "amendment_to": r"^perubahan",  # Check if it is an amendment regulation
        "chapter": r"(## BAB[\S\s]*?)(?=\n+(?:## BAB|agar setiap orang mengetahuinya|ditetapkan di))"  # List of Chapters
    },

    # For every Chapters (BAB)
    "chapter": {
        "about": r"## (BAB [^#]+)##",  # Chapter name
        "part": r"(## Bagian [\S\s]*?)(?=\n+(?:## Bagian|$))",  # List of Parts
        "paragraph": r"(## Paragraf [\S\s]*?)(?=\n+(?:## Paragraf|$))",  # List of Paragraphs
        "article": r"(## Pasal \w+[\S\s]*?)(?=(?:##|ditetapkan di|$))"  # List of Articles
    },

    # For every Parts (Bagian)
    "part": {
        "about": r"## (Bagian [^#]+)##",  # Part name
        "number": r"Bagian (\w+) -"  # Part number
    },

    # For every Paragraphs (Paragraf)
    "paragraph": {
        "about": r"## (Paragraf [^#]+)##",  # Paragraph name
        "number": r"Paragraf (\w+) -"  # Paragraph number

    },

    # For every articles (Pasal)
    "article": {
        "number": r"## Pasal (\d+\w*)",  # Article number
        "text": r"## Pasal \w+\n*([\S\s]*)",  # Article content
        "no_ref": r"\*{2}NO_REF\*{2}",  # Article marked with **NO_REF**
        "reference_1": r"Pasal (\d+\w*)",  # Type 1 article reference
        "reference_2": r"Pasal (\d+\w*) sampai dengan Pasal (\d+\w*)",  # Type 2 article reference
        "check_definition": r"^dalam (?:undang-undang|peraturan)",  # Check if Article 1 contains definitions
        "definition": r"\(\d+[a-z]?\) ((.*?) (?=(?:adalah|yang selanjutnya|selanjutnya)).*)"  # List of definitions in Article 1
    },

    # For amendment regulation
    "amendment_to": {
        "amendment_point_1": r"(## \d+\.[\S\s]*?)(?=\n+(?:## \d+\.|## Pasal II))", # Type 1 amendment point
        "amendment_point_2": r"(?<=## Pasal I)([\s\S]*?)(?=## Pasal II)",  # Type 2 amendment point
        "chapter": r"(## BAB[\S\s]*?)(?=\n+(?:## BAB|agar setiap orang mengetahuinya|ditetapkan di))",  # List of Chapters
        "part": r"(## Bagian [\S\s]*?)(?=\n+(?:## Bagian|$))",  # List of Parts
        "paragraph": r"(## Paragraf [\S\s]*?)(?=\n+(?:## Paragraf|$))",  # List of Paragraphs
    }
}