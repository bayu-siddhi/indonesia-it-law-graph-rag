"""Class for parsing regulation content from text files and extracting structured data."""

import os
import re
import json
import time
import string
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from .constants import PARSING_REGEX_PATTERNS
from ..utils import list_of_dict_to_json
from ..encodings import REGULATION_CODES, WORD_TO_NUMBER


class RegulationParser:
    """
    A class for parsing regulation content from text files and extracting structured 
    data.
    """

    def __init__(self):
        """
        Initializes the RegulationParser with constants and encoding dictionaries.
        """
        self.WORD_TO_NUMBER = WORD_TO_NUMBER
        self.REGULATION_ENCODING = REGULATION_CODES
        self.REGEX_PATTERNS = PARSING_REGEX_PATTERNS

    def parse_regulations_content(
        self, input_dir: str, json_input: str, json_output: str, verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Parses regulation content from text files and extracts structured data.

        Args:
            input_dir (str): Directory containing regulation Markdown files.
            json_input (str): Path to JSON file containing regulation metadata.
            json_output (str): Path to save the parsed regulation content as JSON.
            verbose (bool, optional): Whether to print progress and error messages.
                Defaults to True.

        Returns:
            List[Dict[str, Any]]: A list of parsed regulation dictionaries.
        """

        # Initialize variables
        article_dict = {}
        durations = []
        result = []
        files = []
        success = 0
        failed = 0

        # Collect Markdown files from input directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".md"):
                files.append((os.path.join(input_dir, filename), filename))

        # Iterate over each Markdown file
        for regulation_file in tqdm(
            iterable=files, desc="Parsing regulations content", disable=not verbose
        ):
            start_time = time.time()
            filepath, filename = regulation_file

            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    # Initialize data
                    text = file.read()
                    regulation_dict = {}
                    definition_list = []

                    # Extract metadata from filename
                    metadata = re.search(
                        self.REGEX_PATTERNS["document"]["metadata"], filename
                    )
                    regulation_type = self.REGULATION_ENCODING["type"][metadata[1]]
                    regulation_year = metadata[2]
                    regulation_num = int(metadata[3])

                    # Create template ID for regulations
                    id_template = (
                        f"{regulation_year}"
                        f"{regulation_type}"
                        f"{str(regulation_num).zfill(3)}"
                        "{reg_section}"
                        "{section_num}"
                        "{extra_section_number}"
                    )

                    # Generate regulation ID
                    regulation_id = id_template.format(
                        reg_section=self.REGULATION_ENCODING["section"]["document"],
                        section_num="000",
                        extra_section_number="00",
                    )

                    # Load regulation metadata from JSON
                    with open(json_input, "r", encoding="utf-8") as json_data:
                        for regulation_data in json.load(json_data):
                            if regulation_data["id"] == regulation_id:
                                regulation_dict = regulation_data
                                break

                    # Initialize content dictionary
                    regulation_dict["content"] = {}

                    # Extract "Menimbang" (consideration) section
                    regulation_dict["content"]["consideration"] = {
                        "id": id_template.format(
                            reg_section=self.REGULATION_ENCODING["section"][
                                "consideration"
                            ],
                            section_num="000",
                            extra_section_number="00",
                        ),
                        "text": re.search(
                            self.REGEX_PATTERNS["main"]["consideration"],
                            string=text,
                            flags=re.IGNORECASE,
                        )[1].strip(),
                    }

                    # Extract "Mengingat" (observation) section
                    regulation_dict["content"]["observation"] = {
                        "id": id_template.format(
                            reg_section=self.REGULATION_ENCODING["section"][
                                "observation"
                            ],
                            section_num="000",
                            extra_section_number="00",
                        ),
                        "text": re.search(
                            self.REGEX_PATTERNS["main"]["observation"],
                            string=text,
                            flags=re.IGNORECASE,
                        )[1].strip(),
                    }

                    # Check if the regulation is an amendment
                    is_amendment = re.search(
                        self.REGEX_PATTERNS["main"]["amendment_to"],
                        string=regulation_dict["about"],
                        flags=re.IGNORECASE,
                    )

                    if is_amendment:
                        parse_amendment_reg_result = self._parse_amendment_regulation(
                            text=text,
                            id_template=id_template,
                            regulation_dict=regulation_dict,
                            definition_list=definition_list,
                            article_dict=article_dict,
                            amended_regulations=regulation_dict["status"]["amend"],
                        )
                        regulation_dict, definition_list, article_dict = (
                            parse_amendment_reg_result
                        )
                    else:
                        parse_base_reg_result = self._parse_base_regulation(
                            text=text,
                            id_template=id_template,
                            regulation_dict=regulation_dict,
                            definition_list=definition_list,
                            article_dict=article_dict,
                        )
                        regulation_dict, definition_list, article_dict = (
                            parse_base_reg_result
                        )

                    result.append(regulation_dict)
                    success += 1

            except Exception as e:
                if verbose:
                    failed += 1
                    print(f"ERROR parsing content of {filename}")
                    print(e)

            durations.append(time.time() - start_time)

        # Save parsed results to JSON file
        list_of_dict_to_json(data=result, output_path=json_output)

        # Print summary if verbose mode is enabled
        if verbose:
            print("=" * 80)
            print(f"{'Input directory':<20}: {input_dir}")
            print(f"{'Input JSON':<20}: {json_input}")
            print(f"{'Output JSON':<20}: {json_output}")
            print(f"{'Total regulations':<20}: {len(files)} regulations")
            print(f"{'Total success':<20}: {success} regulations")
            print(f"{'Total failed':<20}: {failed} regulations")
            print(f"{'Total articles':<20}: {len(article_dict)} articles")
            print(f"{'Total time':<20}: {round(sum(durations), 3)} seconds")
            print(
                f"{'Average time/file':<20}: "
                f"{round(sum(durations) * 1000 / success, 3)} miliseconds"
            )
            print("=" * 80)

        return result

    def _parse_base_regulation(
        self,
        text: str,
        id_template: str,
        regulation_dict: Dict[str, Any],
        definition_list: List[Dict[str, str]],
        article_dict: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
        """
        Parses the base regulation text to extract chapters, parts, paragraphs,
        and articles.

        Args:
            text (str): The regulation text.
            id_template (str): Template for generating regulation IDs.
            regulation_dict (Dict[str, Any]): Dictionary containing regulation
                metadata and parsed content.
            definition_list (List[Dict[str, str]]): List of legal definitions
                extracted from the text.
            article_dict (Dict[str, Dict[str, Any]]): Dictionary to store
                parsed articles.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
                Updated regulation dictionary, definition list, and article dictionary.
        """
        # Initialize variables
        last_article_number = ""
        regulation_dict["content"]["articles"] = {}

        # Extract all chapters from the text
        chapters = re.findall(
            self.REGEX_PATTERNS["main"]["chapter"], string=text, flags=re.IGNORECASE
        )

        if chapters:
            # Process each chapter
            for chapter_num, chapter in enumerate(chapters):
                chapter_number = chapter_num + 1
                chapter_about = (
                    re.search(
                        self.REGEX_PATTERNS["chapter"]["about"],
                        string=chapter,
                        flags=re.IGNORECASE,
                    )[1]
                    .strip()
                    .upper()
                )

                chapter_about = re.sub(
                    r"\n", repl=" - ", string=chapter_about, flags=re.IGNORECASE
                )

                # Extract all parts within the chapter
                parts = re.findall(
                    self.REGEX_PATTERNS["chapter"]["part"],
                    string=chapter.strip() + "\n",
                    flags=re.IGNORECASE,
                )

                if parts:
                    # Process each part
                    for part_num, part in enumerate(parts):
                        part_number = part_num + 1
                        part_about = re.search(
                            self.REGEX_PATTERNS["part"]["about"],
                            string=part,
                            flags=re.IGNORECASE,
                        )[1].strip()

                        part_about = re.sub(
                            r"\n", repl=" - ", string=part_about, flags=re.IGNORECASE
                        )

                        # Extract all paragraphs within the part
                        paragraphs = re.findall(
                            self.REGEX_PATTERNS["chapter"]["paragraph"],
                            string=part.strip() + "\n",
                            flags=re.IGNORECASE,
                        )

                        if paragraphs:
                            # Process each paragraph
                            for paragraph in paragraphs:
                                paragraph_about = re.search(
                                    self.REGEX_PATTERNS["paragraph"]["about"],
                                    string=paragraph,
                                    flags=re.IGNORECASE,
                                )[1].strip()

                                paragraph_about = re.sub(
                                    r"\n",
                                    repl=" - ",
                                    string=paragraph_about,
                                    flags=re.IGNORECASE,
                                )

                                paragraph_number = re.search(
                                    self.REGEX_PATTERNS["paragraph"]["number"],
                                    string=paragraph_about,
                                    flags=re.IGNORECASE,
                                )[1].strip()

                                if paragraph_number == "0":
                                    paragraph_number = ""
                                    paragraph_about = ""

                                # Parse articles within the paragraph
                                (
                                    regulation_dict,
                                    definition_list,
                                    article_dict,
                                    last_article_number,
                                ) = self._parse_articles(
                                    text=paragraph,
                                    chapter_number=str(chapter_number),
                                    chapter_about=chapter_about,
                                    part_number=str(part_number),
                                    part_about=part_about,
                                    paragraph_number=str(paragraph_number),
                                    paragraph_about=paragraph_about,
                                    id_template=id_template,
                                    regulation_dict=regulation_dict,
                                    definition_list=definition_list,
                                    article_dict=article_dict,
                                )
                        else:
                            # If no paragraphs, parse articles within the part
                            (
                                regulation_dict,
                                definition_list,
                                article_dict,
                                last_article_number,
                            ) = self._parse_articles(
                                text=part,
                                chapter_number=str(chapter_number),
                                chapter_about=chapter_about,
                                part_number=str(part_number),
                                part_about=part_about,
                                paragraph_number="",
                                paragraph_about="",
                                id_template=id_template,
                                regulation_dict=regulation_dict,
                                definition_list=definition_list,
                                article_dict=article_dict,
                            )
                else:
                    # If no parts, parse articles within the chapter
                    (
                        regulation_dict,
                        definition_list,
                        article_dict,
                        last_article_number,
                    ) = self._parse_articles(
                        text=chapter,
                        chapter_number=str(chapter_number),
                        chapter_about=chapter_about,
                        part_number="",
                        part_about="",
                        paragraph_number="",
                        paragraph_about="",
                        id_template=id_template,
                        regulation_dict=regulation_dict,
                        definition_list=definition_list,
                        article_dict=article_dict,
                    )
        else:
            # If no chapters, parse the entire text as articles
            regulation_dict, definition_list, article_dict, last_article_number = (
                self._parse_articles(
                    text=text,
                    chapter_number="",
                    chapter_about="",
                    part_number="",
                    part_about="",
                    paragraph_number="",
                    paragraph_about="",
                    id_template=id_template,
                    regulation_dict=regulation_dict,
                    definition_list=definition_list,
                    article_dict=article_dict,
                )
            )

        # Mark the last article's "next_article" field as empty
        regulation_dict["content"]["articles"][last_article_number]["next_article"] = ""

        return regulation_dict, definition_list, article_dict

    def _parse_amendment_regulation(
        self,
        text: str,
        id_template: str,
        regulation_dict: Dict[str, Any],
        definition_list: List[Dict[str, str]],
        article_dict: Dict[str, Dict[str, Any]],
        amended_regulations: List[str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
        """
        Parses an amendment regulation text and updates the regulation dictionary.

        Args:
            text (str): The amendment regulation text.
            id_template (str): Identifier template for articles.
            regulation_dict (Dict[str, Any]): Dictionary to store parsed regulations.
            definition_list (List[Dict[str, str]]): List of definitions extracted
                from the text.
            article_dict (Dict[str, Dict[str, Any]]): Dictionary to store parsed
                articles.
            amended_regulations (List[str]): List of amended regulations.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
                Updated regulation dictionary, definition list, and article dictionary.
        """

        regulation_dict["content"]["articles"] = {}

        # Extract amendment points from the text
        amendment_points = re.findall(
            self.REGEX_PATTERNS["amendment_to"]["amendment_point_1"],
            string=text,
            flags=re.IGNORECASE,
        )

        if not amendment_points:
            # Fallback to another regex pattern if amendment points are not found
            amendment_points = re.search(
                self.REGEX_PATTERNS["amendment_to"]["amendment_point_2"],
                string=text,
                flags=re.IGNORECASE,
            )
            if amendment_points:
                amendment_points = amendment_points[1].strip()
                # Extract first sentence
                first_sentence = re.search(
                    r"^.*", string=amendment_points, flags=re.IGNORECASE
                )[0].strip()
                # Remove first sentence
                amendment_points = [
                    amendment_points.replace(first_sentence, "").strip()
                ]

        # Process each amendment point
        for point in amendment_points:
            # Extract parts within the amendment
            parts = re.findall(
                self.REGEX_PATTERNS["amendment_to"]["part"],
                point.strip() + "\n",
                re.IGNORECASE,
            )

            if parts:
                # Process each part
                for part in parts:
                    part_about = re.search(
                        self.REGEX_PATTERNS["part"]["about"], part, re.IGNORECASE
                    )[1].strip()
                    part_about = re.sub(r"\n", " - ", part_about, flags=re.IGNORECASE)
                    part_number = (
                        re.search(
                            self.REGEX_PATTERNS["part"]["number"],
                            part_about,
                            re.IGNORECASE,
                        )[1]
                        .strip()
                        .lower()
                    )
                    part_number = self.WORD_TO_NUMBER.get(part_number, 0)

                    # Extract paragraphs within the part
                    paragraphs = re.findall(
                        self.REGEX_PATTERNS["chapter"]["paragraph"],
                        part.strip() + "\n",
                        re.IGNORECASE,
                    )

                    if paragraphs:
                        # Process each paragraph
                        for paragraph in paragraphs:
                            paragraph_about = re.search(
                                self.REGEX_PATTERNS["paragraph"]["about"],
                                paragraph,
                                re.IGNORECASE,
                            )[1].strip()
                            paragraph_about = re.sub(
                                r"\n", " - ", paragraph_about, flags=re.IGNORECASE
                            )
                            paragraph_number = re.search(
                                self.REGEX_PATTERNS["paragraph"]["number"],
                                paragraph_about,
                                re.IGNORECASE,
                            )[1].strip()

                            # Parse articles within the paragraph
                            (
                                regulation_dict,
                                definition_list,
                                article_dict,
                                last_article_number,
                            ) = self._parse_articles(
                                text=paragraph,
                                chapter_number="",
                                chapter_about="",
                                part_number=str(part_number),
                                part_about=part_about,
                                paragraph_number=str(paragraph_number),
                                paragraph_about=paragraph_about,
                                id_template=id_template,
                                regulation_dict=regulation_dict,
                                definition_list=definition_list,
                                article_dict=article_dict,
                                amended_regulations=amended_regulations,
                            )

                    else:
                        # If no paragraphs, parse articles within the part
                        (
                            regulation_dict,
                            definition_list,
                            article_dict,
                            last_article_number,
                        ) = self._parse_articles(
                            text=part,
                            chapter_number="",
                            chapter_about="",
                            part_number=str(part_number),
                            part_about=part_about,
                            paragraph_number="",
                            paragraph_about="",
                            id_template=id_template,
                            regulation_dict=regulation_dict,
                            definition_list=definition_list,
                            article_dict=article_dict,
                            amended_regulations=amended_regulations,
                        )

            else:
                # Extract paragraphs if no parts are found
                paragraphs = re.findall(
                    self.REGEX_PATTERNS["amendment_to"]["paragraph"],
                    point.strip() + "\n",
                    re.IGNORECASE,
                )

                if paragraphs:
                    # Process each paragraph
                    for paragraph in paragraphs:
                        paragraph_about = re.search(
                            self.REGEX_PATTERNS["paragraph"]["about"],
                            paragraph,
                            re.IGNORECASE,
                        )[1].strip()
                        paragraph_about = re.sub(
                            r"\n", " - ", paragraph_about, flags=re.IGNORECASE
                        )
                        paragraph_number = re.search(
                            self.REGEX_PATTERNS["paragraph"]["number"],
                            paragraph_about,
                            re.IGNORECASE,
                        )[1].strip()

                        # Parse articles within the paragraph
                        (
                            regulation_dict,
                            definition_list,
                            article_dict,
                            last_article_number,
                        ) = self._parse_articles(
                            text=paragraph,
                            chapter_number="",
                            chapter_about="",
                            part_number="",
                            part_about="",
                            paragraph_number=str(paragraph_number),
                            paragraph_about=paragraph_about,
                            id_template=id_template,
                            regulation_dict=regulation_dict,
                            definition_list=definition_list,
                            article_dict=article_dict,
                            amended_regulations=amended_regulations,
                        )

                else:
                    # Parse articles directly from the amendment point
                    (
                        regulation_dict,
                        definition_list,
                        article_dict,
                        last_article_number,
                    ) = self._parse_articles(
                        text=point,
                        chapter_number="",
                        chapter_about="",
                        part_number="",
                        part_about="",
                        paragraph_number="",
                        paragraph_about="",
                        id_template=id_template,
                        regulation_dict=regulation_dict,
                        definition_list=definition_list,
                        article_dict=article_dict,
                        amended_regulations=amended_regulations,
                    )

        return regulation_dict, definition_list, article_dict

    def _parse_articles(
        self,
        text: str,
        chapter_number: str,
        chapter_about: str,
        part_number: str,
        part_about: str,
        paragraph_number: str,
        paragraph_about: str,
        id_template: str,
        regulation_dict: Dict[str, Any],
        definition_list: List[Dict[str, str]],
        article_dict: Dict[str, Dict[str, Any]],
        amended_regulations: List[str] = [],
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Dict[str, Any]], str]:
        """
        Parses articles from the given legal text and updates the regulation
        dictionary.

        Args:
            text (str): The full legal text to be parsed.
            chapter_number (str): The chapter number.
            chapter_about (str): Description of the chapter.
            part_number (str): The part number.
            part_about (str): Description of the part.
            paragraph_number (str): The paragraph number.
            paragraph_about (str): Description of the paragraph.
            id_template (str): The ID template for generating article IDs.
            regulation_dict (Dict[str, Any]): The dictionary containing regulation
                data.
            definition_list (List[Dict[str, str]]): A list of definitions extracted
                from Article 1.
            article_dict (Dict[str, Dict[str, Any]]): A dictionary to store parsed
                articles.
            amended_regulations (List[str], optional): List of amended regulations.
                Defaults to [].

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Dict[str, Any]], str]:
                Updated regulation dictionary, definition list, and last article number.
        """

        # Store last processed article number
        last_article_number = ""

        # Extract all articles from the text using regex
        articles = re.findall(
            self.REGEX_PATTERNS["chapter"]["article"],
            string=text,
            flags=re.IGNORECASE
        )

        # Process each article found in the text
        for article in articles:
            article_number = re.search(
                self.REGEX_PATTERNS["article"]["number"],
                string=article,
                flags=re.IGNORECASE
            )[1]
            article_text = re.search(
                self.REGEX_PATTERNS["article"]["text"], article, re.IGNORECASE
            )[1].strip()
            article_text = re.sub(r"\n+", "\n", article_text)

            # Generate unique ID for the article
            article_id = self._article_number_to_id(
                article_number, id_template, return_last_six=False
            )

            # Update last processed article number
            last_article_number = article_number

            # Determine previous, next article, and list of amended article IDs
            previous_article = ""
            next_article = ""
            amended_article = []

            if not amended_regulations:
                # If not an amendment, calculate previous and next article IDs
                if article_number != "1":
                    previous_article = id_template.format(
                        reg_section=self.REGULATION_ENCODING["section"]["article"],
                        section_num=str(int(article_number) - 1).zfill(3),
                        extra_section_number="00",
                    )

                next_article = id_template.format(
                    reg_section=self.REGULATION_ENCODING["section"]["article"],
                    section_num=str(int(article_number) + 1).zfill(3),
                    extra_section_number="00",
                )

            else:
                # Handle amendments: Find the next article from amended regulations
                for amended_regulation_id in amended_regulations:
                    # Generate a list containing 2 possible next article IDs
                    amended_regulation_id_template = (
                        amended_regulation_id[:-6]
                        + "{reg_section}{section_num}{extra_section_number}"
                    )
                    pred_next_article_ids = self._get_next_article_ids(
                        article_number, amended_regulation_id_template
                    )

                    # Check if the possible next article ID exists,
                    # save and stop if it does
                    for pred_next_article_id in pred_next_article_ids:
                        if pred_next_article_id in article_dict.keys():
                            next_article = pred_next_article_id
                            break
                    if next_article:
                        break

                # Handle amendments: Find the previous article from amended regulations
                if article_number.isdigit() and article_number != "1":
                    # Find the largest existing previous article ID within the same
                    # regulation by filtering article IDs that have the same numeric
                    # article number (excluding letter variations)
                    filtered_ids = []
                    for regulation_id in [regulation_dict["id"]] + amended_regulations:
                        filtered_ids += list(
                            filter(
                                lambda x: x.startswith(
                                    str(int(regulation_id[:-6] + article_id[9:13]) - 1)
                                ),
                                article_dict.keys(),
                            )
                        )

                    # Create a dictionary with the prefix as the key
                    # and the last two digits as the value.
                    id_groups = {}
                    for entity_id in filtered_ids:
                        prefix, suffix = (
                            entity_id[:-2],
                            entity_id[-2:],
                        )  # Split prefix and suffix (last two digits)
                        if prefix not in id_groups or suffix > id_groups[prefix]:
                            id_groups[prefix] = (
                                suffix  # Store the largest number for each prefix
                            )

                    # Collect article IDs that have the largest suffix values
                    max_suffix_ids = [
                        entity_id for entity_id in filtered_ids
                        if entity_id[-2:] == id_groups[entity_id[:-2]]
                    ]

                    # Select the highest article ID from the results
                    # as the previous article
                    previous_article = max(max_suffix_ids)

                else:
                    # Attempt to find the previous article ID from the lettered article
                    for regulation_id in [regulation_dict["id"]] + amended_regulations:
                        # Generate possible previous article IDs
                        prev_regulation_id_template = (
                            regulation_id[:-6]
                            + "{reg_section}{section_num}{extra_section_number}"
                        )
                        pred_prev_article_id = self._get_previous_article_id(
                            article_number, prev_regulation_id_template
                        )

                        # Check if the possible previous article ID exists,
                        # save and stop if it does
                        if pred_prev_article_id in article_dict.keys():
                            previous_article = pred_prev_article_id
                            break

                # If the previous article belongs to the same regulation,
                # update its next_article field
                if previous_article.startswith(regulation_dict["id"][:-6]):
                    previous_article_number = self._id_to_article_number(
                        previous_article
                    )
                    regulation_dict["content"]["articles"][previous_article_number][
                        "next_article"
                    ] = article_id
                    article_dict[previous_article]["next_article"] = article_id

                # Retrieving data for the amendment article
                fetched = False
                for amended_regulation_id in amended_regulations:
                    amended_article_id = amended_regulation_id[:-6] + article_id[-6:]
                    if amended_article_id in article_dict.keys():
                        if not fetched:
                            # Update the chapter/part/paragraph number/about to follow
                            # the amended article only if the current amandment article
                            # does not already have a new number/about
                            chapter_number = (
                                chapter_number
                                if chapter_number
                                else article_dict[amended_article_id]["chapter_number"]
                            )
                            chapter_about = (
                                chapter_about
                                if chapter_about
                                else article_dict[amended_article_id]["chapter_about"]
                            )
                            part_number = (
                                part_number
                                if part_number
                                else article_dict[amended_article_id]["part_number"]
                            )
                            part_about = (
                                part_about
                                if part_about
                                else article_dict[amended_article_id]["part_about"]
                            )
                            paragraph_number = (
                                paragraph_number
                                if paragraph_number
                                else article_dict[amended_article_id][
                                    "paragraph_number"
                                ]
                            )
                            paragraph_about = (
                                paragraph_about
                                if paragraph_about
                                else article_dict[amended_article_id]["paragraph_about"]
                            )
                            fetched = True

                        # Store amended article ID
                        amended_article.append(amended_article_id)

                if previous_article:
                    # If the chapter/part/paragraph data is still empty, copy it
                    # from the previous article. This usually happens with newly
                    # added articles due to amendments (e.g., Pasal 40A UU_2024_001).
                    chapter_number = (
                        chapter_number
                        if chapter_number
                        else article_dict[previous_article]["chapter_number"]
                    )
                    chapter_about = (
                        chapter_about
                        if chapter_about
                        else article_dict[previous_article]["chapter_about"]
                    )
                    part_number = (
                        part_number
                        if part_number
                        else article_dict[previous_article]["part_number"]
                    )
                    part_about = (
                        part_about
                        if part_about
                        else article_dict[previous_article]["part_about"]
                    )
                    paragraph_number = (
                        paragraph_number
                        if paragraph_number
                        else article_dict[previous_article]["paragraph_number"]
                    )
                    paragraph_about = (
                        paragraph_about
                        if paragraph_about
                        else article_dict[previous_article]["paragraph_about"]
                    )

            if article_number == "1":
                # Extract definitions if the article contains legal definitions
                if re.search(
                    self.REGEX_PATTERNS["article"]["check_definition"],
                    article_text,
                    re.IGNORECASE,
                ):
                    definitions = re.findall(
                        self.REGEX_PATTERNS["article"]["definition"],
                        article_text,
                        re.IGNORECASE,
                    )

                    for index, definition_data in enumerate(definitions):
                        definition, name = definition_data
                        definition_list.append(
                            {
                                "id": id_template.format(
                                    reg_section=self.REGULATION_ENCODING["section"][
                                        "definition"
                                    ],
                                    section_num=str(index + 1).zfill(3),
                                    extra_section_number="00",
                                ),
                                "name": name.strip(),
                                "definition": definition.strip(),
                            }
                        )

            # Store definition list
            regulation_dict["content"]["definitions"] = definition_list

            # Extract references to other articles within the text
            # if the article does NOT contain the **NO_REF** marker
            all_article_references = []
            if not re.search(
                self.REGEX_PATTERNS["article"]["no_ref"], article_text, re.IGNORECASE
            ):
                all_article_references = self._get_article_id_references(
                    article_text=article_text,
                    current_regulation_id=regulation_dict["id"],
                    id_template=id_template,
                    amended_regulations=amended_regulations,
                    article_dict=article_dict,
                )
            else:
                article_text = re.sub(
                    self.REGEX_PATTERNS["article"]["no_ref"],
                    "",
                    article_text,
                    flags=re.IGNORECASE,
                ).strip()

            # Store article data
            regulation_dict["content"]["articles"][article_number] = {
                "id": article_id,
                "chapter_number": chapter_number,
                "chapter_about": chapter_about,
                "part_number": part_number,
                "part_about": part_about,
                "paragraph_number": paragraph_number,
                "paragraph_about": paragraph_about,
                "article_number": article_number,
                "text": article_text,
                "previous_article": previous_article,
                "next_article": next_article,
                "references": all_article_references,
                "amend": amended_article,
            }

            # Store article to article_dict
            article_dict[article_id] = regulation_dict["content"]["articles"][
                article_number
            ]

        return regulation_dict, definition_list, article_dict, last_article_number

    def _previous_label(self, label: str) -> str:
        """
        Generate the previous label in a sequence where letters decrement
        alphabetically.

        If a letter is not "A", it is decremented. If it is "A", it becomes
        "Z", and the decrementation continues to the left.

        Args:
            label (str): The input label consisting of uppercase letters.

        Returns:
            str: The previous label in the sequence.

        Example:
            _previous_label("B") -> "A"
            _previous_label("C") -> "B"
            _previous_label("ABC") -> "ABB"
            _previous_label("AAA") -> "ZZ"
            _previous_label("BAA") -> "AZZ"
        """
        # Convert string to a list of characters for manipulation
        label = list(label)

        # Iterate backwards to handle letter decrement
        for i in range(len(label) - 1, -1, -1):
            if label[i] != "A":
                # Decrement the letter
                label[i] = chr(ord(label[i]) - 1)
                return "".join(label)
            # If "A", change to "Z"
            # and continue to the previous letter
            label[i] = "Z"

        # If all were "A", remove the first character
        return "".join(label[1:])

    def _next_label(self, label: str) -> str:
        """
        Generate the next label in a sequence where letters increment
        alphabetically.

        If a letter is not "Z", it is incremented. If it is "Z", it becomes
        "A", and the incrementation continues to the left.

        Args:
            label (str): The input label consisting of uppercase letters.

        Returns:
            str: The next label in the sequence.

        Example:
            _next_label("A") -> "B"
            _next_label("B") -> "C"
            _next_label("ABC") -> "ABD"
            _next_label("ZZZ") -> "AAAA"
            _next_label("AZZ") -> "BAA"
        """
        # Convert string to a list of characters for manipulation
        label = list(label)

        # Iterate backwards to handle letter increment
        for i in range(len(label) - 1, -1, -1):
            if label[i] != "Z":
                # Increment the letter
                label[i] = chr(ord(label[i]) + 1)
                return "".join(label)
            # If "Z", change to "A"
            # and continue to the previous letter
            label[i] = "A"

        # If all were "Z", add "A" in front
        return "A" + "".join(label)

    def _letter_to_string_number(self, letter: str, default: str = "00") -> str:
        """
        Convert an alphabetical string into a two-digit numerical representation.

        This function follows an Excel-like numbering system:
        - "A" → "01", "B" → "02", ..., "Z" → "26"
        - "AA" → "27", "AB" → "28", ..., "CU" → "99"
        - If the input contains non-alphabet characters, it returns the default
            value.
        - If the converted number exceeds 99, a ValueError is raised.

        Args:
            letter (str): The alphabetical string to convert (e.g., "A", "Z",
                "AA", "CU").
            default (str, optional): The default value to return if input is
                invalid. Defaults to "00".

        Returns:
            str: The corresponding two-digit numerical representation as a string.

        Raises:
            ValueError: If the converted number exceeds 99.

        Examples:
            - _letter_to_string_number("A") -> "01"
            - _letter_to_string_number("Z") -> "26"
            - _letter_to_string_number("AA") -> "27"
            - _letter_to_string_number("CU") -> "99"
            - _letter_to_string_number("") -> "00"
            - _letter_to_string_number("A1") -> "00"
            - _letter_to_string_number("XYZ") -> ValueError
        """
        # Return default if letter is empty
        if not letter.isalpha():
            return default

        result = 0
        for char in letter:
            result = result * 26 + (ord(char) - ord("A") + 1)

        # Maximum limit 99 ("CU")
        if result > 99:
            raise ValueError(
                f"Letter '{letter}' exceeds the maximum allowed value of 99 ('CU')."
            )

        return str(result).zfill(2)

    def _article_number_to_id(
        self, article_number: str, id_template: str, return_last_six: bool = False
    ) -> str:
        """
        Generate a formatted article ID based on the given article number.

        This function converts an article number (which may contain a numerical
        part and an optional alphabetical suffix) into a standardized ID format
        using a provided template.

        If `return_last_six` is set to True, the function returns only the last
        six characters of the generated article ID.

        Args:
            article_number (str): The article number as a string (e.g., "10A",
                "12").
            id_template (str): A template string for generating the article ID.
            return_last_six (bool, optional): If True, return only the last six
                characters. Defaults to False.

        Returns:
            str: The formatted article ID.

        Examples:
            id_template = "202401001" + "{reg_section}{section_num}{extra_section_number}"
            - _article_number_to_id("10A", id_template) → "202401001101001"
            - _article_number_to_id("12", id_template) → "202401001101200"
            - _article_number_to_id("10A", id_template, return_last_six=True) → "101001"
        """

        # Initialize the article ID
        article_id = ""

        # Check if the article number contains an alphabetical suffix
        article_alphabet = re.search(r"\d+([A-Z]+)", str(article_number), re.IGNORECASE)

        # Format the article ID with the alphabet
        # suffix converted to its numerical equivalent
        if article_alphabet:
            # Extract the numeric part
            number = re.search(r"\d+", str(article_number), re.IGNORECASE)[0]
            article_id = id_template.format(
                reg_section=self.REGULATION_ENCODING["section"]["article"],
                section_num=number.zfill(3),
                extra_section_number=self._letter_to_string_number(article_alphabet[1]),
            )
        # Format the article ID without a letter suffix
        # (defaulting to "00")
        else:
            article_id = id_template.format(
                reg_section=self.REGULATION_ENCODING["section"]["article"],
                section_num=str(article_number).zfill(3),
                extra_section_number="00",
            )

        # Return the last six characters if required
        return article_id[-6:] if return_last_six else article_id

    def _id_to_article_number(self, article_id: str) -> str:
        """
        Convert an article ID into a formatted article representation.

        This function processes the last 5 digits of the article ID by extracting its
        numerical and alphabetical components. The first three digits represent the
        main article number, while the last two digits indicate an alphabetical suffix
        following an Excel-like numbering system (e.g., 1 → A, 26 → Z, 27 → AA).

        Args:
            article_id (str): The 15-digit article ID (e.g., "202401001101001",
                "202401001101200").

        Returns:
            str: The formatted article string (e.g., "10A", "12").

        Examples:
            - _id_to_article_number("202401001101001") -> "10A"
            - _id_to_article_number("202401001101200") -> "12"
        """

        # Extract the last 5 characters to ensure proper formatting
        article_id = article_id[-5:]

        # Extract the first three digits as the numerical part,
        # removing leading zeros
        number_part = str(int(article_id[:3]))

        # Extract the last two digits as the alphabet
        # index like in Excel (1 → A, 26 → Z, 27 → AA, dst.)
        alphabet_index = int(article_id[-2:])

        # If "00", return only the number part
        if alphabet_index == 0:
            return number_part

        # Convert the numeric index to an alphabetical
        # suffix (Excel-like system)
        alphabet_part = ""
        while alphabet_index > 0:
            alphabet_index -= 1
            alphabet_part = chr(ord("A") + (alphabet_index % 26)) + alphabet_part
            alphabet_index //= 26

        return number_part + alphabet_part

    def _get_previous_article_id(self, article_number: str, id_template: str) -> str:
        """
        Generate the previous article ID based on the current article number.

        The function extracts the numeric and alphabetic parts of the article number.
        If the article ends in "A", it removes the letter (e.g., "10A" → "10").
        Otherwise, it decrements the alphabetic part (e.g., "10C" → "10B").
        The input must have an alphabetic part (e.g., "10A", "10B"), not just a number.

        Args:
            article_number (str): The current article number (e.g., "10A", "10B").
            id_template (str): A template string for formatting the output ID.

        Returns:
            str: The previous article ID, formatted using the provided template.

        Examples:
            - _get_previous_article_id("10C", template) -> "10B"
            - _get_previous_article_id("10B", template) -> "10A"
            - _get_previous_article_id("10A", template) -> "10"
        """

        # Extract numeric and alphabet parts
        match = re.match(r"(\d+)([A-Z]+)", article_number, re.IGNORECASE)

        # Return empty if the input format is invalid
        # e.g., "10" without a letter
        if not match:
            return ""

        # Extract and convert numeric part
        article_number = int(match.group(1))
        # Extract alphabet part
        article_alphabet = match.group(2)

        if article_alphabet == "A":
            # If the article ends with "A", remove the letter
            # to return only the number
            prev_section_num = article_number
            prev_extra_section = "00"
        else:
            # Otherwise, decrement the alphabetic part
            # e.g., "10C" → "10B"
            prev_section_num = article_number
            prev_extra_section = self._previous_label(article_alphabet)

        # Format and return the previous article ID
        return id_template.format(
            reg_section=self.REGULATION_ENCODING["section"]["article"],
            section_num=str(prev_section_num).zfill(3),
            extra_section_number=self._letter_to_string_number(
                prev_extra_section, default="00"
            ),
        )

    def _get_next_article_ids(self, article_number: str, id_template: str) -> List[str]:
        """
        Generate a list containing 2 possible next article IDs

        This function determines the possible next article IDs by either:
        - Incrementing the alphabetic part (e.g., "10A" → "10B").
        - Moving to the next number (e.g., "10A" → "11").

        Args:
            article_number (str): The current article number (e.g., "10", "10A").
            id_template (str): A template string for formatting the output IDs.

        Returns:
            list: A list of the next possible article IDs formatted using the template.

        Examples:
            - _get_next_article_ids("10A", template) -> [template("10B"), template("11")]
            - _get_next_article_ids("10", template) -> [template("10A"), template("11")]
        """

        next_article_ids = []

        # Extract numeric and alphabet parts
        match = re.match(r"(\d+)([A-Z]*)", article_number, re.IGNORECASE)
        if not match:
            return []  # Return empty if the input format is invalid

        # Extract and convert numeric part
        article_number = int(match.group(1))
        # Extract alphabet part (if any)
        article_alphabet = match.group(2)

        if article_alphabet:
            # Generate next articles for cases
            # like "10A" → "10B" or "11"
            next_sections = [
                (
                    article_number,
                    self._next_label(article_alphabet),
                ),  # "10A" → "10B"
                (article_number + 1, "00"),  # "10A" → "11"
            ]
        else:
            # Generate next articles for cases
            # like "10" → "10A" or "11"
            next_sections = [
                (article_number, "A"),  # "10" → "10A"
                (article_number + 1, "00"),  # "10" → "11"
            ]

        # Format and store the generated IDs
        for section_num, extra_section in next_sections:
            next_article_ids.append(
                id_template.format(
                    reg_section=self.REGULATION_ENCODING["section"]["article"],
                    section_num=str(section_num).zfill(3),
                    extra_section_number=self._letter_to_string_number(
                        extra_section, default="00"
                    ),
                )
            )

        return next_article_ids

    def _generate_article_range(self, list1: List[str], list2: List[str]) -> List[str]:
        """
        Generate a list of article numbers based on direct references and article ranges.

        This function processes two lists:
        - `list1`: A list of directly referenced article numbers.
        - `list2`: A list of article ranges, where each tuple contains a start and end value
        (e.g., [("10", "12B")] → generates ["10", "11", "12", "12A", "12B"]).

        The function ensures that all numbers in the specified range are included,
        along with letter suffixes (e.g., "A", "B", etc.) if present in the end value.
        The final list is sorted naturally (numerical + alphabetical order).

        Args:
            list1 (list): A list of individual article numbers as strings.
            list2 (list): A list of tuples representing article number ranges (start, end).

        Returns:
            list: A sorted list of all referenced article numbers.

        Examples:
            _generate_article_range(["5", "8"], [("10", "12B")])
                → ["5", "8", "10", "11", "12", "12A", "12B"]
        """

        # Convert list1 to a set for unique references
        set1 = set(list1)
        set2 = set()

        for start, end in list2:
            # Extract the numeric part of the start value
            start_num = int(re.match(r"\d+", start).group())

            # Extract the numeric and optional letter part of the end value
            end_match = re.match(r"(\d+)([A-Z]?)", end, re.IGNORECASE)
            # Extract numeric part of the end range
            end_num = int(end_match.group(1))
            # Extract letter suffix (if any)
            end_letter = end_match.group(2)

            # Add all numeric values within the range
            for i in range(start_num, end_num + 1):
                set2.add(str(i))

            # If the end value has a letter,
            # generate letter suffixes from "A" to the end letter
            if end_letter:
                for letter in string.ascii_uppercase[: ord(end_letter) - ord("A") + 1]:
                    set2.add(f"{end_num}{letter}")

        # Natural sorting function
        # (handles numerical and alphabetical sorting)
        def natural_sort_key(s):
            return [
                int(text) if text.isdigit() else text
                for text in re.split(r"(\d+)", s)
                if text
            ]

        # Combine both sets and return the sorted result
        return sorted(set1.union(set2), key=natural_sort_key)

    def _get_article_id_references(
        self,
        article_text: str,
        current_regulation_id: str,
        id_template: str,
        amended_regulations: List[str],
        article_dict: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        Extract referenced article IDs from the given article text.

        This function identifies article references within the given text, processes
        them into valid article IDs, and attempts to match them with the regulation
        database, considering amendments.

        Args:
            article_text (str): The text containing article references.
            current_regulation_id (str): The regulation ID of the current article.
            id_template (str): The template used for formatting article IDs.
            amended_regulations (list): A list of regulation IDs that amended by the
                current regulation.
            article_dict (dict): A dictionary mapping valid article IDs to their
                details.

        Returns:
            list: A list of referenced article IDs.

        """

        all_article_references = []

        # Extract article reference numbers
        # using predefined regex patterns
        reference_type_1 = list(
            set(
                re.findall(
                    self.REGEX_PATTERNS["article"]["reference_1"],
                    article_text,
                    re.IGNORECASE,
                )
            )
        )
        reference_type_2 = list(
            set(
                re.findall(
                    self.REGEX_PATTERNS["article"]["reference_2"],
                    article_text,
                    re.IGNORECASE,
                )
            )
        )

        if reference_type_1 or reference_type_2:
            # Generate a list of article numbers,
            # considering possible ranges
            article_references = self._generate_article_range(
                reference_type_1, reference_type_2
            )

            # Convert article numbers
            # into formatted article IDs
            article_references = [
                self._article_number_to_id(number, id_template, return_last_six=True)
                for number in article_references
            ]

            if amended_regulations:
                # Iterate through each referenced article number
                for article_reference_num in article_references:
                    # Check the most recent regulation first,
                    # then move to older amendments
                    for regulation_id in sorted(
                        [current_regulation_id] + amended_regulations, reverse=True
                    ):
                        # Generate a possible article ID by combining
                        # the regulation ID and article number
                        other_article_id = regulation_id[:-6] + article_reference_num

                        # If the generated article ID exists in the article
                        # dictionary, store it and stop searching
                        if other_article_id in article_dict.keys():
                            all_article_references.append(other_article_id)
                            break
            else:
                # If no amendments exist, assume all references
                # belong to the current regulation
                for article_reference_num in article_references:
                    all_article_references.append(
                        current_regulation_id[:-6] + article_reference_num
                    )

        return all_article_references
