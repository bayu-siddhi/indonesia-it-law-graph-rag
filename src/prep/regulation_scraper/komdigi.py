"""Scraper class for extracting regulation content from the JDIH Komdigi website."""

import os
import re
import time
from typing import Dict, List
from tqdm import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .constants import ALPHABET, KOMDIGI_SELECTORS, KOMDIGI_REGEX_PATTERNS
from ..encodings import OL_TYPES


class KomdigiScraper:
    """
    A scraper class for extracting regulation content from the JDIH Komdigi website.
    """

    def __init__(self, web_driver: WebDriver) -> None:
        """
        Initializes the KomdigiScraper with a Selenium WebDriver and configuration
        constants.

        Args:
            web_driver (WebDriver): A Selenium WebDriver instance (e.g., Firefox,
                Chrome).

        Returns:
            None
        """
        self.web_driver = web_driver
        self.ALPHABET = ALPHABET
        self.OL_TYPES = OL_TYPES
        self.KOMDIGI_SELECTORS = KOMDIGI_SELECTORS
        self.KOMDIGI_REGEX_PATTERNS = KOMDIGI_REGEX_PATTERNS

    def _check_ol_tag(self, web_element: WebElement) -> str:
        """
        Checks the type of an ordered list (`ol`) HTML tag.

        Extracts the `<ol>` type attribute (e.g., "lower-alpha", "decimal") from
        the provided WebElement and returns a standardized string representation
        of the type.

        Args:
            web_element (WebElement): The WebElement representing the `<ol>` tag.

        Returns:
            result (str): A string representing the standardized type of the ordered
                list (e.g., "decimal", "lower-alpha").
        """
        REGEX_PATTERNS = self.KOMDIGI_REGEX_PATTERNS[self._check_ol_tag.__name__]

        # Get the full <ol> HTML content from the element
        outer_html = web_element.get_attribute("outerHTML")
        html_tag = re.search(REGEX_PATTERNS["html_tag"], outer_html)[0]

        # Extract the <ol> type: ["a", "lower-alpha", "decimal"]
        ol_type = re.search(REGEX_PATTERNS["ol_type"], html_tag)
        ol_type = ol_type[1] if ol_type is not None else "decimal"

        return self.OL_TYPES[ol_type]

    def _process_parent_element_text(
        self, web_element: WebElement, level: int, index: int
    ) -> str:
        """
        Processes the text content of a parent WebElement based on its level.

        Formats the text content of a parent element (e.g., `<p>`, `<div>`)
        according to its level in the document structure. Adds appropriate
        prefixes (e.g., numbering, indentation) and wrappers (e.g., Markdown
        headings).

        Args:
            web_element (WebElement): The WebElement to process.
            level (int): The hierarchical level of the element (1, 2, or 3).
            index (int): The index of the element within its parent.

        Returns:
            result (str): The formatted text content of the WebElement.
        """
        REGEX_PATTERNS = self.KOMDIGI_REGEX_PATTERNS[
            self._process_parent_element_text.__name__
        ]
        text = web_element.text.strip()

        if level == 1:
            if re.search(
                REGEX_PATTERNS["special_token_pattern_1"], text, re.IGNORECASE
            ):
                return f"\n\n## {text}"
            if re.search(
                REGEX_PATTERNS["special_token_pattern_2"], text, re.IGNORECASE
            ):
                return f"\n\n{text}"
            return f"\n{text}"
        if level == 2:
            return f"\n({index}) {text}"
        if level == 3:
            return f"\n\t{self.ALPHABET[index - 1]}. {text}"

    def _process_child_element_text(
        self, web_element: WebElement, ol_type: str, level: int, index: int
    ) -> str:
        """
        Processes the text content of a child WebElement within an ordered list.

        Formats the text content of a child element (e.g., `<li>`) within an `<ol>`
        according to the list's type and the element's level. Adds appropriate
        prefixes (e.g., numbering, lettering, indentation).

        Args:
            web_element (WebElement): The WebElement to process.
            ol_type (str): The type of the ordered list (e.g., "decimal",
                "lower-alpha").
            level (int): The hierarchical level of the element (1, 2, or 3).
            index (int): The index of the element within its parent.

        Returns:
            result (str): The formatted text content of the WebElement.
        """
        text = web_element.text.strip()

        if level == 1:
            if ol_type == "decimal":
                index = f"({index + 1})"
            else:
                index = f"{self.ALPHABET[index]}."
            if text != "":
                return f"\n{index} {text}"
            return f"\n{text}"
        if level == 2:
            if ol_type == "decimal":
                index = f"{index + 1}."
            else:
                index = f"{self.ALPHABET[index]}."
            return f"\n\t{index} {text}"
        if ol_type == "decimal":
            index = f"{index + 1}."
        else:
            index = f"{self.ALPHABET[index]}."
        return f"\n\t\t{index} {text}"

    def _regulation_product_content_element(
        self, web_element: WebElement, index: int, level: int = 1
    ) -> str:
        """
        Recursively processes an HTML element to extract regulation content.

        This function recursively traverses an HTML element (WebElement),
        extracting relevant content based on the element's tag name and structure.
        It handles ordered lists (`<ol>`), paragraphs (`<p>`), and other elements,
        formatting the extracted content appropriately.

        Args:
            web_element (WebElement): The WebElement to process.
            index (int): The index of the element within its parent.
            level (int, optional): The hierarchical level of the element (1, 2, or
                3). Defaults to 1.

        Returns:
            result (str): The extracted and formatted regulation content.
        """
        result = ""

        # Process <ol> (ordered list) elements
        if web_element.tag_name == "ol":
            # Get the <ol> type: lower-alpha or decimal
            ol_type = self._check_ol_tag(web_element=web_element)
            # Retrieve the list of article contents (sections)
            web_element = web_element.find_elements(By.XPATH, "./*")

            # Iterate over all article contents (sections) within <li>
            for i, sub_element in enumerate(web_element):
                # Get all child elements within <li>
                sub_element_component = sub_element.find_elements(By.XPATH, "./*")
                # Check if <li> contains more than one child element
                num_sub_element_component = len(sub_element_component)

                # Check if <li> contains more than one child element
                if num_sub_element_component > 1:

                    # Iterate over all child elements within <li>
                    for sub_sub_element in sub_element_component:

                        # Check if the child element contains a <br>.
                        # If so, it's raw text
                        if sub_sub_element.tag_name == "br":
                            text = sub_element.text.strip()
                            if ol_type == "decimal":
                                index = f"({i + 1})"
                            else:
                                index = f"{self.ALPHABET[i]}."

                            result += f"\n{index} {text}"
                            # If not break, the output will be copied
                            # as many times as there are <br> tags
                            break

                        # If there's no <br>,
                        # then it must be either <p> or another <ol>
                        result += self._regulation_product_content_element(
                            web_element=sub_sub_element,
                            index=i + 1,
                            level=level + 1,
                        )
                # Check if <li> has no child elements
                elif num_sub_element_component == 0:
                    # If it has no child elements, it's just raw text
                    result += self._process_child_element_text(
                        web_element=sub_element, ol_type=ol_type, level=level, index=i
                    )
                # If <li> has only one child element,
                # it must be either a <p> or a <br>
                else:
                    # Check if there's a <br>
                    for sub_sub_element in sub_element_component:
                        if sub_sub_element.tag_name == "br":
                            text = sub_element.text.strip()
                            if ol_type == "decimal":
                                index = f"({i + 1})"
                            else:
                                index = f"{self.ALPHABET[i]}."
                            result += f"\n{index} {text}"
                            # If not break, the output will be copied
                            # as many times as there are <br> tags
                            break

                    # If there's no <br>, then it must be just a <p>
                    result += self._process_child_element_text(
                        web_element=sub_element_component[0],
                        ol_type=ol_type,
                        level=level,
                        index=i,
                    )
        # Process <p> tags or elements
        # without a specific tag
        else:
            result += self._process_parent_element_text(
                web_element=web_element, level=level, index=index
            )

        return result

    def regulation_product_content(
        self,
        regulation_names_and_links: List[Dict[str, str]],
        output_dir: str,
        verbose: bool = True,
    ) -> None:
        """
        Scrapes regulation content from Komdigi URLs and saves it to txt files.

        This function navigates to each regulation URL in the provided list,
        extracts the main content from the "regulation_box" element, formats it,
        and saves the content to a txt file in the specified output directory.

        Args:
            regulation_names_and_links (List[Dict[str, str]]): A list of 
                dictionaries, where each dictionary contains the regulation name 
                ("name") and its URL ("url").
            output_dir (str): The path to the directory where the output Markdown 
                files should be saved.
            verbose (bool, optional): Whether to display progress information. 
                Defaults to True.

        Returns:
            None
        """
        SELECTORS = self.KOMDIGI_SELECTORS[self.regulation_product_content.__name__]
        os.makedirs(output_dir, exist_ok=True)

        net_durations = []
        success = 0
        failed = 0
        global_start = time.time()

        for regulation in tqdm(
            iterable=regulation_names_and_links,
            desc="Scraping regulation content",
            disable=not verbose,
        ):
            local_start = time.time()

            try:
                result = ""
                self.web_driver.get(regulation["url"])
                wait = WebDriverWait(self.web_driver, timeout=10)
                wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, SELECTORS["regulation_box"])
                    )
                )
                # Get the regulation box
                regulation_box = self.web_driver.find_element(
                    By.CSS_SELECTOR, SELECTORS["regulation_box"]
                )
                # Get access to every element in the regulation: [<p>, <ol>]
                regulation_contents = regulation_box.find_elements(By.XPATH, "./*")

                # Accessing every element in the regulation: [<p>, <ol>]
                for index, regulation_content_element in enumerate(regulation_contents):
                    result += self._regulation_product_content_element(
                        web_element=regulation_content_element, index=index, level=1
                    )

                result = result.strip()
                result = re.sub(r"\n{3,}", "\n\n", result)
                result = re.sub(
                    r"(## pasal \w+)(\n{2,})", r"\1\n", result, flags=re.IGNORECASE
                )
                file_output = os.path.join(output_dir, f"{regulation['name']}.md")
                with open(file_output, "w", encoding="utf-8") as file:
                    file.write(result)

                success += 1
                net_durations.append(time.time() - local_start)
                time.sleep(2)  # Break for 2 seconds
            except Exception as e:
                failed += 1
                if verbose:
                    print(f"ERROR scraping content for {regulation['name']}")
                    print(e)

        gross_time = time.time() - global_start

        if verbose:
            print("=" * 80)
            print(f"{'Output directory':<20}: {os.path.join(output_dir)}")
            print(
                f"{'Total regulations':<20}: {len(regulation_names_and_links)} "
                "regulations"
            )
            print(f"{'Total success':<20}: {success} regulations")
            print(f"{'Total failed':<20}: {failed} regulations")
            print(f"{'Gross time':<20}: {round(gross_time, 3)} seconds")
            print(f"{'Net time':<20}: {round(sum(net_durations), 3)} seconds")
            print(
                f"{'Average gross time':<20}: "
                f"{round(gross_time / len(regulation_names_and_links), 3)} seconds"
            )
            print(
                f"{'Average net time':<20}: "
                f"{round(sum(net_durations) / success, 3)} seconds"
            )
            print("=" * 80)
