import os
import re
import time
import tqdm

from src.prep import encodings
from src.prep.regulation_scraper import constants

from selenium import webdriver
from selenium.webdriver.common import by
from selenium.webdriver.support import ui
from selenium.webdriver.support import expected_conditions as EC


class KomdigiScraper:

    def __init__(self, web_driver: webdriver.remote.webdriver.WebDriver) -> None:
        self.web_driver = web_driver
        self.ALPHABET = constants.ALPHABET
        self.OL_TYPES = encodings.OL_TYPES
        self.KOMDIGI_SELECTORS = constants.KOMDIGI_SELECTORS
        self.KOMDIGI_REGEX_PATTERNS = constants.KOMDIGI_REGEX_PATTERNS
    

    def regulation_product_content(
        self,
        regulation_names_and_links: list[dict],
        output_dir: str,
        verbose: bool = True
    ) -> None:
        
        SELECTORS = self.KOMDIGI_SELECTORS[self.regulation_product_content.__name__]
        os.makedirs(output_dir, exist_ok=True)

        net_durations = []
        success = 0
        failed = 0

        global_start = time.time()

        for regulation in tqdm.tqdm(iterable=regulation_names_and_links, desc="Scraping regulation content",
                                    disable=not verbose):
            
            local_start = time.time()

            try:
                result = ""

                self.web_driver.get(regulation["url"])
                wait = ui.WebDriverWait(self.web_driver, timeout=10)
                wait.until(EC.presence_of_element_located((by.By.CSS_SELECTOR, SELECTORS["regulation_box"])))
                
                # Mendapatkan box peraturan perundang-undangan
                regulation_box = self.web_driver.find_element(by.By.CSS_SELECTOR, SELECTORS["regulation_box"])
                
                # Dapatkan akses ke setiap element di dalam peraturan perundang-undangan: [<p>, <ol>]
                regulation_contents = regulation_box.find_elements(by.By.XPATH, "./*")
                
                # Mengakses setiap element di dalam peraturan perundang-undangan: [<p>, <ol>]
                for index, regulation_content_element in enumerate(regulation_contents):
                    result += self.__regulation_product_content_element(
                        web_element=regulation_content_element,
                        index=index,
                        level=1
                    )
                
                result = result.strip()
                result = re.sub(r"\n{3,}", "\n\n", result)
                result = re.sub(r"(## pasal \w+)(\n{2,})", r"\1\n", result, flags=re.IGNORECASE)

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
            print("=" * 100)
            print(f"{'Output directory':<20}: {os.path.join(output_dir)}")
            print(f"{'Total regulations':<20}: {len(regulation_names_and_links)} regulations")
            print(f"{'Total success':<20}: {success} regulations")
            print(f"{'Total failed':<20}: {failed} regulations")
            print(f"{'Gross time':<20}: {round(gross_time, 3)} seconds")
            print(f"{'Net time':<20}: {round(sum(net_durations), 3)} seconds")
            print(f"{'Average gross time':<20}: {round(gross_time / len(regulation_names_and_links), 3)} seconds")
            print(f"{'Average net time':<20}: {round(sum(net_durations) / success, 3)} seconds")
            print("=" * 100)


    def __check_ol_tag(self, web_element: webdriver.remote.webelement.WebElement) -> str:
        REGEX_PATTERNS = self.KOMDIGI_REGEX_PATTERNS[self.__check_ol_tag.__name__]

        # Get the full <ol> HTML content from the element
        outer_html = web_element.get_attribute("outerHTML")
        html_tag = re.search(REGEX_PATTERNS["html_tag"], outer_html)[0]

        # Extract the <ol> type: ["a", "lower-alpha", "decimal"]
        ol_type = re.search(REGEX_PATTERNS["ol_type"], html_tag)
        ol_type = ol_type[1] if ol_type is not None else "decimal"

        return self.OL_TYPES[ol_type]
    

    def __process_parent_element_text(
            self,
            web_element: webdriver.remote.webelement.WebElement,
            level: int,
            index: int
    ) -> str:
        
        REGEX_PATTERNS = self.KOMDIGI_REGEX_PATTERNS[self.__process_parent_element_text.__name__]
        text = web_element.text.strip()

        if level == 1:            
            if re.search(REGEX_PATTERNS["special_token_pattern_1"], text, re.IGNORECASE):
                return f"\n\n## {text}"
            elif re.search(REGEX_PATTERNS["special_token_pattern_2"], text, re.IGNORECASE):
                return f"\n\n{text}"
            else: 
                return f"\n{text}"
        elif level == 2:
            return f"\n({index}) {text}"
        elif level == 3:
            return f"\n\t{self.ALPHABET[index - 1]}. {text}"
        
    
    def __process_child_element_text(
        self,
        web_element: webdriver.remote.webelement.WebElement,
        ol_type: str,
        level: int,
        index: int
    ) -> str:
        
        text = web_element.text.strip()

        if level == 1:
            index = f"({index + 1})" if ol_type == "decimal" else f"{self.ALPHABET[index]}."
            if text != "":
                return f"\n{index} {text}"
            else:
                return f"\n{text}"
        elif level == 2:
            index = f"{index + 1}." if ol_type == "decimal" else f"{self.ALPHABET[index]}."
            return f"\n\t{index} {text}"
        else:
            index = f"{index + 1}." if ol_type == "decimal" else f"{self.ALPHABET[index]}."
            return f"\n\t\t{index} {text}"


    def __regulation_product_content_element(
        self,
        web_element: webdriver.remote.webelement.WebElement,
        index: int,
        level: int = 1
    ) -> str:
        
        result = ""

        # Process <ol> (ordered list) elements
        if web_element.tag_name == "ol":
            ol_type = self.__check_ol_tag(web_element=web_element)  # Get the <ol> type: lower-alpha or decimal
            web_element = web_element.find_elements(by.By.XPATH, "./*")  # Retrieve the list of article contents (sections)

            for i, sub_element in enumerate(web_element):  # Iterate over all article contents (sections) within <li>
                sub_element_component = sub_element.find_elements(by.By.XPATH, "./*")  # Get all child elements within <li>
                num_sub_element_component = len(sub_element_component)  # Check if <li> contains more than one child element
                
                if num_sub_element_component > 1:  # Check if <li> contains more than one child element
                    for sub_sub_element in sub_element_component:  # Iterate over all child elements within <li>
                        # Check if the child element contains a <br>. If so, it's raw text
                        if sub_sub_element.tag_name == "br":
                            text = sub_element.text.strip()
                            index = f"({i + 1})" if ol_type == "decimal" else f"{self.ALPHABET[i]}."
                            result += f"\n{index} {text}"
                            break  # If not break, the output will be copied as many times as there are <br> tags
                        # If there's no <br>, then it must be either <p> or another <ol>
                        else:
                            result += self.__regulation_product_content_element(
                                web_element=sub_sub_element,
                                index=i + 1,
                                level=level + 1
                            )
                elif num_sub_element_component == 0:  # Check if <li> has no child elements
                    # If it has no child elements, it's just raw text
                    result += self.__process_child_element_text(
                        web_element=sub_element,
                        ol_type=ol_type,
                        level=level,
                        index=i
                    )
                else:  # If <li> has only one child element, it must be either a <p> or a <br>
                    # Check if there's a <br>
                    for sub_sub_element in sub_element_component:
                        if sub_sub_element.tag_name == "br":
                            text = sub_element.text.strip()
                            index = f"({i + 1})" if ol_type == "decimal" else f"{self.ALPHABET[i]}."
                            result += f"\n{index} {text}"
                            break  # If not break, the output will be copied as many times as there are <br> tags
                    
                    # If there's no <br>, then it must be just a <p>
                    result += self.__process_child_element_text(
                        web_element=sub_element_component[0],
                        ol_type=ol_type,
                        level=level,
                        index=i
                    )
        # Process <p> tags or elements without a specific tag
        else:
            result += self.__process_parent_element_text(
                web_element=web_element,
                level=level,
                index=index
            )
        
        return result
        