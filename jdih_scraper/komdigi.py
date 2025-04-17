import os
import re
# import json
import time
# import dateparser
# import pandas as pd
from tqdm import tqdm
from encoding import OL_TYPE
from encoding import ALPHABET
# from encoding import REGULATION_ENCODING
# from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
# from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC



class KomdigiScraper:

    def __init__(self, web_driver: WebDriver):
        self.web_driver = web_driver
        self.ALPHABET = ALPHABET
        self.OL_TYPE = OL_TYPE


    def __check_ol_tag(self, web_element: WebElement) -> str:
        # Mendapatkan isi full tag <ol> HTML dari element tersebut
        outer_html = web_element.get_attribute('outerHTML')
        tag_html = re.search(r'<\s*([a-zA-Z0-9]+)([^>]*)>', outer_html)[0]
        # Mendapatkan jenis <ol>: ['a', 'lower-alpha', 'decimal'], di mana 'a' == 'lower-alpha'
        ol_type = re.search(r'\b(lower-alpha|decimal|a)\b', tag_html)
        ol_type = ol_type[1] if ol_type is not None else 'decimal'
        return self.OL_TYPE[ol_type]
    

    def __process_parent_element_text(self, web_element: WebElement, level: int, index: int) -> str:
        text = web_element.text.strip()

        if level == 1:
            # |(^\d+. .*)
            special_token_pattern_1 = \
                r'(^bab \w+)|(^pasal \w+)|(^bagian \w+)|(^paragraf \w+)|(^menimbang)|(^mengingat)|(^memutuskan)|(^menetapkan)'
            special_token_pattern_2 = \
                r'(^agar setiap orang mengetahuinya)|(^ditetapkan di)|(^dengan rahmat Tuhan Yang Maha Esa)'
            
            if re.search(special_token_pattern_1, text, re.IGNORECASE):
                return f'\n\n## {text}'
            elif re.search(special_token_pattern_2, text, re.IGNORECASE):
                return f'\n\n{text}'
            else: 
                return f'\n{text}'
            
        elif level == 2:
            return f'\n({index}) {text}'
        elif level == 3:
            return f'\n\t{self.ALPHABET[index - 1]}. {text}'
        
    
    def __process_child_element_text(self, web_element: WebElement, ol_type: str, level: int, index: int) -> str:
        text = web_element.text.strip()
        if level == 1:
            index = f'({index + 1})' if ol_type == 'decimal' else f'{self.ALPHABET[index]}.'
            if text != '':
                return f'\n{index} {text}'
            else:
                return f'\n{text}'
        elif level == 2:
            index = f'{index + 1}.' if ol_type == 'decimal' else f'{self.ALPHABET[index]}.'
            return f'\n\t{index} {text}'
        else:
            index = f'{index + 1}.' if ol_type == 'decimal' else f'{self.ALPHABET[index]}.'
            return f'\n\t\t{index} {text}'


    def __regulation_product_content_element(self, web_element: WebElement, index: int, level: int = 1) -> str:
        # Hasil akhir
        result = ''

        # Untuk tag <ol>
        if web_element.tag_name == 'ol':
            ol_type = self.__check_ol_tag(web_element=web_element)                  # Dapatkan jenis tag <ol>: lower-alpha ata decimal
            web_element = web_element.find_elements(By.XPATH, './*')                # Dapatkan list isi pasal (ayat)

            for i, sub_element in enumerate(web_element):                           # Iterasi semua isi pasal (ayat): <li>
                sub_element_component = sub_element.find_elements(By.XPATH, './*')  # Ambil semua child element di dalam ayat <li>
                num_sub_element_component = len(sub_element_component)              # Cek apakah ayat <li> punya > 1 child element
                
                if num_sub_element_component > 1:  # Cek apakah setiap ayat <li> punya > 1 child element

                    for sub_sub_element in sub_element_component:  # Iterasi semua child element di dalam ayat <li>
                        # Cek apakah child element nya ada <br>, jika ada maka hanya raw text
                        if sub_sub_element.tag_name == 'br':
                            text = sub_element.text.strip()
                            index = f'({i + 1})' if ol_type == 'decimal' else f'{self.ALPHABET[i]}.'
                            result += f'\n{index} {text}'
                            break  # Jika tidak break, maka akan copy output sebanyak jumlah br
                        # Jika tidak ada <br>, maka pasti <p> atau <ol> lagi
                        else:
                            result += self.__regulation_product_content_element(
                                web_element=sub_sub_element,
                                index=i + 1,
                                level=level + 1
                            )
                
                elif num_sub_element_component == 0:  # Cek apakah ayat <li> tidak punya child element
                    # Jika tidak punya child element, maka hanya raw text saja
                    result += self.__process_child_element_text(
                        web_element=sub_element,
                        ol_type=ol_type,
                        level=level,
                        index=i
                    )
                
                else:  # Jika ayat <li> hanya punya 1 child element, maka pasti <p> saja atau <br> saja
                    # Cek apakah ada <br>                               
                    for sub_sub_element in sub_element_component:
                        if sub_sub_element.tag_name == 'br':
                            text = sub_element.text.strip()
                            index = f'({i + 1})' if ol_type == 'decimal' else f'{self.ALPHABET[i]}.'
                            result += f'\n{index} {text}'
                            break  # Jika tidak break, maka akan copy output sebanyak jumlah br
                    
                    # Jika tidak ada <br> maka pasti <p> saja
                    result += self.__process_child_element_text(
                        web_element=sub_element_component[0],
                        ol_type=ol_type,
                        level=level,
                        index=i
                    )
                
        # Untuk tag <p> atau no-tag
        else:
            result += self.__process_parent_element_text(
                web_element=web_element,
                level=level,
                index=index
            )
        
        return result
    

    def regulation_product_content(self, regulation_names_and_links: list[dict], output_dir: str, verbose: bool = True) -> None:
        
        regulation_box_css_selector = 'div#produk-content'
        os.makedirs(output_dir, exist_ok=True)
        durations = list()
        success = 0
        failed = 0

        for regulation in tqdm(iterable=regulation_names_and_links, desc='Scraping regulation content', disable=not verbose):        
            start = time.time()

            try:
                result = ''
                self.web_driver.get(regulation['url'])
                wait = WebDriverWait(self.web_driver, timeout=10)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, regulation_box_css_selector)))
                # Mendapatkan box peraturan perundang-undangan
                regulation_box = self.web_driver.find_element(By.CSS_SELECTOR, regulation_box_css_selector)
                # Dapatkan akses ke setiap element di dalam peraturan perundang-undangan: [<p>, <ol>]
                regulation_contents = regulation_box.find_elements(By.XPATH, './*')
                # Mengakses setiap element di dalam peraturan perundang-undangan: [<p>, <ol>]
                
                for index, regulation_content_element in enumerate(regulation_contents):
                    result += self.__regulation_product_content_element(
                        web_element=regulation_content_element,
                        index=index,
                        level=1
                    )

                result = result.strip()
                result = re.sub(r'\n{3,}', '\n\n', result)
                result = re.sub(r'(## pasal \w+)(\n{2})', r'\1\n', result, flags=re.IGNORECASE)

                # https://stackoverflow.com/questions/27092833/unicodeencodeerror-charmap-codec-cant-encode-characters
                file_output = os.path.join(output_dir, f'{regulation["name"]}.md')
                with open(file_output, 'w', encoding='utf-8') as file:
                    file.write(result)
            
                success += 1
                durations.append(time.time() - start)
                time.sleep(2)  # Break for 2 seconds
            
            except Exception as e:
                failed += 1
                if verbose:
                    print(f'ERROR scraping content for {regulation["name"]}')
                    print(f'MESSAGE ERROR: {e}')
        
        self.web_driver.quit()

        if verbose:
            print('=' * 76)
            print(f'Output directory  : {os.path.join(output_dir)}')
            print(f'Total regulations : {len(regulation_names_and_links)} regulations')
            print(f'Total success     : {success} regulations')
            print(f'Total failed      : {failed} regulations')
            print(f'Total time        : {round(sum(durations), 3)} seconds')
            print(f'Average time      : {round(sum(durations) / success, 3)} seconds')
            print('NOTE! Time records do not include the 2 seconds break between each regulation')
            print('=' * 76)
            