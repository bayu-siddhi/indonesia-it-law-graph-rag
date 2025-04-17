# import os
import re
import json
import time
import dateparser
import pandas as pd
from tqdm import tqdm
# from encoding import OL_TYPE
# from encoding import ALPHABET
from encoding import REGULATION_ENCODING
# from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webdriver import WebDriver
# from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC



class BPKScraper:

    def __init__(self, web_driver: WebDriver):
        self.web_driver = web_driver
        self.REGULATION_ENCODING = REGULATION_ENCODING
    

    @staticmethod  # https://stackoverflow.com/questions/735975/static-methods-in-python
    def list_of_dict_to_json(regulation_data: list[dict], output_path: str) -> None:
        if not output_path.endswith('.json'):
            output_path = output_path + '.json'
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(regulation_data, file, indent=4)
    

    @staticmethod
    def list_of_dict_to_excel(regulation_data: list[dict], output_path: str) -> None:
        if not output_path.endswith('.xlsx'):
            output_path = output_path + '.xlsx'
        df = pd.DataFrame(regulation_data)
        df.to_excel(output_path, index=False)


    def active_regulation(self, url: str, regulation_type: str, verbose: bool = True) -> list[dict]:
        regulations_box_xpath = '/html/body/div/div/div[2]/div[2]/div[2]'
        regulations_css_selector = 'div.row.mb-8[class="row mb-8"]'
        regulation_href_css_selector = 'div.col-lg-10.fs-2.fw-bold.pe-4 a'
        pagination_box_css_selector = 'ul.pagination.justify-content-center'
        regulation_number_css_selector = 'div.col-lg-8.fw-semibold.fs-5.text-gray-600'
        regulation_title_css_selector = 'div.col-lg-10.fs-2.fw-bold.pe-4'
        regulation_subjects_css_selector = 'span.badge.badge-light-primary.mb-2'
        page_pattern = r'p=(\d+)'
        
        # Final result
        active_regulations = list()
        durations = list()
        new_url = ''

        # Check page numbering in URL
        if re.search(page_pattern, url):
            new_url = re.sub(page_pattern, 'p={page}', url)
        else:
            new_url = url + '&p={page}'

        # Get last page number
        self.web_driver.get(new_url.format(page=1))
        wait = WebDriverWait(self.web_driver, timeout=10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, pagination_box_css_selector)))
        
        pagination_box = self.web_driver.find_element(By.CSS_SELECTOR, pagination_box_css_selector)
        last_page_button = pagination_box.find_element(By.XPATH, "./*[last()]")
        last_page_href = last_page_button.find_element(By.CSS_SELECTOR, '[href]').get_attribute('href')
        last_page_number = int(re.search(page_pattern, last_page_href)[1])

        # Iterate for every page
        for page in tqdm(iterable=range(1, last_page_number + 1), desc='Scraping active regulations', disable=not verbose):
            start = time.time()
            
            # Go to the page
            access_page = False
            trial_number = 10

            for _ in range(trial_number):
                try:
                    # Try access the page
                    self.web_driver.get(new_url.format(page=page))
                    wait = WebDriverWait(self.web_driver, timeout=10)
                    wait.until(EC.presence_of_element_located((By.XPATH, regulations_box_xpath)))
                    access_page = True
                    break
                except TimeoutException:
                    # If timeout, wait for 2 seconds
                    time.sleep(2)
            
            if not access_page:
                if verbose:
                    print(f'Unable to access {url} on page={page} after {trial_number} attempts')
                    print(f'Skip the scraping process to page={page + 1}')
                continue
            
            # Get all regulation instance
            regulations_box = self.web_driver.find_element(By.XPATH, regulations_box_xpath)
            regulations_all = regulations_box.find_elements(By.CSS_SELECTOR, regulations_css_selector)

            # Iterate for every regulation
            for regulation in regulations_all:
                # Ignore all ineffective regulations
                if not re.findall(r'Dicabut dengan', regulation.text):
                    
                    # Get regulation number and year element
                    regulation_number_and_year = regulation.find_element(By.CSS_SELECTOR, regulation_number_css_selector).text.lower()
                    
                    # Get regulation number
                    new_regulation_number = re.search(r'\b(?:nomor|no\.)\s+(\d+)', regulation_number_and_year)
                    old_regulation_number = re.search(r'(\d+)\/', new_regulation_number[0])
                    regulation_number = new_regulation_number[1] if old_regulation_number is None else old_regulation_number[1]

                    # Get regulation year
                    regulation_year = re.search(r'tahun\s+(\d+)', regulation_number_and_year)
                    regulation_year = regulation_year[1] if regulation_year is not None else ''

                    # Get regulation title
                    regulation_title = regulation.find_element(By.CSS_SELECTOR, regulation_title_css_selector).text.strip()
                    
                    # Get regulation subjects
                    regulation_subjects = list()
                    regulation_subject_elements = regulation.find_elements(By.CSS_SELECTOR, regulation_subjects_css_selector)
                    if regulation_subject_elements:
                        for subject in regulation_subject_elements:
                            regulation_subjects.append(subject.text)
                    
                    # Get regulation URL link
                    regulation_href = regulation.find_element(By.CSS_SELECTOR, regulation_href_css_selector).get_attribute("href")
                    
                    # Create regulation temporary ID, just for ordering
                    number = f'{regulation_year}{regulation_number.zfill(3)}'
                    regulation_id = f'{regulation_type}_{regulation_number.zfill(3)}_{regulation_year}'

                    # Append all data
                    active_regulations.append({
                        'no': number,
                        'name': regulation_id,
                        'about': regulation_title,
                        'subjects': regulation_subjects,
                        'url': regulation_href,
                        'used': False
                    })
            
            durations.append(time.time() - start)
            time.sleep(2)  # Break for 2 seconds
        
        self.web_driver.quit()

        if verbose:
            print('=' * 76)
            print(f'URL               : {url}')
            print(f'Regulation type   : {regulation_type}')
            print(f'Total regulations : {len(active_regulations)} regulations')
            print(f'Total time        : {round(sum(durations), 3)} seconds')
            print(f'Average time      : {round(sum(durations) / len(active_regulations), 3)} seconds')
            print('NOTE! Time records do not include the 2 seconds break between each regulation')
            print('=' * 76)

        return active_regulations
    

    def regulation_metadata(self, regulation_links: list[str], verbose: bool = True) -> list[dict]:
        metadata_box_xpath = '/html/body/div/div/div[2]/div[2]/div/div[1]/div[2]/div'
        download_box_xpath = '/html/body/div/div/div[2]/div[2]/div/div[2]/div[1]'
        status_box_xpath = '/html/body/div/div/div[2]/div[2]/div/div[2]/div[2]'
        metadata_inner_box_css_selector = 'div.container.fs-6'
        status_inner_box_css_selector = 'div.container.fs-6'
        status_type_patterns = r'(Dicabut dengan :|Diubah dengan :|Mengubah :|Mencabut :)'

        # Final result
        regulation_metadata = list()
        durations = list()

        # Iterate for all regulation links
        for regulation_link in tqdm(iterable=regulation_links, desc='Scraping regulation metadata', disable=not verbose):
            start = time.time()
            
            # Go to the page
            access_page = False
            trial_number = 10

            for _ in range(trial_number):
                try:
                    # Try access the page
                    self.web_driver.get(regulation_link)
                    wait = WebDriverWait(self.web_driver, timeout=10)
                    wait.until(EC.presence_of_element_located((By.XPATH, metadata_box_xpath)))
                    wait.until(EC.presence_of_element_located((By.XPATH, download_box_xpath)))
                    access_page = True
                    break
                except TimeoutException:
                    # If timeout, wait for 2 seconds
                    time.sleep(2)
            
            if not access_page:
                if verbose:
                    print(f'Unable to access {regulation_link} after {trial_number} attempts')
                    print('Skip the scraping process to the next regulation link')
                continue

            # Extract metadata
            ineffective = False
            metadata_box = self.web_driver.find_element(By.XPATH, metadata_box_xpath)
            metadata_inner_box = metadata_box.find_element(By.CSS_SELECTOR, metadata_inner_box_css_selector)
            metadata_elements = metadata_inner_box.find_elements(By.XPATH, './*')[:-2]

            for index, element in enumerate(metadata_elements):
                if index == 1:  # Regulation title and about
                    title = re.search(r'Judul\s(.*)', element.text)
                    title = title[1] if title is not None else ''
                    about = re.search(r'[Tt]entang (.*)', title)
                    about = about[1] if about is not None else ''
                elif index == 3:  # Regulation number
                    number = re.search(r'Nomor\s(\d+)', element.text)
                    number = number[1] if number is not None else ''
                elif index == 4:  # Regulation type
                    regulation_type = re.search(r'Bentuk\s(.*)', element.text)
                    regulation_type = regulation_type[1] if regulation_type is not None else ''
                elif index == 5:  # Regulation short type
                    short_type = re.search(r'Bentuk Singkat\s(.*)', element.text)
                    short_type = short_type[1].upper() if short_type is not None else ''
                elif index == 6:  # Regulation year
                    year = re.search(r'Tahun\s(.*)', element.text)
                    year = year[1] if year is not None else ''
                elif index == 7:  # Regulation issue palce
                    issue_place = re.search(r'Tempat Penetapan\s(.*)', element.text)
                    issue_place = issue_place[1] if issue_place is not None else ''
                elif index == 8:  # Regulation issue date
                    issue_date = re.search(r'Tanggal Penetapan\s(.*)', element.text)
                    if issue_date is not None:
                        issue_date = dateparser.parse(date_string=issue_date[1], languages=['id'])
                        issue_date = issue_date.strftime('%Y-%m-%d')
                    else:
                        issue_date = ''
                elif index == 10:  # Regulation effective date
                    effective_date = re.search(r'Tanggal Berlaku\s(.*)', element.text)
                    if effective_date is not None:
                        effective_date = dateparser.parse(date_string=effective_date[1], languages=['id'])
                        effective_date = effective_date.strftime('%Y-%m-%d')
                    else:
                        effective_date = ''
                elif index == 12:  # Regulation subjects
                    subjects = re.search(r'Subjek\s(.*)', element.text)
                    subjects = subjects[1] if subjects is not None else ''
                    subjects = subjects.split('-')
                    subjects = [subject.strip() for subject in subjects]
                elif index == 13:  # Regulation status
                    status = re.search(r'Status\s(.*)', element.text)
                    status = status[1] if status is not None else ''
                    if status.lower() == 'tidak berlaku':
                        print(f'INEFFECTIVE REGULATION: {regulation_link}')
                        ineffective = True
                elif index == 15:  # Regulation institution
                    institution = re.search(r'Lokasi\s(.*)', element.text)
                    institution = institution[1] if institution is not None else ''

            if ineffective:
                continue

            # Create regulation ID
            regulation_id = '{year}{type}{number}{section}{section_number}'.format(
                year=year,
                type=self.REGULATION_ENCODINGE['type'][short_type],
                number=str(number).zfill(3),
                section=self.REGULATION_ENCODINGE['section']['document'],
                section_number='000'
            )

            # Extract download link and name
            download_box = self.web_driver.find_element(By.XPATH, download_box_xpath)
            download_link = download_box.find_element(By.CSS_SELECTOR, '[href]').get_attribute('href')
            download_name = f'{short_type}_{str(number).zfill(3)}_{year}'

            # Extract regulation status references
            status_box = self.web_driver.find_element(By.XPATH, status_box_xpath)
            
            try:
                status_inner_box = status_box.find_element(By.CSS_SELECTOR, status_inner_box_css_selector)
            except NoSuchElementException as e:
                status_inner_box = None
            
            repealed = list()
            repeal = list()
            amended = list()
            amend = list()
            
            if status_inner_box is not None:
                status_elements = status_inner_box.find_elements(By.XPATH, './*')
                current_status = None
                next_status = None

                for element in status_elements:
                    text = element.text.strip()
                    current_status = next_status
                    next_status = None
                    
                    if re.search(status_type_patterns, text):
                        current_status = text
                        next_status = text
                        continue

                    regulation_references = element.find_elements(By.CSS_SELECTOR, '[href]')
                    for regulation_reference in regulation_references:
                        href = regulation_reference.get_attribute('href')
                        if current_status == 'Dicabut dengan :':
                            repealed.append(href)
                        elif current_status == 'Mencabut :':
                            repeal.append(href)
                        elif current_status == 'Diubah dengan :':
                            amended.append(href)
                        elif current_status == 'Mengubah :':
                            amend.append(href)
            
            # Combine and append all metadata to regulation_metadata
            regulation_metadata.append({
                'id': regulation_id,                # ID peraturan
                'url': regulation_link,             # Link Web Peraturan
                'download_link': download_link,     # Link Download Peraturan
                'download_name': download_name,     # Nama File Download
                'title': title,                     # Judul Lengkap Peraturan
                'about': about,                     # Judul Isi Peraturan
                'type': regulation_type,            # Jenis Peraturan
                'short_type': short_type,           # Jenis Peraturan (Singkatan)
                'number': number,                   # Nomor Peraturan
                'year': year,                       # Tahun Peraturan
                'institution': institution,         # Lembaga
                'issue_place': issue_place,         # Tempat Penetapan
                'issue_date': issue_date,           # Tanggal Penetapan
                'effective_date': effective_date,   # Tanggal Diberlakukan
                'subjects': subjects,               # Subjek
                'status': {
                    'repealed': repealed,           # Dicabut dengan ..
                    'repeal': repeal,               # Mencabut ...
                    'amended': amended,             # Diubah dengan ...
                    'amend': amend                  # Mengubah ...
                }
            })

            durations.append(time.time() - start)
            time.sleep(2)  # Break for 2 seconds
        
        self.web_driver.quit()

        if verbose:
            print('=' * 76)
            print(f'Total regulations : {len(regulation_links)} regulations')
            print(f'Total time        : {round(sum(durations), 3)} seconds')
            print(f'Average time      : {round(sum(durations) / len(regulation_links), 3)} seconds')
            print('NOTE! Time records do not include the 2 seconds break between each regulation')
            print('=' * 76)

        return regulation_metadata
    