import os
import re
import time
import tqdm
import dateparser
import urllib.parse

from tools import encodings
from tools.regulation_scraper import constants

from selenium import webdriver
from selenium.common import exceptions 
from selenium.webdriver.common import by
from selenium.webdriver.support import ui
from selenium.webdriver.support import expected_conditions as EC


class BPKScraper:

    def __init__(self, web_driver: webdriver.remote.webdriver.WebDriver) -> None:
        self.web_driver = web_driver
        self.REGULATION_CODES = encodings.REGULATION_CODES
        self.WORD_TO_NUMBER = encodings.WORD_TO_NUMBER
        self.BPK_SELECTORS = constants.BPK_SELECTORS
        self.BPK_REGEX_PATTERNS = constants.BPK_REGEX_PATTERNS


    def active_regulation(self, url: str, regulation_type: str, verbose: bool = True) -> list[dict]:
        SELECTORS = self.BPK_SELECTORS[self.active_regulation.__name__]
        REGEX_PATTERNS = self.BPK_REGEX_PATTERNS[self.active_regulation.__name__]
        active_regulations = []
        net_durations = []

        # Format URL page numbering
        if re.search(REGEX_PATTERNS["page_number"], url, re.IGNORECASE):
            new_url = re.sub(REGEX_PATTERNS["page_number"], "p={page}", url, flags=re.IGNORECASE)
        else:
            new_url = url + "&p={page}"

        # Get last page number
        self.web_driver.get(new_url.format(page=1))
        wait = ui.WebDriverWait(self.web_driver, timeout=10)
        wait.until(EC.presence_of_element_located((by.By.CSS_SELECTOR, SELECTORS["pagination_box"])))
        pagination_box = self.web_driver.find_element(by.By.CSS_SELECTOR, SELECTORS["pagination_box"])
        last_page_button = pagination_box.find_element(by.By.XPATH, "./*[last()]")
        last_page_href = last_page_button.find_element(by.By.CSS_SELECTOR, "[href]").get_attribute("href")
        last_page_number = int(re.search(REGEX_PATTERNS["page_number"], last_page_href, re.IGNORECASE)[1])

        global_start = time.time()

        # Iterate for every page
        for page in tqdm.tqdm(iterable=range(1, last_page_number + 1), desc="Scraping active regulations", disable=not verbose):
            local_start = time.time()
            access_page = False
            trial_number = 10

            # Try access the page
            for _ in range(trial_number):
                try:
                    self.web_driver.get(new_url.format(page=page))
                    wait = ui.WebDriverWait(self.web_driver, timeout=10)
                    wait.until(EC.presence_of_element_located((by.By.XPATH, SELECTORS["reg_box"])))
                    access_page = True
                    break
                except exceptions.TimeoutException:
                    time.sleep(2)
            
            if not access_page:
                if verbose:
                    print(f"Unable to access {url} on page={page} after {trial_number} attempts")
                    print(f"Skip the scraping process to page={page + 1}")
                continue
            
            # Get all regulation instances
            regulations_box = self.web_driver.find_element(by.By.XPATH, SELECTORS["reg_box"])
            regulations_all = regulations_box.find_elements(by.By.CSS_SELECTOR, SELECTORS["reg_items"])

            # Iterate for every regulation instances
            # Ignore all ineffective regulations
            for regulation in regulations_all:
                if not re.findall(REGEX_PATTERNS["ineffective_reg"], regulation.text, re.IGNORECASE):        
                    # Get regulation number and year element
                    regulation_number_and_year = regulation.find_element(by.By.CSS_SELECTOR, SELECTORS["reg_number"]).text.lower()
                    
                    # Get regulation number
                    new_regulation_number = re.search(REGEX_PATTERNS["reg_new_number"], regulation_number_and_year, re.IGNORECASE)
                    old_regulation_number = re.search(REGEX_PATTERNS["reg_old_number"], new_regulation_number[0], re.IGNORECASE)
                    regulation_number = new_regulation_number[1] if old_regulation_number is None else old_regulation_number[1]

                    # Get regulation year
                    regulation_year = re.search(REGEX_PATTERNS["reg_year"], regulation_number_and_year, re.IGNORECASE)
                    regulation_year = regulation_year[1] if regulation_year is not None else ""

                    # Get regulation title
                    regulation_title = regulation.find_element(by.By.CSS_SELECTOR, SELECTORS["reg_title"]).text.strip()
                    
                    # Get regulation subjects
                    regulation_subjects = []
                    regulation_subject_elements = regulation.find_elements(by.By.CSS_SELECTOR, SELECTORS["reg_subject"])
                    if regulation_subject_elements:
                        for subject in regulation_subject_elements:
                            regulation_subjects.append(subject.text)
                    
                    # Get regulation URL link
                    regulation_href = regulation.find_element(by.By.CSS_SELECTOR, SELECTORS["reg_href"]).get_attribute("href")
                    
                    # Create regulation temporary ID, just for ordering
                    regulation_id = f"{regulation_type}_{regulation_year}_{regulation_number.zfill(3)}"

                    # Append all data
                    active_regulations.append({
                        "name": regulation_id,
                        "about": regulation_title,
                        "subjects": regulation_subjects,
                        "url_1": regulation_href,
                        "url_2": "",
                        "good_pdf": False,
                        "used": False,
                        "done": False,
                        "pic": "",
                        "notes": "",
                    })
            
            net_durations.append(time.time() - local_start)
            time.sleep(1)  # Break for 1 seconds

        gross_time = time.time() - global_start

        if verbose:
            print("=" * 100)
            print(f"{'URL':<20}: {url}")
            print(f"{'Regulation type':<20}: {regulation_type}")
            print(f"{'Total regulations':<20}: {len(active_regulations)} regulations")
            print(f"{'Gross time':<20}: {round(gross_time, 3)} seconds")
            print(f"{'Net time':<20}: {round(sum(net_durations), 3)} seconds")
            print(f"{'Average gross time':<20}: {round(gross_time / len(active_regulations), 3)} seconds")
            print(f"{'Average net time':<20}: {round(sum(net_durations) / len(active_regulations), 3)} seconds")
            print("=" * 100)

        return active_regulations


    def regulation_metadata(self, urls: list[str], verbose: bool = True) -> list[dict]:
        SELECTORS = self.BPK_SELECTORS[self.regulation_metadata.__name__]
        REGEX_PATTERNS = self.BPK_REGEX_PATTERNS[self.regulation_metadata.__name__]
        regulation_id_template = "{year}{type}{number}{section}{section_number}{additional_section_number}"
        regulation_metadata = []
        net_durations = []

        global_start = time.time()

        # Iterate for every regulation links
        for url in tqdm.tqdm(iterable=urls, desc="Scraping regulation metadata", disable=not verbose):
            local_start = time.time()
            access_page = False
            trial_number = 10

            # Try access the url
            for _ in range(trial_number):
                try:
                    self.web_driver.get(url)
                    wait = ui.WebDriverWait(self.web_driver, timeout=10)
                    wait.until(EC.presence_of_element_located((by.By.XPATH, SELECTORS["metadata_box"])))
                    wait.until(EC.presence_of_element_located((by.By.XPATH, SELECTORS["download_box"])))
                    access_page = True
                    break
                except exceptions.TimeoutException:
                    time.sleep(2)
            
            if not access_page:
                if verbose:
                    print(f"Unable to access {url} after {trial_number} attempts")
                    print("Skip the scraping process to the next regulation link")
                continue

            # Extract metadata
            ineffective = False
            metadata_box = self.web_driver.find_element(by.By.XPATH, SELECTORS["metadata_box"])
            metadata_inner_box = metadata_box.find_element(by.By.CSS_SELECTOR, SELECTORS["metadata_inner_box"])
            metadata_elements = metadata_inner_box.find_elements(by.By.XPATH, "./*")[:-2]

            for index, element in enumerate(metadata_elements):
                if index == 1:
                    # Extract regulation title
                    title = re.search(REGEX_PATTERNS["reg_title"], element.text, re.IGNORECASE)
                    title = title[1] if title is not None else ""

                    # Extract regulation about
                    about = re.search(REGEX_PATTERNS["reg_about"], title, re.IGNORECASE)
                    about = about[1] if about is not None else ""

                    # Extract regulation amendment number
                    amendment = "0"
                    if re.search(REGEX_PATTERNS["reg_amendment_number_1"], about, re.IGNORECASE):
                        amendment = "1"
                    elif re.search(REGEX_PATTERNS["reg_amendment_number_2"], about, re.IGNORECASE):
                        amendment = re.search(REGEX_PATTERNS["reg_amendment_number_2"], about, re.IGNORECASE)[1]
                        amendment = str(self.WORD_TO_NUMBER[amendment.strip().lower()])

                # Extract regulation number
                elif index == 3:
                    number = re.search(REGEX_PATTERNS["reg_number"], element.text, re.IGNORECASE)
                    number = number[1] if number is not None else ""
                # Extract regulation type
                elif index == 4:
                    regulation_type = re.search(REGEX_PATTERNS["reg_type"], element.text, re.IGNORECASE)
                    regulation_type = regulation_type[1] if regulation_type is not None else ""
                # Extract regulation short type
                elif index == 5:
                    short_type = re.search(REGEX_PATTERNS["reg_short_type"], element.text, re.IGNORECASE)
                    short_type = short_type[1].upper() if short_type is not None else ""
                # Extract regulation year
                elif index == 6:
                    year = re.search(REGEX_PATTERNS["reg_year"], element.text, re.IGNORECASE)
                    year = year[1] if year is not None else ""
                # Extract regulation issue place
                elif index == 7:
                    issue_place = re.search(REGEX_PATTERNS["reg_issue_place"], element.text, re.IGNORECASE)
                    issue_place = issue_place[1] if issue_place is not None else ""
                # Extract regulation issue date
                elif index == 8:
                    issue_date = re.search(REGEX_PATTERNS["reg_issue_date"], element.text, re.IGNORECASE)
                    if issue_date is not None:
                        issue_date = dateparser.parse(date_string=issue_date[1], languages=["id"])
                        issue_date = issue_date.strftime("%Y-%m-%d")
                    else:
                        issue_date = ""
                # Extract regulation effective date
                elif index == 10:
                    effective_date = re.search(REGEX_PATTERNS["reg_effective_date"], element.text, re.IGNORECASE)
                    if effective_date is not None:
                        effective_date = dateparser.parse(date_string=effective_date[1], languages=["id"])
                        effective_date = effective_date.strftime("%Y-%m-%d")
                    else:
                        effective_date = ""
                # Extract regulation subjects
                elif index == 12:
                    subjects = re.search(REGEX_PATTERNS["reg_subject"], element.text, re.IGNORECASE)
                    subjects = subjects[1] if subjects is not None else ""
                    subjects = subjects.split("-")
                    subjects = [subject.strip() for subject in subjects]
                # Extract regulation status
                elif index == 13:
                    status = re.search(REGEX_PATTERNS["reg_status"], element.text, re.IGNORECASE)
                    status = status[1] if status is not None else ""
                    if status.lower() == "tidak berlaku":
                        print(f"INEFFECTIVE REGULATION: {url}")
                        ineffective = True
                # Extract regulation institution
                elif index == 15:
                    institution = re.search(REGEX_PATTERNS["reg_institution"], element.text, re.IGNORECASE)
                    institution = institution[1] if institution is not None else ""

            if ineffective:
                continue

            # Create regulation ID
            regulation_id = regulation_id_template.format(
                year=year,
                type=self.REGULATION_CODES["type"][short_type],
                number=str(number).zfill(3),
                section=self.REGULATION_CODES["section"]["document"],
                section_number="000",
                additional_section_number="00"
            )

            # Extract regulation download link and name
            download_box = self.web_driver.find_element(by.By.XPATH, SELECTORS["download_box"])
            download_link = download_box.find_element(by.By.CSS_SELECTOR, "[href]").get_attribute("href")
            download_name = f"{short_type}_{year}_{str(number).zfill(3)}"

            # Extract regulation status references
            status_box = self.web_driver.find_element(by.By.XPATH, SELECTORS["status_box"])
            
            try:
                status_inner_box = status_box.find_element(by.By.CSS_SELECTOR, SELECTORS["status_inner_box"])
            except exceptions.NoSuchElementException:
                status_inner_box = None
            
            repealed = []
            repeal = []
            amended = []
            amend = []
            
            if status_inner_box is not None:
                status_elements = status_inner_box.find_elements(by.By.XPATH, "./*")
                current_status = None
                next_status = None

                for element in status_elements:
                    text = element.text.strip().lower()
                    current_status = next_status
                    next_status = None
                    
                    if re.search(REGEX_PATTERNS["status_type"], text, re.IGNORECASE):
                        current_status = text
                        next_status = text
                        continue

                    regulation_references = element.find_elements(by.By.CSS_SELECTOR, "[href]")
                    for regulation_reference in regulation_references:
                        href = regulation_reference.get_attribute("href")
                        if current_status == "dicabut dengan :":
                            repealed.append(href)
                        elif current_status == "mencabut :":
                            repeal.append(href)
                        elif current_status == "diubah dengan :":
                            amended.append(href)
                        elif current_status == "mengubah :":
                            amend.append(href)
            
            # Combine and append all metadata to regulation_metadata
            regulation_metadata.append({
                "id": regulation_id,                # ID peraturan
                "url": url,                         # Link Web Peraturan
                "download_link": download_link,     # Link Download Peraturan
                "download_name": download_name,     # Nama File Download
                "title": title,                     # Judul Lengkap Peraturan
                "about": about,                     # Judul Isi Peraturan
                "type": regulation_type,            # Jenis Peraturan
                "short_type": short_type,           # Jenis Peraturan (Singkatan)
                "amendment": amendment,             # Nomor Amandemen
                "number": number,                   # Nomor Peraturan
                "year": year,                       # Tahun Peraturan
                "institution": institution,         # Lembaga
                "issue_place": issue_place,         # Tempat Penetapan
                "issue_date": issue_date,           # Tanggal Penetapan
                "effective_date": effective_date,   # Tanggal Diberlakukan
                "subjects": subjects,               # Subjek
                "status": {
                    "repealed": repealed,           # Dicabut dengan ..
                    "repeal": repeal,               # Mencabut ...
                    "amended": amended,             # Diubah dengan ...
                    "amend": amend                  # Mengubah ...
                }
            })

            net_durations.append(time.time() - local_start)
            time.sleep(2)  # Break for 2 seconds

        gross_time = time.time() - global_start

        if verbose:
            print("=" * 100)
            print(f"{'Total regulations':<20}: {len(urls)} regulations")
            print(f"{'Gross time':<20}: {round(gross_time, 3)} seconds")
            print(f"{'Net time':<20}: {round(sum(net_durations), 3)} seconds")
            print(f"{'Average gross time':<20}: {round(gross_time / len(urls), 3)} seconds")
            print(f"{'Average net time':<20}: {round(sum(net_durations) / len(urls), 3)} seconds")
            print("=" * 100)

        return regulation_metadata
    
    
    @staticmethod
    def download_regulation_pdf(download_data: list[dict], download_full_dir_path: str, verbose: bool = True) -> None:
    
        relative_download_dir_path = os.path.relpath(download_full_dir_path, start=os.getcwd())
        os.makedirs(relative_download_dir_path, exist_ok=True)

        # Only for Firefox
        # https://stackoverflow.com/questions/60170311/how-to-switch-download-directory-using-selenium-firefox-python
        firefox_options = webdriver.FirefoxOptions()
        firefox_options.set_preference("browser.download.folderList", 2)  # Use custom folder
        firefox_options.set_preference("browser.download.dir", download_full_dir_path)  # Specify the destination folder
        firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")  # Avoid prompts
        firefox_options.set_preference("pdfjs.disabled", True)  # Disable PDF preview in browser
        # firefox_options.add_argument("-headless")  # Does not display browser

        web_driver = webdriver.Firefox(firefox_options)
        web_driver.set_page_load_timeout(5)

        for row in tqdm.tqdm(iterable=download_data, desc="Download PDF files", disable=not verbose):
            try:
                web_driver.get(row["url"])
            except exceptions.TimeoutException:
                pass

        time.sleep(20)

        for row in tqdm.tqdm(iterable=download_data, desc="Rename PDF files", disable=not verbose):
            # https://stackoverflow.com/questions/300445/how-to-unquote-a-urlencoded-unicode-string-in-python
            downloaded_file = re.search(r"\d+\/(.*\.pdf)$", row["url"])[1]
            downloaded_file = urllib.parse.unquote(downloaded_file)
            downloaded_file = re.sub(r"\s{2,}", " ", downloaded_file).strip()
            downloaded_file = os.path.join(relative_download_dir_path, downloaded_file)
            renamed_file = os.path.join(relative_download_dir_path, row["name"] + ".pdf")

            if os.path.exists(downloaded_file):
                os.rename(downloaded_file, renamed_file)
            else:
                print(f"File {downloaded_file} not found.")
        
        time.sleep(5)
        
        web_driver.quit()

        if verbose:
            print(f"Successfully download PDF files to {relative_download_dir_path}")
    