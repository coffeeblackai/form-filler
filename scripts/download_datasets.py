#!/usr/bin/env python3
import os
import subprocess
import requests
import zipfile
from bs4 import BeautifulSoup
import re
import hashlib
import time # For potential delays if needed

# Directory to store raw downloads
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw'))
FUNSD_URL = 'https://github.com/GuillaumeJaume/FUNSD.git'
FORM_NLU_URL = 'https://example.com/formnlu.zip'  # TODO: replace with actual Form-NLU download link
DOCLAYNET_URL = 'https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip'
IRS_FORMS_PAGE = 'https://www.irs.gov/forms-pubs'
OPM_STANDARD_FORMS_PAGE = 'https://www.opm.gov/forms/standard-forms/'
OPM_OPTIONAL_FORMS_PAGE = 'https://www.opm.gov/forms/optional-forms/'
OPM_OPM_FORMS_PAGE = 'https://www.opm.gov/forms/opm-forms/'
OPM_RETIREMENT_FORMS_PAGE = 'https://www.opm.gov/forms/retirement-and-insurance-forms/'
OPM_INVESTIGATION_FORMS_PAGE = 'https://www.opm.gov/forms/federal-investigation-forms/'
OPM_FEGLI_FORMS_PAGE = 'https://www.opm.gov/forms/federal-employees-group-life-insurance-forms/'


def download_funsd():
    out_dir = os.path.join(RAW_DIR, 'FUNSD')
    if not os.path.isdir(out_dir):
        print('[FUNSD] Cloning repository...')
        subprocess.run(['git', 'clone', FUNSD_URL, out_dir], check=True)
    else:
        print('[FUNSD] Already exists, skipping.')


def download_formnlu():
    out_dir = os.path.join(RAW_DIR, 'Form-NLU')
    zip_path = os.path.join(RAW_DIR, 'Form-NLU.zip')
    # Skip if already extracted
    if os.path.isdir(out_dir):
        print('[Form-NLU] Already exists, skipping.')
        return
    # If zip is present, just extract it
    if os.path.isfile(zip_path):
        print(f'[Form-NLU] Found existing zip at {zip_path}, extracting...')
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_dir)
        return
    # Otherwise, attempt automatic download
    print('[Form-NLU] Downloading archive...')
    try:
        r = requests.get(FORM_NLU_URL, stream=True)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[Form-NLU] Automatic download failed: {e}")
        print(f"Please manually download the Form-NLU dataset from https://github.com/adlnlp/form_nlu and place the zip at {zip_path}")
        return
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    # Extract after download
    print('[Form-NLU] Extracting archive...')
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)


def download_doclaynet():
    out_dir = os.path.join(RAW_DIR, 'DocLayNet')
    zip_path = os.path.join(RAW_DIR, 'DocLayNet.zip')
    if not os.path.isdir(out_dir):
        print('[DocLayNet] Downloading archive...')
        r = requests.get(DOCLAYNET_URL, stream=True)
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print('[DocLayNet] Extracting archive...')
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_dir)
    else:
        print('[DocLayNet] Already exists, skipping.')


def download_irs_forms():
    out_dir = os.path.join(RAW_DIR, 'IRS-PDFs')
    os.makedirs(out_dir, exist_ok=True)
    print(f'[IRS] Scraping {IRS_FORMS_PAGE}...')
    try:
        r = requests.get(IRS_FORMS_PAGE)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        content_area = soup.find('main') or soup
        links = set()
        for a in content_area.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE)):
             links.add(a['href'])
        print(f'[IRS] Found {len(links)} unique PDF links.')
        downloaded_count = 0
        for href in links:
            pdf_url = href if href.startswith('http') else f'https://www.irs.gov{href}'
            try:
                 filename = os.path.basename(href).split('?')[0]
                 if not filename: filename = f"irs_form_{hashlib.sha1(pdf_url.encode()).hexdigest()[:8]}.pdf"
            except Exception: filename = f"irs_form_{hashlib.sha1(pdf_url.encode()).hexdigest()[:8]}.pdf"
            save_path = os.path.join(out_dir, filename)
            if not os.path.isfile(save_path):
                try:
                    resp = requests.get(pdf_url, stream=True, timeout=30)
                    resp.raise_for_status()
                    if 'application/pdf' not in resp.headers.get('Content-Type', '').lower():
                        print(f"\nWarning: Skipping non-PDF link {pdf_url} (Content-Type: {resp.headers.get('Content-Type')})")
                        continue
                    with open(save_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded_count += 1
                except requests.exceptions.RequestException as e:
                    print(f"\nWarning: Failed to download {pdf_url}: {e}")
                    if os.path.exists(save_path):
                        try: 
                            os.remove(save_path)
                        except OSError: 
                            pass # Ignore error if file couldn't be removed
    except requests.exceptions.RequestException as e:
        print(f"Error accessing IRS forms page {IRS_FORMS_PAGE}: {e}")
    print(f'[IRS] Download check complete. Newly downloaded: {downloaded_count}')


def scrape_and_download_pdfs(page_url, output_subdir_name, base_url, prefix):
    """Generic function to scrape a page for PDFs and download them."""
    out_dir = os.path.join(RAW_DIR, output_subdir_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'[{prefix}] Scraping {page_url}...')
    downloaded_count = 0
    try:
        r = requests.get(page_url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        links = set()
        # Prioritize links within table cells
        table_cells = soup.find_all('td')
        if table_cells:
            for td in table_cells:
                for a in td.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE)):
                    links.add(a['href'])
        
        # Fallback: Check main content area if no table links found
        if not links:
            content_area = soup.find('main') or soup
            print(f"[{prefix}] Info: No PDF links found in table cells. Searching wider area.")
            for a in content_area.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE)):
                 links.add(a['href'])

        print(f'[{prefix}] Found {len(links)} unique PDF links.')

        for href in links:
            # pdf_url = href if href.startswith('http') else f'{base_url.rstrip("/")}{href}'
            # FIX: Use standard concatenation to avoid f-string syntax issue
            if href.startswith('http'):
                pdf_url = href
            else:
                pdf_url = base_url.rstrip('/') + href
            
            try:
                 filename = os.path.basename(href).split('?')[0]
                 if not filename: filename = f"{prefix.lower()}_form_{hashlib.sha1(pdf_url.encode()).hexdigest()[:8]}.pdf"
            except Exception:
                 filename = f"{prefix.lower()}_form_{hashlib.sha1(pdf_url.encode()).hexdigest()[:8]}.pdf"

            save_path = os.path.join(out_dir, filename)

            if not os.path.isfile(save_path):
                # print(f'[{prefix}] Downloading {filename}...')
                try:
                    # Optional short delay between requests
                    # time.sleep(0.1) 
                    resp = requests.get(pdf_url, stream=True, timeout=30)
                    resp.raise_for_status()
                    if 'application/pdf' not in resp.headers.get('Content-Type', '').lower():
                         print(f"\nWarning: Skipping non-PDF link {pdf_url} (Content-Type: {resp.headers.get('Content-Type')})")
                         continue
                    with open(save_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded_count += 1 # Increment count here
                except requests.exceptions.RequestException as e:
                    print(f"\nWarning: Failed to download {pdf_url}: {e}")
                    if os.path.exists(save_path):
                        try: 
                            os.remove(save_path)
                        except OSError: 
                            pass

    except requests.exceptions.RequestException as e:
        print(f"Error accessing page {page_url}: {e}")
    print(f'[{prefix}] Download check complete. Newly downloaded: {downloaded_count}')


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    # download_funsd() # Skip FUNSD
    # download_formnlu() # Skip Form-NLU
    # download_doclaynet() # Skip DocLayNet
    download_irs_forms() # Only download IRS forms
    
    # Download from various OPM pages using the generic function
    scrape_and_download_pdfs(OPM_STANDARD_FORMS_PAGE, "OPM-Standard", "https://www.opm.gov", "OPM-Std")
    scrape_and_download_pdfs(OPM_OPTIONAL_FORMS_PAGE, "OPM-Optional", "https://www.opm.gov", "OPM-Opt")
    scrape_and_download_pdfs(OPM_OPM_FORMS_PAGE, "OPM-OPM", "https://www.opm.gov", "OPM-OPM")
    scrape_and_download_pdfs(OPM_RETIREMENT_FORMS_PAGE, "OPM-Retirement", "https://www.opm.gov", "OPM-Ret")
    scrape_and_download_pdfs(OPM_INVESTIGATION_FORMS_PAGE, "OPM-Investigation", "https://www.opm.gov", "OPM-Inv")
    scrape_and_download_pdfs(OPM_FEGLI_FORMS_PAGE, "OPM-FEGLI", "https://www.opm.gov", "OPM-FEGLI")


if __name__ == '__main__':
    main() 