import re
import time
import sys
import getopt
import yaml
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from urllib3.util.ssl_ import create_urllib3_context

# Precompile regex patterns for better performance
SIZE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)', re.IGNORECASE)
ACL_PATTERN = re.compile(r"https?://aclanthology\.org/\S+")

# Centralize threshold constants for better maintainability
README_LENGTH_THRESHOLD = 20
README_QUALITY_HIGH_THRESHOLD = 4
README_QUALITY_MEDIUM_THRESHOLD = 2

# Centralize documentation keywords using sets for faster lookup
DOCUMENTATION_KEYWORDS = {
    "usage": {"usage", "how to use"},
    "license": {"license"},
    "examples": {"example", "examples"},
    "citation": {"citation", "how to cite"},
    "description": {"description", "overview"},
    "authors": {"author", "maintainer"}
}

# Disable SSL verification warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Create a custom adapter that disables SSL verification
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.check_hostname = False
        context.verify_mode = 0  # CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

# Create a session with SSL disabled
session = requests.Session()
session.mount('https://', SSLAdapter())

# Patch huggingface_hub to use our session if needed
import huggingface_hub.hf_api
# Try to patch the session, but handle cases where internal structure might have changed
try:
    huggingface_hub.hf_api.HfApi.session = session
except Exception as e:
    pass  # Ignore if session patching fails

from huggingface_hub import list_datasets, list_models, hf_hub_download

from huggingface_hub.utils import logging
logging.set_verbosity_error()


def covert_sizes_to_bytes(size):
    size_map = {
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4
    }

    matches = SIZE_PATTERN.findall(size)
    for value, unit in matches:
        unit_upper = unit.upper()
        bytes_size = float(value) * size_map[unit_upper]
        return bytes_size
    return 0  # Return 0 if no match found



def get_page_source(url):
    # For Windows, ChromeDriver is usually in PATH or at specific location
    CHROMEDRIVER_PATH = "chromedriver.exe"  # Assume it's in PATH or current directory

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)

    try:
        # Go directly to the dataset page
        driver.get(url)
        time.sleep(3)


        return driver.page_source

    finally:
        driver.quit()

def get_readme_score(readme_text):
    text = readme_text.lower()
    score = 0
    
    # Use pre-defined sets for faster keyword matching
    for keywords in DOCUMENTATION_KEYWORDS.values():
        if any(keyword in text for keyword in keywords):
            score += 1
    
    return score



def classify_readme_level(readme_text):
    if not isinstance(readme_text, str) or len(readme_text.strip()) < README_LENGTH_THRESHOLD:
        return "Low"

    score = get_readme_score(readme_text)
    
    # Classification logic using centralized thresholds
    if score >= README_QUALITY_HIGH_THRESHOLD:
        return "High"   # Excellent Threshold
    elif score >= README_QUALITY_MEDIUM_THRESHOLD:
        return "Medium"  # Trusted Threshold
    else:
        return "Low"


def get_dataset_size_info(dataset_name):
    url = f"https://huggingface.co/datasets/{dataset_name}"
   
    
    try:
        page_source = get_page_source(url)        
        soup = BeautifulSoup(page_source, 'html.parser')

        Size_of_downloaded_dataset_files = soup.find_all('a', class_='bg-linear-to-r dark:via-none group mb-1.5 flex max-w-full flex-col overflow-hidden rounded-lg border border-gray-100 from-white via-white to-white px-2 py-1 hover:from-gray-50 dark:from-gray-900 dark:to-gray-925 dark:hover:to-gray-900 md:mr-1.5 pointer-events-none')
        Size_of_downloaded_dataset_files_str = Size_of_downloaded_dataset_files[0].text.strip("Size of downloaded dataset files:").strip("\n")
        Size_of_downloaded_dataset_files_bytes = covert_sizes_to_bytes(Size_of_downloaded_dataset_files_str)
        
        
        Size_of_the_auto_converted_Parquet_files = soup.find_all('div', class_='truncate text-sm group-hover:underline')
        Size_of_the_auto_converted_Parquet_files_str = Size_of_the_auto_converted_Parquet_files[-1].text.strip()
        Size_of_the_auto_converted_Parquet_files_bytes = covert_sizes_to_bytes(Size_of_the_auto_converted_Parquet_files_str)
        

        Number_of_rows = soup.find_all('a', class_='bg-linear-to-r dark:via-none group mb-1.5 flex max-w-full flex-col overflow-hidden rounded-lg border border-gray-100 from-white via-white to-white px-2 py-1 hover:from-gray-50 dark:from-gray-900 dark:to-gray-925 dark:hover:to-gray-900 md:mr-1.5 pointer-events-none')
        Number_of_rows = Number_of_rows[1].text.strip("Number of rows:").strip("\n")
        Number_of_rows = Number_of_rows.strip(",").replace(",", "")    
       

        return Size_of_downloaded_dataset_files_str, Size_of_downloaded_dataset_files_bytes, Size_of_the_auto_converted_Parquet_files_str, Size_of_the_auto_converted_Parquet_files_bytes, Number_of_rows
    
    except IndexError:
        return "unknown", "unknown", "unknown", "unknown", "unknown"
    
    except UnboundLocalError:
        return "unknown", "unknown", "unknown", "unknown", "unknown"

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        return "unknown", "unknown", "unknown", "unknown", "unknown"#


def get_dataset_readme(dataset_id):
    readme_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename="README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    return readme_content


def extract_acl_links(readme_content):
    return ACL_PATTERN.findall(readme_content)



def get_arabic_datasets_by_task_categories(task_mapping):
    from huggingface_hub import list_datasets
    all_rows = []
    
    # Loop through each task
    for user_task, hf_task in task_mapping.items():
        # print(f"🔍 Processing task: {user_task} ({hf_task})")
        try:
            datasets_list = list_datasets(task_categories=hf_task, language="ar")
        except Exception as e:
            print(f"Failed to fetch for task {user_task}: {e}")
            continue
    
        for dataset in datasets_list:
            tags = dataset.tags if hasattr(dataset, "tags") else []
    
            try:
                license = [tag for tag in dataset.tags if "license" in tag][0].split(":")[-1]
            except:
                license = "none"
            try:
                models = len(list(list_models(filter=f"dataset:{dataset.id}")))
            except:
                models = "none"
    
            # size = next((tag.split(":")[-1] for tag in tags if tag.startswith("size:")), "unknown")
            Size_of_downloaded_dataset_files_str, Size_of_downloaded_dataset_files_bytes, Size_of_the_auto_converted_Parquet_files_str, Size_of_the_auto_converted_Parquet_files_bytes, Number_of_rows = get_dataset_size_info(dataset.id)
            
            arxiv_link = next((tag.split(":", 1)[-1] for tag in tags if tag.startswith("arxiv:")), "none")
            if arxiv_link != "none":
                arxiv_link = "https://arxiv.org/abs/"+arxiv_link

            try:
                readme = get_dataset_readme(dataset.id)
                if len(readme) == 0:
                    readme = "none" 
                    
                acl_links = extract_acl_links(readme)
                if len(acl_links) == 0:
                    acl_links = "none" 
            except:
                readme = "none"
                acl_links = "none"  

            readme_quality_level = classify_readme_level(readme)
            readme_quality_score = get_readme_score(readme)

            all_rows.append({
                "Task": user_task,
                "Dataset ID": dataset.id,
                "Likes": dataset.likes,
                "Downloads": dataset.downloads,
                "Last Modified": dataset.lastModified,
                "License": license,
                "Models": models,
                "Size of downloaded files": Size_of_downloaded_dataset_files_str.upper(),
                "Size of downloaded files in bytes": Size_of_downloaded_dataset_files_bytes,
                "Size of Parquet files": Size_of_the_auto_converted_Parquet_files_str.upper(),
                "Size of Parquet files in bytes": Size_of_the_auto_converted_Parquet_files_bytes,
                "Number of Rows" : Number_of_rows,
                "ArXiv Paper": arxiv_link,
                "ACL Paper": acl_links,
                "README file": readme,
                "README Quality Level": readme_quality_level,
                "README Quality Score": readme_quality_score
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(all_rows)
    return df



def get_arabic_datasets_by_keywords(search_keywords, required_tags, required_modality):
    #from huggingface_hub import list_datasets
    # Get only Arabic datasets
    datasets_list = list_datasets(language="ar")
   
    dataset_rows = []

    for dataset in datasets_list:
        dataset_id = dataset.id.lower()
        dataset_tags = dataset.tags or []

        # Check for match in name 
        name_match = any(keyword.lower() in dataset_id for keyword in search_keywords)
        
        # Make sure it's a text dataset
        has_text_modality = required_modality in dataset_tags

        if has_text_modality and (name_match ):
            try:
                license = [tag for tag in dataset_tags if "license:" in tag][0].split(":")[-1]
            except:
                license = "none"
            try:
                models = len(list(list_models(filter=f"dataset:{dataset.id}")))
            except:
                models = "none"

            # size = next((tag.split(":")[-1] for tag in dataset_tags if tag.startswith("size:")), "unknown")
            Size_of_downloaded_dataset_files_str, Size_of_downloaded_dataset_files_bytes, Size_of_the_auto_converted_Parquet_files_str, Size_of_the_auto_converted_Parquet_files_bytes, Number_of_rows = get_dataset_size_info(dataset.id)

            arxiv_link = next((tag.split(":", 1)[-1] for tag in dataset_tags if tag.startswith("arxiv:")), "none")
            if arxiv_link != "none":
                arxiv_link = "https://arxiv.org/abs/"+arxiv_link

            try:
                readme = get_dataset_readme(dataset.id)
                if len(readme) == 0:
                    readme = "none" 
                    
                acl_links = extract_acl_links(readme)
                if len(acl_links) == 0:
                    acl_links = "none" 
            except:
                readme = "none"
                acl_links = "none"  

            readme_quality_level = classify_readme_level(readme)
            readme_quality_score = get_readme_score(readme)
            
            dataset_rows.append({
                "Dataset ID": dataset.id,
                "Likes": dataset.likes,
                "Downloads": dataset.downloads,
                "Last Modified": dataset.lastModified,
                "License": license,
                "Models": models,
                "Size of downloaded files": Size_of_downloaded_dataset_files_str.upper(),
                "Size of downloaded files in bytes": Size_of_downloaded_dataset_files_bytes,
                "Size of Parquet files": Size_of_the_auto_converted_Parquet_files_str.upper(),
                "Size of Parquet files in bytes": Size_of_the_auto_converted_Parquet_files_bytes,
                "Number of Rows" : Number_of_rows,
                "ArXiv Paper": arxiv_link,
                "ACL Paper": acl_links,
                "README file": readme,
                "README Quality Level": readme_quality_level,
                "README Quality Score": readme_quality_score
            })

    # Create and show DataFrame
    df = pd.DataFrame(dataset_rows)
    return df


def load_task_mapping(yaml_file_path):
    """
    Reads the task mapping from a YAML file and returns it as a Python dictionary.
    """
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        task_mapping = yaml.safe_load(f)
    return task_mapping



if __name__ == "__main__":

    #task_mapping = load_task_mapping("task_mapping.yaml")
    task_mapping = {
    "Q&A": "question-answering",
    "Reasoning & Multi-step Thinking": "reasoning",
    "Summarization": "summarization",
    "Cultural Alignment": "cultural-aligned",
    "Dialog/Conversation": "conversational",
    "Personal Ownership/System Prompt": "System Prompt",  
    "Robustness & Safety": "Safety",
    "Function Call": "function-call",  
    "Ethics, Bias, and Fairness": "bias-and-fairness",
    "Code Generation": "Code Generation",
    "Official Documentation": "documentation",
    "Translation": "translation"
}

    # Keyword mapping per category (example)
    keywords_map = {
        "Reasoning & Multi-step Thinking": {
            "search_keywords": ["Reasoning", "Multi-step reasoning"],
            "required_tags": [],
            "required_modality": "modality:text"
        },
        "Cultural Alignment": {
            "search_keywords": ["cultural", "culture", "cidar"],
            "required_tags": ["cultural-aligned"],
            "required_modality": "modality:text"
        },
        "Dialog & Conversation": {
            "search_keywords": ["Dialog", "Conversation"],
            "required_tags": [],
            "required_modality": "modality:text"
        },
        "Personal Ownership/System Prompt": {
            "search_keywords": ["system prompt", "persona"],
            "required_tags": [],
            "required_modality": "modality:text"
        },
        "Robustness & Safety": {
            "search_keywords": ["Robustness", "Safety", "Toxicity", "jailbreak"],
            "required_tags": [],
            "required_modality": "modality:text"
        },
        "Ethics, Bias, and Fairness": {
            "search_keywords": ["Ethics", "Bias", "Fairness"],
            "required_tags": [],
            "required_modality": "modality:text"
        },
        "Function Call": {
            "search_keywords": ["Function Call"],
            "required_tags": [],
            "required_modality": "modality:text"
        },
        "Code Generation": {
            "search_keywords": ["code generation"],
            "required_tags": [],
            "required_modality": "modality:text"
        },
        "Official Documentation": {
            "search_keywords": ["Documentation", "Official Documentation"],
            "required_tags": [],
            "required_modality": "modality:text"
        }
       
    }

    category = None
    save = False
    list_categories = False
    offline = False
    try:
      opts, _ = getopt.getopt(sys.argv[1:], "c:slo", ["category=", "save", "list", "offline"])

    except getopt.GetoptError:
        print("Usage: python functions.py -c <category> [-s] | -l | -o") 
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-c", "--category"):
            category = arg
        elif opt in ("-s", "--save"):
            save = True
        elif opt in ("-l", "--list"):
            list_categories = True
        elif opt in ("-o", "--offline"):
            offline = True

    if list_categories:
        print("Available Categories:")
        for cat in task_mapping.keys():
            print(f" - {cat}")
        sys.exit(0)

    if offline:
        # Load from existing complete dataset
        print("Loading data in offline mode...")
        df = pd.read_csv("all_datasets_by_task_Updated.csv")
        if category:
            df = df[df['Task'] == category]
        print(df)
        if save and category:
            filename = f"{category.replace(' ', '_')}_offline.csv"
            df.to_csv(filename, index=False)
            print(f"Offline data saved to {filename}")
        sys.exit(0)

    if not category:
        print("Please provide a category using -c or --category, or use -l to list categories.")
        sys.exit(2)

    if category not in task_mapping:
        # print(f"Invalid category. Available categories: {list(task_mapping.keys())}")
        print(f"Invalid category: '{category}'")
        print("Available categories:")
        for cat in task_mapping.keys():
            print(f" - {cat}")
        sys.exit(1)
    print(f"Processing category: {category} ...")
        
    mapped_task = task_mapping[category]

    # First try get by task category
    df = get_arabic_datasets_by_task_categories({category: mapped_task})

    # If empty or no data, fallback to keyword search if mapping exists
    if df.empty and category in keywords_map:
        print(f"\nNo data found for category '{category}', trying keyword search fallback...")
        kw_info = keywords_map[category]
        df = get_arabic_datasets_by_keywords(
            kw_info["search_keywords"],
            kw_info["required_tags"],
            kw_info["required_modality"]
        )

        if df.empty:
            print(f"\n No data found for keywords fallback for category '{category}'.")
            print(f"\n No data found for the category '{category}'.")
        else:
            print(df)
            if save:
                filename = f"{category.replace(' ', '_')}_keywords.csv"
                df.to_csv(filename, index=False)
                print(f"Keyword search data saved to {filename}")

    else:
        if df.empty:
            print(f"\n No data found for the category '{category}'.")
        else:
            print(df)
            if save:
                filename = f"{category.replace(' ', '_')}.csv"
                df.to_csv(filename, index=False)
                print(f"Data saved to {filename}")