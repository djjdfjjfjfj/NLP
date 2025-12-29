import sys
import getopt
import pandas as pd
import re

# Precompile regex patterns for better performance
ACL_PATTERN = re.compile(r"https?://aclanthology\.org/\S+")

# Centralize threshold constants for better maintainability
# Documentation score thresholds
DOCUMENTATION_EXCELLENT_THRESHOLD = 4
DOCUMENTATION_TRUSTED_THRESHOLD = 2
DOCUMENTATION_AVG_HIGH_THRESHOLD = 4
DOCUMENTATION_AVG_MEDIUM_THRESHOLD = 2
DOCUMENTATION_TOTAL_HIGH_THRESHOLD = 8
DOCUMENTATION_TOTAL_MEDIUM_THRESHOLD = 4

# Popularity thresholds
POPULARITY_HIGH_THRESHOLD = 200
POPULARITY_MEDIUM_THRESHOLD = 100

# Adoption thresholds
ADOPTION_HIGH_THRESHOLD = 50
ADOPTION_MEDIUM_THRESHOLD = 20

# Recency thresholds
RECENCY_HIGH_THRESHOLD = 6
RECENCY_MEDIUM_THRESHOLD = 12

# Scientific contribution thresholds
SCIENTIFIC_HIGH_THRESHOLD = 3
SCIENTIFIC_MEDIUM_THRESHOLD = 2

# Centralize documentation keywords using sets for faster lookup
DOCUMENTATION_KEYWORDS = {
    "usage": {"usage", "how to use"},
    "license": {"license"},
    "examples": {"example", "examples"},
    "citation": {"citation", "how to cite"},
    "description": {"description", "overview"},
    "authors": {"author", "maintainer"}
}

ANNOTATION_KEYWORDS = {
    "task": {"task categories", "task_categories"},
    "language": {"language"},
    "size": {"size categories", "size_categories"},
    "license": {"license"},
    "source": {"dataset source", "source_datasets"},
    "configs": {"configs", "dataset_info"}
}

def get_documentation_score(readme_text):
    text = readme_text.lower()
    score = 0
    
    # Use pre-defined sets for faster keyword matching
    for keywords in DOCUMENTATION_KEYWORDS.values():
        if any(keyword in text for keyword in keywords):
            score += 1
    
    return score


def get_annotation_score(readme_text):
    text = readme_text.lower()
    score = 0
    
    # Use pre-defined sets for faster keyword matching
    for keywords in ANNOTATION_KEYWORDS.values():
        if any(keyword in text for keyword in keywords):
            score += 1
    
    return score


def classify_documentation_annotation_level(readme_text):
    
    score_doc = get_documentation_score(readme_text)
    score_ann = get_annotation_score(readme_text)
    
    avg_score = (score_doc + score_ann) / 2

    if avg_score >= DOCUMENTATION_AVG_HIGH_THRESHOLD:
        return "High"   # Excellent Threshold
    elif avg_score >= DOCUMENTATION_AVG_MEDIUM_THRESHOLD:
        return "Medium"  # Trusted Threshold
    else:
        return "Low"
    

def classify_documentation_annotation_level_t(readme_text):
    
    score_doc = get_documentation_score(readme_text)
    score_ann = get_annotation_score(readme_text)
    
    t_score = score_doc + score_ann

    if t_score >= DOCUMENTATION_TOTAL_HIGH_THRESHOLD:
        return "High"   # Excellent Threshold
    elif t_score >= DOCUMENTATION_TOTAL_MEDIUM_THRESHOLD:
        return "Medium"  # Trusted Threshold
    else:
        return "Low"
    

def get_popularity_score(df):
    df['popularity_score'] = df['Likes'].fillna(0) + df['Downloads'].fillna(0)
    return df


def classify_popularity_level(popularity_score):
    if popularity_score >= POPULARITY_HIGH_THRESHOLD:
        return "High"
    elif popularity_score >= POPULARITY_MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"
    

def get_spaces_count(dataset_id):
    from huggingface_hub import HfApi
    api = HfApi()

    spaces = api.list_spaces(datasets=dataset_id)
    spaces = len([space.id for space in spaces])
    
    return spaces


def get_adoption_score(df):
    df['adoption_score'] = df['Models'].fillna(0) + df['Spaces'].fillna(0)
    return df


def classify_adoption_level(adoption_score):
    if adoption_score >= ADOPTION_HIGH_THRESHOLD:
        return "High"
    elif adoption_score >= ADOPTION_MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"
    

def extract_date(timestamp):
    import pandas as pd
    if pd.isnull(timestamp):
        return None
    return str(timestamp).split()[0]


def months_difference(d1, d2):
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)


def get_recency_maintenance_score(df):
    import pandas as pd
    from datetime import datetime

    df['Last Modified'] = df['Last Modified'].apply(extract_date)
    df['Last Modified'] = pd.to_datetime(df['Last Modified'])

    now = pd.to_datetime(datetime.now().date())

    df['recency_maintenance_score'] = df['Last Modified'].apply(lambda x: months_difference(now, x)) 

    return df


def classify_recency_maintenance_level(recency_maintenance_score):
    if recency_maintenance_score <= RECENCY_HIGH_THRESHOLD:
        return "High"
    elif recency_maintenance_score <= RECENCY_MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"
    

def classify_licensing_transparency_level(license):
    if license == "none":
        return "Low"
    elif license == "unknown" or license == "other":
        return "Medium"
    else:        
        return "High"  


def get_doi_info(dataset_name):
    from huggingface_hub import HfApi
    api = HfApi()

    # Get dataset info
    dataset_info = api.dataset_info(dataset_name)

    for tag in dataset_info.tags:
        if "doi" in tag:
            doi = tag
            break
        else:  
            doi = None
    return [doi] if doi else "none"


def get_doi_score(dataset_name):
    from huggingface_hub import HfApi
    api = HfApi()

    # Get dataset info
    dataset_info = api.dataset_info(dataset_name)

    for tag in dataset_info.tags:
        if "doi" in tag:
            doi = tag
            break
        else:  
            doi = None
    return len([doi]) if doi else 0


def safe_count_links(x):
    import ast
    import pandas as pd
    
    if pd.isnull(x):
        return 0
    if isinstance(x, list):
        return len(x)
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return len(parsed)
        else:
            return 0
    except Exception:
        return 0


def get_scientific_contribution_scores(df):
    import pandas as pd

    df['arXiv_score'] = df['ArXiv Paper'].apply(lambda x: 1 if pd.notnull(x) and str(x).strip() != '' else 0)
    df["acl_score"] = df["ACL Paper"].apply(safe_count_links)
    df['doi_score'] = df['DOIs'].apply(lambda x: 1 if pd.notnull(x) and str(x).strip().lower() not in ['', 'none'] else 0)
    
    df['scientific_contribution_score'] = df['arXiv_score'].fillna(0) + df['acl_score'].fillna(0) + df['doi_score'].fillna(0)

    return df


def classify_scientific_contribution_level(scientific_contribution_score):
    if scientific_contribution_score >= SCIENTIFIC_HIGH_THRESHOLD:
        return "High"
    elif scientific_contribution_score >= SCIENTIFIC_MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"
    

def extract_acl_links(readme_content):
    return set(ACL_PATTERN.findall(readme_content))


def get_acl_links_from_readme(df):
    df['ACL Papers'] = df['README file'].apply(extract_acl_links)
    return 


def safe_get_doi_info(dataset_id):
    try:
        return get_doi_info(dataset_id)
    except Exception as e:
        if "401" in str(e) or "Repository Not Found" in str(e):
            print(f"⚠️ Skipping DOI for {dataset_id} (not found or private)")
        else:
            print(f"Warning: Failed to get DOI for {dataset_id}: {e}")
        return None


def main(argv):
    csv_file = None
    eval_column = None
    save_csv = False
    offline = False

    # --- Parse command line arguments ---
    try:
        opts, args = getopt.getopt(argv, "f:e:so", ["file=", "eval=", "save", "offline"])
    except getopt.GetoptError:
        print("Usage: python evals.py -f <csvfile> -e <column_name|all> [-s] [--offline]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-f", "--file"):
            csv_file = arg
        elif opt in ("-e", "--eval"):
            eval_column = arg
        elif opt in ("-s", "--save"):
            save_csv = True
        elif opt in ("-o", "--offline"):
            offline = True

    if not csv_file:
        print("CSV file is required. Usage: python evals.py -f <csvfile> -e <column_name|all> [-s]")
        sys.exit(2)

    # --- Load CSV ---
    df = pd.read_csv(csv_file)

    # --- Preprocessing ---
    if offline:
        print("Running in offline mode - skipping Hugging Face API calls")
        # Initialize fields that would normally come from API
        df['Spaces'] = None
        df['DOIs'] = None
    else:
        df['Spaces'] = df['Dataset ID'].apply(lambda x: get_spaces_count(x) if isinstance(x, str) else None)
        df['DOIs'] = df['Dataset ID'].apply(lambda x: safe_get_doi_info(x) if isinstance(x, str) else None)

    # Keep relevant columns
    columns_to_keep = [
        'Task', 'Dataset ID', 'Likes', 'Downloads', 'Last Modified', 'License',
        'Models', 'Spaces', 'DOIs', 'Size of downloaded files',
        'Size of downloaded files in bytes', 'Size of Parquet files',
        'Size of Parquet files in bytes', 'Number of Rows', 'ArXiv Paper',
        'ACL Paper', 'README file'
    ]
    df = df[[col for col in columns_to_keep if col in df.columns]]

    # --- Derived scores and levels ---
    df['documentation_score'] = df['README file'].apply(lambda x: get_documentation_score(x) if isinstance(x, str) else None)
    df['annotation_score'] = df['README file'].apply(lambda x: get_annotation_score(x) if isinstance(x, str) else None)
    df['documentation_annotation_level'] = df['README file'].apply(lambda x: classify_documentation_annotation_level(x) if isinstance(x, str) else "Uncategorized")

    df = get_popularity_score(df)
    df['popularity_level'] = df['popularity_score'].apply(lambda x: classify_popularity_level(x))

    df = get_adoption_score(df)
    df['adoption_level'] = df['adoption_score'].apply(lambda x: classify_adoption_level(x))

    df = get_recency_maintenance_score(df)
    df['recency_maintenance_level'] = df['recency_maintenance_score'].apply(lambda x: classify_recency_maintenance_level(x))

    df['licensing_transparency_level'] = df['License'].apply(lambda x: classify_licensing_transparency_level(x))

    df = get_scientific_contribution_scores(df)
    df['scientific_contribution_level'] = df['scientific_contribution_score'].apply(lambda x: classify_scientific_contribution_level(x))

    # --- Evaluation metrics list ---
    eval_metrics = [
        "documentation_annotation_level",
        "popularity_level",
        "adoption_level",
        "recency_maintenance_level",
        "licensing_transparency_level",
        "scientific_contribution_level"
    ]

    # --- Print counts ---
    if eval_column:
        if eval_column.lower() == "all":
            for metric in eval_metrics:
                if metric in df.columns:
                    print(f"\n{metric} Counts:")
                    print(df[metric].value_counts(dropna=False))
        elif eval_column in df.columns:
            print(f"\n{eval_column} Counts:")
            print(df[eval_column].value_counts(dropna=False))
        else:
            print(f"Error: Column '{eval_column}' not found in CSV.")
    else:
        # Default: print all metrics
        for metric in eval_metrics:
            if metric in df.columns:
                print(f"\n{metric} Counts:")
                print(df[metric].value_counts(dropna=False))

    # --- Save to CSV if -s is used ---
    if save_csv:
        out_file = csv_file.replace(".csv", "_evaluated.csv")
        df.to_csv(out_file, index=False)
        print(f"\n✅ Evaluation completed. Results saved to {out_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
