import pandas as pd
import os
from collections import deque

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "postings.csv")

# A global queue to store recently viewed job IDs
recently_viewed_queue = deque(maxlen=5) 

def load_jobs(skills=None):
    """
    Loads the jobs dataset from the specified CSV file and filters by skills.
    """
    print(f"Loading jobs dataset from {DATA_PATH}...")
    try:
        jobs = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ File not found at {DATA_PATH}.")
        return pd.DataFrame()

    required_columns = ["title", "company_name", "location", "skills_desc", "min_salary", "max_salary", "med_salary", "pay_period"]
    if not all(col in jobs.columns for col in required_columns):
        missing = [col for col in required_columns if col not in jobs.columns]
        raise ValueError(f"❌ Missing required columns: {missing}")

    # Drop rows where 'skills_desc' is empty before filtering
    jobs = jobs[required_columns].dropna(subset=['skills_desc'])

    # Filter jobs based on the presence of a skill from the input list
    if skills:
        jobs = jobs[jobs['skills_desc'].str.contains('|'.join(skills), case=False, na=False)]
    
    # Check if a 'job_id' column exists, otherwise create one
    if 'job_id' not in jobs.columns:
        jobs['job_id'] = range(len(jobs))

    return jobs

def add_to_recently_viewed(job_id):
    """Adds a job ID to the front of the queue."""
    recently_viewed_queue.append(job_id)

def get_recently_viewed():
    """Returns the jobs from the queue in order from newest to oldest."""
    jobs = load_jobs()
    if jobs.empty:
        return pd.DataFrame()
    
    viewed_ids = list(recently_viewed_queue)
    return jobs[jobs['job_id'].isin(viewed_ids)]