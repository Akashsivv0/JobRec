import os
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque # Import deque for a more efficient queue

# ---------------------------
# File paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "postings.csv")

# A global queue to store recently viewed job IDs
# Deque is more efficient for append and pop operations than a list
recently_viewed_queue = deque(maxlen=5) 

# ---------------------------
# Load jobs
# ---------------------------
def load_jobs():
    print(f"Loading jobs dataset from {DATA_PATH}...")
    try:
        jobs = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ File not found at {DATA_PATH}. Please download the 'job_postings.csv' file from Kaggle and place it in a 'data' folder.")
        return pd.DataFrame()

    required_columns = ["title", "company_name", "location", "skills_desc"]
    if not all(col in jobs.columns for col in required_columns):
        missing = [col for col in required_columns if col not in jobs.columns]
        raise ValueError(f"❌ Missing required columns: {missing}")

    jobs = jobs[required_columns].dropna()
    # Add a unique ID for each job to track them in the queue
    jobs['job_id'] = jobs.index
    return jobs

# ---------------------------
# Recommend jobs
# ---------------------------
def recommend_jobs(user_skills, top_n=5):
    jobs = load_jobs()
    if jobs.empty:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(jobs["skills_desc"])

    user_vec = vectorizer.transform([user_skills])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]

    recommendations = jobs.iloc[top_indices].copy()
    recommendations["similarity"] = cosine_sim[top_indices]

    return recommendations

# ---------------------------
# Data structure function
# ---------------------------
def add_to_recently_viewed(job_id):
    """Adds a job ID to the front of the queue."""
    # Add the ID to the queue
    recently_viewed_queue.append(job_id) #the queue function

def get_recently_viewed():
    """Returns the jobs from the queue in order from newest to oldest."""
    jobs = load_jobs()
    if jobs.empty:
        return pd.DataFrame()
    
    # Get the unique job IDs from the queue, maintaining order
    viewed_ids = list(recently_viewed_queue)
    # Get the corresponding job data from the main DataFrame
    return jobs[jobs['job_id'].isin(viewed_ids)]

# ---------------------------
# Run as script
# ---------------------------
if __name__ == "__main__":
    user_input = input("Enter your skills (comma-separated): ")
    if not user_input:
        print("Please enter some skills to get recommendations.")
    else:
        try:
            recs = recommend_jobs(user_input)

            if not recs.empty:
                print("\n🔎 Recommended Jobs:")
                for index, row in recs.iterrows():
                    print(
                        f"- {row['title']} at {row['company_name']} ({row['location']}) | match={row['similarity']:.2f}"
                    )
                    # Add the job to the recently viewed queue
                    add_to_recently_viewed(row['job_id'])
                
                print("\n👁️ Recently Viewed Jobs:")
                recent_jobs = get_recently_viewed()
                for _, row in recent_jobs.iterrows():
                     print(f"- {row['title']} at {row['company_name']} ({row['location']})")
            else:
                print("No jobs found or no matches could be made with the provided dataset.")
        except ValueError as e:
            print(e) 
