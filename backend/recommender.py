import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# File paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_PATH = os.path.join(BASE_DIR, "data", "all_job_post.csv")


# ---------------------------
# Load jobs
# ---------------------------
def load_jobs():
    print(f"Loading jobs dataset from {DATA_PATH}...")
    try:
        jobs = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ File not found at {DATA_PATH}")
        return pd.DataFrame()

    # Keep only relevant columns
    required_columns = ["job_title", "category", "job_skill_set"]
    if not all(col in jobs.columns for col in required_columns):
        missing = [col for col in required_columns if col not in jobs.columns]
        raise ValueError(f"❌ Missing required columns: {missing}")

    jobs = jobs[required_columns].dropna()
    return jobs


# ---------------------------
# Recommend jobs
# ---------------------------
def recommend_jobs(user_skills, top_n=5):
    jobs = load_jobs()

    if jobs.empty:
        return pd.DataFrame()

    # Vectorize job skills
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(jobs["job_skill_set"])

    # Transform user input
    user_vec = vectorizer.transform([user_skills])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()

    # Get top matches
    top_indices = cosine_sim.argsort()[-top_n:][::-1]

    recommendations = jobs.iloc[top_indices].copy()
    recommendations["similarity"] = cosine_sim[top_indices]

    return recommendations


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
                for _, row in recs.iterrows():
                    print(
                        f"- {row['job_title']} ({row['category']}) | match={row['similarity']:.2f}"
                    )
            else:
                print("No jobs found or no matches could be made with the provided dataset.")
        except ValueError as e:
            print(e)
