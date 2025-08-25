import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load job dataset
def load_jobs():
    # dynamically build path: backend/../data/jobs.csv
    base_dir = os.path.dirname(__file__)  
    file_path = os.path.join(base_dir, "..", "data", "jobs.csv")
    jobs = pd.read_csv(file_path)
    return jobs
# Build recommender system
def recommend_jobs(user_skills, top_n=3):
    jobs = load_jobs()

    # TF-IDF on job skills column
    vectorizer = TfidfVectorizer()
    job_skill_matrix = vectorizer.fit_transform(jobs['skills'])

    # Convert user input into vector
    user_vector = vectorizer.transform([user_skills])

    # Cosine similarity between user and jobs
    similarity_scores = cosine_similarity(user_vector, job_skill_matrix).flatten()

    # Rank jobs by similarity
    jobs['score'] = similarity_scores
    ranked_jobs = jobs.sort_values(by="score", ascending=False)

    return ranked_jobs[['job_title', 'skills', 'location', 'salary', 'score']].head(top_n)

# Entry point
if __name__ == "__main__":
    user_input = input("Enter your skills (comma-separated): ")
    recs = recommend_jobs(user_input)
    print("\nRecommendations for:", user_input)
    print(recs)
