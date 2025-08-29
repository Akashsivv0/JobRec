import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_jobs, add_to_recently_viewed, get_recently_viewed
from graph_recommender import build_job_graph, find_related_jobs
from pdf_parser import parse_pdf_resume # Import the new parser

# ---------------------------
# Recommend jobs
# ---------------------------
def recommend_jobs(user_skills, top_n=5):
    # Pass user skills to the load_jobs function to filter the dataset
    jobs = load_jobs(skills=user_skills.split(',')) 
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
# Run as script
# ---------------------------
if __name__ == "__main__":
    # -------------------------------------------------------------
    # THIS PART WILL BE HANDLED BY THE STREAMLIT FRONTEND LATER
    # For now, we'll simulate the user's skills from a parsed resume
    # -------------------------------------------------------------
    resume_path = "path/to/your/resume.pdf" # This would be a dynamic path from the UI
    try:
        resume_text = parse_pdf_resume(resume_path)
    except FileNotFoundError as e:
        print(e)
        resume_text = ""

    if not resume_text:
        print("\n❌ Could not parse resume. Please ensure the path is correct and the file is a valid PDF.")
        print("💡 Since parsing failed, please enter your skills manually for now.")
        user_input = input("Enter your skills (comma-separated): ")
        if not user_input:
            print("Please enter some skills to get recommendations.")
        else:
            user_skills = user_input
    else:
        # We will make this part smarter later, but for now, we'll
        # just use a sample string of skills that would be extracted
        print("\n✅ Resume text parsed successfully. Using sample skills for now.")
        user_skills = "java, python, machine learning, deep learning, excel"

    # --- End of frontend simulation ---
    
    if user_skills:
        try:
            # Load the data and build the graph once for efficiency
            all_jobs = load_jobs()
            if all_jobs.empty:
                print("Cannot proceed without a valid dataset.")
            else:
                job_graph = build_job_graph(all_jobs)
                
                recs = recommend_jobs(user_skills)

                if not recs.empty:
                    print("\n🔎 Recommended Jobs:")
                    for _, row in recs.iterrows():
                        salary_info = ""
                        if pd.notna(row.get('med_salary')):
                            salary_info = f"({row['med_salary']:.2f} {row.get('pay_period', 'N/A')})"
                        elif pd.notna(row.get('min_salary')) and pd.notna(row.get('max_salary')):
                            salary_info = f"({row['min_salary']:.2f} - {row['max_salary']:.2f} {row.get('pay_period', 'N/A')})"
                        
                        print(
                            f"- {row['title']} at {row['company_name']} ({row['location']}) {salary_info} | match={row['similarity']:.2f}"
                        )
                        # Add the job to the recently viewed queue
                        add_to_recently_viewed(row['job_id'])
                    
                    # Get the ID of the first recommended job to find related jobs
                    top_job_id = recs['job_id'].iloc[0]
                    related_jobs = find_related_jobs(top_job_id, job_graph, all_jobs)

                    print("\n🔗 Jobs You Might Also Like:")
                    if not related_jobs.empty:
                        for _, row in related_jobs.iterrows():
                            print(
                                f"- {row['title']} at {row['company_name']} ({row['location']}) | match={row['similarity']:.2f}"
                            )
                    else:
                        print("No related jobs found based on graph similarity.")

                    print("\n👁️ Recently Viewed Jobs:")
                    recent_jobs = get_recently_viewed()
                    for _, row in recent_jobs.iterrows():
                         print(f"- {row['title']} at {row['company_name']} ({row['location']})")

                else:
                    print("No jobs found or no matches could be made with the provided dataset.")
        except ValueError as e:
            print(e)