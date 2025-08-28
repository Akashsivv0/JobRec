import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_job_graph(jobs_df, similarity_threshold=0.55):
    """
    Builds a graph where jobs are nodes and edges connect jobs with
    a high cosine similarity based on their skills.
    
    Args:
        jobs_df (pd.DataFrame): DataFrame containing job data with a 'skills_desc' column.
        similarity_threshold (float): The minimum similarity score to create an edge.
        
    Returns:
        networkx.Graph: A graph of jobs connected by skill similarity.
    """
    if jobs_df.empty:
        return nx.Graph()

    # Create a TF-IDF vectorizer and transform job skills
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(jobs_df['skills_desc'].fillna(''))
    
    # Compute cosine similarity for all pairs of jobs
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Initialize a new graph
    job_graph = nx.Graph()
    job_graph.add_nodes_from(jobs_df['job_id'])

    # Add edges between jobs that are highly similar
    for i in range(len(jobs_df)):
        for j in range(i + 1, len(jobs_df)):
            similarity = cosine_sim_matrix[i, j]
            if similarity > similarity_threshold:
                job_graph.add_edge(jobs_df['job_id'].iloc[i], jobs_df['job_id'].iloc[j], weight=similarity)
                
    return job_graph

def find_related_jobs(job_id, job_graph, jobs_df):
    """
    Finds jobs related to a specific job by traversing the graph.
    
    Args:
        job_id: The ID of the job to find related jobs for.
        job_graph (networkx.Graph): The graph of job similarities.
        jobs_df (pd.DataFrame): The original DataFrame with job details.
        
    Returns:
        pd.DataFrame: A DataFrame of related jobs.
    """
    if job_id not in job_graph.nodes:
        return pd.DataFrame()
        
    # Get neighbors (related jobs) from the graph
    related_ids = list(job_graph.neighbors(job_id))
    
    # Get the details for the related jobs from the main DataFrame
    related_jobs = jobs_df[jobs_df['job_id'].isin(related_ids)].copy()
    
    # The fix is here:
    if not related_jobs.empty:
        related_jobs['similarity'] = related_jobs['job_id'].apply(lambda x: job_graph[job_id][x]['weight'])
        return related_jobs.sort_values(by='similarity', ascending=False)
    else:
        # Return an empty DataFrame immediately if no related jobs are found
        return pd.DataFrame()