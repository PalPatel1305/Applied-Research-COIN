import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
import json

# Specify the path to your JSON file
json_file_path = 'data.json'

# Open the JSON file and load its contents using UTF-8 encoding
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Extract titles from the JSON object
titles = [entry["title"] for entry in data]

# Encode titles into embeddings
title_embeddings = embed(titles)

def semantic_search(query, titles, title_embeddings, top_n=15, similarity_threshold=0.1):
    # Encode query
    query_embedding = embed([query])[0]

    # Compute cosine similarity between query and titles
    similarity_scores = np.inner(query_embedding, title_embeddings)

    # Filter titles with similarity above threshold
    similar_titles = [(titles[i], similarity_scores[i]) for i in range(len(titles)) if similarity_scores[i] > similarity_threshold]

    # Sort by similarity score
    similar_titles.sort(key=lambda x: x[1], reverse=True)

    # Initialize set to store seen titles
    seen_titles = set()

    # Get top similar titles, ignoring duplicates
    top_titles = []
    for title, score in similar_titles:
        if title not in seen_titles:
            top_titles.append(title)
            seen_titles.add(title)
            if len(top_titles) >= top_n:
                break

    return top_titles
# Example usage
if __name__ == "__main__":
    query = "lessons"
    top_titles = semantic_search(query, titles, title_embeddings)
      
    print("Top Titles:")
    for title in top_titles:
        print(title)

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import json

# # Load Universal Sentence Encoder
# model = SentenceTransformer("all-mpnet-base-v2")

# # Specify the path to your JSON file
# json_file_path = 'data.json'

# # Open the JSON file and load its contents using UTF-8 encoding
# with open(json_file_path, 'r', encoding='utf-8') as json_file:
#     data = json.load(json_file)

# # Extract titles from the JSON object
# titles = [entry["title"] for entry in data]

# # Encode titles into embeddings
# title_embeddings = model.encode(titles)

# def semantic_search(query, titles, title_embeddings, top_n=10, similarity_threshold=0.1):
#     # Encode query
#     query_embedding = model.encode([query])[0]

#     # Compute cosine similarity between query and titles
#     similarity_scores = [cosine_similarity([query_embedding], [title_embedding])[0][0] for title_embedding in title_embeddings]

#     # Filter titles with similarity above threshold
#     similar_titles = [(titles[i], similarity_scores[i]) for i in range(len(titles)) if similarity_scores[i] > similarity_threshold]

#     # Sort by similarity score
#     similar_titles.sort(key=lambda x: x[1], reverse=True)

#     # Initialize set to store seen titles
#     seen_titles = set()

#     # Get top similar titles, ignoring duplicates
#     top_titles = []
#     for title, score in similar_titles:
#         if title not in seen_titles:
#             top_titles.append(title)
#             seen_titles.add(title)
#             if len(top_titles) >= top_n:
#                 break

#     return top_titles

# # Example usage
# if __name__ == "__main__":
#     query = "your_query_here"
#     top_titles = semantic_search(query, titles, title_embeddings)

#     print("Top Titles:")
#     for title in top_titles:
#         print(title)
