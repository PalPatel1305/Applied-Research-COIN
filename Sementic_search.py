from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# Sample JSON object
data = [
    {
        "month": "January",
        "title": "President's Message",
        "author": "",
        "page": "1"
    },
    {
        "month": "January",
        "title": "Editor's Page",
        "author": "",
        "page": "2"
    },
    {
        "month": "January",
        "title": "C.N.A. Chapter & Club Reports",
        "author": "",
        "page": "3"
    },
    {
        "month": "January",
        "title": "Bank of Nova Scotia contains information by the Librarian of the Bank of Nova Scotia on the history and note issues of the bank from 1832 - 1935. Includes a list of Bank Presidents and General Managers",
        "author": "by Betty Hearn",
        "page": "6"
    },
    {
        "month": "January",
        "title": "More Chapters, please !",
        "author": "",
        "page": "8"
    },
    {
        "month": "January",
        "title": "New Members, etc.",
        "author": "",
        "page": "9"
    },
    {
        "month": "January",
        "title": "The old Masters can be wrong! an article concerning cataloguing errors made concerning Br 528 and 529 and LeRoux 532 and 533 (Quebec Bank tokens)",
        "author": "by Major S. S. Carroll",
        "page": "10"
    },
    {
        "month": "January",
        "title": "1956 C.N.A. Convention",
        "author": "",
        "page": "11"
    },
    {
        "month": "January",
        "title": "Acknowledgements",
        "author": "",
        "page": "11"
    },
    {
        "month": "January",
        "title": "Metallurgical Aspects of Coinage with Special Reference to Nickel contains information on the metallurgical and physical characteristics of nickel, especially as these relate to minting coins.",
        "author": "by Aubrey A. Tuttle",
        "page": "12"
    }]


# Extract titles from the JSON object
titles = [entry["title"] for entry in data]

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="path_to_your_cache_directory")
model = BertModel.from_pretrained('bert-base-uncased')

# Encode titles using BERT tokenizer and get embeddings
encoded_titles = tokenizer(titles, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    output = model(**encoded_titles)

# Extract embeddings from BERT output
title_embeddings = output.last_hidden_state[:, 0, :].numpy()

# Perform semantic search
def semantic_search(query, titles, title_embeddings, top_n=5):
    # Encode query using BERT tokenizer
    encoded_query = tokenizer(query, padding=True, truncation=True, return_tensors='pt')

    # Get query embedding
    with torch.no_grad():
        output = model(**encoded_query)
    query_embedding = output.last_hidden_state[:, 0, :].numpy()

    # Calculate cosine similarity between query embedding and title embeddings
    similarities = cosine_similarity(query_embedding, title_embeddings)

    # Get indices of top-n most similar titles
    top_indices = np.argsort(similarities[0])[-top_n:][::-1]

    # Return top-n most similar titles
    results = [(titles[i], similarities[0][i]) for i in top_indices]
    return results

# Example usage
query = "lesson"
results = semantic_search(query, titles, title_embeddings)
for result in results:
    print(result)
