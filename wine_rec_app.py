# Imports
import pandas as pd
import streamlit as st
import pickle
import os
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Define file paths for saved models
TFIDF_PATH = '/app/wine_recommender/tfidf_matrix.pkl'
COSINE_SIM_PATH = '/app/wine_recommender/cosine_sim_matrix.pkl'

# Load data
file = 'https://raw.githubusercontent.com/cfortunska/wine-recommender/main/wine_final.csv'
df1 = pd.read_csv(file)

# Function to build similarity matrix
def build_similarity_matrix():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df1['new_description'].fillna(''))
    cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Save the matrices to disk
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open(COSINE_SIM_PATH, 'wb') as f:
        pickle.dump(cosine_sim_matrix, f)
    
    return tfidf_matrix, cosine_sim_matrix

# Check if pre-computed matrices exist
try:
    if os.path.exists(TFIDF_PATH) and os.path.exists(COSINE_SIM_PATH):
        with open(TFIDF_PATH, 'rb') as f:
            tfidf_matrix = pickle.load(f)

        with open(COSINE_SIM_PATH, 'rb') as f:
            cosine_sim_matrix = pickle.load(f)
        st.write("✅ Loaded pre-computed matrices successfully.")
    else:
        st.write("⏳ Computing similarity matrix. This may take a while...")
        tfidf_matrix, cosine_sim_matrix = build_similarity_matrix()
        st.write("✅ Computation complete and models saved successfully.")
except Exception as e:
    st.error(f"Error loading or saving matrices: {e}")
    tfidf_matrix, cosine_sim_matrix = build_similarity_matrix()

# Function to search for best match
def search_item(user_input, choices, threshold=80):
    best_match = process.extractOne(user_input, choices)
    return best_match[0] if best_match and best_match[1] >= threshold else None

# Function to recommend wines
def recommend(title):
    if title not in df1["title"].values:
        return pd.DataFrame()  # Return empty if title not found
    
    idx = df1[df1["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    wine_indices = [i[0] for i in sim_scores]

    return df1.iloc[wine_indices][["title", "price", "country", "points", "province", "variety"]]

# Streamlit UI
st.title("🍷 Wine Recommender")

user_input = st.text_input("Enter a wine name:")

if user_input:
    matched_item = search_item(user_input, df1["title"].tolist())

    if matched_item:
        st.write(f"Showing results for: **{matched_item}**")
        recommendations = recommend(matched_item)

        if not recommendations.empty:
            for _, row in recommendations.iterrows():
                st.markdown(f"**{row['title']}**  \n"
                            f"**Price:** ${row['price']}  \n"
                            f"**Country:** {row['country']}  \n"
                            f"**Points:** {row['points']}  \n"
                            f"**Province:** {row['province']}  \n"
                            f"**Variety:** {row['variety']}")
        else:
            st.warning("No similar wines found.")
    else:
        st.warning("No close match found. Try another search.")
