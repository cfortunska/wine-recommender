# Imports
import pandas as pd
import streamlit as st
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
from io import StringIO

# Function to load data from Google Drive
def load_data(file_url):
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        data = pd.read_csv(csv_data)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()  # Return empty DataFrame if loading fails

# Load Data
file_url = "https://drive.google.com/uc?export=download&id=13_2Y30Xz6AXgaemW70-UL6r5DBzlaZDD"
df1 = load_data(file_url)

if df1.empty:
    st.error("Failed to load data. Please check the file URL or file accessibility.")
    st.stop()

# Ensure necessary columns exist
if "title" not in df1.columns or "new_description" not in df1.columns:
    st.error("The dataset must contain 'title' and 'new_description' columns.")
    st.stop()

# Build Recommendation System
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df1['new_description'].fillna(''))  # Fill NaN descriptions with empty strings
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

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
