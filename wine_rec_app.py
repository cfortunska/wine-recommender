import pandas as pd
import streamlit as st
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
from io import StringIO

# Function to load data from Google Drive
@st.cache_data
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
file_url = "https://drive.google.com/uc?export=download&id=1a9W-WfQTfe1XS5Kw7rDmQjllaOs9PCLy"
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
    matches = process.extract(user_input, choices, limit=5)  # Limit to 5 suggestions
    filtered_matches = [match[0] for match in matches if match[1] >= threshold]  # Only keep matches above threshold
    return filtered_matches

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

# Handle user input dynamically using session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Text input with dynamic updates
user_input = st.text_input("Enter a wine name:", value=st.session_state.user_input)

# Update session state when the user types
if user_input != st.session_state.user_input:
    st.session_state.user_input = user_input

# Only trigger search when user has input
if st.session_state.user_input:
    matched_items = search_item(st.session_state.user_input, df1["title"].tolist())

    if matched_items:
        # Show suggestions dynamically in a selectbox
        selected_item = st.selectbox("Select a wine from the suggestions:", matched_items)

        if selected_item:
            st.write(f"Showing results for: **{selected_item}**")
            recommendations = recommend(selected_item)

            if not recommendations.empty:
                for _, row in recommendations.iterrows():
                    st.markdown(f"**{row['title']}**  \n"
                                f"**Price:** ${row['price']}  \n"
                                f"**Country:** {row['country']}  \n"
                                f"**Points:** {row['points']}  \n"
                                f"**Province:** {row['province']}  \n"
                                f"**Variety:** {row['variety']}")
            else:
                st.warning(f"No similar wines found for **{selected_item}**.")
    else:
        st.warning("No close match found. Try another search.")
