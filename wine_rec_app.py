import pandas as pd
import streamlit as st
from fuzzywuzzy import process
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
if "title" not in df1.columns:
    st.error("The dataset must contain a 'title' column.")
    st.stop()

# Function to search for best match using fuzzywuzzy
def search_item(user_input, choices, threshold=80):
    matches = process.extract(user_input, choices, limit=5)  # Limit to 5 suggestions
    filtered_matches = [match[0] for match in matches if match[1] >= threshold]  # Only keep matches above threshold
    return filtered_matches

# Streamlit UI
st.title("🍷 Wine Recommender")

# Text input with dynamic updates
user_input = st.text_input("Enter a wine name:")

if user_input:
    matched_items = search_item(user_input, df1["title"].tolist())

    if matched_items:
        # Show suggestions dynamically in a selectbox
        selected_item = st.selectbox("Select a wine from the suggestions:", matched_items)

        if selected_item:
            st.write(f"Showing results for: **{selected_item}**")
            # Here, you can add your wine recommendation functionality for the selected item
            st.write("Display recommendations for the selected wine here.")
    else:
        st.warning("No close match found. Try another search.")
