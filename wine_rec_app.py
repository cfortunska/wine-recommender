import pandas as pd
import streamlit as st
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

# Streamlit UI
st.title("🍷 Wine Recommender")

# Text input to type a wine name
user_input = st.text_input("Enter a wine name:")

# Filter wines based on the user input
if user_input:
    filtered_titles = df1[df1["title"].str.contains(user_input, case=False, na=False)]["title"].tolist()

    if filtered_titles:
        # Display matching titles in a selectbox
        selected_item = st.selectbox("Select a wine from the suggestions:", filtered_titles)

        if selected_item:
            st.write(f"Showing results for: **{selected_item}**")
            # You can add recommendation logic for the selected item here
            st.write("Display recommendations for the selected wine here.")
    else:
        st.warning("No close match found. Try another search.")
