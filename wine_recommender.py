{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fd1d6f45-e3c3-42e5-91cc-253059a29125",
   "metadata": {},
   "outputs": [],
   "source": [
    "#imports, uploads, inspection of data\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "pd.set_option(\"display.max_rows\", None, \"display.max_columns\", None)\n",
    "np.random.seed(5)\n",
    "file=r'/Users/cecylia/Desktop/Notes/winedata.csv'\n",
    "df=pd.read_csv(file)\n",
    "#this is a large dataset-lets take a smaller sample and drop anything with a null country, price, province\n",
    "df=df.dropna(subset=['country', 'price', 'province', 'variety'])\n",
    "#drop duplicates\n",
    "df.drop_duplicates(subset=['title'], inplace=True)\n",
    "#this is still a rather large dataset so lets take a  random sample of 40000\n",
    "df1 = df.sample(n=40000)\n",
    "df1.reset_index(inplace=True)\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import linear_kernel\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "#lets start building a simple recommender\n",
    "tfidf = TfidfVectorizer(stop_words='english')\n",
    "tfidf_matrix = tfidf.fit_transform(df1['description'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "52bdce1d-e02b-4b53-8f74-f858ac4489a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "cosine_sim_matrix = linear_kernel(tfidf_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3c2fff5b-9cdb-4393-9a2d-7cbf98d4009e",
   "metadata": {},
   "outputs": [],
   "source": [
    "indices = pd.Series(df1.index, index=df1['title']).drop_duplicates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a1b9471b-db4d-4019-852a-81a13dbb2e62",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from fuzzywuzzy import process\n",
    "\n",
    "# Search function using fuzzy matching\n",
    "def search_item(user_input, choices, threshold=80):\n",
    "    best_match = process.extractOne(user_input, choices)\n",
    "    return best_match[0] if best_match and best_match[1] >= threshold else None\n",
    "\n",
    "# Recommendation function\n",
    "def recommend(title):\n",
    "    if title not in df1[\"title\"].values:\n",
    "        return pd.DataFrame()  # Return empty if title not found\n",
    "    \n",
    "    idx = df1[df1[\"title\"] == title].index[0]\n",
    "    sim_scores = list(enumerate(cosine_sim_matrix[idx]))\n",
    "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]\n",
    "    wine_indices = [i[0] for i in sim_scores]\n",
    "    \n",
    "    return df1.iloc[wine_indices][[\"title\", \"price\", \"country\", \"points\", \"province\", \"variety\"]]\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"🍷 Wine Recommender\")\n",
    "\n",
    "user_input = st.text_input(\"Enter a wine name:\")\n",
    "\n",
    "if user_input:\n",
    "    matched_item = search_item(user_input, df1[\"title\"].tolist())\n",
    "    \n",
    "    if matched_item:\n",
    "        st.write(f\"Showing results for: **{matched_item}**\")\n",
    "\n",
    "        recommendations = recommend(matched_item)\n",
    "        \n",
    "        if not recommendations.empty:\n",
    "            # Format tooltip details\n",
    "            recommendations[\"details\"] = recommendations.apply(\n",
    "                lambda row: f\"**Price:** ${row['price']}  \\n\"\n",
    "                            f\"**Country:** {row['country']}  \\n\"\n",
    "                            f\"**Points:** {row['points']}  \\n\"\n",
    "                            f\"**Province:** {row['province']}  \\n\"\n",
    "                            f\"**Variety:** {row['variety']}\", axis=1\n",
    "            )\n",
    "            \n",
    "            # Show recommendations\n",
    "            for _, row in recommendations.iterrows():\n",
    "                st.markdown(f\"**{row['title']}**  \\n{row['details']}\")\n",
    "        else:\n",
    "            st.warning(\"No similar wines found.\")\n",
    "    else:\n",
    "        st.warning(\"No close match found. Try another search.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c4fac447-31f6-4881-a1d5-6720b4ccd57f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
