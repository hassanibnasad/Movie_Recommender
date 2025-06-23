# movie_recommender/app.py

import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

# Load movie data
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.xls")

    def extract_genres(genre_str):
        try:
            genre_list = ast.literal_eval(genre_str)
            return ', '.join([g['name'] for g in genre_list if isinstance(g, dict) and 'name' in g])
        except Exception as e:
            return ""
    
    movies['genres'] = movies['genres'].apply(extract_genres)

    return movies.dropna(subset=['original_title','overview','genres','weighted_average_ratings','popularity'])

movies = load_data()

# TF-IDF similarity based on overview
@st.cache_resource
def compute_similarity():
    tfidf = TfidfVectorizer(stop_words='english', ngram_range= (1,3))
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    return cosine_similarity(tfidf_matrix)

cosine_sim = compute_similarity()

# TF-IDF similarity based on geners
@st.cache_resource
def compute_genre_similarity():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    return cosine_similarity(tfidf_matrix)

cosine_sim_genre = compute_genre_similarity()

#Hybrid recommender
def hybrid_recommend(title, top_n=10, alpha=0.2, beta=0.05, gamma=0.4, delta=0.35):
    try:
        idx = movies[movies['original_title'] == title].index[0]
    except:
        return pd.DataFrame()

    overview_scores = list(enumerate(cosine_sim[idx]))
    genre_scores = list(enumerate(cosine_sim_genre[idx]))

    combined_scores = []
    for i, (ov_score, gn_score) in enumerate(zip(overview_scores, genre_scores)):
        if i == idx:
            continue  # skip the selected movie itself

        weighted_score = movies.iloc[i]['weighted_average_ratings']
        popularity_score = movies.iloc[i]['popularity']

        # Normalize rating and popularity
        norm_rating = (weighted_score - movies['weighted_average_ratings'].min()) / (movies['weighted_average_ratings'].max() - movies['weighted_average_ratings'].min())
        norm_popularity = (popularity_score - movies['popularity'].min()) / (movies['popularity'].max() - movies['popularity'].min())

        final = alpha * norm_rating + beta * norm_popularity + gamma * ov_score[1] + delta * gn_score[1]
        combined_scores.append((i, final))

    top_indices = [i[0] for i in sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_n]]
    return movies[['original_title', 'genres', 'weighted_average_ratings', 'popularity']].iloc[top_indices]


# UI
movie_list = sorted(movies['original_title'].unique())
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Sliders for weight adjustment
st.markdown("### 🎚️ Adjust Recommender Weights")
st.markdown("Adjust weights for getting movie recommendations based on hybrid or sole parameter.")

alpha = st.slider("⭐ Rating Weight (α)", 0.0, 1.0, 0.2, step=0.05)
beta = st.slider("🔥 Popularity Weight (β)", 0.0, 1.0, 0.05, step=0.05)
gamma = st.slider("🧠 Overview Similarity Weight (γ)", 0.0, 1.0, 0.5, step=0.05)
delta = st.slider("🎭 Genre Similarity Weight (δ)", 0.0, 1.0, 0.25, step=0.05)

# Normalize weights
total = alpha + beta + gamma + delta
if total == 0:
    st.error("At least one weight must be greater than 0.")
    st.stop()

alpha, beta, gamma, delta = alpha / total, beta / total, gamma / total, delta / total

st.markdown(f"🔢 Normalized Weights → Rating: `{alpha:.2f}`, Popularity: `{beta:.2f}`, Overview: `{gamma:.2f}`, Genre: `{delta:.2f}`")

# Recommend button
if st.button("Recommend"):
    results = hybrid_recommend(selected_movie, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
    if results.empty:
        st.warning("No recommendations found for that genre.")
    else:
        st.subheader("Top Recommendations:")
        for _, row in results.iterrows():
            st.markdown(f"**🎥 {row['original_title']}**")
            st.markdown(f"• 🎭 Genre: `{row['genres']}`")
            st.markdown(f"• ⭐ Rating: `{row['weighted_average_ratings']:.2f}` | 🔥 Popularity: `{row['popularity']:.2f}`")
            st.markdown("---")
