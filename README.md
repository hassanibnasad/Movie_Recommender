# 🎬 Hybrid Movie Recommendation System

A smart movie recommender web app that suggests personalized movies by blending content-based filtering, popularity, and genre similarity using TF-IDF and weighted scoring techniques. Built with Python, Pandas, Scikit-learn, and Streamlit.
- 📄 Content-based filtering (overview and genre similarity)
- ⭐ Weighted average ratings
- 🔥 Popularity metrics

Users can select a movie and adjust similarity weights to receive personalized recommendations.

![image](https://github.com/user-attachments/assets/3116c221-c2fa-41fd-8ccb-585c43221038)


---

## 🚀 Features
-🔎 Hybrid Recommendations combining:
-    📖 Overview-based similarity (TF-IDF + cosine)
-    🎭 Genre-based similarity
-    ⭐ IMDb-style weighted average ratings
-    🔥 Popularity metric

- 🎚 Adjustable sliders to customize weightage
- 🧠 Clean, responsive UI built with Streamlit

---

## 🌐 Live Demo

-Try the app hosted on Streamlit:
[Live Demo](https://movierecommender-fcjhwbacdxr2djthfrhkjf.streamlit.app/)

---

## 📊 How It Works

1. Data Preprocessing
- Reads movie metadata from movies.xls
- Parses genre strings using ast.literal_eval()

2. Similarity Computation
- Overview: TF-IDF + cosine similarity on movie plots
- Genre: TF-IDF + cosine similarity on genres
- Ratings/Popularity: Normalized 

3. Hybrid Score
- The system uses a hybrid scoring formula:
- final_score = α * rating_score + β * popularity_score + γ * overview_similarity + δ * genre_similarity
- Adjustable with sliders on the UI.

---

## 📁 Project Structure

- movie_recommender/
- │
- ├── app.py                 
- ├── movies.xls             
- ├── requirements.txt
- ├── movie_recommendation.ipynb  
- └── README.md

---

## 📦 Installation

- 1.Clone the repository:
-    git clone https://github.com/your-username/movie_recommender.git
-    cd movie_recommender
- 2.Install dependencies:
-    pip install -r requirements.txt
- 3.Run the App
-    streamlit run app.py


