import streamlit as st
import pickle
import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
import time
from pathlib import Path
import base64
import os
import tensorflow as tf

# Cache the model loading function to avoid reloading on every run
@st.cache_resource
def load_model():
    model_url = "https://moviesrecommendationsystem.s3.eu-north-1.amazonaws.com/similarity.pkl"
    model_path = "similarity.pkl"

    # Check if the model file exists locally
    if not os.path.exists(model_path):
        # Download the model from the provided URL
        response = requests.get(model_url)
        if response.status_code == 200:
            # Save the model to the local path
            with open(model_path, 'wb') as f:
                f.write(response.content)
        else:
            st.error("Failed to download model. Please check the URL.")
            return None

    # Load the model using TensorFlow's Keras model loader
    return pickle.load(open(model_path, 'rb'))

# Load the similarity model
similarity = load_model()

# Check if the model was loaded successfully
if similarity is None:
    st.error("Model could not be loaded. Exiting.")
    st.stop()

# Define paths for the image and CSS files
image_path = Path("images/img.png")
css_template_path = Path("styles/styles_template.css")

# Serve the local image using Streamlit
def serve_local_image(image_path):
    if not image_path.is_file():
        st.error(f"Image not found: {image_path}")
        return ""
    
    # Read the image file and encode it as base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    return f"data:image/png;base64,{encoded_image}"

# Get the image URL
background_image_url = serve_local_image(image_path)

# Read the CSS template file
if css_template_path.is_file():
    with open(css_template_path, "r") as css_file:
        css_content = css_file.read()
        # Replace the placeholder with the actual background image URL
        css_content = css_content.replace("__BACKGROUND_IMAGE_URL__", background_image_url)
else:
    st.error(f"CSS template file not found: {css_template_path}")

# Inject the CSS code into the Streamlit app
st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Movie recommendation function using the similarity matrix
def recommendation(text):
    idx = movies[movies['title'] == text].index[0]
    cs = similarity[idx]  # Using the loaded similarity matrix
    neighbors = sorted(list(enumerate(cs)), reverse=True, key=lambda x: x[1])[1:11]
    neighbors_movies = []
    neighbors_movies_posters = []
    neighbors_movies_ids = []
    for i in neighbors:
        neighbors_movies.append(movies.iloc[i[0]].title)
        neighbors_movies_posters.append(movie_poster(movies.iloc[i[0]].movie_id))
        neighbors_movies_ids.append(movies.iloc[i[0]].movie_id)
    return neighbors_movies, neighbors_movies_posters, neighbors_movies_ids

# Fetch movie poster from TMDb API
def movie_poster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=f19a59316ffc4477ffcb4affa952f1e9'
    attempts = 3
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  
            json_data = response.json()
            poster_path = json_data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
        except ConnectionError:
            st.error("Network problem. Retrying...")
            time.sleep(2)  
        except Timeout:
            st.error("Request timed out. Retrying...")
            time.sleep(2)
        except RequestException as e:
            st.error(f"An error occurred: {e}")
            return "https://via.placeholder.com/500"  
    return "https://via.placeholder.com/500"  

# Fetch movie details from TMDb API
def movie_details(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=f19a59316ffc4477ffcb4affa952f1e9'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Fetch movie trailer from TMDb API
def movie_trailer(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=f19a59316ffc4477ffcb4affa952f1e9'
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    for video in data['results']:
        if video['type'] == 'Trailer' and video['site'] == 'YouTube':
            return f"https://www.youtube.com/embed/{video['key']}"
    return None

# Load movie data
movies_data = pickle.load(open('movies_data.pkl', 'rb'))
movies = pd.DataFrame(movies_data)

# Streamlit App Title
st.title('Movies Recommendation System')

# Check if a movie ID is passed as a query parameter
query_params = st.query_params
if 'movie_id' in query_params:
    movie_id = query_params['movie_id']
    movie = movie_details(movie_id)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(f"https://image.tmdb.org/t/p/w500/{movie['poster_path']}")
        
    with col2:
        st.header(movie['title'])
        st.write(movie['overview'])
        st.write(f"Release Date: {movie['release_date']}")
        st.write(f"Rating: {movie['vote_average']}")
    
        # Fetch and display the movie trailer
        trailer_url = movie_trailer(movie_id)
    if trailer_url:
        st.markdown(
            '<div style="text-align: center;"><h5>Movie Trailer</h5></div>',
            unsafe_allow_html=True)
        st.markdown(f'<iframe width="720" height="400" src="{trailer_url}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', unsafe_allow_html=True)
    else:
        st.write("Trailer not available.")
else:
    # Display a movie selection box
    selected_movie = st.selectbox("Select a movie", movies['title'].values)
    if st.button('Select'):
        # Fetch recommendations
        names, posters, ids = recommendation(selected_movie)
        
        # First row with top 5 movies
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.text(names[i])
                movie_url = f"?movie_id={ids[i]}"
                st.markdown(f"[![Movie Poster]({posters[i]})]({movie_url})", unsafe_allow_html=True)
        
        # Second row with next 5 movies
        cols = st.columns(5)
        for i in range(5, 10):
            with cols[i - 5]:
                st.text(names[i])
                movie_url = f"?movie_id={ids[i]}"
                st.markdown(f"[![Movie Poster]({posters[i]})]({movie_url})", unsafe_allow_html=True)
