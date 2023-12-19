import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
import pickle

with open('music_df.pkl', 'rb') as f:
    music_df = pickle.load(f)

# Function to calculate weighted popularity scores based on release date
def calculate_weight_probability(release_date):
    release_date = datetime.strptime(release_date, '%Y-%m-%d')
    timespan = datetime.now() - release_date
    weight = 1 / (timespan.days + 1)
    return weight

def clean_text(text):
    text = text.strip()
    text = text.replace('.', '').replace('"', '')
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to get content-based recommendations
def content_based_recommendation(input_song, num_rec=5):
    if input_song not in music_df['Track Name'].values:
        st.warning(f"{input_song} not in the dataset. Please enter a valid song name")
        exit()
    
    inp_song_index = music_df[music_df['Track Name'] == input_song].index[0]
    similarity_scores = cosine_similarity([music_df.iloc[inp_song_index][['Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
                                                                          'Speechiness', 'Acousticness', 'Instrumentalness',
                                                                          'Liveness', 'Valence', 'Tempo']].values],
                                         music_df[['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
                                                    'Acousticness', 'Instrumentalness', 'Liveness', 'Valence',
                                                    'Tempo']].values)
    similar_song_index = similarity_scores.argsort()[0][::-1][1:num_rec + 1]
    content_based_recommendations = music_df.iloc[similar_song_index][['Track Name', 'Artists', 'Album Name',
                                                                       'Release Date', 'Popularity']]
    return content_based_recommendations

# Function to get hybrid recommendations
def hybrid_recommendations(input_song_name, num_recommendations=5, alpha=0.5):
    if input_song_name not in music_df['Track Name'].values:
        st.warning(f"'{input_song_name}' not found in the dataset. Please enter a valid song name.")
        exit()

    scaler = MinMaxScaler()
    music_feature = music_df[['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']].values
    scaler.fit_transform(music_feature)

    content_based_rec = content_based_recommendation(input_song_name, num_recommendations)

    popularity_score = music_df.loc[music_df['Track Name'] == input_song_name, 'Popularity'].values[0]

    weighted_popularity_score = popularity_score * calculate_weight_probability(
        music_df.loc[music_df['Track Name'] == input_song_name, 'Release Date'].values[0])

    hybrid_recommendations = content_based_rec
    input_song_df = pd.DataFrame({
        'Track Name': [input_song_name],
        'Artists': [music_df.loc[music_df['Track Name'] == input_song_name, 'Artists'].values[0]],
        'Album Name': [music_df.loc[music_df['Track Name'] == input_song_name, 'Album Name'].values[0]],
        'Release Date': [music_df.loc[music_df['Track Name'] == input_song_name, 'Release Date'].values[0]],
        'Popularity': [weighted_popularity_score]
    })

    hybrid_recommendations = pd.concat([content_based_rec, input_song_df], ignore_index=True)

    hybrid_recommendations = hybrid_recommendations.sort_values(by='Popularity', ascending=False)

    hybrid_recommendations = hybrid_recommendations[hybrid_recommendations['Track Name'].str.lower() != input_song_name]

    return hybrid_recommendations

# Streamlit app
def main():
    st.title("Music Recommendation App")
    # User input for the song name
    input_song = st.text_input("Enter a song name:", "Flowers")
    input_song = clean_text(input_song.lower())

    # Display hybrid recommendations
    st.header("Hybrid Recommendations")
    hybrid_rec = hybrid_recommendations(input_song)
    if input_song:
        st.table(hybrid_rec.set_index([[i for i in range(1,6)]], drop=True))
    
if __name__ == '__main__':
    main()
