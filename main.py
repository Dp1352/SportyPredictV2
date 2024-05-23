import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from nba_api.stats.endpoints import playergamelog, commonallplayers
from nba_api.stats.static import players
import pandas as pd
import requests_cache

# Enable caching
requests_cache.install_cache('nba_api_cache', expire_after=3600)  # Cache expires after 1 hour

# Function to get player ID from name
def get_player_id(player_name):
    player_dict = players.get_players()
    for player in player_dict:
        if player['full_name'].lower() == player_name.lower():
            return player['id']
    return None

# Function to fetch game logs
def fetch_player_game_logs(player_id, season='2023-24'):
    # Fetch regular season game logs
    regular_season_game_logs = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    regular_season_game_logs_df = regular_season_game_logs.get_data_frames()[0]
    
    # Fetch postseason game logs
    postseason_game_logs = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Playoffs')
    postseason_game_logs_df = postseason_game_logs.get_data_frames()[0]
    
    # Concatenate regular season and postseason game logs
    game_logs_df = pd.concat([regular_season_game_logs_df, postseason_game_logs_df], ignore_index=True)
    
    # Sort game logs by date in descending order
    game_logs_df['GAME_DATE'] = pd.to_datetime(game_logs_df['GAME_DATE'])
    game_logs_df = game_logs_df.sort_values(by='GAME_DATE', ascending=False).reset_index(drop=True)

    game_logs_df = game_logs_df.iloc[::-1].reset_index(drop=True)
    
    return game_logs_df


# Function to get players who played in the current season
def get_current_season_players(season='2023-24'):
    all_players = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
    current_season_player_ids = all_players['PERSON_ID'].tolist()
    current_season_players = [player for player in players.get_players() if player['id'] in current_season_player_ids]
    return current_season_players

# Class that will calculate likelihood of player achieving prediction
class Calculation:
    def __init__(self, sample_data):
        self.sample_data = sample_data
        self.model_points = None
        self.model_assists = None
        self.model_rebounds = None
        self.model_steals = None
        self.model_blocks = None

    def _extract_stats(self):
        points = []
        assists = []
        rebounds = []
        steals = []
        blocks = []

        for index, game in self.sample_data.iterrows():
            points.append(game['PTS'])
            assists.append(game['AST'])
            rebounds.append(game['REB'])
            steals.append(game['STL'])
            blocks.append(game['BLK'])

        self.points = np.array(points)
        self.assists = np.array(assists)
        self.rebounds = np.array(rebounds)
        self.steals = np.array(steals)
        self.blocks = np.array(blocks)

    def _prepare_feature_sets(self):
        self.X_train_points = np.column_stack((self.assists, self.rebounds, self.steals, self.blocks))
        self.X_train_assists = np.column_stack((self.points, self.rebounds, self.steals, self.blocks))
        self.X_train_rebounds = np.column_stack((self.points, self.assists, self.steals, self.blocks))
        self.X_train_steals = np.column_stack((self.points, self.assists, self.rebounds, self.blocks))
        self.X_train_blocks = np.column_stack((self.points, self.assists, self.rebounds, self.steals))

    def _train_models(self):
        self.model_points = self.train_model(self.X_train_points, self.points)
        self.model_assists = self.train_model(self.X_train_assists, self.assists)
        self.model_rebounds = self.train_model(self.X_train_rebounds, self.rebounds)
        self.model_steals = self.train_model(self.X_train_steals, self.steals)
        self.model_blocks = self.train_model(self.X_train_blocks, self.blocks)

    def train_model(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def _predict_points(self, assists, rebounds, steals, blocks):
        user_input = np.array([[assists, rebounds, steals, blocks]])
        prediction = self.model_points.predict(user_input)[0]
        return prediction

    def _predict_assists(self, points, rebounds, steals, blocks):
        user_input = np.array([[points, rebounds, steals, blocks]])
        prediction = self.model_assists.predict(user_input)[0]
        return prediction

    def _predict_rebounds(self, points, assists, steals, blocks):
        user_input = np.array([[points, assists, steals, blocks]])
        prediction = self.model_rebounds.predict(user_input)[0]
        return prediction

    def _predict_steals(self, points, assists, rebounds, blocks):
        user_input = np.array([[points, assists, rebounds, blocks]])
        prediction = self.model_steals.predict(user_input)[0]
        return prediction

    def _predict_blocks(self, points, assists, rebounds, steals):
        user_input = np.array([[points, assists, rebounds, steals]])
        prediction = self.model_blocks.predict(user_input)[0]
        return prediction

    def _evaluate_likelihood(self, predicted, actual):
        if predicted >= actual:
            return "Very likely"
        elif 0.9 * actual <= predicted < actual:
            return "Probable"
        else:
            return "Not very likely"

    def predict_stat(self, stat, value):
        if stat == "Points":
            predicted_value = self._predict_points(self.assists[-1], self.rebounds[-1], self.steals[-1], self.blocks[-1])
        elif stat == "Assists":
            predicted_value = self._predict_assists(self.points[-1], self.rebounds[-1], self.steals[-1], self.blocks[-1])
        elif stat == "Rebounds":
            predicted_value = self._predict_rebounds(self.points[-1], self.assists[-1], self.steals[-1], self.blocks[-1])
        elif stat == "Steals":
            predicted_value = self._predict_steals(self.points[-1], self.assists[-1], self.rebounds[-1], self.blocks[-1])
        elif stat == "Blocks":
            predicted_value = self._predict_blocks(self.points[-1], self.assists[-1], self.rebounds[-1], self.steals[-1])
        else:
            raise ValueError("Invalid stat choice.")

        likelihood = self._evaluate_likelihood(predicted_value, value)

        return predicted_value, likelihood

# Streamlit UI
st.title("SportyPredict")

# Get current season players
season = '2023-24'
current_season_players = get_current_season_players(season)
player_names = [player['full_name'] for player in current_season_players]

# Select player
player_name = st.selectbox("Select a Player", options=player_names)

# Select stat and value
stat = st.selectbox("Select a Statistic", ["Points", "Assists", "Rebounds", "Steals", "Blocks"])
value = st.number_input("Enter the Value")

# Get player ID
player_id = get_player_id(player_name)

if player_id is None:
    st.error("Player not found.")
else:
    # Fetch game logs
    game_logs_df = fetch_player_game_logs(player_id)

    if game_logs_df.empty:
        st.error("No data found for the player.")
    else:
        total_games = len(game_logs_df)

        # Add a slider for selecting the number of previous games
        num_previous_games = st.slider("Select the number of previous games to base analysis on", min_value=1, max_value=total_games, value=10)

        if st.button("Calculate"):
            games = game_logs_df.tail(num_previous_games)

            if games.empty:
                st.error("No data found for the selected number of games.")
            else:
                predictor = Calculation(games)
                predictor._extract_stats()
                predictor._prepare_feature_sets()
                predictor._train_models()

                predicted_value, likelihood = predictor.predict_stat(stat, value)

                st.write(f"Prediction: {predicted_value:.2f}")
                st.write(f"Likelihood: {likelihood}")

                # Display table for selected number of previous games
                st.subheader(f"Performance Table for Last {num_previous_games} Games")
                descending_games = games.iloc[::-1]  # Reverse the order of rows
                st.table(descending_games[['PTS', 'AST', 'REB', 'STL', 'BLK']])