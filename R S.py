import numpy as np
import pandas as pd

# Sample user-item interaction matrix (ratings)
data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Movie1': [5, 4, np.nan, 2, 1],
    'Movie2': [4, np.nan, 5, 3, 2],
    'Movie3': [np.nan, 2, 3, 4, 5],
    'Movie4': [1, 2, 4, np.nan, 5],
    'Movie5': [np.nan, 5, 2, 3, 4],
}
df = pd.DataFrame(data)
df.set_index('User', inplace=True)
df
from sklearn.metrics.pairwise import cosine_similarity

# Fill NaN values with 0 for similarity calculation
df_filled = df.fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(df_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=df.index, columns=df.index)
user_similarity_df
def get_recommendations(target_user, df, user_similarity_df, top_n=2):
    # Get the ratings of the target user
    target_user_ratings = df.loc[target_user]

    # Get the similarity scores for the target user
    similar_users = user_similarity_df[target_user]

    # Calculate the weighted average of ratings from similar users
    weighted_ratings = np.dot(similar_users, df_filled) / np.sum(similar_users)

    # Convert to a series and filter out movies already rated by the target user
    weighted_ratings_series = pd.Series(weighted_ratings, index=df.columns)
    recommendations = weighted_ratings_series[target_user_ratings.isna()]

    # Sort the recommendations and return the top N
    recommendations = recommendations.sort_values(ascending=False)
    return recommendations.head(top_n)

# Get recommendations for 'User1'
recommendations = get_recommendations('User1', df, user_similarity_df)
recommendations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item interaction matrix (ratings)
data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Movie1': [5, 4, np.nan, 2, 1],
    'Movie2': [4, np.nan, 5, 3, 2],
    'Movie3': [np.nan, 2, 3, 4, 5],
    'Movie4': [1, 2, 4, np.nan, 5],
    'Movie5': [np.nan, 5, 2, 3, 4],
}

df = pd.DataFrame(data)
df.set_index('User', inplace=True)

# Fill NaN values with 0 for similarity calculation
df_filled = df.fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(df_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=df.index, columns=df.index)

def get_recommendations(target_user, df, user_similarity_df, top_n=2):
    # Get the ratings of the target user
    target_user_ratings = df.loc[target_user]

    # Get the similarity scores for the target user
    similar_users = user_similarity_df[target_user]

    # Calculate the weighted average of ratings from similar users
    weighted_ratings = np.dot(similar_users, df_filled) / np.sum(similar_users)

    # Convert to a series and filter out movies already rated by the target user
    weighted_ratings_series = pd.Series(weighted_ratings, index=df.columns)
    recommendations = weighted_ratings_series[target_user_ratings.isna()]

    # Sort the recommendations and return the top N
    recommendations = recommendations.sort_values(ascending=False)
    return recommendations.head(top_n)

# Get recommendations for 'User1'
recommendations = get_recommendations('User1', df, user_similarity_df)
print("Recommendations for User1:")
print(recommendations)
