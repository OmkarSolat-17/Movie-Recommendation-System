import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_PATH = 'C:\\Users\\omkar\\OneDrive\\Desktop\\data\\ml-100k/'  # Update to your path
NUM_FACTORS = 50  # Latent factors for SVD
EMBEDDING_DIM = 32  # Dimension for neural embeddings
TOP_K = 10  # Number of recommendations

# Step 1: Data Loading and Preprocessing

def load_data():
    # Load ratings: user_id, movie_id, rating, timestamp
    ratings = pd.read_csv(DATA_PATH + 'u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    # Load movies: movie_id, title, release_date, video_release_date, IMDb_URL, unknown, Action, Adventure, ..., Horror
    movies = pd.read_csv(DATA_PATH + 'u.item', sep='|', encoding='latin-1', usecols=range(24),
                         names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + 
                               ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                                'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                                'Thriller', 'War', 'Western'])
    # Load users: user_id, age, gender, occupation, zip_code
    users = pd.read_csv(DATA_PATH + 'u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    
    # Create genre string for movies (for content-based filtering)
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                  'Thriller', 'War', 'Western']
    movies['genres'] = movies[genre_cols].apply(lambda x: ' '.join([col for col in genre_cols if x[col] == 1]), axis=1)
    
    # Merge ratings with movies and users
    data = ratings.merge(movies, on='movie_id').merge(users, on='user_id')
    
    # Preprocess: Drop unnecessary columns, handle missing values
    data = data[['user_id', 'movie_id', 'rating', 'title', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                 'Sci-Fi', 'Thriller', 'War', 'Western', 'age', 'gender', 'occupation']]
    data.fillna(0, inplace=True)  # Simple imputation
    
    return data, movies, users


# Step 2: Collaborative Filtering with SVD

def collaborative_filtering_svd(ratings_matrix, train_ratings, test_ratings):
    # Fit SVD on training data
    svd = TruncatedSVD(n_components=NUM_FACTORS, random_state=42)
    svd.fit(train_ratings.fillna(0))  # Fill NaNs with 0 for matrix factorization
    
    # Predict ratings for test set (fill NaNs with 0 for transformation)
    predicted_ratings = svd.inverse_transform(svd.transform(test_ratings.fillna(0)))
    
    # Compute RMSE only on non-NaN test ratings (mask out unrated items)
    mask = ~np.isnan(test_ratings.values)  # True for rated items
    if mask.sum() > 0:  # Ensure there are rated items to evaluate
        rmse = np.sqrt(mean_squared_error(test_ratings.values[mask], predicted_ratings[mask]))
        print(f"Collaborative Filtering SVD RMSE: {rmse:.4f}")
    else:
        print("No rated items in test set for RMSE calculation.")
        rmse = float('nan')  # Or handle as needed
    
    return svd, predicted_ratings


# Step 3: Content-Based Filtering with TF-IDF and Cosine Similarity
def content_based_filtering(movies):
    # Vectorize genres using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

# Step 4: Neural Embeddings with Autoencoder
def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def train_autoencoder(ratings_matrix):
    # Normalize ratings to [0,1]
    ratings_norm = ratings_matrix / 5.0
    
    autoencoder, encoder = build_autoencoder(ratings_matrix.shape[1], EMBEDDING_DIM)
    autoencoder.fit(ratings_norm.fillna(0), ratings_norm.fillna(0), epochs=50, batch_size=64, shuffle=True, verbose=0)
    
    # Get embeddings
    embeddings = encoder.predict(ratings_norm.fillna(0))
    return embeddings

# Step 5: Hybrid Recommendation

def get_hybrid_recommendations(user_id, svd_model, cosine_sim, embeddings, data, movies, top_k=TOP_K):
    # Get all movie IDs for reindexing
    all_movies = movies['movie_id'].tolist()
    
    # Get user's ratings and reindex to all movies (fill unrated with 0)
    user_ratings = data[data['user_id'] == user_id].set_index('movie_id')['rating']
    user_ratings_full = user_ratings.reindex(all_movies, fill_value=0)
    
    # Collaborative: Get predicted ratings for user
    user_pred = svd_model.inverse_transform(svd_model.transform(user_ratings_full.to_frame().T))[0]
    
    # Content-Based: Find similar movies to user's top-rated
    top_movie = user_ratings.idxmax()
    sim_scores = list(enumerate(cosine_sim[top_movie - 1]))  # Movie IDs start from 1
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
    content_recs = [movies.iloc[i[0]]['movie_id'] for i in sim_scores]
    
    # Embeddings: Find nearest neighbors in embedding space
    user_embedding = embeddings[user_id - 1]  # User IDs start from 1
    nbrs = NearestNeighbors(n_neighbors=top_k, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors([user_embedding])
    embedding_recs = [idx + 1 for idx in indices[0]]  # Convert back to user IDs
    
    # Hybrid: Combine scores (simple weighted average)
    all_movies_set = set(all_movies)
    watched = set(user_ratings.index)
    candidates = list(all_movies_set - watched)
    
    scores = {}
    for movie in candidates:
        cf_score = user_pred[movie - 1] if movie - 1 < len(user_pred) else 0
        cb_score = 1 if movie in content_recs else 0
        emb_score = 1 if movie in embedding_recs else 0
        scores[movie] = 0.5 * cf_score + 0.3 * cb_score + 0.2 * emb_score
    
    top_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [movies[movies['movie_id'] == mid]['title'].values[0] for mid, _ in top_recs]


# Step 6: Evaluation

def evaluate_model(test_ratings, predicted_ratings):
    # Compute RMSE only on non-NaN test ratings (mask out unrated items)
    mask = ~np.isnan(test_ratings.values)  # True for rated items
    if mask.sum() > 0:  # Ensure there are rated items to evaluate
        rmse = np.sqrt(mean_squared_error(test_ratings.values[mask], predicted_ratings[mask]))
        print(f"Overall Hybrid RMSE: {rmse:.4f}")
    else:
        print("No rated items in test set for RMSE calculation.")
        rmse = float('nan')  # Or handle as needed



# Main Function

# Main Function (Updated)
def main():
    print("Loading and preprocessing data...")
    data, movies, users = load_data()
    
    # Get all unique movies to ensure consistent columns
    all_movies = sorted(movies['movie_id'].unique())
    
    # Create full user-item matrix and reindex to include all movies
    ratings_matrix = data.pivot_table(index='user_id', columns='movie_id', values='rating').reindex(columns=all_movies)
    
    # Split data into train/test BEFORE pivoting to avoid dimension mismatch
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create train and test matrices, reindexing to all_movies for consistency
    train_matrix = train_data.pivot_table(index='user_id', columns='movie_id', values='rating').reindex(columns=all_movies)
    test_matrix = test_data.pivot_table(index='user_id', columns='movie_id', values='rating').reindex(columns=all_movies)
    
    print("Training Collaborative Filtering (SVD)...")
    svd_model, pred_ratings = collaborative_filtering_svd(ratings_matrix, train_matrix, test_matrix)
    
    print("Computing Content-Based Similarities...")
    cosine_sim = content_based_filtering(movies)
    
    print("Training Neural Embeddings (Autoencoder)...")
    embeddings = train_autoencoder(ratings_matrix)
    
    print("Evaluating Model...")
    evaluate_model(test_matrix, pred_ratings)
    
    # Example Inference
    user_id = 2  # Example user
    print(f"\nTop {TOP_K} Recommendations for User {user_id}:")
    recs = get_hybrid_recommendations(user_id, svd_model, cosine_sim, embeddings, data, movies)
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec}")
    
    # Optional: Plot embedding visualization (2D PCA for simplicity)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.title("User Embeddings (2D PCA)")
    plt.show()

if __name__ == "__main__":
    main()