import os

import kagglehub
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

products_path = kagglehub.dataset_download(
    "sofikulislam/amazon-eco-friendly-products-dataset"
)

users_path = kagglehub.dataset_download(
    "naveennas/sustainable-lifestyle-rating-dataset"
)

products_df = pd.read_csv(
    os.path.join(products_path, "amazon_eco-friendly_products.csv")
).dropna(subset=["title", "description"])

users_df = pd.read_csv(
    os.path.join(users_path, "lifestyle_sustainability_data.csv")
).dropna()

user_features = users_df.copy()

encoders = {}

for col in user_features.columns:
    if not pd.api.types.is_numeric_dtype(user_features[col]):
        le = LabelEncoder()
        user_features[col] = le.fit_transform(user_features[col].astype(str))
        encoders[col] = le


scaler = StandardScaler()
scaled_features = scaler.fit_transform(user_features)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
users_df["cluster"] = kmeans.fit_predict(scaled_features)

products_df["text"] = (
    products_df["title"].astype(str) + " " + products_df["description"].astype(str)
)


vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
X = vectorizer.fit_transform(products_df["text"])

kmeans_products = KMeans(n_clusters=10, random_state=42, n_init=10)

products_df["cluster"] = kmeans_products.fit_predict(X)
cluster_product_map = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}


def match_user_to_cluster(user_input):
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if col in encoders:
            value = str(input_df[col].iloc[0])

            if value not in encoders[col].classes_:
                value = encoders[col].classes_[0]

            input_df[col] = encoders[col].transform([value])

    input_scaled = scaler.transform(input_df)
    cluster = kmeans.predict(input_scaled)[0]

    return cluster


def recommend_products(user_input, top_n=5):
    user_cluster = match_user_to_cluster(user_input)
    mapped_clusters = cluster_product_map[user_cluster]
    recommendations = products_df[products_df["cluster"].isin(mapped_clusters)]
    recommendations = recommendations.sample(
        min(top_n, len(recommendations)), random_state=None
    )

    return recommendations
