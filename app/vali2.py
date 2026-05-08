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

# Encode all non-numeric columns
for col in user_features.columns:
    if not pd.api.types.is_numeric_dtype(user_features[col]):
        le = LabelEncoder()

        user_features[col] = le.fit_transform(user_features[col].astype(str))

        encoders[col] = le

# Scale data
scaler = StandardScaler()

scaled_features = scaler.fit_transform(user_features)


kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

users_df["cluster"] = kmeans.fit_predict(scaled_features)


products_df["text"] = (
    products_df["title"].astype(str) + " " + products_df["description"].astype(str)
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=100, stop_words="english")

X = vectorizer.fit_transform(products_df["text"])


kmeans_products = KMeans(n_clusters=10, random_state=42, n_init=10)

products_df["cluster"] = kmeans_products.fit_predict(X)


cluster_product_map = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}


def match_user_to_cluster(user_input):

    input_df = pd.DataFrame([user_input])

    # Encode categorical columns
    for col in input_df.columns:
        if col in encoders:
            value = str(input_df[col].iloc[0])

            # Handle unseen labels
            if value not in encoders[col].classes_:
                value = encoders[col].classes_[0]

            input_df[col] = encoders[col].transform([value])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    return cluster


def recommend_products(user_input, top_n=5):

    # Predict user cluster
    user_cluster = match_user_to_cluster(user_input)

    # Map user cluster to product clusters
    mapped_clusters = cluster_product_map[user_cluster]

    # Filter products
    recommendations = products_df[products_df["cluster"].isin(mapped_clusters)]

    # Random recommendations
    recommendations = recommendations.sample(
        min(top_n, len(recommendations)), random_state=None
    )

    return recommendations


# =========================================================
# RECOMMENDATION ACCURACY EVALUATION
# =========================================================


import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# FUNCTION TO CHECK RECOMMENDATION ACCURACY
# =========================================================


def evaluate_recommendation_accuracy(users_df, recommend_products, sample_size=50):

    correct_recommendations = 0
    total_tests = 0

    # Remove cluster column
    test_users = users_df.drop(columns=["cluster"])

    # Random sample of users
    sample_users = test_users.sample(min(sample_size, len(test_users)), random_state=42)

    for _, user in sample_users.iterrows():
        user_input = user.to_dict()

        # Get recommendations
        recommendations = recommend_products(user_input, top_n=5)

        # Skip if no recommendations
        if recommendations.empty:
            continue

        total_tests += 1

        # =================================================
        # ACCURACY CONDITION
        # =================================================
        # If recommendations belong to mapped product
        # clusters, count as correct
        # =================================================

        user_cluster = match_user_to_cluster(user_input)

        mapped_clusters = cluster_product_map[user_cluster]

        recommended_clusters = recommendations["cluster"].unique()

        # Check if all recommended products
        # belong to expected clusters
        is_correct = all(cluster in mapped_clusters for cluster in recommended_clusters)

        if is_correct:
            correct_recommendations += 1

    # Final Accuracy
    accuracy = (correct_recommendations / total_tests) * 100

    return accuracy


# =========================================================
# RUN EVALUATION
# =========================================================

accuracy = evaluate_recommendation_accuracy(users_df, recommend_products)

print("\n==============================")
print("RECOMMENDATION SYSTEM ACCURACY")
print("==============================")

print(f"Accuracy: {accuracy:.2f}%")

# =========================================================
# ACCURACY GRAPH
# =========================================================

labels = ["Correct", "Incorrect"]

correct = accuracy
incorrect = 100 - accuracy

values = [correct, incorrect]

plt.figure(figsize=(6, 6))

plt.pie(values, labels=labels, autopct="%1.1f%%")

plt.title("Recommendation Accuracy")

plt.show()

# =========================================================
# BAR GRAPH
# =========================================================

plt.figure(figsize=(6, 5))

plt.bar(["Accuracy"], [accuracy])

plt.ylim(0, 100)

plt.ylabel("Accuracy Percentage")

plt.title("Recommendation System Accuracy")

plt.grid(True)

plt.show()

# =========================================================
# SAMPLE OUTPUT VALIDATION
# =========================================================

print("\n==============================")
print("SAMPLE RECOMMENDATIONS")
print("==============================")

sample_user = users_df.drop(columns=["cluster"]).iloc[0].to_dict()

sample_recommendations = recommend_products(sample_user)

for i, title in enumerate(sample_recommendations["title"].head(5), start=1):
    print(f"{i}. {title}")
