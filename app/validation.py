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
# VALIDATION + ACCURACY ANALYSIS
# =========================================================

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================================================
# 1. USER CLUSTER VALIDATION
# =========================================================

print("\n==============================")
print("USER CLUSTER VALIDATION")
print("==============================")

# Silhouette Score
user_score = silhouette_score(scaled_features, users_df["cluster"])

print(f"User Clustering Silhouette Score: {user_score:.4f}")

# PCA Visualization
pca = PCA(n_components=2)

user_pca = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(user_pca[:, 0], user_pca[:, 1], c=users_df["cluster"])

plt.title("User Cluster Map")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.colorbar(scatter)

plt.grid(True)

plt.show()

# =========================================================
# 2. PRODUCT CLUSTER VALIDATION
# =========================================================

print("\n==============================")
print("PRODUCT CLUSTER VALIDATION")
print("==============================")

# Silhouette Score
product_score = silhouette_score(X, products_df["cluster"])

print(f"Product Clustering Silhouette Score: {product_score:.4f}")

# PCA Visualization
product_pca = PCA(n_components=2).fit_transform(X.toarray())

plt.figure(figsize=(8, 6))

scatter = plt.scatter(product_pca[:, 0], product_pca[:, 1], c=products_df["cluster"])

plt.title("Product Cluster Map")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.colorbar(scatter)

plt.grid(True)

plt.show()

# =========================================================
# 3. CLUSTER DISTRIBUTION GRAPH
# =========================================================

print("\n==============================")
print("CLUSTER DISTRIBUTION")
print("==============================")

# User cluster distribution
user_cluster_counts = users_df["cluster"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

user_cluster_counts.plot(kind="bar")

plt.title("User Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Users")

plt.grid(True)

plt.show()

# Product cluster distribution
product_cluster_counts = products_df["cluster"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

product_cluster_counts.plot(kind="bar")

plt.title("Product Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Products")

plt.grid(True)

plt.show()

# =========================================================
# 4. RECOMMENDATION DIVERSITY TEST
# =========================================================

print("\n==============================")
print("RECOMMENDATION DIVERSITY TEST")
print("==============================")

# Test using 2 users from dataset
profile1 = users_df.drop(columns=["cluster"]).iloc[0].to_dict()
profile2 = users_df.drop(columns=["cluster"]).iloc[10].to_dict()

rec1 = recommend_products(profile1)
rec2 = recommend_products(profile2)

print("\nRecommendations for User 1:")
print(rec1["title"].head(5).tolist())

print("\nRecommendations for User 2:")
print(rec2["title"].head(5).tolist())

# =========================================================
# 5. ACCURACY VISUALIZATION GRAPH
# =========================================================

print("\n==============================")
print("MODEL QUALITY GRAPH")
print("==============================")

scores = pd.DataFrame(
    {
        "Model": ["User Clustering", "Product Clustering"],
        "Silhouette Score": [user_score, product_score],
    }
)

plt.figure(figsize=(7, 5))

plt.bar(scores["Model"], scores["Silhouette Score"])

plt.title("Clustering Quality Comparison")
plt.ylabel("Silhouette Score")

plt.ylim(0, 1)

plt.grid(True)

plt.show()

# =========================================================
# 6. INTERPRETATION
# =========================================================

print("\n==============================")
print("INTERPRETATION")
print("==============================")

if user_score > 0.5:
    print("User clustering quality is GOOD")
elif user_score > 0.25:
    print("User clustering quality is ACCEPTABLE")
else:
    print("User clustering quality is WEAK")

if product_score > 0.5:
    print("Product clustering quality is GOOD")
elif product_score > 0.25:
    print("Product clustering quality is ACCEPTABLE")
else:
    print("Product clustering quality is WEAK")
