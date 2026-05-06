import os

import kagglehub
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

products_path = kagglehub.dataset_download(
    "sofikulislam/amazon-eco-friendly-products-dataset"
)

print("Path to dataset files:", products_path)

users_path = kagglehub.dataset_download(
    "naveennas/sustainable-lifestyle-rating-dataset"
)

print("Path to dataset files:", users_path)

products_df = pd.read_csv(
    os.path.join(products_path, "amazon_eco-friendly_products.csv")
).dropna(subset=["title", "description"])

users_df = pd.read_csv(
    os.path.join(users_path, "lifestyle_sustainability_data.csv")
).dropna()


# Preprocess users_df for clustering
le = LabelEncoder()
user_features = users_df.copy()
for col in users_df.columns:
    user_features[col] = le.fit_transform(users_df[col])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(user_features)
kmeans = KMeans(n_clusters=5, random_state=42)
users_df["cluster"] = kmeans.fit_predict(scaled_features)


# Combine title and description
products_df["text"] = products_df["title"] + " " + products_df["description"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=20, stop_words="english")
X = vectorizer.fit_transform(products_df["text"])

# Cluster
kmeans = KMeans(n_clusters=10, random_state=42)
products_df["cluster"] = kmeans.fit_predict(X)


def match_user_to_cluster(user_input, users_df, scaler, kmeans):
    # Convert user_input to a DataFrame row
    st.write(user_features.columns)
    input_df = pd.DataFrame([user_input], columns=user_features.columns)
    for col in input_df.columns:
        input_df[col] = le.fit_transform(input_df[col])
    input_scaled = scaler.transform(input_df)
    cluster = kmeans.predict(input_scaled)[0]
    return cluster


# products_df["ensemble"] = products_df["title"] + " " + products_df["description"]
# corpus = products_df["ensemble"].tolist()


# removes a list of words (ie. stopwords) from a tokenized list.
# def removeWords(listOfTokens, listOfWords):
#     return [token for token in listOfTokens if token not in listOfWords]
#
#
# # applies stemming to a list of tokenized words
# def applyStemming(listOfTokens, stemmer):
#     return [stemmer.stem(token) for token in listOfTokens]
#
#
# # removes any words composed of less than 2 or more than 21 letters
# def twoLetters(listOfTokens):
#     twoLetterWord = []
#     for token in listOfTokens:
#         if len(token) <= 2 or len(token) >= 21:
#             twoLetterWord.append(token)
#     return twoLetterWord
#
#
# def processCorpus(corpus, language):
#     stopwords = nltk.corpus.stopwords.words(language)
#     param_stemmer = SnowballStemmer(language)
#
#     for document in corpus:
#         index = corpus.index(document)
#         corpus[index] = corpus[index].replace(
#             "\ufffd", "8"
#         )  # Replaces the ASCII '�' symbol with '8'
#         corpus[index] = corpus[index].replace(",", "")  # Removes commas
#         corpus[index] = corpus[index].rstrip("\n")  # Removes line breaks
#         corpus[index] = corpus[index].casefold()  # Makes all letters lowercase
#
#         corpus[index] = re.sub(
#             "\W_", " ", corpus[index]
#         )  # removes specials characters and leaves only words
#         corpus[index] = re.sub(
#             "\S*\d\S*", " ", corpus[index]
#         )  # removes numbers and words concatenated with numbers
#         corpus[index] = re.sub(
#             "\S*@\S*\s?", " ", corpus[index]
#         )  # removes emails and mentions (words with @)
#         corpus[index] = re.sub(r"http\S+", "", corpus[index])  # removes URLs with http
#         corpus[index] = re.sub(r"www\S+", "", corpus[index])  # removes URLs with www
#         corpus[index] = re.sub(r"[^\x00-\x7F]+", "", corpus[index])  # remove non-ASCII
#
#         listOfTokens = word_tokenize(corpus[index])
#         twoLetterWord = twoLetters(listOfTokens)
#
#         listOfTokens = removeWords(listOfTokens, stopwords)
#         listOfTokens = removeWords(listOfTokens, twoLetterWord)
#
#         listOfTokens = applyStemming(listOfTokens, param_stemmer)
#
#         corpus[index] = " ".join(listOfTokens)
#
#     return corpus
#
#
# language = "english"
# corpus = processCorpus(corpus, language)
#
# # creating count_vect object
# count_vect = CountVectorizer()
#
# # Create Matrix
# count_matrix = count_vect.fit_transform(corpus)
#
# # Compute the cosine similarity matrix
# cosine_sim = cosine_similarity(count_matrix, count_matrix)
# # st.write(cosine_sim)
# indices = pd.Series(products_df.index, index=products_df["title"]).drop_duplicates()
# # st.write(indices)
#
#
# # Function that takes in product title as input and gives recommendations
# def content_recommender(title, cosine_sim=cosine_sim, df=products_df, indices=indices):
#
#     # Obtain the index of the product that matches the title
#     idx = indices[title]
#
#     # Get the pairwsie similarity scores of all products with that product
#     # And convert it into a list of tuples as described above
#     sim_scores = list(enumerate(cosine_sim[idx]))
#
#     # Sort the products based on the cosine similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#
#     # Get the scores of the 30 most similar products. Ignore the first product.
#     sim_scores = sim_scores[1:30]
#
#     # Get the product indices
#     product_indices = [i[0] for i in sim_scores]
#
#     # Return the top 30 most similar products
#     return df["title"].iloc[product_indices]
#
#
# # Define the product we want to recommend other items from
# product_title = "Agfabric Natural Jute Erosion Control, 16yard(50 feet Long) Jute Netting -8ft Wide Soil Saver Mesh Blanket-400 Sq.Ft.Coverage"
#
# # Launching the content_recommender function
# recommendations = content_recommender(product_title)
#
# # Associating titles to recommendations
# asin_recommendations = products_df[products_df["title"].isin(recommendations)]
#
# # Merging datasets
# recommendations = pd.merge(
#     recommendations, asin_recommendations, on="title", how="left"
# )

# Showing top 5 recommended products
# st.write(recommendations["title"].head())

# TFIDF-based, was not working that well
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# tf_idf = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names_out())
#
# final_df = tf_idf
#
# print("{} rows".format(final_df.shape[0]))
# st.write(final_df.T.nlargest(5, 0))
