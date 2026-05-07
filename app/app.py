import numpy as np
import pandas as pd
import streamlit as st
from rec_sys import (
    kmeans,
    match_user_to_cluster,
    products_df,
    scaler,
    users_df,
)

st.set_page_config(
    page_title="Sustainable Product Recommendation System", layout="wide"
)
st.title("Sustainable Product Recommendation System")
mid, right = st.columns(2)
with right:
    st.subheader("Recommended Products")
with mid:
    with st.form("main_form"):
        options_list = [99999]
        for col in users_df.columns[1:19]:
            if col == "cluster":
                continue
            options = users_df[col].unique()
            if type(users_df[col].dtype) is pd.StringDtype:
                options_list.append(st.selectbox(col, options))
            elif users_df[col].dtype == bool:
                options_list.append(st.checkbox(col))
            elif np.issubdtype(users_df[col].dtype, np.number):
                if len(options) <= 5:
                    options_list.append(
                        st.slider(
                            col, users_df[col].min(), users_df[col].max(), width=300
                        )
                    )
                else:
                    options_list.append(
                        st.number_input(
                            col,
                            value=users_df[col].min(),
                            step=1,
                        )
                    )

        options_list.append(1)
        submitted = st.form_submit_button("Submit")
        if submitted:
            user_cluster = match_user_to_cluster(options_list, users_df, scaler, kmeans)
            cluster_products = products_df[products_df["cluster"] == user_cluster]
            with right:
                if not cluster_products.empty:
                    for title in cluster_products["title"].head(5):
                        st.write(f"- {title}")
                else:
                    st.write("No recommendations found for this profile.")

# st.title("Sustainable Product Recommendation System")
# with st.form("main_form"):
#     options_list = list()
#     for col in users_df.columns:
#         options = users_df[col].unique()
#         if len(options.tolist()) < 5:
#             print(col, "->", type(users_df[col].dtype))
#             if type(users_df[col].dtype) is pd.StringDtype:
#                 options_list.append(st.selectbox(col, tuple(options)))
#             elif users_df[col].dtype == np.bool:
#                 options_list.append(st.checkbox(col))
#             elif np.issubdtype(users_df[col].dtype, np.number):
#                 options_list.append(
#                     st.slider(
#                         col, float(users_df[col].min()), float(users_df[col].max())
#                     )
#                 )
#
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         user_cluster = match_user_to_cluster(options_list, users_df, scaler, kmeans)
#         cluster_products = products_df[products_df["cluster"] == user_cluster]
#         if not cluster_products.empty:
#             st.subheader("Recommended Products")
#             for title in cluster_products["title"].head(5):
#                 st.write(f"- {title}")
#         else:
#             st.write("No recommendations found for this profile.")
#
# submitted = st.form_submit_button("Submit")
# if submitted:
#     st.write("submitted")
#     st.write(options_list)

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
#
# # Showing top 5 recommended products
# st.write(recommendations["title"].head())
# st.write(users_df.head())
