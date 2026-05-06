import numpy as np
import pandas as pd
import streamlit as st
from rec_sys import users_df

st.title("Sustainable Product Recommendation System")
with st.form("main_form"):
    options_list = list()
    for col in users_df.columns:
        options = users_df[col].unique()
        if len(options.tolist()) < 5:
            print(col, "->", type(users_df[col].dtype))
            if type(users_df[col].dtype) is pd.StringDtype:
                options_list.append(st.selectbox(col, tuple(options)))
            elif users_df[col].dtype == np.bool:
                options_list.append(st.checkbox(col))

    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("submitted")
        st.write(options_list)

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
