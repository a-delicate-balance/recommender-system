import numpy as np
import pandas as pd
import streamlit as st
from rec_sys import recommend_products, users_df

st.set_page_config(
    page_title="Sustainable Product Recommendation System", layout="wide"
)

st.title("Sustainable Product Recommendation System")

st.subheader("Enter Your Lifestyle Details")

user_input = {}

mid, right = st.columns(2)
with right:
    st.subheader("Recommended Products")
with mid:
    with st.form("main_form"):
        user_input["ParticipantID"] = 99999
        for col in users_df.columns:
            if col.lower() in ["cluster", "participantid", "rating"]:
                continue

            if type(users_df[col].dtype) is pd.StringDtype:
                user_input[col] = st.selectbox(col, users_df[col].unique())

            elif users_df[col].dtype == bool:
                user_input[col] = st.checkbox(col)

            elif np.issubdtype(users_df[col].dtype, np.number):
                min_val = users_df[col].min()
                max_val = users_df[col].max()

                user_input[col] = st.number_input(
                    col,
                    value=min_val,
                )

        user_input["Rating"] = 2.5
        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        recommendations = recommend_products(user_input)

        with right:
            if not recommendations.empty:
                for _, row in recommendations.iterrows():
                    st.write("###", row["title"])

                    st.write(row["description"])

                    st.divider()

            else:
                st.write("No recommendations found.")
