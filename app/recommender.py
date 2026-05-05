import os

import kagglehub
import pandas as pd

lifestyle_data_path = kagglehub.dataset_download(
    "naveennas/sustainable-lifestyle-rating-dataset"
)
print("Path to dataset files:", lifestyle_data_path)


def recursive_traverse(path, ext=None):
    ext_paths = list()
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            recursive_traverse(file_path)
        else:
            print(file_path)
            if ext is not None:
                if file_path.endswith(ext):
                    ext_paths.append(file_path)
                    return ext_paths


recursive_traverse(lifestyle_data_path)

lifestyle_data = pd.read_csv(
    os.path.join(lifestyle_data_path, "lifestyle_sustainability_data.csv")
)

lifestyle_data.drop_duplicates(inplace=True)

lifestyle_data["Rating"] = pd.to_numeric(lifestyle_data["Rating"], errors="coerce")

# Identify top 20% and bottom 20% participants
top = lifestyle_data[lifestyle_data["Rating"] >= lifestyle_data["Rating"].quantile(0.8)]
low = lifestyle_data[lifestyle_data["Rating"] <= lifestyle_data["Rating"].quantile(0.2)]


def lifestyle_recommendations(row):
    recs = []

    if row["TransportationMode"] == "Car":
        recs.append("Consider using public transit, biking, or walking more often.")
    if row["EnergySource"] in ["Non-Renewable", "Mixed"]:
        recs.append("Switch to renewable energy sources like solar or wind.")
    if row["UsingPlasticProducts"] in ["Often", "Sometimes"]:
        recs.append("Reduce single-use plastic and switch to reusable materials.")
    if row["DisposalMethods"] in ["Landfill", "Combination"]:
        recs.append("Adopt composting or proper recycling methods.")
    if row["DietType"] == "Mostly Animal-Based":
        recs.append(
            "Include more plant-based meals in your diet for lower carbon impact."
        )
    if (
        row["MonthlyWaterConsumption"]
        > lifestyle_data["MonthlyWaterConsumption"].median()
    ):
        recs.append("Reduce water usage through efficient fixtures and habits.")
    if (
        row["MonthlyElectricityConsumption"]
        > lifestyle_data["MonthlyElectricityConsumption"].median()
    ):
        recs.append("Use energy-efficient appliances and unplug unused devices.")

    return recs if recs else ["Keep up your great eco-friendly habits!"]


# Apply the recommender to low-rating participants
low["Recommendations"] = low.apply(lifestyle_recommendations, axis=1)

print(
    low[
        [
            "ParticipantID",
            "Rating",
            "Recommendations",
        ]
    ].head()
)

products_data_path = kagglehub.dataset_download(
    "sofikulislam/amazon-eco-friendly-products-dataset"
)

print("Path to dataset files:", products_data_path)
csv_files = recursive_traverse(products_data_path, ext=".csv")

df = pd.read_csv(csv_files[0])
print(df.drop(columns=["url", "img_url", "inStock", "inStockText"]).head())
