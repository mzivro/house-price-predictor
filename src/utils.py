from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
import kagglehub
import pickle
import re
import os

warsaw_districts = [
    "Śródmieście",
    "Mokotów",
    "Ochota",
    "Wola",
    "Żoliborz",
    "Praga-Południe",
    "Praga-Północ",
    "Bemowo",
    "Białołęka",
    "Bielany",
    "Rembertów",
    "Targówek",
    "Ursus",
    "Ursynów",
    "Wawer",
    "Wesoła",
    "Wilanów",
    "Włochy",
]

krakow_districts = [
    "Stare Miasto",
    "Grzegórzki",
    "Prądnik Czerwony",
    "Prądnik Biały",
    "Krowodrza",
    "Bronowice",
    "Zwierzyniec",
    "Dębniki",
    "Łagiewniki-Borek Fałęcki",
    "Swoszowice",
    "Podgórze Duchackie",
    "Bieżanów-Prokocim",
    "Podgórze",
    "Nowa Huta Czyżyny",
    "Mistrzejowice",
    "Bieńczyce",
    "Wzgórza Krzesławickie",
    "Nowa Huta",
]

poznan_districts = [
    "Stare Miasto Piątkowo",
    "Stare Miasto",
    "Grunwald Łazarz",
    "Jeżyce",
    "Wilda",
    "Grunwald",
    "Nowe Miasto",
]


def _extract_districts(data: pd.DataFrame, districts: list):
    """
    Extract districts from addresses.

    Parameters
    ----------
    data : pd.DataFrame
        Transformed data.
    districts : list
        List of districts to extract.
    """
    pattern = r"\b(" + "|".join(map(re.escape, districts)) + r")\b"

    data["district"] = data["address"].str.extract(
        pattern, flags=re.IGNORECASE, expand=False
    )

    data.dropna(subset=["district"], inplace=True)
    data.drop(["address"], axis=1, inplace=True)


def train(path: str = "regressor_model.pkl"):
    """
    Train a house price prediction model and save it to disk.

    The function downloads the House Prices in Poland dataset, transform data,
    trains a One-Hot Encoder + Random Forest pipeline, evaluates its
    performance, and serializes the trained model.

    Parameters
    ----------
    path : str, default="regressor_model.pkl"
        Output path for the serialized model.
    """
    print("Downloading data")

    dataset_path = kagglehub.dataset_download("dawidcegielski/house-prices-in-poland")
    print(f"Dataset path: {dataset_path}")
    data = pd.read_csv(
        os.path.join(dataset_path, "Houses.csv"), encoding="windows-1250"
    )

    print("Transforming and cleaning data")

    data = data[(data["year"] >= 1950) & (data["year"] <= 2021)]
    data = data[(data["longitude"] >= 16) & (data["longitude"] <= 22)]
    data = data[(data["latitude"] >= 48) & (data["latitude"] <= 53)]

    data.drop(data[["Unnamed: 0", "id", "year"]], axis=1, inplace=True)

    # Warsaw
    warsaw_data = data[data["city"] == "Warszawa"]

    warsaw_data = warsaw_data[
        (warsaw_data["longitude"] >= 20.7) & (warsaw_data["longitude"] <= 21.3)
    ]
    warsaw_data = warsaw_data[
        (warsaw_data["latitude"] >= 52) & (warsaw_data["latitude"] <= 52.4)
    ]

    warsaw_data.drop(warsaw_data[["longitude", "latitude"]], axis=1, inplace=True)

    _extract_districts(warsaw_data, warsaw_districts)

    # Krakow
    krakow_data = data[data["city"] == "Kraków"]

    krakow_data = krakow_data[
        (krakow_data["longitude"] >= 19.8) & (krakow_data["longitude"] <= 20.2)
    ]
    krakow_data = krakow_data[
        (krakow_data["latitude"] >= 49.9) & (krakow_data["latitude"] <= 50.2)
    ]

    krakow_data.drop(krakow_data[["longitude", "latitude"]], axis=1, inplace=True)

    _extract_districts(krakow_data, krakow_districts)

    # Poznan
    poznan_data = data[data["city"] == "Poznań"]

    poznan_data = poznan_data[
        (poznan_data["longitude"] >= 16.6) & (poznan_data["longitude"] <= 17.2)
    ]
    poznan_data = poznan_data[
        (poznan_data["latitude"] >= 52.2) & (poznan_data["latitude"] <= 52.5)
    ]

    poznan_data.drop(poznan_data[["longitude", "latitude"]], axis=1, inplace=True)

    _extract_districts(poznan_data, poznan_districts)

    print("Training model")

    training_data = pd.concat(
        [warsaw_data, krakow_data, poznan_data], ignore_index=True
    )

    training_data["city"] = training_data["city"].astype("category")
    training_data["district"] = training_data["district"].astype("category")

    X = training_data.drop(["price"], axis=1)
    y = training_data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=8, shuffle=True
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "district"]),
            ("num", "passthrough", ["floor", "rooms", "sq"]),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=None,
                    min_samples_leaf=1,
                    min_samples_split=3,
                    max_features="sqrt",
                    random_state=8,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Evaluating model")
    y_pred = pipeline.predict(X_test)

    print("Root mean squared error:", mean_squared_error(y_test, y_pred) ** 0.5)
    print("Mean absolute error:", mean_absolute_error(y_test, y_pred))
    print("r2 score:", r2_score(y_test, y_pred))

    print("Saving model")

    with open(path, "wb") as file:
        pickle.dump(pipeline, file)
