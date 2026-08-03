import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

city_data = {
    "Warszawa": [
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
    ],
    "Kraków": [
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
    ],
    "Poznań": [
        "Stare Miasto Piątkowo",
        "Stare Miasto",
        "Grunwald Łazarz",
        "Jeżyce",
        "Wilda",
        "Grunwald",
        "Nowe Miasto",
    ],
}

st.title("House Price Predictor")

# city - district

upper_col_1, upper_col_2 = st.columns(2)

with upper_col_1:
    city = st.selectbox("City", list(city_data.keys()))

with upper_col_2:
    district = st.selectbox("District", city_data[city])

# floor - rooms - square meters

lower_col_1, lower_col_2, lower_col_3 = st.columns(3)

with lower_col_1:
    floor = st.number_input("Floor", min_value=0, value=1, step=1)

with lower_col_2:
    rooms = st.number_input("Rooms", min_value=1, value=1, step=1)

with lower_col_3:
    sq = st.number_input(
        "Square meters", min_value=1.0, value=50.0, step=0.1, format="%.2f"
    )

if st.button("Predict", width="stretch"):
    payload = {
        "city": city,
        "district": district,
        "floor": floor,
        "rooms": rooms,
        "sq": sq,
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        prediction = response.json()["prediction"]

        st.success(f"Prediction: {prediction:.2f} PLN")
    else:
        st.error(response.text)
