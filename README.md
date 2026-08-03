# SMS Spam Classifier

A machine learning web application for house price prediction in Poland built with **FastAPI**, **Streamlit**, and **scikit-learn**. The application predicts a house price in Warsaw, Kraków or Poznań in **PLN** currency using One-Hot Encoder and Random Forest regressor.

**Warning** \
This project was created for educational purposes and uses a dataset from 2021, so predictions may not be relevant today.

## Features

* House Price prediction in Warsaw, Kraków and Poznań
* The model takes into account districts, floors, number of rooms and square meters
* REST API built with FastAPI
* Interactive web interface built with Streamlit
* Automatic model training if no serialized model is found

## Tech Stack

* scikit-learn
* FastAPI
* Pydantic
* Streamlit
* Pandas
* KaggleHub
* Pickle

## Model Pipeline

1. Download the House Prices in Poland dataset from Kaggle.
2. Clear outliers by filtering year and coordinate features.
3. Extract districts from addresses.
4. Convert city and district features to one-hot features.
5. Train a Random Forest regressor.
6. Save the trained model for future inference.

## Installation

Clone the repository:

```bash
git clone https://github.com/mzivro/house-price-predictor.git
cd house-price-predictor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

The model will be automatically trained with first start-up of API.

```bash
fastapi run src/server.py
```

The API will be available at:

```
http://localhost:8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

## Running the Streamlit Client

```bash
streamlit run src/client.py
```

The application will be available at:

```
http://localhost:8501
```

## API

### POST `/predict`

Request

```json
{
  "city": "Warszawa",
  "district": "Mokotów",
  "floor": 5,
  "rooms": 2,
  "sq": 43.64
}
```

Response

```json
{
  "prediction": 562790.19
}
```

## Docker

Build the image:

```bash
docker build -t house-price-predictor .
```

Run the container:

```bash
docker run -p 8000:8000 house-price-predictor
```

## Dataset

The project uses the **House Prices in Poland** dataset from Kaggle, downloaded via KaggleHub. Dataset is made by **Dawid Cegielski**.

Dataset source:
https://www.kaggle.com/datasets/dawidcegielski/house-prices-in-poland

## License

MIT License. Feel free to use, modify, and build upon this project.
