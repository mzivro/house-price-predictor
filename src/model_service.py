import src.utils as utils
import pandas as pd
import pickle


class ModelService:
    """
    Service responsible for loading and serving the house price prediction model.

    If the serialized model file does not exist, a new model is trained and
    saved automatically.

    Parameters
    ----------
    path : str
        Path to the serialized model file.
    """

    def __init__(self, path: str):
        """
        Load the trained model from disk or train a new one if necessary.

        Parameters
        ----------
        path : str
            Path to the serialized model file.
        """
        try:
            with open(path, "rb") as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            print("Model file not found, beginning training")
            utils.train(path)

            with open(path, "rb") as file:
                self.model = pickle.load(file)

    def predict(
        self, city: str, district: str, floor: int, rooms: int, sq: float
    ) -> float:
        """
        Predict whether a house price.

        The input data is packed into dataframe before being passed to the trained model.

        Parameters
        ----------
        city : str
            City name.
        district : int
            District name.
        floor : int
            Floor number.
        rooms : int
            Room numbers.
        sq : float
            Square meters of house.

        Returns
        -------
        float
            Predicted house price.
        """
        input_data = pd.DataFrame(
            [
                {
                    "city": city,
                    "district": district,
                    "floor": floor,
                    "rooms": rooms,
                    "sq": sq,
                }
            ]
        )

        return float(self.model.predict(input_data)[0])
