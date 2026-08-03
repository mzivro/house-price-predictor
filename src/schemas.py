from pydantic import BaseModel


class HouseData(BaseModel):
    """
    Request schema containing a house data.

    Attributes
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
    """

    city: str
    district: str
    floor: int
    rooms: int
    sq: float


class PredictionResponse(BaseModel):
    """
    Response schema containing the predicted value.

    Attributes
    ----------
    prediction : float
        Predicted price of the house
    """

    prediction: float
