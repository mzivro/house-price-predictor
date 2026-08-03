from src.schemas import HouseData, PredictionResponse
from fastapi import APIRouter, Request

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, data: HouseData) -> PredictionResponse:
    """
    Predict a house price.

    Parameters
    ----------
    request : Request
        FastAPI request object containing the application state.
    data : HouseData
        Request body containing the input house data.

    Returns
    -------
    PredictionResponse
        Prediction result of a house price.
    """
    model = request.app.state.model_service

    prediction = model.predict(
        data.city, data.district, data.floor, data.rooms, data.sq
    )

    return PredictionResponse(prediction=prediction)
