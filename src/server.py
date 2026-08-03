from contextlib import asynccontextmanager
from src.model_service import ModelService
from fastapi import FastAPI
from src.api import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle.

    The machine learning model is loaded during application startup and stored
    in the application state for reuse across requests.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance.

    Yields
    ------
    None
        Control back to the application after initialization.
    """
    app.state.model_service = ModelService("regressor_model.pkl")

    yield


app = FastAPI(lifespan=lifespan)

app.include_router(router)
