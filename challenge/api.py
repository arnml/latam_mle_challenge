from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import pandas as pd

from challenge.model import DelayModel

app = FastAPI()

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int = Field(..., ge=1, le=12)

class FlightRequest(BaseModel):
    flights: List[Flight]

# Initialize the model
model = DelayModel()

@app.get("/health")
async def get_health():
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(request: FlightRequest):
    try:
        flights_to_process = [flight.model_dump() for flight in request.flights]
        flights_df = pd.DataFrame(flights_to_process)
        features = model.preprocess(data=flights_df)
        predictions = model.predict(features=features)
        return {"predict": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body},
    )