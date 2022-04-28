from typing import Any

import cv2
import imutils
import numpy as np

import joblib
from core.config import TOKEN, CHAT_ID
from core.errors import PredictException
from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger
from models.prediction import HealthResponse, MachineLearningResponse, Detection
from services.predict import MachineLearningModelHandlerScore as model

from services import blink_detection
from aiogram import Bot, types


router = APIRouter()
bot = Bot(token=TOKEN)

get_prediction = lambda data_input: MachineLearningResponse(
    model.predict(data_input, load_wrapper=joblib.load, method="predict_proba")
)

EYE_AR_THRESH = 0.15

@router.get("/predict", response_model=MachineLearningResponse, name="predict:get-data")
async def predict(data_input: Any = None):
    if not data_input:
        raise HTTPException(status_code=404, detail=f"'data_input' argument invalid!")
    try:
        prediction = get_prediction(data_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return MachineLearningResponse(prediction=prediction)


# @router.get(
#     "/health", response_model=HealthResponse, name="health:get-data",
# )
# async def health():
#     is_health = False
#     try:
#         get_prediction("lorem ipsum")
#         is_health = True
#         return HealthResponse(status=is_health)
#     except Exception:
#         raise HTTPException(status_code=404, detail="Unhealthy")


# get base64 image
@router.get("/detect", response_model=Detection ,name="image:get-data")
async def image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # print(contents)
        nparr = np.fromstring(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        frame = imutils.resize(frame, width=700)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        EAR = blink_detection.calculate_ear(frame, gray)

        await bot.send_photo(chat_id=CHAT_ID, photo=contents, caption=f"EAR: {EAR}")

        if EAR is not None :
            if EAR <= EYE_AR_THRESH:
                return Detection(status=False)
            else:
                return Detection(status=True)



    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")