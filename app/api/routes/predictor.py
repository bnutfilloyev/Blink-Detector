import cv2
import imutils
import numpy as np

from core.config import TOKEN, CHAT_ID
from fastapi import APIRouter, File, HTTPException, UploadFile
from models.prediction import Detection
from services.predict import MachineLearningModelHandlerScore as model

from services import blink_detection
from aiogram import Bot, types


router = APIRouter()
bot = Bot(token=TOKEN)

EYE_AR_THRESH = 0.15

# get base64 image
@router.post("/detect", response_model=Detection ,name="image:get-data")
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
        else:
            return Detection(status=False)



    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")