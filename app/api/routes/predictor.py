import cv2
import numpy as np
# from services import blink_detection
from aiogram import Bot
from core.config import CHAT_ID, TOKEN
from fastapi import APIRouter, File, HTTPException, UploadFile
from models.prediction import Detection
from services.eye_detector import eye_blink_detection
from services.predict import MachineLearningModelHandlerScore as model

router = APIRouter()
bot = Bot(token=TOKEN)

EYE_AR_THRESH = 0.15

# # get base64 image
# @router.post("/detect", response_model=Detection ,name="image:get-data")
# async def image(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         # print(contents)
#         nparr = np.fromstring(contents, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         frame = imutils.resize(frame, width=700)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         EAR = blink_detection.calculate_ear(frame, gray)

#         await bot.send_photo(chat_id=CHAT_ID, photo=contents, caption=f"EAR: {EAR}")

#         if EAR is not None :
#             if EAR <= EYE_AR_THRESH:
#                 return Detection(status=False)
#             else:
#                 return Detection(status=True)
#         else:
#             return Detection(status=False)


#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Exception: {e}")

# get base64 image
@router.post("/detect", response_model=Detection, name="image:get-data")
async def image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # print(contents)
        nparr = np.fromstring(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(frame.shape)

        result, pred_l, pred_r = eye_blink_detection(frame)

        await bot.send_photo(
            chat_id=CHAT_ID,
            photo=contents,
            caption=f"<b>result</b>: <code>{result}, </code>\n\n<b>Left: </b> <code>{pred_l[0][0]}, </code>\n<b>Right: </b> <code>{pred_r[0][0]}</code>",
            parse_mode="HTML",
        )

        if result is not None:
            if result == "Blink":
                return Detection(status=False)
            else:
                return Detection(status=True)
        else:
            return Detection(status=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")
