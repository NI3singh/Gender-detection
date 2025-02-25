from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image
from enum import Enum
import io
import torch
from mtcnn import MTCNN
import numpy as np
from transformers import AutoModelForImageClassification, AutoImageProcessor
import cv2
from fastapi.responses import JSONResponse

app = FastAPI(title="Gender Detection API")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Gender Detection API",
        "usage": "Send a POST request with an image file to /predict-gender endpoint"
    }

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


# Initialize MTCNN for face detection
detector = MTCNN()

# Load the gender classification model
model_name = "rizvandwiki/gender-classification-2"
model = AutoModelForImageClassification.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)

def process_image(image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert PIL Image to numpy array for MTCNN
    image_np = np.array(image)
    
    # Detect faces using MTCNN
    faces = detector.detect_faces(image_np)
    
    if not faces:
        return None
    
    # Get the first face (assuming single person)
    face = faces[0]
    x, y, width, height = face['box']
    confidence = face['confidence']

    if confidence < 0.9:  # Using a high threshold to filter out non-human faces
        raise HTTPException(status_code=400, detail="No human face detected. Please upload an image with a clear human face.")
    
    # Extract face region
    face_image = image.crop((x, y, x + width, y + height))
    
    return face_image

def predict_gender(face_image):
    # Prepare image for the model
    inputs = image_processor(face_image, return_tensors="pt")
    # print(inputs)

    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.softmax(dim=-1)
        
    # Get predicted class
    predicted_class = model.config.id2label[predictions.argmax().item()]
    
    return predicted_class

# @app.post("/predict-gender/<predicted_class>")
# async def predict_gender_endpoint(file: UploadFile = File(...)):
#     try:
#         # Read image file
#         image_bytes = await file.read()
        
#         # Process image and get face
#         face_image = process_image(image_bytes)
        
#         if face_image is None:
#             return JSONResponse(
#                 status_code=400,
#                 content={"detail": "No face detected in the image. Please upload a different image."}
#             )
        
#         # Predict gender
#         gender = predict_gender(face_image)
        
#         if gender.lower() == predicted_class.lower():
#             return jsonify({"gender": gender}), 200
#         else:
#             return jsonify({"error": "Gender mismatch"}), 400
#         # return {"gender": gender}
        
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"detail": f"An error occurred: {str(e)}"}
#         )

@app.post("/predict-gender")
async def predict_gender_endpoint(
    file: UploadFile = File(...),
    selected_gender: Gender = Gender.MALE
):
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Process image and get face
        face_image = process_image(image_bytes)
        
        # Predict gender
        detected_gender = predict_gender(face_image)
        
        # If user selected a gender, compare with detected
        if selected_gender:
            # Check if they match
            if selected_gender.value == detected_gender.lower():
                return {"gender": detected_gender}
            else:
                return {"result": "mismatch", "selected_gender": selected_gender.value, "detected_gender": detected_gender}
        else:
            # Just return detected gender
            return {"gender": detected_gender}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")