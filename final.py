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
# detector = MTCNN()

detector = MTCNN(min_face_size=30, scale_factor=0.709)


try:
    # Try to load the pre-trained face cascade (comes with OpenCV)
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Warning: Could not load Haar cascade: {e}")
    haar_cascade = None

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

    # If MTCNN fails, try Haar Cascade as fallback
    if not faces and haar_cascade is not None:
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Detect faces using Haar Cascade
        haar_faces = haar_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert Haar Cascade results to MTCNN-like format for consistent processing
        if len(haar_faces) > 0:
            faces = []
            for (x, y, w, h) in haar_faces:
                faces.append({
                    'box': (x, y, w, h),
                    'confidence': 0.95  # Placeholder confidence
                })
    
    if not faces:
        raise HTTPException(status_code=422, detail="No human face detected. Please upload an image with a clear human face.")
    
    if len(faces) > 1:
        raise HTTPException(status_code=422, detail="Multiple faces detected. Please upload an image with a single clear human face.")
    
    # Get the first face (assuming single person)
    face = faces[0]
    x, y, width, height = face['box']
    confidence = face['confidence']

    if confidence < 0.85:  # Using a high threshold to filter out non-human faces
        raise HTTPException(status_code=422, detail="No human face detected. Please upload an image with a clear human face.")
    
    # Make sure box coordinates are valid
    x, y = max(0, x), max(0, y)
    width = min(width, image_np.shape[1] - x)
    height = min(height, image_np.shape[0] - y)
    
    # Extract face region with some margin
    margin_percent = 0.2
    margin_x = int(width * margin_percent)
    margin_y = int(height * margin_percent)
    
    # Ensure the expanded region doesn't go outside the image bounds
    x_expanded = max(0, x - margin_x)
    y_expanded = max(0, y - margin_y)
    width_expanded = min(width + 2 * margin_x, image_np.shape[1] - x_expanded)
    height_expanded = min(height + 2 * margin_y, image_np.shape[0] - y_expanded)
    
    # Extract face region
    # face_image = image.crop((x, y, x + width, y + height))
    
    face_image = image.crop((x_expanded, y_expanded, x_expanded + width_expanded, y_expanded + height_expanded))

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
    selected_gender: Gender = Query(..., description=" ")
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
                # return {"result": "mismatched gender please try again"}
                raise HTTPException(status_code=422, detail="mismatched gender please try again.")
        else:
            # Just return detected gender
            return {"gender": detected_gender}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")