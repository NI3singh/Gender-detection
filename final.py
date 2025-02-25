from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
from mtcnn import MTCNN
import numpy as np
from transformers import AutoModelForImageClassification, AutoImageProcessor
import cv2

app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Gender Detection API",
        "usage": "Send a POST request with an image file to /predict-gender endpoint"
    }

# Initialize MTCNN for face detection
detector = MTCNN()

# Load the gender classification model
model_name = "rizvandwiki/gender-classification-2"
model = AutoModelForImageClassification.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)

def process_image(image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert PIL Image to numpy array for MTCNN
    image_np = np.array(image)
    
    # Detect faces using MTCNN
    faces = detector.detect_faces(image_np)
    
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image. Please upload a different image.")
    
    # Get the first face (assuming single person)
    face = faces[0]
    x, y, width, height = face['box']
    
    # Extract face region
    face_image = image.crop((x, y, x + width, y + height))
    
    return face_image

def predict_gender(face_image):
    # Prepare image for the model
    inputs = image_processor(face_image, return_tensors="pt")
    print(inputs)

    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.softmax(dim=-1)
        
    # Get predicted class
    predicted_class = model.config.id2label[predictions.argmax().item()]
    
    return predicted_class

@app.post("/predict-gender")
async def predict_gender_endpoint(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Process image and get face
        face_image = process_image(image_bytes)
        
        # Predict gender
        gender = predict_gender(face_image)
        
        return {"gender": gender}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)