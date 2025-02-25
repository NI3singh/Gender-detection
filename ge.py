# # main.py
# import cv2
# import numpy as np
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import os
# from typing import List, Dict, Any
# import logging
# from datetime import datetime
# from mtcnn import MTCNN

# detector = MTCNN()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(
#     title="Gender Detection API",
#     description="API for detecting gender from facial images using OpenCV",
#     version="1.0.0"
# )

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, replace with specific origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Constants
# ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
# MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# # Model paths
# MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
# GENDER_MODEL_PATH = os.path.join(MODEL_PATH, "gender_net.caffemodel")
# GENDER_PROTO_PATH = os.path.join(MODEL_PATH, "gender_deploy.prototxt")

# try:
#     # Initialize face detection model
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     if face_cascade.empty():
#         raise Exception("Error loading face cascade classifier")

#     # Initialize gender detection model
#     if not (os.path.exists(GENDER_MODEL_PATH) and os.path.exists(GENDER_PROTO_PATH)):
#         raise Exception("Gender model files not found")
    
#     gender_model = cv2.dnn.readNet(GENDER_MODEL_PATH, GENDER_PROTO_PATH)
    
# except Exception as e:
#     logger.error(f"Error initializing models: {str(e)}")
#     raise

# # Model parameters
# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# GENDER_LIST = ['Male', 'Female']

# def validate_image(file: UploadFile) -> bool:
#     """Validate the uploaded image file."""
#     # Check file extension
#     file_ext = os.path.splitext(file.filename)[1].lower()
#     if file_ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
#         )
#     return True

# def get_gender_prediction(face_img: np.ndarray) -> tuple[str, float]:
#     """Predict gender from face image."""
#     try:
#         # Prepare image for gender detection
#         blob = cv2.dnn.blobFromImage(
#             face_img, 1.0, (227, 227),
#             MODEL_MEAN_VALUES, swapRB=False
#         )
        
#         # Gender detection
#         gender_model.setInput(blob)
#         gender_preds = gender_model.forward()
#         gender = GENDER_LIST[gender_preds[0].argmax()]
#         confidence = float(gender_preds[0][gender_preds[0].argmax()])
        
#         return gender, confidence
        
#     except Exception as e:
#         logger.error(f"Error in gender prediction: {str(e)}")
#         raise HTTPException(status_code=500, detail="Error processing image")

# @app.get("/")
# async def root():
#     """Root endpoint returning API information."""
#     return {
#         "api": "Gender Detection API",
#         "version": "1.0.0",
#         "status": "active"
#     }

# @app.post("/predict-gender")
# async def predict_gender(file: UploadFile = File(...)) -> Dict[str, Any]:
#     """
#     Predict gender from uploaded image.
#     Returns gender predictions for all detected faces.
#     """
#     try:
#         # Validate file
#         validate_image(file)
        
#         # Read image file
#         contents = await file.read()
#         if len(contents) > MAX_IMAGE_SIZE:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"File size exceeds maximum limit of {MAX_IMAGE_SIZE/1024/1024}MB"
#             )
        
#         # Convert to numpy array
#         nparr = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if img is None:
#             raise HTTPException(status_code=400, detail="Invalid image file")
        
#         # Convert to grayscale for face detection
#         # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Detect faces
#         faces = detector.detect_faces(img)
        
#         if len(faces) == 0:
#             return JSONResponse(content={"prediction": "No face detected"})
        
#         results = []
#         # Process each detected face
#         for face in faces:
#             if face['confidence'] > 0.9:  # Filter by MTCNN confidence (adjustable)
#                 x, y, w, h = face['box']  # MTCNN returns [x, y, width, height]
#                 face_img = img[y:y+h, x:x+w]  # Crop face
#                 gender, confidence = get_gender_prediction(face_img)
#                 confidence_pct = round(confidence * 100, 2)
#                 results.append({
#                     "gender": gender,
#                     "confidence": confidence_pct
#                 })

#         # If no faces were processed (e.g., all MTCNN confidences < 0.9), return no faces detected
#         if not results:
#             return JSONResponse(content={"prediction": "No face detected"})

#         # Find the prediction with the highest confidence
#         best_prediction = max(results, key=lambda x: x["confidence"])

#         # Log successful prediction
#         logger.info(f"Processed image with highest confidence prediction: {best_prediction}")

#         return JSONResponse(content={
#             "predictions": [best_prediction]  # Return only the highest confidence prediction
#         })
        
#     except HTTPException as he:
#         logger.error(f"HTTP Exception: {str(he)}")
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)