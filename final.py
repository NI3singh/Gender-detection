from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image
from enum import Enum
import io
import numpy as np
import cv2
from fastapi.responses import JSONResponse
import os
from retinaface import RetinaFace

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

# Update these paths to be relative or use environment variables
# You should place your model files in a directory accessible to your application
MODEL_DIR = os.getenv("MODEL_DIR", r"/home/gender/Gender-detection/models")

faceProto = os.path.join(MODEL_DIR, "face_detection/opencv_face_detector_uint8.pb")
faceModel = os.path.join(MODEL_DIR, "face_detection/opencv_face_detector.pbtxt")
faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderProto = os.path.join(MODEL_DIR, "face_recognistion/gender_deploy.prototxt")
genderModel = os.path.join(MODEL_DIR, "face_recognistion/gender_net.caffemodel")
genderNet = cv2.dnn.readNet(genderModel, genderProto)

genderList = ['Male', 'Female']

def process_image(faceDetectionModel, image_bytes, conf_threshold=0.7):
    # Convert bytes to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    # Convert PIL image to OpenCV format (numpy array)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # Convert PIL image to numpy array for OpenCV
    image = np.array(pil_image)
    
    # Get image dimensions from numpy array
    # frameHeight, frameWidth = image.shape[:2]
    
    # # Create a blob from the image
    # blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    
    # faceDetectionModel.setInput(blob)
    # detections = faceDetectionModel.forward()
    
    # bounding_boxes = []
    # for i in range(detections.shape[2]):
    #     confidence = detections[0, 0, i, 2]
        
    #     if confidence > conf_threshold:
    #         x1 = int(detections[0, 0, i, 3] * frameWidth)
    #         y1 = int(detections[0, 0, i, 4] * frameHeight)
    #         x2 = int(detections[0, 0, i, 5] * frameWidth)
    #         y2 = int(detections[0, 0, i, 6] * frameHeight)
            
    #         # Ensure coordinates are within image boundaries
    #         x1 = max(0, x1)
    #         y1 = max(0, y1)
    #         x2 = min(frameWidth, x2)
    #         y2 = min(frameHeight, y2)
            
    #         bounding_boxes.append([x1, y1, x2, y2])
            
    #         # Draw rectangle on the image (optional for debugging)
    #         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # RetinaFace detection
    faces = RetinaFace.detect_faces(image)

    bounding_boxes = []
    for key in faces:
        face_data = faces[key]
        x1, y1, x2, y2 = face_data["facial_area"]
        
        bounding_boxes.append([x1, y1, x2, y2])

            
    if len(faces) > 1:
        return JSONResponse(
            status_code=422,
            content={"detail": "Multiple faces detected, please upload an image with a single face."}
        )
    
    return image, bounding_boxes

def predict_gender(image, bounding_boxes):
    results = []
    
    if not bounding_boxes:
        return JSONResponse(
                status_code=422,
                content={"detail": "No human face detected. Please upload an image with a clear human face."}
            )
    
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        
        # Check if the bounding box is valid
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 <= 0 or y2 <= 0:
            continue
            
        # Extract the face ROI
        face_roi = image[y1:y2, x1:x2]
        
        # Check if face_roi is empty
        if face_roi.size == 0:
            continue
            
        # Preprocess for gender classification
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                     [78.4263377603, 87.7689143744, 114.895847746], 
                                     swapRB=False)
        
        genderNet.setInput(blob)
        gender_preds = genderNet.forward()
        gender_idx = gender_preds[0].argmax()
        gender = genderList[gender_idx]
        confidence = float(gender_preds[0][gender_idx])
        
        results.append({
            "gender": gender,
            "confidence": confidence,
        })
    
    return results

@app.post("/predict-gender")
async def predict_gender_endpoint(
    file: UploadFile = File(...),
    selected_gender: Gender = Query(..., description=" ")
):
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Process image and get faces
        processed_image, bounding_boxes = process_image(faceNet, image_bytes)
        
        if not bounding_boxes:
            return JSONResponse(
                status_code=422,
                content={"detail": "Face is not detected, try to upload clear image again"}
            )
        
        if len(bounding_boxes) > 1:
            return JSONResponse(
                status_code=422,
                content={"detail": "Multiple faces detected, please upload an image with a single face."}
            )
        
        # Predict gender for all detected faces
        gender_predictions = predict_gender(processed_image, bounding_boxes)
        
        if not gender_predictions:
            return JSONResponse(
                status_code=422,
                content={"detail": "Failed to classify gender for detected faces"}
            )
        
        # Get gender from first face (taking the most prominent face in the image)
        detected_gender = gender_predictions[0]["gender"]
        
        # If user selected a gender to verify
        if selected_gender:
            # Check if detected gender matches the selected gender
            if detected_gender.lower() == selected_gender.value:
                return {"gender": detected_gender}
            else:
                return JSONResponse(
                    status_code=422,
                    content={"detail": "mismatched gender please try again."}
                )
        else:
            # Just return detected gender in the required format
            return {"gender": detected_gender}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Please Upload a High Quality Face Image and try again.")