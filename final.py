# # from fastapi import FastAPI, File, UploadFile, HTTPException, Query
# # from PIL import Image
# # from enum import Enum
# # import io
# # import numpy as np
# # import cv2
# # from fastapi.responses import JSONResponse
# # import os

# # app = FastAPI(title="Gender Detection API")

# # @app.get("/")
# # async def root():
# #     return {
# #         "message": "Welcome to the Gender Detection API",
# #         "usage": "Send a POST request with an image file to /predict-gender endpoint"
# #     }

# # class Gender(str, Enum):
# #     MALE = "male"
# #     FEMALE = "female"

# # # Update these paths to be relative or use environment variables
# # # You should place your model files in a directory accessible to your application
# # MODEL_DIR = os.getenv("MODEL_DIR", r"/home/gender/Gender-detection/models")

# # faceProto = os.path.join(MODEL_DIR, "face_detection/opencv_face_detector_uint8.pb")
# # faceModel = os.path.join(MODEL_DIR, "face_detection/opencv_face_detector.pbtxt")
# # faceNet = cv2.dnn.readNet(faceModel, faceProto)

# # genderProto = os.path.join(MODEL_DIR, "face_recognistion/gender_deploy.prototxt")
# # genderModel = os.path.join(MODEL_DIR, "face_recognistion/gender_net.caffemodel")
# # genderNet = cv2.dnn.readNet(genderModel, genderProto)

# # genderList = ['Male', 'Female']

# # def process_image(faceDetectionModel, image_bytes, conf_threshold=0.7):
# #     # Convert bytes to PIL Image
# #     pil_image = Image.open(io.BytesIO(image_bytes))
    
# #     # Convert PIL image to OpenCV format (numpy array)
# #     if pil_image.mode != "RGB":
# #         pil_image = pil_image.convert("RGB")
    
# #     # Convert PIL image to numpy array for OpenCV
# #     image = np.array(pil_image)
    
# #     # Get image dimensions from numpy array
# #     frameHeight, frameWidth = image.shape[:2]
    
# #     # Create a blob from the image
# #     blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    
# #     faceDetectionModel.setInput(blob)
# #     detections = faceDetectionModel.forward()
    
# #     bounding_boxes = []
# #     for i in range(detections.shape[2]):
# #         confidence = detections[0, 0, i, 2]
        
# #         if confidence > conf_threshold:
# #             x1 = int(detections[0, 0, i, 3] * frameWidth)
# #             y1 = int(detections[0, 0, i, 4] * frameHeight)
# #             x2 = int(detections[0, 0, i, 5] * frameWidth)
# #             y2 = int(detections[0, 0, i, 6] * frameHeight)
            
# #             # Ensure coordinates are within image boundaries
# #             x1 = max(0, x1)
# #             y1 = max(0, y1)
# #             x2 = min(frameWidth, x2)
# #             y2 = min(frameHeight, y2)
            
# #             bounding_boxes.append([x1, y1, x2, y2])
            
# #             # Draw rectangle on the image (optional for debugging)
# #             cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
# #     return image, bounding_boxes

# # def predict_gender(image, bounding_boxes):
# #     results = []
    
# #     if not bounding_boxes:
# #         return JSONResponse(
# #                 status_code=422,
# #                 content={"detail": "No human face detected. Please upload an image with a clear human face."}
# #             )
    
# #     for bbox in bounding_boxes:
# #         x1, y1, x2, y2 = bbox
        
# #         # Check if the bounding box is valid
# #         if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 <= 0 or y2 <= 0:
# #             continue
            
# #         # Extract the face ROI
# #         face_roi = image[y1:y2, x1:x2]
        
# #         # Check if face_roi is empty
# #         if face_roi.size == 0:
# #             continue
            
# #         # Preprocess for gender classification
# #         blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
# #                                      [78.4263377603, 87.7689143744, 114.895847746], 
# #                                      swapRB=False)
        
# #         genderNet.setInput(blob)
# #         gender_preds = genderNet.forward()
# #         gender_idx = gender_preds[0].argmax()
# #         gender = genderList[gender_idx]
# #         confidence = float(gender_preds[0][gender_idx])
        
# #         results.append({
# #             "gender": gender,
# #             "confidence": confidence,
# #         })
    
# #     return results

# # @app.post("/predict-gender")
# # async def predict_gender_endpoint(
# #     file: UploadFile = File(...),
# #     selected_gender: Gender = Query(..., description=" ")
# # ):
# #     try:
# #         # Read image file
# #         image_bytes = await file.read()
        
# #         # Process image and get faces
# #         processed_image, bounding_boxes = process_image(faceNet, image_bytes)
        
# #         if not bounding_boxes:
# #             return JSONResponse(
# #                 status_code=422,
# #                 content={"detail": "Face is not detected, try to upload clear image again"}
# #             )
        
# #         if len(bounding_boxes) > 1:
# #             return JSONResponse(
# #                 status_code=422,
# #                 content={"detail": "Multiple faces detected, please upload an image with a single face."}
# #             )
        
# #         # Predict gender for all detected faces
# #         gender_predictions = predict_gender(processed_image, bounding_boxes)
        
# #         if not gender_predictions:
# #             return JSONResponse(
# #                 status_code=422,
# #                 content={"detail": "Failed to classify gender for detected faces"}
# #             )
        
# #         # Get gender from first face (taking the most prominent face in the image)
# #         detected_gender = gender_predictions[0]["gender"]
        
# #         # If user selected a gender to verify
# #         if selected_gender:
# #             # Check if detected gender matches the selected gender
# #             if detected_gender.lower() == selected_gender.value:
# #                 return {"gender": detected_gender}
# #             else:
# #                 return JSONResponse(
# #                     status_code=422,
# #                     content={"detail": "mismatched gender please try again."}
# #                 )
# #         else:
# #             # Just return detected gender in the required format
# #             return {"gender": detected_gender}
            
# #     except Exception as e:
# #         import traceback
# #         traceback.print_exc()
# #         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# #--------------RetinaFace, DeepFace, CLIP, -----------------

# # import cv2
# # import numpy as np
# # from fastapi import FastAPI, File, UploadFile, HTTPException, Query
# # from transformers import CLIPProcessor, CLIPModel
# # from fastapi.responses import JSONResponse
# # from retinaface import RetinaFace
# # from deepface import DeepFace
# # from io import BytesIO
# # from PIL import Image
# # from enum import Enum

# # # FastAPI app instance
# # app = FastAPI()

# # @app.get("/")
# # async def root():
# #     return {
# #         "message": "Welcome to the Gender Detection API",
# #         "usage": "Send a POST request with an image file to /predict-gender endpoint"
# #     }

# # class Gender(str, Enum):
# #     MALE = "male"
# #     FEMALE = "female"

# # # Load pre-trained CLIP model and processor
# # model_name = "openai/clip-vit-base-patch32"
# # processor = CLIPProcessor.from_pretrained(model_name)
# # model = CLIPModel.from_pretrained(model_name)

# # def is_real_image(image: Image.Image) -> str:
# #     """Classify if the image is real or cartoon using CLIP."""
# #     try:
# #         # Prepare inputs for CLIP
# #         inputs = processor(
# #             text=["male", "female", "Human", "Non-Human", "Real Person", "Cartoon", "Sketch", "Anime", "Painting", "Pixel Art"],
# #             images=image,
# #             return_tensors="pt",
# #             padding=True
# #         )

# #         # Get predictions
# #         outputs = model(**inputs)
# #         logits_per_image = outputs.logits_per_image
# #         probs = logits_per_image.softmax(dim=1).detach().numpy()

# #         # Check if the image is classified as real
# #         male_conf = probs[0][0]  # Probability of being a male image
# #         female_conf = probs[0][1]
# #         real_person_conf = probs[0][4]

# #         if real_person_conf > max(probs[0][5:]):
# #             # If the image is real, check human vs non-human confidence
# #             if probs[0][2] > probs[0][3]: 
# #                 return "real-human", male_conf, female_conf, 0.3, 0.7
# #             else:
# #                 raise ValueError("No human face detected, only human faces are supported")
# #         elif real_person_conf < max(probs[0][5:]):
# #             if probs[0][2] > probs[0][3]:
# #                 return "cartoon-human", male_conf, female_conf, 0.7, 0.3
# #             else:
# #                 raise ValueError("No human face detected, only human faces are supported")
# #         else:
# #             raise ValueError("an error occurred while processing the image try with another image again")

# #         # return real_prob > 0.5  # Threshold for real image classification
# #     except Exception as e:
# #         return JSONResponse(status_code=422, content={"message": "an error occurred while processing the image try with another image again"})

# # # Helper function to preprocess image for DeepFace
# # def preprocess_image_for_deepface(image: np.ndarray):
# #     # DeepFace requires image in shape of (224, 224, 3)
# #     image_resized = cv2.resize(image, (224, 224))
# #     image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
# #     return image_rgb

# # # Face detection using RetinaFace
# # def detect_face(image: np.ndarray):
# #     # Using RetinaFace to detect faces
# #     faces = RetinaFace.extract_faces(image)
    
# #     if len(faces) == 0:
# #         raise ValueError("No face detected in the image")
# #     elif len(faces) > 1:
# #         raise ValueError("Multiple faces detected, please upload an image with only one face")
    
# #     # Ensure the detected face is a human face (this is inferred from RetinaFace output metadata)
# #     face = faces[0]
# #     if 'confidence' in face and face['confidence'] < 0.5:  # Low confidence can imply it's not a human face
# #         raise ValueError("No human face detected, only human faces are supported")
    
# #     return face  # Return the first detected face

# # @app.post("/predict_gender/")
# # async def predict_gender(selected_gender: Gender = Query(..., description=" "), file: UploadFile = File(...)):
# #     try:    
# #         # Read the image file
# #         image_bytes = await file.read()
# #         image = Image.open(BytesIO(image_bytes))
# #         image_np = np.array(image)

# #         # Step 1: Detect face in the image using RetinaFace
# #         try:
# #             face = detect_face(image_np)
# #         except ValueError as e:
# #             return JSONResponse(status_code=422, content={"message": "an error occurred while processing the image try with another image again"})

# #         # Step 2: Preprocess the image for DeepFace gender classification
# #         preprocessed_image = preprocess_image_for_deepface(face)

# #         # Step 3: Predict gender using DeepFace
# #         try:
# #             result = DeepFace.analyze(preprocessed_image, actions=['gender'], enforce_detection=False)
# #             predicted_gender_scores = result[0]["gender"]
# #             predicted_gender = "male" if predicted_gender_scores["Man"] > predicted_gender_scores["Woman"] else "female"
# #             print(predicted_gender)
# #         except Exception as e:
# #             return JSONResponse(status_code=422, content={"message": "an error occurred while processing the image try with another image again"})
        
# #         # Step 4: Get CLIP classification result
# #         classification, male_conf, female_conf, clip_weight, model2_weight = is_real_image(image)
        
# #         if classification == "Error: Not a human image":
# #             return JSONResponse(status_code=422, content={"message": "Not a Human Image. Please upload an image with a human face."})

# #         # Step 5: Weighted Gender Result
# #         clip_result = "male" if male_conf > female_conf else "female"
# #         total_weight = 1
# #         final_clip_weight = clip_weight * total_weight
# #         final_model2_weight = model2_weight * total_weight
# #         print(clip_result)
# #         print(final_clip_weight)
# #         print(final_model2_weight)
# #         # Final gender decision based on weighted score
# #         # if final_clip_weight > final_model2_weight:
# #         #     return JSONResponse(content={"predicted_gender": clip_result, "weight_from_clip_model": final_clip_weight})
# #         # else:
# #         #     return JSONResponse(content={"predicted_gender": predicted_gender, "weight_from_deepface_model": final_model2_weight})

# #         if final_clip_weight > final_model2_weight:
# #             final_gender = clip_result
# #         else:
# #             final_gender = predicted_gender

# #         # Check if predicted gender matches selected gender
# #         if final_gender.lower() != selected_gender.lower():
# #             return JSONResponse(status_code=422, content={"message": "Mismatched gender. Please upload an image with the selected gender."})

# #         # Return the result
# #         return JSONResponse(content={"gender": final_gender})
        
# #         # # Step 4: Check if the predicted gender matches the user-selected gender
# #         # if predicted_gender.lower() != selected_gender.lower():
# #         #     return JSONResponse(status_code=400, content={"message": "Mismatched gender. Please upload an image with the selected gender."})

# #         # # Return the result
# #         # return JSONResponse(content={"predicted_gender": predicted_gender, "user_selected_gender": selected_gender})
    
# #     except Exception as e:
# #             return JSONResponse(status_code=422, content={"message": "an error occurred while processing the image try with another image again"})

# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="0.0.0.0", port=8000)


# #--------------RetinaFace, CLIP -----------------

# import cv2
# import numpy as np
# from fastapi import FastAPI, File, UploadFile, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from retinaface import RetinaFace
# from io import BytesIO
# from PIL import Image
# from enum import Enum

# import clip
# import torch
# from PIL import Image


# # FastAPI app instance
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change this to a specific domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {
#         "message": "Welcome to the Gender Detection API",
#         "usage": "Send a POST request with an image file to /predict-gender endpoint"
#     }

# class Gender(str, Enum):
#     MALE = "male"
#     FEMALE = "female"


# device = 'cpu'
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# def prediction_of_gender(image: Image.Image) -> str:
#     """Classify if the image is real or cartoon using CLIP."""
#     try:
#         device = 'cpu'
#         # Prepare inputs for CLIP
#         image_input = preprocess(image).unsqueeze(0).to(device)
#         texts_input = clip.tokenize(["man", "woman", "male", "female", "boy", "girl", "Human", "Non-Human"]).to(device)

#         with torch.no_grad():
#             image_features = model.encode_image(image_input)
#             text_features = model.encode_text(texts_input)

#         # Get predictions
#         # Compare which text matches the image
#         logits_per_image, logits_per_text = model(image_input, texts_input)
#         probs = logits_per_image.softmax(dim=-1).detach().to(device).numpy()[0]
#         print("Label probs:", probs)

#         man_conf = probs[0]
#         woman_conf = probs[1]
#         male_conf = probs[2]
#         female_conf = probs[3]
#         boy_conf = probs[4]
#         girl_conf = probs[5]
#         human_conf = probs[6]
#         non_human_conf = probs[7]
        
#         print(f"man_conf: {man_conf}, woman_conf: {woman_conf}, male_conf: {male_conf}, female_conf: {female_conf}")
#         print(f"boy_conf: {boy_conf}, girl_conf: {girl_conf}, human_conf: {human_conf}, non_human_conf: {non_human_conf}")

#         if human_conf > non_human_conf:
#             # If the image is real, check human vs non-human confidence
#             if max(male_conf, man_conf, boy_conf) > max(female_conf, woman_conf, girl_conf):
#                 return "male"
#             else:
#                 return "female"    
#         else:
#             raise ValueError("No human face detected, only human faces are supported.")

#     except Exception as e:
#         raise ValueError("No human face detected, only human faces are supported.")

# # Helper function to preprocess image for DeepFace
# def preprocess_image_for_deepface(image: np.ndarray):
#     # DeepFace requires image in shape of (224, 224, 3)
#     image_resized = cv2.resize(image, (224, 224))
#     image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
#     return image_rgb

# # Face detection using RetinaFace
# def detect_face(image: np.ndarray):
#     # Using RetinaFace to detect faces
#     faces = RetinaFace.detect_faces(image)
    

#     if len(faces) > 1:
#         raise ValueError("Multiple faces detected, please upload an image with only one face.")
    
#     # Ensure the detected face is a human face (this is inferred from RetinaFace output metadata)
#     face_data = list(faces.values())[0]
#     if face_data['score'] < 0.5:
#         raise ValueError("No human face detected, only human faces are supported.")
    
#     return True  # Return the first detected face

# @app.post("/predict_gender/")
# async def predict_gender(selected_gender: Gender = Query(..., description=" "), file: UploadFile = File(...)):
#     try:    
#         # Read the image file
#         image_bytes = await file.read()
#         image = Image.open(BytesIO(image_bytes))
#         # Check if the image is in RGB mode
#         if image.mode != "RGB":
#             image = image.convert("RGB")  # Convert to RGB

#         image_np = np.array(image)

#         # Step 1: Detect face in the image using RetinaFace
#         try:
#             face = detect_face(image_np)
#         except ValueError as e:
#             return JSONResponse(status_code=422, content={"message": "Multiple faces detected, please upload an image with only one face."})
        
#         # Step 4: Get CLIP classification result
#         try:
#             predicted_gender = prediction_of_gender(image)
#         except ValueError as e:
#             return JSONResponse(
#                 status_code=422,
#                 content={"message": "No human face detected, only human faces are supported."}
#             )
        
#         # Step 3: Check if the predicted gender matches the user-selected gender
#         if predicted_gender != selected_gender.value.lower():
#             return JSONResponse(
#                 status_code=422,
#                 content={"message": "Mismatched gender, Please upload an image with the selected gender."}
#             )

#         # Return the result
#         return JSONResponse(content={
#             "gender": predicted_gender
#         })
    
#     except Exception as e:
#         return JSONResponse(
#             status_code=422,
#             content={"message": "No human face detected, only human faces are supported."}
#         )



#--------------RetinaFace, CLIP, Mediapie -----------------

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from retinaface import RetinaFace
from io import BytesIO
from PIL import Image
from enum import Enum

import clip
import torch
from PIL import Image

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from typing import Optional, List, Tuple

BOUNDARY_MARGIN = 0.05 

# FastAPI app instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to a specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Gender Detection API",
        "usage": "Send a POST request with an image file to /predict-gender endpoint"
    }

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


device = 'cpu'
# device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # Helper function to get face landmarks
# def get_face_landmarks(image: np.ndarray) -> Optional[landmark_pb2.NormalizedLandmarkList]:
#     results = face_mesh.process(image)
#     if results.multi_face_landmarks:
#         return results.multi_face_landmarks[0]
#     return None

# # Helper function to check if the face is within the image boundaries
# def is_face_within_boundaries(face_landmarks: landmark_pb2.NormalizedLandmarkList, image_shape: Tuple[int, int]) -> bool:
#     h, w, _ = image_shape
#     for landmark in face_landmarks.landmark:
#         x = int(landmark.x * w)
#         y = int(landmark.y * h)
#         if x < 0 or x >= w or y < 0 or y >= h:
#             return False
#     return True

def check_face_completeness_mediapipe(
    image_np: np.ndarray,
    margin: float = BOUNDARY_MARGIN
    ) -> Tuple[bool, Optional[List[landmark_pb2.NormalizedLandmark]]]:
    """
    Checks if a face detected in an image is reasonably complete using MediaPipe Face Mesh.

    Detects face landmarks using MediaPipe Face Mesh. It then checks if key
    landmarks defining the face's outer boundary (forehead, chin, left/right jaw)
    are too close to the image edges, indicating potential cropping.

    Args:
        image_np: The input image as a NumPy array (BGR format).
        margin: Normalized distance (0.0-1.0) from image edges. If key boundary
                landmarks fall within this distance, the face is flagged as
                potentially incomplete. Default is BOUNDARY_MARGIN (0.05).

    Returns:
        A tuple:
        - bool: True if a face is detected and deemed sufficiently complete
                based on landmark proximity to edges, False otherwise (no face
                detected, or landmarks suggest cropping).
        - list | None: A list containing the mediapipe NormalizedLandmark objects
                       for the first detected face if successful and complete,
                       None otherwise.
    """
    if not 0.0 < margin < 0.5:
        print(f"Warning: Invalid margin {margin}. Using default {BOUNDARY_MARGIN}.")
        margin = BOUNDARY_MARGIN

    if image_np is None or image_np.size == 0:
         print("Error: Invalid image data provided to MediaPipe check.")
         return False, None

    try:
        # Convert BGR (from OpenCV/PIL np.array) to RGB for MediaPipe
        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = img_rgb.shape
    except Exception as e:
        print(f"Error converting image for MediaPipe: {e}")
        return False, None

    # Initialize and use Face Mesh within a 'with' block for resource management
    try:
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,      # Optimize for single images
                max_num_faces=1,             # Assume one primary face
                refine_landmarks=True,       # Get the detailed 478 landmarks
                min_detection_confidence=0.5 # Standard confidence threshold
                ) as face_mesh:

            results = face_mesh.process(img_rgb)

    except Exception as e:
          print(f"Error during MediaPipe Face Mesh processing: {e}")
          return False, None

    # Check if any face landmarks were detected
    if not results.multi_face_landmarks:
        # This means MediaPipe didn't find a face meeting its criteria
        # It could be no face, or a very heavily cropped/occluded one.
        # We treat this as incomplete for our purpose.
        return False, None

    # Get landmarks for the first detected face
    face_landmarks = results.multi_face_landmarks[0].landmark

    # Optional check: Ensure expected number of landmarks were found
    # if len(face_landmarks) < 468:
    #     print("Warning: Did not detect the full set of landmarks.")
        # Decide how to handle this - for completeness, we might require the full set.
        # return False, None

    # Key landmark indices for boundary checks
    try:
        forehead_pt = face_landmarks[10]     # Top of forehead center
        chin_pt = face_landmarks[152]    # Bottom of chin
        left_jaw_pt = face_landmarks[234]    # Left edge of jaw/cheek
        right_jaw_pt = face_landmarks[454]   # Right edge of jaw/cheek
    except IndexError:
        print("Error: Could not access expected landmark indices. Landmarks might be incomplete.")
        return False, None
    except Exception as e:
        print(f"Error accessing landmark data: {e}")
        return False, None

    # Perform the Completeness Check based on Edge Proximity
    # Check Top Crop
    if forehead_pt.y < margin:
        print(f"Face Completeness Check Failed: Forehead too close to top edge (y={forehead_pt.y:.2f} < {margin}).")
        return False, None
    # Check Bottom Crop
    if chin_pt.y > (1.0 - margin):
        print(f"Face Completeness Check Failed: Chin too close to bottom edge (y={chin_pt.y:.2f} > {1.0-margin:.2f}).")
        return False, None
    # Check Left Crop
    if left_jaw_pt.x < margin:
        print(f"Face Completeness Check Failed: Left Jaw too close to left edge (x={left_jaw_pt.x:.2f} < {margin}).")
        return False, None
    # Check Right Crop
    if right_jaw_pt.x > (1.0 - margin):
        print(f"Face Completeness Check Failed: Right Jaw too close to right edge (x={right_jaw_pt.x:.2f} > {1.0-margin:.2f}).")
        return False, None

    # If all checks passed, the face is considered complete
    return True, face_landmarks

def prediction_of_gender(image: Image.Image) -> str:
    """Classify if the image is real or cartoon using CLIP."""
    try:
        device = 'cpu'
        # Prepare inputs for CLIP
        image_input = preprocess(image).unsqueeze(0).to(device)
        texts_input = clip.tokenize(["man", "woman", "male", "female", "boy", "girl", "Human", "Non-Human"]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(texts_input)

        # Get predictions
        # Compare which text matches the image
        logits_per_image, logits_per_text = model(image_input, texts_input)
        probs = logits_per_image.softmax(dim=-1).detach().to(device).numpy()[0]
        print("Label probs:", probs)

        man_conf = probs[0]
        woman_conf = probs[1]
        male_conf = probs[2]
        female_conf = probs[3]
        boy_conf = probs[4]
        girl_conf = probs[5]
        human_conf = probs[6]
        non_human_conf = probs[7]
        
        print(f"man_conf: {man_conf}, woman_conf: {woman_conf}, male_conf: {male_conf}, female_conf: {female_conf}")
        print(f"boy_conf: {boy_conf}, girl_conf: {girl_conf}, human_conf: {human_conf}, non_human_conf: {non_human_conf}")

        if human_conf > non_human_conf:
            # If the image is real, check human vs non-human confidence
            if max(male_conf, man_conf, boy_conf) > max(female_conf, woman_conf, girl_conf):
                return "male"
            else:
                return "female"    
        else:
            raise ValueError("No human face detected, only human faces are supported.")

    except Exception as e:
        raise ValueError("No human face detected, only human faces are supported.")

# Helper function to preprocess image for DeepFace
def preprocess_image_for_deepface(image: np.ndarray):
    # DeepFace requires image in shape of (224, 224, 3)
    image_resized = cv2.resize(image, (224, 224))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    return image_rgb

# Face detection using RetinaFace
def detect_face_retina(image: np.ndarray):
    # Using RetinaFace to detect faces
    faces = RetinaFace.detect_faces(image)
    

    if len(faces) > 1:
        raise ValueError("Multiple faces detected, please upload an image with only one face.")
    
    # Ensure the detected face is a human face (this is inferred from RetinaFace output metadata)
    face_data = list(faces.values())[0]
    if face_data['score'] < 0.5:
        raise ValueError("No human face detected, only human faces are supported.")
    
    return True  # Return the first detected face

@app.post("/predict_gender/")
async def predict_gender(selected_gender: Gender = Query(..., description=" "), file: UploadFile = File(...)):
    try:    
        # Read the image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        # Check if the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")  # Convert to RGB

        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Explicit conversion just in case PIL loads differently

        # Step 1: Detect face in the image using RetinaFace
        # try:
        #     face = detect_face(image_np)
        # except ValueError as e:
        #     return JSONResponse(status_code=422, content={"message": "Multiple faces detected, please upload an image with only one face."})
        try:
            detect_face_retina(image_np) # Raises ValueError on failure
            print("RetinaFace Check: Passed (Single confident face detected).")
        except ValueError as e:
            print(f"RetinaFace Check Failed: {e}")
            # Provide a more specific error based on the exception message if needed
            return JSONResponse(status_code=422, content={"message": str(e)})
        
         # --- Step 3: Face Completeness Check using MediaPipe ---
        try:
            is_complete, _ = check_face_completeness_mediapipe(image_np) # Pass the numpy array
            if not is_complete:
                # The function check_face_completeness_mediapipe already prints the reason
                print("MediaPipe Check Failed: Face deemed incomplete or cropped.")
                return JSONResponse(
                    status_code=422,
                    content={"message": "Please ensure your entire face is clearly visible in the profile photo"}
                )
            print("MediaPipe Check: Passed (Face seems complete).")
        except Exception as e:
            print(f"Error during MediaPipe completeness check: {e}")
            return JSONResponse(status_code=422, content={"message": "An error occurred during face analysis."})
        
        # Step 4: Get CLIP classification result
        try:
            predicted_gender = prediction_of_gender(image)
        except ValueError as e:
            return JSONResponse(
                status_code=422,
                content={"message": "No human face detected, only human faces are supported."}
            )
        
        # Step 3: Check if the predicted gender matches the user-selected gender
        if predicted_gender != selected_gender.value.lower():
            return JSONResponse(
                status_code=422,
                content={"message": "Mismatched gender, Please upload an image with the selected gender."}
            )

        # Return the result
        return JSONResponse(content={
            "gender": predicted_gender
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={"message": "No human face detected, only human faces are supported."}
        )