# # # from fastapi import FastAPI, File, UploadFile, HTTPException
# # # from transformers import AutoImageProcessor, AutoModelForImageClassification
# # # import torch
# # # from PIL import Image
# # # import io

# # # app = FastAPI()

# # # # Load the model and processor
# # # try:
# # #     processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification-2")
# # #     model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification-2")
# # #     model.eval()  # Set model to evaluation mode
# # # except Exception as e:
# # #     print(f"Error loading model: {e}")
# # #     raise

# # # @app.post("/predict-gender/")
# # # async def predict_gender(file: UploadFile = File(...)):
# # #     # Validate file type
# # #     if not file.content_type.startswith(r"C:\Users\itsni\Desktop\Gender-Detection\IMG_20230803_102419877 (1).jpg"):
# # #         raise HTTPException(status_code=400, detail="File must be an image")
    
# # #     try:
# # #         # Read and process image
# # #         image_data = await file.read()
# # #         image = Image.open(io.BytesIO(image_data))
        
# # #         # Convert RGBA to RGB if necessary
# # #         if image.mode == 'RGBA':
# # #             image = image.convert('RGB')
        
# # #         # Process image and get prediction
# # #         inputs = processor(images=image, return_tensors="pt")
        
# # #         with torch.no_grad():
# # #             outputs = model(**inputs)
# # #             predictions = outputs.logits.softmax(dim=-1)
# # #             predicted_class = predictions.argmax().item()
# # #             confidence = predictions[0][predicted_class].item() * 100
        
# # #         # Get prediction label
# # #         predicted_label = model.config.id2label[predicted_class]
        
# # #         return {
# # #             "gender": predicted_label,
# # #             "confidence": f"{confidence:.2f}%",
# # #             "status": "success"
# # #         }
        
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# # # # Health check endpoint
# # # @app.get("/health")
# # # async def health_check():
# # #     return {"status": "healthy"}

# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app)









# # # # import pandas as pd
# # # # import numpy as np
# # # # import os
# # # # import matplotlib.pyplot as plt
# # # # import cv2
# # # # from keras.models import Sequential,load_model,Model
# # # # from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
# # # # from sklearn.model_selection import train_test_split

# # # # path = "UTKFace"
# # # # pixels = []
# # # # age = []
# # # # gender = []
# # # # for img in os.listdir(path):
# # # #   ages = img.split("_")[0]
# # # #   genders = img.split("_")[1]
# # # #   img = cv2.imread(str(path)+"/"+str(img))
# # # #   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# # # #   pixels.append(np.array(img))
# # # #   age.append(np.array(ages))
# # # #   gender.append(np.array(genders))
# # # # age = np.array(age,dtype=np.int64)
# # # # pixels = np.array(pixels)
# # # # gender = np.array(gender,np.uint64)

# # # # x_train,x_test,y_train,y_test = train_test_split(pixels,age,random_state=100)
# # # # x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(pixels,gender,random_state=100)

# # # # input = Input(shape=(200,200,3))
# # # # conv1 = Conv2D(140,(3,3),activation="relu")(input)
# # # # conv2 = Conv2D(130,(3,3),activation="relu")(conv1)
# # # # batch1 = BatchNormalization()(conv2)
# # # # pool3 = MaxPool2D((2,2))(batch1)
# # # # conv3 = Conv2D(120,(3,3),activation="relu")(pool3)
# # # # batch2 = BatchNormalization()(conv3)
# # # # pool4 = MaxPool2D((2,2))(batch2)
# # # # flt = Flatten()(pool4)
# # # # #age
# # # # age_l = Dense(128,activation="relu")(flt)
# # # # age_l = Dense(64,activation="relu")(age_l)
# # # # age_l = Dense(32,activation="relu")(age_l)
# # # # age_l = Dense(1,activation="relu")(age_l)
# # # # #gender
# # # # gender_l = Dense(128,activation="relu")(flt)
# # # # gender_l = Dense(80,activation="relu")(gender_l)
# # # # gender_l = Dense(64,activation="relu")(gender_l)
# # # # gender_l = Dense(32,activation="relu")(gender_l)
# # # # gender_l = Dropout(0.5)(gender_l)
# # # # gender_l = Dense(2,activation="softmax")(gender_l)

# # # # model = Model(inputs=input,outputs=[age_l,gender_l])
# # # # model.compile(optimizer="adam",loss=["mse","sparse_categorical_crossentropy"],metrics=['mae','accuracy'])
# # # # save = model.fit(x_train,[y_train,y_train_2],validation_data=(x_test,[y_test,y_test_2]),epochs=50)
# # # # model.save("model.h5")



# import tensorflow as tf
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import layers, Model

# # Load pre-trained InceptionV3 model
# base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# # Add custom layers for gender classification
# x = base_model.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(128, activation='relu')(x)
# predictions = layers.Dense(2, activation='softmax')(x)  # 2 classes: male, female

# # Create the model
# model = Model(inputs=base_model.input, outputs=predictions)

# # Freeze base model layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Data generators for training and validation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
# )
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     'path/to/train',  # Directory with train/male and train/female
#     target_size=(299, 299),
#     batch_size=32,
#     class_mode='categorical'
# )
# test_generator = test_datagen.flow_from_directory(
#     'path/to/test',  # Directory with test/male and test/female
#     target_size=(299, 299),
#     batch_size=32,
#     class_mode='categorical'
# )

# # Train the model
# model.fit(
#     train_generator,
#     validation_data=test_generator,
#     epochs=10
# )

# # Save the model
# model.save('gender_classification_model.h5')
