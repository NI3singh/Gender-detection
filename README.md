# Gender Detection API

This project provides a FastAPI-based API for detecting the gender of a person in an uploaded image. It uses RetinaFace for face detection and OpenAI's CLIP model for gender classification. The API ensures high accuracy for real human faces and handles errors gracefully for non-human or low-quality images.

## Features

- **Face Detection**: Uses RetinaFace to detect and validate human faces in the image.
- **Gender Classification**: Uses OpenAI's CLIP model to classify the gender as "male" or "female".
- **Error Handling**: Handles cases like no face detected, multiple faces, or non-human images.
- **FastAPI**: Provides a fast and scalable API for gender detection.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### Clone the Repository:

```bash
git clone https://github.com/your-username/gender-detection-api.git
cd gender-detection-api
```

### Set Up a Virtual Environment (Optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

### Start the FastAPI Server:

```bash
uvicorn final:app --reload
```

The API will start running at http://127.0.0.1:8000.

### Access the API Docs:
Open your browser and navigate to:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## API Endpoints

### 1. Root Endpoint
**URL**: GET /

**Description**: Welcome message and usage instructions.

**Response**:
```json
{
  "message": "Welcome to the Gender Detection API",
  "usage": "Send a POST request with an image file to /predict-gender endpoint"
}
```

### 2. Predict Gender
**URL**: POST /predict_gender/

**Description**: Predicts the gender of the person in the uploaded image.

**Parameters**:
- `selected_gender` (Query Parameter): Expected gender (male or female).
- `file` (Form Data): Image file to analyze.

**Response**:

Success:
```json
{
  "gender": "male" "or" "female",
}
```

Error:
```json
{
  "message": "No faces detected in the image. Please upload an image with a clear human face."
}
```

## Example Requests

### Using curl
```bash
curl -X POST -F "file=@test_image.jpg" "http://127.0.0.1:8000/predict_gender/?selected_gender=male"
```

### Using Python (requests library)
```python
import requests

url = "http://127.0.0.1:8000/predict_gender/?selected_gender=male"
files = {"file": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Project Structure
```
gender-detection-api/
├── final.py               # Main FastAPI application
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

## Dependencies

The project uses the following Python libraries:

- **fastapi**: For building the API.
- **uvicorn**: For running the FastAPI server.
- **retina-face**: For face detection.
- **clip**: For gender classification using OpenAI's CLIP model.
- **torch**: For deep learning operations.
- **Pillow**: For image processing.
- **numpy**: For numerical operations.
- **opencv-python**: For image manipulation.

All dependencies are listed in `requirements.txt`.

## Error Handling

The API handles the following errors gracefully:

### No Face Detected:
```json
{
  "message": "No faces detected in the image. Please upload an image with a clear human face."
}
```

### Multiple Faces Detected:
```json
{
  "message": "Multiple faces detected. Please upload an image with only one face."
}
```

### Non-Human Image:
```json
{
  "message": "Non-human image detected. Please upload a real human face."
}
```

### Gender Mismatch:
```json
{
  "message": "Gender mismatch. Detected: male, Expected: female"
}
```

### Internal Server Error:
```json
{
  "message": "Internal server error: <error details>"
}
```

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.


## Acknowledgments

- **RetinaFace**: For robust face detection.
- **OpenAI CLIP**: For accurate gender classification.
- **FastAPI**: For building a fast and scalable API.

## Contact

For any questions or feedback, feel free to reach out:

- **Name**: Nitin Singh
- **Email**: ni3.singh.r@gmail.com
- **GitHub**: NI3singh

Enjoy using the Gender Detection API!
