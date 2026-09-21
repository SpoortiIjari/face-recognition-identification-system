# Face Recognition Identification System

A computer vision-based Face Recognition Identification System that detects faces, generates face embeddings, identifies enrolled individuals using similarity-based matching, and rejects unknown faces.

Built using Python, OpenCV YuNet, OpenCV SFace, NumPy, Pandas, and Streamlit.

---

## Project Overview

This project implements an end-to-end face recognition pipeline that allows users to enroll individuals and identify new faces by comparing them against an enrolled face database.

The system supports both image upload and camera capture.

## Key Features

- Face detection using YuNet
- Face alignment using facial landmarks
- 128-dimensional face embeddings using SFace
- Face enrollment
- Multiple face samples per person
- Image upload
- Camera capture
- Cosine similarity-based matching
- Similarity score display
- Unknown face rejection
- Local face embedding database
- Evaluation using genuine and impostor pairs
- Streamlit web interface

---


<img width="1366" height="532" alt="Screenshot 2026-09-21 164435" src="https://github.com/user-attachments/assets/b4a40782-b52a-414d-9a8f-d1e62ef985a2" />
<img width="1235" height="763" alt="Screenshot 2026-09-21 164454" src="https://github.com/user-attachments/assets/8811e7ef-6751-4b39-8440-694ef46cb4c2" />
<img width="1022" height="497" alt="Screenshot 2026-09-21 164508" src="https://github.com/user-attachments/assets/e8e885ed-f458-439a-81dd-c60c27afb336" />



## System Architecture

    Input Image / Camera
            |
            v
    Face Detection - YuNet
            |
            v
    Face Alignment
            |
            v
    Face Embedding - SFace
            |
            v
    128-Dimensional Embedding
            |
            v
    Cosine Similarity Matching
            |
            v
    Best Database Match
            |
            v
    Threshold Check
          /   \
         /     \
        v       v
      Known   Unknown

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Face Detection | OpenCV YuNet |
| Face Recognition | OpenCV SFace |
| Face Embedding | 128-dimensional |
| Matching Method | Cosine Similarity |
| Web Interface | Streamlit |
| Data Storage | JSON |
| Numerical Processing | NumPy |
| Data Processing | Pandas |
| Model Format | ONNX |

---

## Models Used

### YuNet - Face Detection

YuNet is used to detect faces in the input image.

It provides the face bounding box and facial landmarks required for further processing.

Model:

`face_detection_yunet_2026may.onnx`

### SFace - Face Recognition

SFace is used to generate a numerical representation of each detected face.

Each face is represented using a 128-dimensional embedding.

Model:

`face_recognition_sface_2021dec.onnx`

---

## How the System Works

### 1. Face Detection

The input image is processed using YuNet to detect the face and obtain its bounding box and facial landmarks.

### 2. Face Alignment

The detected face is aligned using its facial landmarks before generating the face embedding.

Face alignment helps provide a consistent input to the recognition model.

### 3. Face Embedding

The aligned face is passed to SFace.

SFace converts the face into a 128-dimensional numerical representation.

Face Image -> SFace -> 128-Dimensional Embedding

### 4. Enrollment

During enrollment:

1. The user enters a person's name.
2. An image is uploaded or captured using the camera.
3. The face is detected.
4. The face is aligned.
5. The face embedding is generated.
6. The embedding is stored in the local JSON database.

Multiple face samples can be stored for the same person.

### 5. Recognition

During recognition:

1. An image is uploaded or captured.
2. The face is detected.
3. The face is aligned.
4. A query embedding is generated.
5. The query embedding is compared with stored embeddings.
6. Cosine similarity is calculated.
7. The highest similarity score is selected.
8. The score is compared with the recognition threshold.

### 6. Unknown Rejection

The system does not automatically accept the highest-scoring person.

The best similarity score must meet the configured threshold.

Similarity >= 0.40 -> Known Person

Similarity < 0.40 -> Unknown Person

This prevents low-similarity matches from being automatically accepted.

---

## Similarity Matching

The system uses cosine similarity to compare face embeddings.

A higher similarity score indicates greater similarity between two face embeddings.

For each enrolled person, the system compares the query embedding against their stored face samples and retains the best similarity score for that person.

The person with the highest valid similarity score becomes the candidate match.

The threshold is then applied to determine whether the result should be accepted or rejected.

---

## Matching Threshold

The current recognition threshold is:

`0.40`

Multiple threshold values were evaluated using genuine and impostor face pairs.

On the evaluation dataset used for this project, thresholds from 0.30 to 0.55 produced zero false accepts and zero false rejects.

A threshold of 0.40 was selected as the operating threshold for the application.

The selected threshold is specific to the evaluation dataset and should not be interpreted as a universal threshold for all environments.

---

## Evaluation

The system was evaluated using a total of 28 face pairs.

| Evaluation Type | Number of Pairs |
|---|---:|
| Genuine Pairs | 12 |
| Impostor Pairs | 16 |
| Total Pairs | 28 |

### Genuine Pairs

Genuine pairs contain two images belonging to the same person.

### Impostor Pairs

Impostor pairs contain images belonging to different people.

### Results at Threshold = 0.40

| Metric | Result |
|---|---:|
| Genuine Pairs | 12 |
| Impostor Pairs | 16 |
| True Accepts | 12 |
| True Rejects | 16 |
| False Accepts | 0 |
| False Rejects | 0 |
| False Accept Rate (FAR) | 0% |
| False Reject Rate (FRR) | 0% |

### Observed Similarity Ranges

Genuine pairs: 0.5581 - 0.7659

Impostor pairs: 0.0233 - 0.2614

The evaluation dataset showed clear separation between genuine and impostor similarity scores.

> Note: These results are specific to the evaluation dataset used in this project and should not be interpreted as universal real-world performance.

---

## Failure Cases

The system may experience reduced performance under the following conditions:

### Poor Lighting

Very dark, overexposed, or unevenly illuminated images may affect detection and recognition.

### Large Pose Variations

Faces viewed from significantly different angles may produce lower similarity scores.

### Low-Quality Images

Blurred, noisy, or low-resolution images can reduce recognition performance.

### Face Occlusion

Masks, sunglasses, hands, or other objects covering parts of the face may affect recognition.

### Multiple Faces

The current application expects one face during enrollment and recognition. Images containing multiple faces are rejected.

### Unregistered Individuals

Individuals who are not present in the enrolled database should be rejected when their similarity score is below the configured threshold.

### Limited Evaluation Dataset

The evaluation was performed using a relatively small dataset. A larger and more diverse dataset would provide a stronger assessment of real-world performance.

---

## Limitations

- The evaluation dataset is relatively small.
- Recognition performance can vary with lighting, pose, image quality, and occlusion.
- The current application expects a single face for enrollment and recognition.
- JSON storage is suitable for a small prototype but is not intended for large-scale deployment.
- The system does not currently implement liveness detection.
- The selected threshold is based on the evaluation dataset used in this project.
- The system has not been evaluated across a large population or wide range of environmental conditions.

---

## Future Improvements

Potential improvements include:

- Liveness detection to reduce photo and screen spoofing
- Larger and more diverse evaluation datasets
- Multiple enrollment samples under different poses and lighting conditions
- More systematic threshold calibration
- Improved multiple-face handling
- Face-quality assessment before recognition
- Secure and encrypted biometric data storage
- Vector database support for large-scale deployments
- Improved performance under challenging environmental conditions

---

## Project Structure

    face-recognition-identification-system/
    |
    +-- database/
    |   +-- faces.json
    |
    +-- evaluation/
    |   +-- images/
    |   +-- pairs.csv
    |   +-- generate_pairs.py
    |   +-- evaluate.py
    |
    +-- models/
    |   +-- face_detection_yunet_2026may.onnx
    |   +-- face_recognition_sface_2021dec.onnx
    |
    +-- src/
    |   +-- __init__.py
    |   +-- database.py
    |   +-- detector.py
    |   +-- recognizer.py
    |   +-- utils.py
    |
    +-- app.py
    +-- requirements.txt
    +-- README.md
    +-- .gitignore

---

## Installation

### Clone the Repository

    git clone https://github.com/SpoortiIjari/face-recognition-identification-system.git

### Navigate to the Project

    cd face-recognition-identification-system

### Create a Virtual Environment

    python -m venv venv

### Activate the Virtual Environment on Windows

    venv\Scripts\activate

### Install Dependencies

    pip install -r requirements.txt

---

## Running the Application

Start the Streamlit application:

    streamlit run app.py

The application will open in the browser.

### Enrollment

Users can:

1. Enter a person's name.
2. Upload an image or capture an image using the camera.
3. Detect the face.
4. Generate the face embedding.
5. Store the embedding in the local database.

### Recognition

Users can:

1. Upload an image or capture an image using the camera.
2. Detect the face.
3. Generate the query embedding.
4. Compare it with enrolled embeddings.
5. Identify the closest matching person.
6. Reject the input as unknown when the similarity score is below the threshold.

---

## Cost

Development Cost: ₹0 / $0

The project uses locally running pre-trained models and open-source libraries without requiring paid AI APIs or cloud services.

---

## Privacy and Security

Face embeddings are stored locally in the project database.

This project is intended as an assignment prototype.

For production deployment, appropriate security controls should be implemented for:

- Biometric data protection
- Encryption
- Access control
- User consent
- Data retention
- Secure storage
- Secure deletion of biometric information

---

## Security Note

The public repository should not contain real personal biometric data.

For public deployment, the database should contain only test/demo data or an empty database.

Production systems should use appropriate security and privacy controls for biometric information.

---

## Conclusion

This project implements an end-to-end face recognition identification pipeline using YuNet for face detection and SFace for face embedding generation.

The system uses 128-dimensional face embeddings and cosine similarity for matching. A threshold-based rejection mechanism prevents low-similarity matches from being accepted as known identities.

The application supports:

- Face enrollment
- Image upload
- Camera capture
- Face detection
- Face alignment
- Face embedding generation
- Similarity-based recognition
- Unknown face rejection
- Evaluation using genuine and impostor pairs

The system demonstrates a complete face recognition workflow using open-source tools with a development cost of ₹0/$0.
