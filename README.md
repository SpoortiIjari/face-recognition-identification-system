# Face Recognition Identification System

A computer vision-based **Face Recognition Identification System** that detects faces, generates face embeddings, identifies enrolled individuals using similarity-based matching, and rejects unknown faces when the similarity score is below a defined threshold.

**Technology:** Python • OpenCV • YuNet • SFace • NumPy • Pandas • Streamlit

---

## Key Features

- Face detection using OpenCV YuNet
- Face embedding generation using OpenCV SFace
- Face enrollment with multiple samples per person
- Similarity-based face matching
- Unknown face rejection mechanism
- Configurable similarity threshold
- JSON-based local face database
- Streamlit web interface
- Evaluation using genuine and impostor face pairs
- FAR and FRR based threshold evaluation

---

## Application Screenshots

### Home Page

![Home Page](home.png)

### Face Enrollment

![Face Enrollment](enrollment.png)

### Face Recognition

![Face Recognition](recognition.png)

---

## System Architecture

    User
      |
      v
    Streamlit Web Interface
      |
      +-----------------------------+
      |                             |
      v                             v
    Enrollment                  Recognition
      |                             |
      v                             v
    YuNet Face Detection        YuNet Face Detection
      |                             |
      v                             v
    SFace Embedding              SFace Embedding
      |                             |
      v                             v
    Store Embedding              Compare Embeddings
      |                             |
      v                             v
    faces.json                   Similarity Score
                                    |
                                    v
                              Threshold = 0.40
                                    |
                         +----------+----------+
                         |                     |
                         v                     v
                       Known                 Unknown

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Face Detection | OpenCV YuNet |
| Face Recognition | OpenCV SFace |
| Numerical Processing | NumPy |
| Evaluation | Pandas |
| Frontend | Streamlit |
| Database | JSON |
| Model Format | ONNX |

---

## Models Used

### 1. YuNet Face Detector

**Model:** `face_detection_yunet_2026may.onnx`

YuNet is used to detect faces from input images or camera frames.

It provides the face bounding box and facial landmark information required for the recognition pipeline.

Model source:

https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

### 2. SFace Face Recognition Model

**Model:** `face_recognition_sface_2021dec.onnx`

SFace converts a detected face into a numerical feature vector called a **face embedding**.

In this project, the generated embedding has the shape:

`(1, 128)`

Model source:

https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

---

## How the System Works

The system follows these main steps:

### Step 1: Face Detection

The input image is passed to YuNet.

YuNet identifies the face region and provides the bounding box and facial landmarks.

### Step 2: Face Embedding

The detected face is passed to SFace.

SFace converts the face into a 128-dimensional numerical representation.

This representation is called a **face embedding**.

### Step 3: Enrollment

During enrollment:

1. User enters a person's name.
2. A face image is provided.
3. YuNet detects the face.
4. SFace generates the embedding.
5. The embedding is stored in `database/faces.json`.
6. Additional samples can be stored for the same person.

### Step 4: Recognition

During recognition:

1. A new face is provided.
2. YuNet detects the face.
3. SFace generates a new embedding.
4. The embedding is compared with stored embeddings.
5. Similarity scores are calculated.
6. The highest similarity score is selected.
7. The score is compared with the configured threshold.
8. The system returns either a known person's name or `Unknown`.

---

## Face Embeddings

A face embedding is a numerical representation of facial characteristics.

Instead of comparing two images directly pixel-by-pixel, the system compares their embeddings.

Example:

    Input Face
        |
        v
    SFace Model
        |
        v
    128-Dimensional Embedding
        |
        v
    [0.12, -0.04, 0.31, ...]

Two images of the same person should generally produce more similar embeddings than images of different people.

---

## Similarity Matching

The system uses similarity-based comparison between face embeddings.

A higher similarity score indicates that two embeddings are more similar.

Example:

    Same Person:
    Similarity = 0.6080

    Different Person:
    Similarity = 0.2036

The similarity score is **not a confidence percentage**.

It is a numerical similarity measurement used to determine whether the detected face matches an enrolled identity.

---

## Matching Threshold

The project currently uses:

**Threshold = 0.40**

Decision logic:

    If similarity >= 0.40
            |
            v
        Known Face

    If similarity < 0.40
            |
            v
          Unknown

The threshold was selected after testing different threshold values on the evaluation dataset.

A threshold of `0.40` produced `0% FAR` and `0% FRR` on the tested 28-pair evaluation dataset.

This result applies only to the evaluated dataset and should not be interpreted as universal performance.

---

## Evaluation

The evaluation dataset contains:

- 8 images
- 12 genuine pairs
- 16 impostor pairs
- 28 total pairs

### Genuine Pair

A genuine pair contains two images of the same person.

### Impostor Pair

An impostor pair contains images belonging to different people.

---

## Evaluation Results

At threshold `0.40`:

| Metric | Result |
|---|---:|
| Genuine Pairs | 12 |
| Impostor Pairs | 16 |
| Total Pairs | 28 |
| True Accepts | 12 |
| True Rejects | 16 |
| False Accepts | 0 |
| False Rejects | 0 |
| FAR | 0% |
| FRR | 0% |

### Observed Similarity Ranges

**Genuine pairs:**

- Minimum: `0.5581`
- Maximum: `0.7659`

**Impostor pairs:**

- Minimum: `0.0233`
- Maximum: `0.2614`

The evaluation shows a clear separation between genuine and impostor similarity scores for this dataset.

---

## Threshold Analysis

Different threshold values were evaluated.

| Threshold | FAR | FRR |
|---:|---:|---:|
| 0.20 | 12.50% | 0% |
| 0.25 | 6.25% | 0% |
| 0.30 | 0% | 0% |
| 0.35 | 0% | 0% |
| 0.40 | 0% | 0% |
| 0.45 | 0% | 0% |
| 0.50 | 0% | 0% |
| 0.55 | 0% | 0% |
| 0.60 | 0% | 8.33% |

The threshold of `0.40` is used in the current implementation because it produced zero false accepts and zero false rejects on the evaluated dataset.

---

## Failure Cases

The system may fail or produce an incorrect result under certain conditions.

### 1. Poor Lighting

Very dark or overexposed images may reduce face recognition quality.

### 2. Large Pose Changes

Faces viewed from extreme angles may produce less reliable embeddings.

### 3. Occlusion

Masks, sunglasses, hands, or other objects covering facial features can affect recognition.

### 4. Low Image Quality

Blurred or low-resolution images may reduce detection and recognition quality.

### 5. Multiple Faces

When multiple faces are present, the system must correctly process the detected face regions.

### 6. Similar-Looking Individuals

People with similar facial characteristics may produce relatively similar embeddings.

### 7. Limited Enrollment Samples

Recognition performance can be affected when only a small number of enrollment samples are stored for a person.

---

## Limitations

- The current database uses JSON storage.
- The system is designed as a recognition prototype.
- Performance depends on image quality and environmental conditions.
- The evaluation dataset is small.
- The threshold is dataset-dependent.
- No liveness detection is implemented.
- No advanced demographic or environmental bias evaluation has been performed.
- Large-scale production deployment would require a more robust database and security architecture.

---

## Future Improvements

Possible improvements include:

- Increase the size and diversity of the evaluation dataset.
- Store more enrollment samples per person.
- Use a production database such as PostgreSQL.
- Add liveness detection.
- Add camera-based real-time recognition.
- Improve handling of multiple faces.
- Add face alignment and preprocessing.
- Add automatic threshold calibration.
- Add ROC and precision-recall analysis.
- Add secure biometric data storage.
- Encrypt stored embeddings.
- Add authentication and authorization.
- Add audit logs.
- Improve performance for large face databases.

---

## Project Structure

    face-recognition-system/
    |
    +-- data/
    |
    +-- database/
    |   +-- faces.json
    |
    +-- evaluation/
    |   +-- evaluate.py
    |   +-- generate_pairs.py
    |   +-- images/
    |   +-- pairs.csv
    |   +-- results.csv
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
    +-- venv/
    |
    +-- app.py
    +-- requirements.txt
    +-- README.md
    +-- test_detection.py
    +-- test_embedding.py
    +-- test_enrollment.py
    +-- test_models.py
    +-- test_similarity.py

---

## Installation

### 1. Clone the Repository

    git clone https://github.com/SpoortiIjari/face-recognition-identification-system.git
    cd face-recognition-identification-system

### 2. Create a Virtual Environment

    python -m venv venv

### 3. Activate the Virtual Environment

Windows:

    venv\Scripts\activate

### 4. Install Dependencies

    pip install -r requirements.txt

---

## Requirements

The main dependencies are:

- opencv-python
- numpy
- pandas
- streamlit

The exact versions are available in `requirements.txt`.

---

## Running the Application

Start the Streamlit application using:

    streamlit run app.py

The application provides navigation for:

- Home
- Enrollment
- Recognition

---

## Face Enrollment

To enroll a person:

1. Open the Enrollment page.
2. Enter the person's name.
3. Provide an image containing the face.
4. The system detects the face using YuNet.
5. The face embedding is generated using SFace.
6. The embedding is stored in the database.
7. Additional samples can be enrolled for the same person.

The database is stored at:

`database/faces.json`

---

## Face Recognition

To recognize a person:

1. Open the Recognition page.
2. Provide an input image.
3. The system detects the face.
4. The face embedding is generated.
5. The embedding is compared with enrolled embeddings.
6. The highest similarity score is selected.
7. The score is compared against the threshold.
8. The system displays the matching identity or `Unknown`.

---

## Database

The face database stores embeddings in JSON format.

Conceptually:

    {
        "Person_Name": [
            [embedding values],
            [embedding values]
        ]
    }

Multiple embeddings can therefore be stored for a single enrolled person.

---

## Recognition Logic

The core recognition process can be summarized as:

    Input Image
         |
         v
    Face Detection
         |
         v
    Face Crop
         |
         v
    SFace Embedding
         |
         v
    Compare With Database
         |
         v
    Calculate Similarity
         |
         v
    Highest Similarity
         |
         v
    Compare With Threshold
         |
         +----------------------+
         |                      |
         v                      v
    Similarity >= 0.40    Similarity < 0.40
         |                      |
         v                      v
       Known                  Unknown

---

## Technical Design Decisions

### Why YuNet?

YuNet was selected as the face detector because it is lightweight and integrates directly with OpenCV.

### Why SFace?

SFace was selected because it provides face embeddings specifically designed for face recognition and can be used through OpenCV.

### Why Embeddings?

Embeddings allow the system to represent faces numerically and compare them using mathematical similarity rather than comparing raw image pixels.

### Why a Threshold?

A threshold is required to implement unknown-face rejection.

Without a threshold, the system could always return the closest enrolled identity even when the input face does not belong to anyone in the database.

### Why JSON?

JSON was used for the prototype because it is simple and easy to inspect during development.

A production system should use a secure and scalable database.

---

## Important Terminology

### Face Detection

The process of locating faces inside an image.

### Face Recognition

The process of determining which enrolled identity corresponds to a detected face.

### Face Embedding

A numerical representation of facial characteristics generated by a recognition model.

### Similarity Score

A numerical value indicating how similar two face embeddings are.

### Threshold

The minimum similarity score required to accept a recognition result.

### Genuine Pair

Two images belonging to the same person.

### Impostor Pair

Two images belonging to different people.

### False Accept

An unknown or incorrect identity is incorrectly accepted as a match.

### False Reject

A genuine identity is incorrectly rejected.

### FAR

**False Acceptance Rate**

`FAR = False Accepts / Total Impostor Attempts × 100`

### FRR

**False Rejection Rate**

`FRR = False Rejects / Total Genuine Attempts × 100`

---

## Privacy and Security

Face embeddings are biometric-related data and should be handled carefully.

For a production system:

- Do not expose the face database publicly.
- Protect stored embeddings.
- Use access control.
- Encrypt sensitive data.
- Avoid storing unnecessary raw face images.
- Obtain appropriate consent before collecting biometric data.
- Use secure authentication.
- Maintain audit logs.
- Follow applicable privacy and data-protection requirements.

The public repository should not contain private or sensitive biometric data.

---

## Cost

This project was developed using free and open-source tools.

| Item | Cost |
|---|---:|
| Development Cost | ₹0 |
| Software Cost | ₹0 |
| Model Cost | ₹0 |

---

## Testing

The project includes separate test scripts for validating individual components.

### Model Testing

`test_models.py`

Used to verify that the YuNet and SFace models load successfully.

### Face Detection Testing

`test_detection.py`

Used to verify face detection.

### Embedding Testing

`test_embedding.py`

Used to verify face embedding generation.

### Similarity Testing

`test_similarity.py`

Used to verify similarity calculation between face embeddings.

### Enrollment Testing

`test_enrollment.py`

Used to verify that face embeddings can be stored in the database.

These tests validate:

- Model loading
- Face detection
- Embedding generation
- Similarity calculation
- Face enrollment
- Recognition behavior

---

## Project Information

**Project:** Face Recognition Identification System

**Domain:** Computer Vision / Artificial Intelligence

**Programming Language:** Python

**Frontend:** Streamlit

**Face Detector:** YuNet

**Face Recognition Model:** SFace

**Database:** JSON

**Matching Threshold:** 0.40

**Development Cost:** ₹0

---

## Conclusion

This project demonstrates an end-to-end face recognition identification pipeline using OpenCV YuNet for face detection and OpenCV SFace for face embedding generation.

The system supports face enrollment, similarity-based identification, and unknown-face rejection through a configurable similarity threshold.

The evaluation performed on the available 28-pair dataset produced `0% FAR` and `0% FRR` at a threshold of `0.40`. These results are specific to the tested dataset and should not be treated as a universal accuracy measurement.

The project provides a foundation that can be extended with real-time recognition, liveness detection, secure biometric storage, larger-scale evaluation, and production-grade database infrastructure.

---

## GitHub Repository

https://github.com/SpoortiIjari/face-recognition-identification-system

---

## Author

**Spoorti Ijari**

Computer Science Engineering Student

**Project Domain:** Artificial Intelligence / Computer Vision
