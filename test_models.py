import cv2

# Model paths
yunet_model = "models/face_detection_yunet_2026may.onnx"
sface_model = "models/face_recognition_sface_2021dec.onnx"

print("Loading YuNet...")

detector = cv2.FaceDetectorYN.create(
    yunet_model,
    "",
    (320, 320)
)

print("✓ YuNet loaded successfully!")

print("Loading SFace...")

recognizer = cv2.FaceRecognizerSF.create(
    sface_model,
    ""
)

print("✓ SFace loaded successfully!")

print("\nAll models loaded successfully! 🎉")
