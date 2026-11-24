
import io
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# 1. MODEL CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_movie_object_detection.onnx"

CLASS_NAMES = {
    0: "person",
    1: "tree",
    2: "window",
    3: "light",
    4: "building",
    5: "tie",
    6: "chair",
    7: "vehicle"
}

IMAGE_SIZE = 640
MAX_IMAGE_SIZE = 10 * 1024 * 1024


# --------------------------------------------------
# 2. LOAD ONNX MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("YOLOv8 ONNX model loaded successfully!")
print("Execution provider:", session.get_providers())


# --------------------------------------------------
# 3. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Movie Object Detection API",
    description="Object detection using trained YOLOv8 ONNX model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):

    height, width = image.shape[:2]

    ratio = min(
        IMAGE_SIZE / height,
        IMAGE_SIZE / width
    )

    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    dw = (IMAGE_SIZE - new_width) / 2
    dh = (IMAGE_SIZE - new_height) / 2

    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    input_tensor = rgb.astype(np.float32) / 255.0

    input_tensor = np.transpose(
        input_tensor,
        (2, 0, 1)
    )

    input_tensor = np.expand_dims(
        input_tensor,
        axis=0
    )

    return input_tensor, ratio, left, top


# --------------------------------------------------
# 5. NON-MAXIMUM SUPPRESSION
# --------------------------------------------------

def apply_nms(boxes, scores, class_ids, iou_threshold):

    selected = []

    order = np.argsort(scores)[::-1]

    while len(order) > 0 and len(selected) < 100:

        current = order[0]
        selected.append(current)

        remaining = order[1:]

        if len(remaining) == 0:
            break

        x1 = np.maximum(
            boxes[current, 0],
            boxes[remaining, 0]
        )

        y1 = np.maximum(
            boxes[current, 1],
            boxes[remaining, 1]
        )

        x2 = np.minimum(
            boxes[current, 2],
            boxes[remaining, 2]
        )

        y2 = np.minimum(
            boxes[current, 3],
            boxes[remaining, 3]
        )

        intersection = (
            np.maximum(0, x2 - x1) *
            np.maximum(0, y2 - y1)
        )

        area_current = (
            (boxes[current, 2] - boxes[current, 0]) *
            (boxes[current, 3] - boxes[current, 1])
        )

        area_remaining = (
            (boxes[remaining, 2] - boxes[remaining, 0]) *
            (boxes[remaining, 3] - boxes[remaining, 1])
        )

        union = area_current + area_remaining - intersection

        iou = intersection / np.maximum(union, 1e-7)

        # Suppress overlapping boxes of the same class only
        keep = (
            (iou <= iou_threshold) |
            (class_ids[remaining] != class_ids[current])
        )

        order = remaining[keep]

    return selected


# --------------------------------------------------
# 6. OBJECT DETECTION
# --------------------------------------------------

def detect_objects(
    image,
    confidence_threshold=0.25,
    iou_threshold=0.45
):

    height, width = image.shape[:2]

    tensor, ratio, pad_x, pad_y = preprocess_image(image)

    outputs = session.run(
        None,
        {input_name: tensor}
    )[0]

    predictions = outputs[0]

    # YOLOv8 output: 4 box values + 8 class scores
    if predictions.shape[0] == 4 + len(CLASS_NAMES):
        predictions = predictions.T
    elif predictions.shape[1] != 4 + len(CLASS_NAMES):
        raise RuntimeError(
            f"Unexpected YOLO output shape: {outputs.shape}"
        )

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1
    )

    scores = np.max(
        class_scores,
        axis=1
    )

    valid = scores >= confidence_threshold

    boxes_xywh = boxes_xywh[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if len(scores) == 0:
        return []

    boxes = np.zeros_like(boxes_xywh)

    # Convert center coordinates to corner coordinates
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    selected = apply_nms(
        boxes,
        scores,
        class_ids,
        iou_threshold
    )

    detections = []

    for index in selected:

        x1, y1, x2, y2 = boxes[index]

        # Convert coordinates back to original image
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        x1 = float(np.clip(x1, 0, width))
        y1 = float(np.clip(y1, 0, height))
        x2 = float(np.clip(x2, 0, width))
        y2 = float(np.clip(y2, 0, height))

        class_id = int(class_ids[index])

        detections.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": round(float(scores[index]), 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        })

    return detections


# --------------------------------------------------
# 7. API ROUTES
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Movie Object Detection API is running!",
        "model": "YOLOv8 ONNX",
        "task": "Object Detection",
        "available_classes": list(CLASS_NAMES.values()),
        "execution_provider": "CPUExecutionProvider"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": session is not None,
        "model": "YOLOv8 ONNX"
    }


@app.post("/predict/dlp")
async def predict_dlp(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45
):

    if confidence_threshold < 0 or confidence_threshold > 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence threshold must be between 0 and 1."
        )

    if iou_threshold < 0 or iou_threshold > 1:
        raise HTTPException(
            status_code=400,
            detail="IoU threshold must be between 0 and 1."
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image."
        )

    try:
        detections = detect_objects(
            image,
            confidence_threshold,
            iou_threshold
        )

    except Exception as error:
        print("Detection error:", str(error))
        raise HTTPException(
            status_code=500,
            detail="Object detection failed."
        )

    return {
        "model": "YOLOv8 ONNX",
        "task": "Object Detection",
        "image": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0])
        },
        "total_objects": len(detections),
        "detections": detections
    }


print("Movie Object Detection API initialized!")
