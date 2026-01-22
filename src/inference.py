import cv2
import numpy as np
import onnxruntime as ort

def load_onnx_model(onnx_path):
    """
    Loads ONNX model using ONNX Runtime.
    """
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    return session


def preprocess_video(video_path, num_frames=16, size=(112, 112)):
    """
    Converts a video into a normalized input tensor
    suitable for ONNX inference.
    """

    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    video_data = np.array(frames).astype(np.float32)
    video_data = video_data / 255.0
    video_data = np.transpose(video_data, (3, 0, 1, 2))  # (C, T, H, W)
    video_data = np.expand_dims(video_data, axis=0)      # (1, C, T, H, W)

    return video_data



def predict_video(session, input_tensor):
    """
    Runs ONNX inference and returns prediction,
    confidence score, and probabilities.
    """

    inputs = {
        session.get_inputs()[0].name: input_tensor
    }

    outputs = session.run(None, inputs)
    logits = outputs[0]

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    prediction = np.argmax(probs)
    label = "FAKE" if prediction == 1 else "REAL"
    confidence = probs[0][prediction] * 100

    return label, confidence, probs
