import cv2
import numpy as np
import random


def fast_sample_clip(video_path, clip_len=16, frame_size=112):
    """
    fixed-length clip from a video.

    Args:
        video_path (str): Path to input video file
        clip_len (int): Number of frames per clip
        frame_size (int): Height and width of resized frames

    Returns:
        np.ndarray or None:
            - Shape (T, H, W, C) if successful
            - None if video is too short or frame read fails
    """

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < clip_len or total_frames <= 0:
        cap.release()
        return None

   
    start = random.randint(0, total_frames - clip_len)
    indices = range(start, start + clip_len)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_size, frame_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) != clip_len:
        return None

    return np.array(frames)


if __name__ == "__main__":
    print("Video to clip preprocess done succesfully")