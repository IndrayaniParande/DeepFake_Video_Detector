import os
import numpy as np
import torch
from torch.utils.data import Dataset


def print_clip_summary(clips_root):
    real_dir = os.path.join(clips_root, "real")
    fake_dir = os.path.join(clips_root, "fake")

    real_clips = [f for f in os.listdir(real_dir) if f.endswith(".npy")]
    fake_clips = [f for f in os.listdir(fake_dir) if f.endswith(".npy")]

    print("CLIP COUNT SUMMARY")
    print(f" REAL clips : {len(real_clips)}")
    print(f" FAKE clips : {len(fake_clips)}")
    print(f" TOTAL clips: {len(real_clips) + len(fake_clips)}")


class DeepfakeClipDataset(Dataset):
    """
    Each clip is expected in shape:
    (T, H, W, C) and converted to (C, T, H, W)
    """

    def __init__(self, clips_root, transform=None):
        self.samples = []
        self.transform = transform

        real_dir = os.path.join(clips_root, "real")
        fake_dir = os.path.join(clips_root, "fake")

        for f in os.listdir(real_dir):
            if f.endswith(".npy"):
                self.samples.append((os.path.join(real_dir, f), 0))  # REAL = 0

        for f in os.listdir(fake_dir):
            if f.endswith(".npy"):
                self.samples.append((os.path.join(fake_dir, f), 1))  # FAKE = 1

        print(f"Loaded {len(self.samples)} total clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        clip = np.load(path) 
        clip = torch.tensor(clip, dtype=torch.float32)
        clip = clip.permute(3, 0, 1, 2) 
        clip = clip / 255.0

        if self.transform:
            clip = self.transform(clip)

        return clip, label


if __name__ == "__main__":
    print("Dataset module loaded successfully")
