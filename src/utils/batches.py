import cv2
import tensorflow as tf
import numpy as np
import torch

from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


#### ---------------------------------------------
#### Funkcie tykajuce sa uprave dat
#### ---------------------------------------------

# riesenie oulierov
# SMALL - vytvorenie black canvasu 256x256 a vlozenie obrazka do stredu (zachovame ratio)
# LARGE - zmensenie ak max_dim > 500, vycentrovanie do stredu a orezanie bokov (predpokladame ze hlavne features su v strede)

def center_pad_or_crop(img, target, max_size):
    img = tf.cast(img, tf.uint8).numpy()
    h, w, _ = img.shape

    # 1) Downscale extrémne veľkých obrázkov
    max_dim = max(h, w)
    if max_dim > max_size:
        scale = max_size / max_dim
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = tf.image.resize(img, (new_h, new_w), antialias=True).numpy().astype(np.uint8)
        h, w = new_h, new_w

    # 2) Teraz cropneme, ak treba
    if h > target:
        top = (h - target) // 2
        img = img[top:top + target, :, :]
        h = target

    if w > target:
        left = (w - target) // 2
        img = img[:, left:left + target, :]
        w = target

    # 3) Padding ak menšie
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    y0 = (target - h) // 2
    x0 = (target - w) // 2
    canvas[y0:y0+h, x0:x0+w] = img

    return canvas

def augment(img, target_size):
    # always cast to float for TF augment ops
    img = tf.cast(img, tf.float32)

    # standard augment
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)

    # random zoom (crop)
    if tf.random.uniform(()) > 0.5:

        # GET SHAPE FROM NUMPY, NOT TF
        h, w = img.shape[0], img.shape[1]

        crop_frac = tf.random.uniform((), 0.85, 1.0)

        new_h = int(crop_frac * h)
        new_w = int(crop_frac * w)

        img = tf.image.resize_with_crop_or_pad(img, new_h, new_w)
        img = tf.image.resize(img, target_size)

    # output back to uint8 numpy
    return img.numpy().astype(np.uint8)

def is_outlier(img):
    h, w = img.shape[:2]
    aspect = w / h
    area = w * h

    return (
        aspect < 0.5 or
        aspect > 2.0 or
        area < 50000 or
        area > 2_000_000
    )
def create_sampler(df, class_mapping):
    #weigths for underrepresented classes
    labels = df["label"].map(class_mapping).values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

class AnimalDataset(Dataset):
    def __init__(self, df, class_mapping, target_size=(256, 256), augment=False):
        self.filepaths = df["filepath"].values
        self.labels = df["label"].map(class_mapping).values
        self.target_size = target_size
        self.augment = augment

        # image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(target_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]

        # read image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        label = int(self.labels[idx])
        return img, label

class SingleImageInput():
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    def read_image(self, path) -> torch.Tensor:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img