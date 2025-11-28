import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import Sequence
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

class AnimalImageGenerator(Sequence):
    def __init__(self, df, batch_size, target_size, num_classes, class_mapping,
                 shuffle=True, augment=False, aug_strength=None, max_size=500):
        super(AnimalImageGenerator, self).__init__()

        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.target_size = target_size
        self.max_size = max_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = num_classes
        self.aug_strength = aug_strength
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
        self.class_mapping = class_mapping

        print("---")
        print(f"Total images: {len(self.df)}")
        print(f"Num classes: {self.num_classes}")
        print("---")

    def __len__(self):
        return len(self.df) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        images, labels = [], []

        for _, row in batch_df.iterrows():
            label = row.label
            index = self.class_mapping[label] # TODO: Dont know for sure if this will work
            repeats = 1

            # minority-boost augementation
            if self.augment and self.aug_strength is not None:
                repeats = int(self.aug_strength[label])

            for _ in range(repeats):

                # Image loading
                img = cv2.imread(row.filepath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Outlier detection and processing
                if is_outlier(img):
                    img = center_pad_or_crop(img, self.target_size[0], self.max_size)
                else:
                    img = cv2.resize(img, self.target_size)

                # Augmentation
                if self.augment:
                    img = augment(img, self.target_size)

                # ImageNet Scaling
                img = tf.keras.applications.efficientnet.preprocess_input(img)
                images.append(img)

                # One-hot Encoding
                oh = np.zeros(self.num_classes, dtype="float32")
                oh[index] = 1.0
                labels.append(oh)

        return np.array(images), np.array(labels)
