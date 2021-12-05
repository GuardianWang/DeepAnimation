import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2
from pathlib import Path
from math import floor, ceil
import re


def load_test_dataset():
    x = tf.data.Dataset.from_tensor_slices(tf.random.normal([4, 112, 112, 3]))
    x = x.batch(2)

    return x


class DataSet(Sequence):
    def __init__(self, img_paths, batch_size=64, shuffle=False):
        super().__init__()
        self.img_paths = img_paths
        self.first_frame_paths = list(map(self.get_first_frame_path, self.img_paths))
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_first_frame_path(self, img_path):
        return re.sub(r"-.", "-0", img_path)

    def __len__(self):
        return int(ceil(len(self.img_paths) / self.batch_size))

    def __getitem__(self, item):
        l = self.batch_size * item
        r = self.batch_size * (item + 1)
        data = []
        data_first = []
        for path, path2 in zip(self.img_paths[l: r], self.first_frame_paths[l: r]):
            img = self.read_svg(path)
            img_first = self.read_svg(path2)
            data.append(img)
            data_first.append(img_first)
        return tf.convert_to_tensor(data), tf.convert_to_tensor(data_first)

    def read_svg(self, path):
        img = cv2.imread(path).astype(float)  # bgr
        return img


def load_list_ds():
    frame_dir = r"C:\Users\zichu\Downloads\transformed_svgs"
    p = Path(frame_dir)
    # list_ds = tf.data.Dataset.from_tensor_slices(list(p.glob("*.svg")))
    list_ds = tf.data.Dataset.list_files(str(p / "*.svg"))

    return list_ds


def parse_image(path):
    img = cv2.imread(path)  # bgr
    img = tf.convert_to_tensor(img)
    return img


def process_path(path):
    # img = tf.io.decode_image(path)
    img = parse_image(tf.strings.strip(path).numpy())

    return img


def configure_for_performance(ds, batch_size=64):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=len(ds))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


if __name__ == "__main__":
    frame_dir = r"C:\Users\zichu\Downloads\transformed_svgs"
    p = Path(frame_dir)
    ds = DataSet(list(map(str, p.glob("*.svg"))))
    # ds = ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print()
