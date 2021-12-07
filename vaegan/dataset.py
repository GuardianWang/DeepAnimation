import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2
from pathlib import Path
from math import floor, ceil
import re


def load_toy_dataset():
    x = tf.data.Dataset.from_tensor_slices(tf.random.normal([4, 112, 112, 3]))
    x = x.batch(2)

    return x


class DataSet(Sequence):
    def __init__(self, img_paths, batch_size=4, shuffle=False, h=112, w=112):
        super().__init__()
        self.img_paths = img_paths
        self.first_frame_paths = list(map(self.get_first_frame_path, self.img_paths))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = [h, w]

    def get_first_frame_path(self, img_path):
        return re.sub(r"-\d+\.", "-0.", img_path)

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
            img = tf.image.resize(img, self.img_size)
            img_first = tf.image.resize(img_first, self.img_size)
            data.append(img)
            data_first.append(img_first)
        return tf.convert_to_tensor(data) / 255., tf.convert_to_tensor(data_first) / 255.

    def read_svg(self, path):
        img = cv2.imread(path).astype(float)  # bgr
        return img


def load_list_ds(frame_dir):
    p = Path(frame_dir)
    list_ds = tf.data.Dataset.list_files(str(p / "*.png"))

    return list_ds


def parse_image(path, find_first_frame=False):
    if find_first_frame:
        path = tf.strings.regex_replace(path, r"-\d+\.", "-0.")
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    return img


def process_path(path):
    img = parse_image(path)
    first_img = parse_image(path, True)

    return img, first_img


def configure_for_performance(ds, batch_size=256):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=8)
    return ds


def make_dataset(frame_dir):
    ds = load_list_ds(frame_dir)
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1.)
    resize = tf.keras.layers.experimental.preprocessing.Resizing(112, 112)
    ds = ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda x, y: (rescale(resize(x)), rescale(resize(y))), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = configure_for_performance(ds)
    return ds


if __name__ == "__main__":
    # frame_dir = r"C:\Users\zichu\Downloads\transformed_svgs"
    frame_dir = r"C:/Users/zichu/Downloads/icons/target_svg/pngs"
    # frame_dir = "../icon_svg/pngs"
    # p = Path(frame_dir)
    # ds = DataSet(list(map(str, p.glob("*.png"))))
    ds = make_dataset()
    print()
