import tensorflow as tf
from tensorflow.keras import Model
from pathlib import Path
from datetime import datetime


def save_weights(model: Model, **kwargs):
    if 'path' in kwargs:
        p = Path(kwargs['path'])
    else:
        p = Path()
    p = p / Path(f"ckpts/{kwargs['name']}")
    p.mkdir(parents=True, exist_ok=True)
    p = p / parse_model_name(**kwargs)
    model.save_weights(str(p))


def load_weights(model: Model, input_shape, **kwargs):
    if 'path' in kwargs:
        p = Path(kwargs['path'])
    else:
        p = Path()
    p = p / Path(f"ckpts/{kwargs['name']}")
    p = p / parse_model_name(**kwargs)

    data = tf.zeros(input_shape)
    model(data)
    model.load_weights(str(p))


def get_writer():
    p = Path("logs")
    p.mkdir(parents=True, exist_ok=True)
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = p / cur_time / "train"
    train_writer = tf.summary.create_file_writer(str(p))

    return train_writer


def parse_model_name(**kwargs):
    return f"model_{kwargs['name']}_epoch_{kwargs['epoch']:04d}_batch_{kwargs['batch']:04d}"
