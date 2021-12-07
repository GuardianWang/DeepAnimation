import tensorflow as tf
from tensorflow.keras import Model
from pathlib import Path


def save_weights(model: Model, **kwargs):
    p = Path(f"ckpts/{kwargs['name']}")
    p.mkdir(parents=True, exist_ok=True)
    p = p / parse_model_name(**kwargs)
    model.save_weights(str(p))


def load_weights(model: Model, input_shape, **kwargs):
    p = Path(f"ckpts/{kwargs['name']}")
    p = p / parse_model_name(**kwargs)

    data = tf.zeros(input_shape)
    model(data)
    model.load_weights(str(p))


def parse_model_name(**kwargs):
    return f"model_{kwargs['name']}_epoch_{kwargs['epoch']:04d}_batch_{kwargs['batch']:04d}"
