from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.experimental import CosineDecayRestarts


def get_adam():
    lr_schedule = get_scheduler('cos')
    optimizer = Adam(learning_rate=lr_schedule)
    return optimizer


def get_scheduler(name="cos"):
    lr = None
    if name == "cos":
        lr = CosineDecayRestarts(
            initial_learning_rate=1e-3,
            first_decay_steps=150,
            t_mul=2.,
            m_mul=0.95
        )
    elif name == "exp":
        lr = ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=5000,
            decay_rate=0.9,
            staircase=True,
        )
    else:
        raise NotImplementedError("scheduler not defined.")

    return lr


if __name__ == '__main__':
    import tensorflow as tf
    lrs = CosineDecayRestarts(1., 5, 3., 0.9)
    for i in range(40):
        print(i, lrs(tf.convert_to_tensor([i], dtype=tf.int32)))
