from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def get_adam():
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=5000,
        decay_rate=0.9,
        staircase=True,
    )
    optimizer = Adam(learning_rate=lr_schedule)
    return optimizer
