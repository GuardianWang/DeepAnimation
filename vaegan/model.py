"""
For normal image formats
"""
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.layers import LeakyReLU, Reshape, Conv2DTranspose, Conv2D, BatchNormalization, \
    Dense, Flatten, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam


class VAEGAN(Model):
    def __init__(self, content_latent_size=100, motion_latent_size=100):
        super().__init__()
        self.content_latent_size = content_latent_size
        self.motion_latent_size = motion_latent_size
        self.generator = Generator()
        self.discriminator = Discriminator(latent_size=content_latent_size)
        self.content_encoder = Encoder(latent_size=content_latent_size)
        self.motion_encoder = Encoder(latent_size=motion_latent_size)

        self.concat = Concatenate()

        self.optim_g = Adam(1e-4)
        self.optim_d = Adam(3e-5)

    def encode_image(self, x):
        return self.content_encoder(x)

    def encode_motion(self, x):
        return self.motion_encoder(x)

    @tf.function
    def generate(self, x, is_x_image=True, m=None, is_m_image=True):
        if is_x_image:
            x = self.encode_image(x)

        if m is None:
            m = tf.random.normal([x.shape[0], self.motion_latent_size])
        elif is_m_image:
            m = self.encode_motion(m)

        x = self.concat([x, m])
        return self.generator(x)

    def discriminate(self, x):
        return self.discriminator(x)

    @property
    def generator_encoder_trainable_vars(self):
        res = []
        res.extend(self.generator.trainable_variables)
        res.extend(self.content_encoder.trainable_variables)
        res.extend(self.motion_encoder.trainable_variables)
        return res


class EncoderHead(Model):
    def __init__(self, latent_size=100):
        super().__init__()
        self.latent_size = latent_size
        self.mu_layer = Dense(self.latent_size)
        self.logvar_layer = Dense(self.latent_size)

    def call(self, x):
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        x = reparametrize(mu, logvar)

        return x


class Generator(Model):
    def __init__(self):
        super().__init__()
        self.model = Sequential([
            Dense(7 * 7 * 256),
            Reshape((7, 7, 256)),
            Conv2DTranspose(128, (5, 5), padding='same', use_bias=False),  # [7, 7, 128]
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(64, (3, 3), (2, 2), padding='same', use_bias=False),  # [14, 14, 64]
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(32, (3, 3), (2, 2), padding='same', use_bias=False),  # [28, 28, 32]
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(16, (3, 3), (2, 2), padding='same', use_bias=False),  # [56, 56, 16]
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(3, (3, 3), (2, 2), padding='same', use_bias=False, activation='tanh'),  # [112, 112, 3]
        ])

    def call(self, x):
        return self.model(x)


class Discriminator(Model):
    def __init__(self, latent_size=100):
        super().__init__()
        self.latent_size = latent_size
        self.base_model = make_discriminator_encoder_base()
        self.discriminator_head = Dense(1)

    def call(self, x):
        x = self.base_model(x)
        return self.discriminator_head(x)


class Encoder(Model):
    def __init__(self, latent_size=100):
        super().__init__()
        self.latent_size = latent_size
        self.base_model = make_discriminator_encoder_base()
        self.encoder_head = EncoderHead(latent_size)

    def call(self, x):
        x = self.base_model(x)
        return self.encoder_head(x)


class VGG(Model):
    def __init__(self, layer_names=('block5_conv2',)):
        super().__init__()

        self.model = self.make_layers(layer_names)
        self.model.trainable = False
        self.trainable = False

    def make_layers(self, layer_names):
        vgg = VGG16(include_top=False)
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def call(self, x):
        return self.model(x)


@tf.function
def reparametrize(mu, logvar):
    z = tf.random.normal(mu.shape)
    z = mu + z * tf.exp(logvar * 0.5)
    return z


def make_discriminator_encoder_base():
    model = Sequential([
            Conv2D(32, (5, 5), (2, 2), padding='same'),  # [56, 56, 32]
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(64, (3, 3), (2, 2), padding='same'),  # [28, 28, 64]
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(128, (3, 3), (2, 2), padding='same'),  # [14, 14, 128]
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(256, (3, 3), (2, 2), padding='same'),  # [7, 7, 256]
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(512, (3, 3), (2, 2), padding='same'),  # [3, 3, 512]
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(1024, (3, 3), (2, 2), padding='same'),  # [1, 1, 1024]
            LeakyReLU(),
            Dropout(0.3),
            Flatten(),
        ])

    return model


if __name__ == "__main__":
    noise = tf.random.normal([1, 100])
    g = Generator()
    d = Discriminator()
    x = g(noise)
    x = d(x)
    print(x)
