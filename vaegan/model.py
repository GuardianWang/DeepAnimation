"""
For normal image formats
"""
import tensorflow as tf
from tensorflow.keras.applications import VGG16, vgg16
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
        self.motion_encoder = None  # Encoder(latent_size=motion_latent_size)

        self.concat = Concatenate()

        self.optim_e = Adam(1e-4)
        self.optim_g = Adam(1e-4)
        self.optim_d = Adam(3e-5)

    def encode_image(self, x, training=True, mask=None):
        return self.content_encoder(x, training=training, mask=mask)

    def encode_motion(self, x, training=True, mask=None):
        return self.motion_encoder(x, training=training, mask=mask)

    def generate(self, x, m=None, training=True, mask=None):
        is_x_image = True if len(x.shape) == 4 else False
        is_m_image = True if m is not None and len(m.shape) == 4 else False

        if is_x_image:
            x, _, _ = self.encode_image(x, training=training, mask=mask)

        if m is None:
            m = tf.random.normal([x.shape[0], self.motion_latent_size])
        elif is_m_image:
            m, _, _ = self.encode_motion(m, training=training, mask=mask)

        x = self.concat([x, m])
        return self.generator(x, training=training, mask=mask)

    def random_generate(self, batch_sz, training=True, mask=None):
        x = tf.random.normal([batch_sz, self.content_latent_size + self.motion_latent_size])
        return self.generator(x, training=training, mask=mask)

    def discriminate(self, x, training=True, mask=None):
        return self.discriminator(x, training=training, mask=mask)

    @property
    def generator_encoder_trainable_vars(self):
        res = []
        res.extend(self.generator.trainable_variables)
        res.extend(self.content_encoder.trainable_variables)
        res.extend(self.motion_encoder.trainable_variables)
        return res

    @property
    def encoder_trainable_params(self):
        res = []
        res.extend(self.content_encoder.trainable_variables)
        if self.motion_encoder is not None:
            res.extend(self.motion_encoder.trainable_variables)
        return res


class EncoderHead(Model):
    def __init__(self, latent_size=100):
        super().__init__()
        self.latent_size = latent_size
        self.mu_layer = Dense(self.latent_size)
        self.logvar_layer = Dense(self.latent_size)

    def call(self, x, training=True, mask=None):
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        x = reparametrize(mu, logvar)

        return x, mu, logvar


class Generator(Model):
    def __init__(self):
        super().__init__()
        self.model = Sequential([
            Dense(7 * 7 * 256),
            Reshape((7, 7, 256)),
            Conv2DTranspose(128, (5, 5), padding='same', use_bias=False),  # [7, 7, 128]
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(64, (5, 5), (2, 2), padding='same', use_bias=False),  # [14, 14, 64]
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(32, (5, 5), (2, 2), padding='same', use_bias=False),  # [28, 28, 32]
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(16, (5, 5), (2, 2), padding='same', use_bias=False),  # [56, 56, 16]
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(3, (5, 5), (2, 2), padding='same', use_bias=False, activation='tanh'),  # [112, 112, 3]
        ])

    def call(self, x, training=True, mask=None):
        return self.model(x)


class Discriminator(Model):
    def __init__(self, latent_size=100):
        super().__init__()
        self.latent_size = latent_size
        self.base_model = make_discriminator_encoder_base()
        self.discriminator_head = Dense(1)

    def call(self, x, training=True, mask=None):
        x = self.base_model(x)
        return self.discriminator_head(x)


class Encoder(Model):
    def __init__(self, latent_size=100):
        super().__init__()
        self.latent_size = latent_size
        self.base_model = make_discriminator_encoder_base()
        self.encoder_head = EncoderHead(latent_size)

    def call(self, x, training=True, mask=None):
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

    def call(self, x, training=True, mask=None):
        x *= 255.
        x = vgg16.preprocess_input(x)
        return self.model(x)


@tf.function
def reparametrize(mu, logvar):
    z = tf.random.normal(mu.shape)
    z = mu + z * tf.exp(logvar * 0.5)
    return z


def make_discriminator_encoder_base():
    model = Sequential([
        Conv2D(32, (5, 5), (2, 2), padding='same'),  # [56, 56, 32]
        BatchNormalization(),
        LeakyReLU(),
        # Dropout(0.3),
        Conv2D(64, (3, 3), (2, 2), padding='same'),  # [28, 28, 64]
        BatchNormalization(),
        LeakyReLU(),
        # Dropout(0.3),
        Conv2D(128, (3, 3), (2, 2), padding='same'),  # [14, 14, 128]
        BatchNormalization(),
        LeakyReLU(),
        # Dropout(0.3),
        Conv2D(256, (3, 3), (2, 2), padding='same'),  # [7, 7, 256]
        BatchNormalization(),
        LeakyReLU(),
        # Dropout(0.3),
        Conv2D(512, (3, 3), (2, 2), padding='same'),  # [3, 3, 512]
        BatchNormalization(),
        LeakyReLU(),
        # Dropout(0.3),
        Conv2D(1024, (3, 3), (2, 2), padding='same'),  # [1, 1, 1024]
        BatchNormalization(),
        LeakyReLU(),
        # Dropout(0.3),
        Flatten(),
        ])

    return model


if __name__ == "__main__":
    noise = tf.random.normal([1, 100])
    g = Generator()
    d = Discriminator()
    x = g(noise)
    print(x.shape)
    x = d(x)
    print(x)
    tf.keras.losses.binary_crossentropy()
