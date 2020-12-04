import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Conv2DTranspose

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE():
    def __init__(self, data_shape, latent_dim, batch_size=50, epochs=50,
                 learning_rate=0.0005,
                 optimizer=tf.keras.optimizers.Adam,
                 verbose=True):
        self.model_name = 'VAE'
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.verbose = verbose

        self.init_encoder()
        self.init_decoder()
        self.init_full_model()

    def init_encoder(self):
        self.encoder_inputs = Input(shape=self.data_shape, name='encoder_input')
        x = Conv2D(32, 3, activation="relu", strides=2, padding="same", name='Conv1_encoder')(self.encoder_inputs)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same", name='Conv2_encoder')(x)
        x = Flatten()(x)
        x = Dense(16, activation="relu", name='dense1_encoder')(x)
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        
        z = Sampling()([z_mean, z_log_var])
        self.encoder = tf.keras.Model(self.encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.summary()

    def init_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        x = Dense(7 * 7 * 64, activation="relu", name='dense1_decoder')(latent_inputs)
        x = Reshape((7, 7, 64))(x)
        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        self.decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same", name='decoder_output')(x)
        self.decoder = tf.keras.Model(latent_inputs, self.decoder_outputs, name="decoder")
        self.decoder.summary()

    def init_full_model(self):
        _, _, z = self.encoder(self.encoder_inputs)
        out = self.decoder(z)
        self.full_model = tf.keras.Model(self.encoder_inputs, out)
        self.full_model.compile(optimizer=self.optimizer(lr=self.learning_rate), loss=None)
        self.full_model.summary()

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.full_model.trainable_weights)
        self.full_model.optimizer.apply_gradients(zip(grads, self.full_model.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
    def train_vae(self, data, save_model: bool = False):
        self.full_model.train_step = self.train_step
        self.full_model.fit(
            x=data,
            shuffle=True,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
        )

        if save_model:
            self.full_model.save('vae', save_format='tf')
            self.encoder.save('vae_encoder', save_format='tf')
            self.decoder.save('vae_decoder', save_format='tf')

