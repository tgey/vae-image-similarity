from time import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from vae import VAE

def plot_label_clusters(encoder, decoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

def main(training: bool = False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    labels = np.concatenate([y_train, y_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

    if training:
        vae = VAE(data_shape=(28,28,1), latent_dim=2, epochs=20, batch_size=128, optimizer=tf.keras.optimizers.Adam)
        vae.train_vae(mnist_digits, save_model=True)
    else:
        vae = VAE(data_shape=(28,28,1), latent_dim=2)
        vae.full_model = tf.keras.models.load_model('vae')
        vae.encoder = tf.keras.models.load_model('vae_encoder')
        vae.decoder = tf.keras.models.load_model('vae_decoder')
        
    plot_label_clusters(vae.encoder, vae.decoder, mnist_digits, labels)

    _,_,Latent = vae.encoder.predict(mnist_digits)
    # Clusters = KMeans(n_clusters=10, random_state=42)
    # X_ClusterLabels = Clusters.fit_predict(Latent)
    
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(Latent)
    test = x_test[0]
    plt.imshow(test)
    plt.show()
    start = time()
    test = np.expand_dims(test, 0)
    test = np.expand_dims(test, -1).astype("float32") / 255


    _, _, latent = vae.encoder.predict(test)
    # label = Clusters.predict(latent)
    # filtered_digits = Latent[np.argwhere(X_ClusterLabels == label)]
 
    closest = neigh.kneighbors(latent, 1, False)
    print(time() - start)
    plt.imshow(mnist_digits[closest[0]][0, :, :, 0])
    plt.show()

    
if __name__ == '__main__':
    main(training=False)