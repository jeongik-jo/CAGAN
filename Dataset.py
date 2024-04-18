import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp

if hp.is_mnist:
    (imgs0, lbls0), (imgs1, lbls1) = kr.datasets.mnist.load_data()
    imgs = tf.concat([imgs0, imgs1], axis=0)
    imgs = tf.cast(imgs, 'float32')[:, :, :, tf.newaxis] / 127.5 - 1
    lbls = tf.concat([lbls0, lbls1], axis=0)
    assert hp.ctg_dim == 10

    imgs_sets = [[] for _ in range(hp.ctg_dim)]
    for img, lbl in zip(imgs, lbls):
        imgs_sets[lbl].append(img)
    imgs_sets = [tf.convert_to_tensor(i) for i in imgs_sets]
    imgs = tf.concat(imgs_sets, axis=0)

    lbls = []
    probs = []
    cnd_probs = []
    for i in range(hp.ctg_dim):
        n = imgs_sets[i].shape[0]
        lbls.append(tf.fill([n], i))
        probs.append(tf.fill([n], 1 / n * hp.data_ctg_prob[i]))
        cnd_probs.append(tf.fill([n], 1 / n))
    lbls = tf.one_hot(tf.concat(lbls, axis=0), depth=hp.ctg_dim)
    probs = tf.concat(probs, axis=0)
    log_probs = tf.math.log(probs)
    cnd_log_probs = [tf.math.log(cnd_prob) for cnd_prob in cnd_probs]

    def data_dist(batch_size):
        indexes = tf.random.categorical([log_probs], batch_size)[0]
        return tf.gather(imgs, indexes), tf.gather(lbls, indexes)

    def cnd_data_dist(batch_size, lbl):
        indexes = tf.random.categorical([cnd_log_probs[lbl]], batch_size)[0]
        return tf.gather(imgs_sets[lbl], indexes)


else:
    cluster_coords = [(-1.0, 1.0), (0.0, -2.0), (1.0, -1.0), (2.0, 1.)]
    cluster_rad = 0.3
    assert hp.ctg_dim == 4

    log_prob = tf.math.log(hp.data_ctg_prob)

    def data_dist(batch_size):
        indexes = tf.random.categorical([log_prob], batch_size)[0]
        coords = tf.gather(cluster_coords, indexes)
        coords += tf.random.normal(stddev=cluster_rad, shape=coords.shape)

        return coords, tf.one_hot(indexes, depth=hp.ctg_dim)
