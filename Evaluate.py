from tensorflow import keras as kr
import tensorflow as tf
import tensorflow_probability as tfp
import HyperParameters as hp
from scipy.linalg import sqrtm
import numpy as np
import Dataset
import matplotlib.pyplot as plt
import os

inception_model = tf.keras.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)


@tf.function
def _get_batch_results(gen: kr.Model, lbl=None):
    if lbl == None:
        real_imgs, ctg_vecs = Dataset.data_dist(hp.batch_size)
        real_imgs = tf.tile(real_imgs, [1, 1, 1, 3])
    else:
        ctg_vecs = tf.one_hot(tf.repeat(lbl, hp.batch_size), depth=hp.ctg_dim)
        real_imgs = tf.tile(Dataset.cnd_data_dist(hp.batch_size, lbl), [1, 1, 1, 3])
    fake_imgs = tf.tile(tf.clip_by_value(gen([ctg_vecs, hp.cnt_ltn_dist(hp.batch_size)]), clip_value_min=-1, clip_value_max=1), [1, 1, 1, 3])

    real_features = inception_model(tf.image.resize(real_imgs, [299, 299]))
    fake_features = inception_model(tf.image.resize(fake_imgs, [299, 299]))

    return {'real_features': real_features, 'fake_features': fake_features}


def _pairwise_distances(U, V):
    norm_u = tf.reduce_sum(tf.square(U), 1)
    norm_v = tf.reduce_sum(tf.square(V), 1)

    norm_u = tf.reshape(norm_u, [-1, 1])
    norm_v = tf.reshape(norm_v, [1, -1])

    D = tf.maximum(norm_u - 2 * tf.matmul(U, V, False, True) + norm_v, 0.0)

    return D


def _get_fid(real_features, fake_features):
    real_features_mean = tf.reduce_mean(real_features, axis=0)
    fake_features_mean = tf.reduce_mean(fake_features, axis=0)

    mean_difference = tf.reduce_sum((real_features_mean - fake_features_mean) ** 2)
    real_cov, fake_cov = tfp.stats.covariance(real_features), tfp.stats.covariance(fake_features)
    cov_mean = sqrtm(tf.matmul(real_cov, fake_cov))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    cov_difference = tf.linalg.trace(real_cov + fake_cov - 2.0 * cov_mean)

    fid = mean_difference + cov_difference

    return fid


@tf.function
def _get_pr(ref_features, eval_features, nhood_size=3):
    thresholds = -tf.math.top_k(-_pairwise_distances(ref_features, ref_features), k=nhood_size + 1, sorted=True)[0]
    thresholds = thresholds[tf.newaxis, :, -1]

    distance_pairs = _pairwise_distances(eval_features, ref_features)
    return tf.reduce_mean(tf.cast(tf.math.reduce_any(distance_pairs <= thresholds, axis=1), 'float32'))


def eval(gen):
    results = {}
    if hp.is_mnist:
        temp_dict = {}
        for _ in range(hp.step_per_epoch):
            batch_results = _get_batch_results(gen)
            for key in batch_results:
                try:
                    temp_dict[key].append(batch_results[key])
                except KeyError:
                    temp_dict[key] = [batch_results[key]]

        real_features = tf.concat(temp_dict['real_features'], axis=0)
        fake_features = tf.concat(temp_dict['fake_features'], axis=0)

        fid = _get_fid(real_features, fake_features)
        precision = _get_pr(real_features, fake_features)
        recall = _get_pr(fake_features, real_features)

        fids = []
        precisions = []
        recalls = []
        for lbl in range(hp.ctg_dim):
            temp_dict = {}
            for _ in range(hp.step_per_epoch // hp.ctg_dim):
                batch_results = _get_batch_results(gen, lbl)
                for key in batch_results:
                    try:
                        temp_dict[key].append(batch_results[key])
                    except KeyError:
                        temp_dict[key] = [batch_results[key]]
            real_features = tf.concat(temp_dict['real_features'], axis=0)
            fake_features = tf.concat(temp_dict['fake_features'], axis=0)

            fids.append(_get_fid(real_features, fake_features))
            precisions.append(_get_pr(real_features, fake_features))
            recalls.append(_get_pr(fake_features, real_features))

        results = {'fid': fid, 'precision': precision, 'recall': recall,
                   'fids': tf.reduce_mean(fids), 'precisions': tf.reduce_mean(precisions), 'recalls': tf.reduce_mean(recalls)}

    for key in results:
        print('%-20s:' % key, '%13.6f' % results[key].numpy())
    return results


def save_samples(gen, epoch):
    if not os.path.exists('samples'):
        os.makedirs('samples')
    if hp.is_mnist:
        imgs = []
        cnt_ltn_vecs = hp.cnt_ltn_dist(hp.batch_size)
        for ctg_value in range(hp.ctg_dim):
            ctg_vecs = tf.one_hot(tf.fill([hp.batch_size], ctg_value), hp.ctg_dim)
            fake_x = gen([ctg_vecs, cnt_ltn_vecs])
            imgs.append(np.vstack(fake_x))
        kr.preprocessing.image.save_img('samples/%d.png' % epoch,
                                        tf.clip_by_value(np.hstack(imgs), clip_value_min=-1, clip_value_max=1))

    else:
        plt.clf()
        real_x, real_lbls = Dataset.data_dist(1000)
        plt.scatter(real_x[:, 0], real_x[:, 1], label='real', s=5)

        for ctg_val in range(hp.ctg_dim):
            sample_size = tf.cast(tf.round(1000 * hp.data_ctg_prob[ctg_val]), 'int64')
            cnt_vecs = hp.cnt_ltn_dist(sample_size)
            ctg_vecs = tf.one_hot(tf.fill([sample_size], ctg_val), hp.ctg_dim)
            fake_x = gen([ctg_vecs, cnt_vecs])
            plt.scatter(fake_x[:, 0], fake_x[:, 1], label=ctg_val, s=5)
        plt.xlim((-2, 3))
        plt.ylim((-3, 2))
        plt.legend()
        plt.savefig('samples/%d.png' % epoch, dpi=300)


results = {}
def save_figures(train_results, eval_results):
    if not os.path.exists('figures'):
        os.makedirs('figures')

    for key in train_results:
        try:
            results[key].append(train_results[key])
        except KeyError:
            results[key] = [train_results[key]]
    for key in eval_results:
        try:
            results[key].append(eval_results[key])
        except KeyError:
            results[key] = [eval_results[key]]

    for key in results:
        np.savetxt('figures/%s.txt' % key, results[key], fmt='%f')
        plt.title(key)
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.plot([(i + 1) * hp.epoch_per_eval for i in range(len(results[key]))], results[key])
        plt.savefig('figures/%s.png' % key)
        plt.clf()

