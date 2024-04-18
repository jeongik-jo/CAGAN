import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras as kr

dis_opt = kr.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001, beta_1=0.0, beta_2=0.99)
gen_opt = kr.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001, beta_1=0.0, beta_2=0.99,
                              use_ema=True, ema_momentum=0.999, ema_overwrite_frequency=None)

cnt_dim = 256
adv_reg_w = 1.0

is_mnist = True
if is_mnist:
    ctg_ltn_prob = [0.1 for _ in range(10)]
    ctg_dim = 10
else:
    ctg_ltn_prob = [0.1, 0.2, 0.3, 0.4]
    ctg_dim = 4

is_acgan = False
if is_acgan:
    d_real_ctg_w = 1.0
    d_fake_ctg_w = 0.0
    g_ctg_w = 1.0

is_train_label_biased = False
if is_train_label_biased:
    if is_mnist:
        data_ctg_prob = [0.55] + [0.05 for _ in range(9)]
    else:
        data_ctg_prob = [0.4, 0.3, 0.2, 0.1]
else:
    data_ctg_prob = ctg_ltn_prob


is_dis_batch_op = False
mix_rate_per_epoch = 0.0

batch_size = 32
step_per_epoch = 1000
epochs = 200

eval_model = True
epoch_per_eval = 10


def cnt_ltn_dist(batch_size):
    return tf.random.normal([batch_size, cnt_dim])


def ctg_ltn_dist(batch_size):
    return tf.one_hot(tf.random.categorical(logits=[tf.math.log(ctg_ltn_prob)], num_samples=batch_size)[0], depth=ctg_dim)
