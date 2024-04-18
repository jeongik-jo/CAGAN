import tensorflow as tf
import Dataset
import HyperParameters as hp


@tf.function
def _train_step(dis, gen, epoch):
    real_xs, real_ctg_vecs = Dataset.data_dist(hp.batch_size)
    cnt_vecs = hp.cnt_ltn_dist(hp.batch_size)
    fake_ctg_vecs = hp.ctg_ltn_dist(hp.batch_size)

    fake_xs = gen([fake_ctg_vecs, cnt_vecs])

    with tf.GradientTape(persistent=True) as dis_tape:
        slice_index = tf.cast(tf.minimum(hp.mix_rate_per_epoch * epoch, 0.5) * hp.batch_size, dtype='int32')
        real_xs0, real_xs1 = real_xs[:slice_index], real_xs[slice_index:]
        real_ctg_vecs0, real_ctg_vecs1 = real_ctg_vecs[:slice_index], real_ctg_vecs[slice_index:]
        fake_xs0, fake_xs1 = fake_xs[:slice_index], fake_xs[slice_index:]
        fake_ctg_vecs0, fake_ctg_vecs1 = fake_ctg_vecs[:slice_index], fake_ctg_vecs[slice_index:]
        xs0 = tf.concat([real_xs0, fake_xs1], axis=0)
        ctg_vecs0 = tf.concat([real_ctg_vecs0, fake_ctg_vecs1], axis=0)
        xs1 = tf.concat([fake_xs0, real_xs1], axis=0)
        ctg_vecs1 = tf.concat([fake_ctg_vecs0, real_ctg_vecs1], axis=0)

        with tf.GradientTape(persistent=True) as reg_tape:
            reg_tape.watch([xs0, xs1])
            adv_vals0, ctg_logs0 = dis(xs0)
            adv_vals1, ctg_logs1 = dis(xs1)
            if hp.is_acgan:
                reg_scrs0 = adv_vals0
                reg_scrs1 = adv_vals1
            else:
                reg_scrs0 = tf.reduce_sum(ctg_logs0 * ctg_vecs0, axis=-1)
                reg_scrs1 = tf.reduce_sum(ctg_logs1 * ctg_vecs1, axis=-1)
        if hp.is_mnist:
            reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(reg_scrs0, xs0)), axis=[1, 2, 3]) + \
                         tf.reduce_sum(tf.square(reg_tape.gradient(reg_scrs1, xs1)), axis=[1, 2, 3])
        else:
            reg_losses = tf.reduce_sum(tf.square(reg_tape.gradient(reg_scrs0, xs0)), axis=[1]) + \
                         tf.reduce_sum(tf.square(reg_tape.gradient(reg_scrs1, xs1)), axis=[1])

        if hp.is_acgan:
            real_adv_vals = tf.concat([adv_vals0[:slice_index], adv_vals1[slice_index:]], axis=0)
            fake_adv_vals = tf.concat([adv_vals1[:slice_index], adv_vals0[slice_index:]], axis=0)
            real_ctg_logs = tf.concat([ctg_logs0[:slice_index], ctg_logs1[slice_index:]], axis=0)
            fake_ctg_logs = tf.concat([ctg_logs1[:slice_index], ctg_logs0[slice_index:]], axis=0)
            dis_adv_losses = tf.nn.softplus(-real_adv_vals) + tf.nn.softplus(fake_adv_vals)

            real_ctg_losses = tf.losses.categorical_crossentropy(real_ctg_vecs, real_ctg_logs, from_logits=True)
            fake_ctg_losses = tf.losses.categorical_crossentropy(fake_ctg_vecs, fake_ctg_logs, from_logits=True)

            dis_losses = dis_adv_losses + hp.d_real_ctg_w * real_ctg_losses + hp.d_fake_ctg_w * fake_ctg_losses + hp.adv_reg_w * reg_losses

        else:
            adv_vals0 = tf.reduce_sum(ctg_logs0 * ctg_vecs0, axis=-1)
            adv_vals1 = tf.reduce_sum(ctg_logs1 * ctg_vecs1, axis=-1)
            real_adv_vals = tf.concat([adv_vals0[:slice_index], adv_vals1[slice_index:]], axis=0)
            fake_adv_vals = tf.concat([adv_vals1[:slice_index], adv_vals0[slice_index:]], axis=0)
            dis_adv_losses = tf.nn.softplus(-real_adv_vals) + tf.nn.softplus(fake_adv_vals)

            dis_losses = dis_adv_losses + hp.adv_reg_w * reg_losses

        dis_loss = tf.reduce_mean(dis_losses)
    hp.dis_opt.minimize(dis_loss, dis.trainable_variables, dis_tape)

    real_xs, real_ctg_vecs = Dataset.data_dist(hp.batch_size)
    cnt_vecs = hp.cnt_ltn_dist(hp.batch_size)
    fake_ctg_vecs = hp.ctg_ltn_dist(hp.batch_size)

    with tf.GradientTape(persistent=True) as gen_tape:
        fake_xs = gen([fake_ctg_vecs, cnt_vecs])

        slice_index = tf.cast(tf.minimum(hp.mix_rate_per_epoch * epoch, 0.5) * hp.batch_size, dtype='int32')
        real_xs0, real_xs1 = real_xs[:slice_index], real_xs[slice_index:]
        real_ctg_vecs0, real_ctg_vecs1 = real_ctg_vecs[:slice_index], real_ctg_vecs[slice_index:]
        fake_xs0, fake_xs1 = fake_xs[:slice_index], fake_xs[slice_index:]
        fake_ctg_vecs0, fake_ctg_vecs1 = fake_ctg_vecs[:slice_index], fake_ctg_vecs[slice_index:]
        xs0 = tf.concat([real_xs0, fake_xs1], axis=0)
        ctg_vecs0 = tf.concat([real_ctg_vecs0, fake_ctg_vecs1], axis=0)
        xs1 = tf.concat([fake_xs0, real_xs1], axis=0)
        ctg_vecs1 = tf.concat([fake_ctg_vecs0, real_ctg_vecs1], axis=0)

        adv_vals0, ctg_logs0 = dis(xs0)
        adv_vals1, ctg_logs1 = dis(xs1)

        if hp.is_acgan:
            real_adv_vals = tf.concat([adv_vals0[:slice_index], adv_vals1[slice_index:]], axis=0)
            fake_adv_vals = tf.concat([adv_vals1[:slice_index], adv_vals0[slice_index:]], axis=0)
            fake_ctg_logs = tf.concat([ctg_logs1[:slice_index], ctg_logs0[slice_index:]], axis=0)
            gen_adv_losses = tf.nn.softplus(-fake_adv_vals)

            fake_ctg_losses = tf.losses.categorical_crossentropy(fake_ctg_vecs, fake_ctg_logs, from_logits=True)

            gen_losses = gen_adv_losses + hp.g_ctg_w * fake_ctg_losses

        else:
            adv_vals0 = tf.reduce_sum(ctg_logs0 * ctg_vecs0, axis=-1)
            adv_vals1 = tf.reduce_sum(ctg_logs1 * ctg_vecs1, axis=-1)
            real_adv_vals = tf.concat([adv_vals0[:slice_index], adv_vals1[slice_index:]], axis=0)
            fake_adv_vals = tf.concat([adv_vals1[:slice_index], adv_vals0[slice_index:]], axis=0)
            gen_adv_losses = tf.nn.softplus(-fake_adv_vals)

            gen_losses = gen_adv_losses
        gen_loss = tf.reduce_mean(gen_losses)

    hp.gen_opt.minimize(gen_loss, gen.trainable_variables, gen_tape)

    results = {
        'real_adv_vals': real_adv_vals, 'fake_adv_vals': fake_adv_vals,
        'reg_losses': reg_losses
    }
    if hp.is_acgan:
        results['real_ctg_losses'] = real_ctg_losses
        results['fake_ctg_losses'] = fake_ctg_losses

    return results


def train(dis, gen, epoch):
    results = {}
    for i in range(hp.step_per_epoch):
        batch_results = _train_step(dis, gen, epoch)

        for key in batch_results:
            try:
                results[key].append(batch_results[key])
            except KeyError:
                results[key] = [batch_results[key]]

    temp_results = {}
    for key in results:
        mean, variance = tf.nn.moments(tf.concat(results[key], axis=0), axes=0)
        temp_results[key + '_mean'] = mean
        temp_results[key + '_variance'] = variance
    results = temp_results

    for key in results:
        print('%-30s:' % key, '%13.6f' % results[key].numpy())

    return results
