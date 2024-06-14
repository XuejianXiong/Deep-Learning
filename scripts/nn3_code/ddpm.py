#
# 2024 Compute Ontario Summer School
# Artificial Neural Networks
# Day 3, morning
# Erik Spence.
#
# This is a script which builds and trains a variational autoencoder,
# and applies it to MNIST data.
#
#
# This file, ddpm.py, contains the code which builds the noise
# schedule used to train the diffusion network.
#

#######################################################################


import tensorflow as tf


#######################################################################


## Now create the noise schedule.
#beta1, beta2 = 1e-4, 0.02
#T = 1000
   
def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * tf.range(0, T + 1,
                                        dtype = tf.float32) / T + beta1
    sqrt_beta_t = tf.math.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = tf.math.log(alpha_t)
    alphabar_t = tf.math.exp(tf.math.cumsum(log_alpha_t))

    sqrtab = tf.math.sqrt(alphabar_t)
    oneover_sqrta = 1 / tf.math.sqrt(alpha_t)

    sqrtmab = tf.math.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


#######################################################################
