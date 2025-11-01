# In dag_gflownet/utils/jnp_utils.py
import jax.numpy as jnp

from jax import random


def batch_random_choice(key, probas, masks):
    # Sample from the distribution
    uniform = random.uniform(key, shape=(probas.shape[0], 1))
    cum_probas = jnp.cumsum(probas, axis=1)

    # FIX: Clip samples to ensure they are within the max index,
    # preventing out-of-bounds indexing due to float errors.
    # The maximum index is probas.shape[1] - 1.
    max_index = probas.shape[1] - 1
    samples = jnp.clip(
        jnp.sum(cum_probas < uniform, axis=1, keepdims=True),
        a_min=0,
        a_max=max_index
    )

    # In rare cases, the sampled actions may be invalid, despite having
    # probability 0. In those cases, we select the stop action by default.
    stop_mask = jnp.ones((masks.shape[0], 1), dtype=masks.dtype)  # Stop action is always valid

    masks = masks.reshape(masks.shape[0], -1)
    masks = jnp.concatenate((masks, stop_mask), axis=1)

    # FIX: Explicitly convert the mask to integer (jnp.int32) before looking up the validity.
    # This prevents floating point errors from the mask values (e.g., a tiny non-zero float)
    # from being treated as "valid" (True) when it should be 0.
    is_valid = jnp.take_along_axis(masks.astype(jnp.int32), samples, axis=1)

    # The stop action is the last index
    stop_action = masks.shape[1] - 1

    samples = jnp.where(is_valid, samples, stop_action)

    return jnp.squeeze(samples, axis=1)
