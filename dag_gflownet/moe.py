# In dag_gflownet/moe.py

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
from jax import lax, random, jit, vmap
from functools import partial

# --- FIX: Import base classes and correct utils ---
from dag_gflownet.gflownet import DAGGFlowNet, DAGGFlowNetParameters, DAGGFlowNetState
from dag_gflownet.utils.gflownet import detailed_balance_loss, uniform_log_policy
from dag_gflownet.utils.jnp_utils import batch_random_choice
# --- FIX: Import MASKED_VALUE, not MASKED_LOGIT ---
from dag_gflownet.utils.gflownet import MASKED_VALUE
# Import new model
from dag_gflownet.nets.moe_gflownet import moe_gflownet


# === 1. Utility functions for MoE Policy ===


def log_policy(logits, stop, mask):
    """Computes the log-policy."""
    # --- START FIX: Correct mask reshaping ---
    # The mask comes in as (N, N), e.g., (10, 10).
    # The logits come in as (K, N*N), e.g., (5, 100) OR (N*N,) e.g. (100,)
    # We must flatten the mask to (N*N,) and let it broadcast.
    mask_flat = mask.reshape(-1) # Shape (N*N,)

    # Broadcast mask_flat to match logits shape (handles both K, N*N and N*N)
    mask = jnp.broadcast_to(mask_flat, logits.shape)
    # --- END FIX ---

    # --- Original code was this ---
    logits = jnp.where(mask, logits, MASKED_VALUE)
    logits_stop = jnp.concatenate((logits, stop), axis=-1)
    return jax.nn.log_softmax(logits_stop, axis=-1)


@jax.vmap
def _vmapped_log_policy(logits, stop, mask):
    """Vmapped version of the original log_policy."""
    return log_policy(logits, stop, mask)

def marginal_log_policy(logits_dict, mask):
    """Computes the marginal log_policy P_F(a|s) = log(sum_k G_k(s) P_k(a|s))."""

    # --- START FIX: Apply correct vmap for marginal_log_policy ---
    # _vmapped_log_policy is vmapped over the Batch dimension (B).
    # We now need to vmap it over the Expert dimension (K).
    # in_axes=(0, 0, None) maps over K for logits/stop and broadcasts the (B, N, N) mask.
    vmapped_over_experts_fn = jax.vmap(
        _vmapped_log_policy, in_axes=(0, 0, None)
    )

    # We must transpose logits/stop from (B, K, ...) to (K, B, ...)
    # to match the (K) vmap.

    # logits_dict['action_logits'] shape is (B, K, N*N)
    action_logits_k_first = jnp.transpose(logits_dict['action_logits'], (1, 0, 2))
    # logits_dict['stop_logits'] shape is (B, K, 1)
    stop_logits_k_first = jnp.transpose(logits_dict['stop_logits'], (1, 0, 2))

    # log_pi_all_experts shape [K, B, N*N+1]
    log_pi_all_experts = vmapped_over_experts_fn(
        action_logits_k_first,
        stop_logits_k_first,
        mask
    )
    # --- END FIX ---

    pi_all_experts = jnp.exp(log_pi_all_experts)

    # Gating probabilities [B, K]
    expert_probs = jax.nn.softmax(logits_dict['gating_logits'], axis=-1)

    # Compute marginal probability: (B, K) * (K, B, A) -> (B, A)
    marginal_pi = jnp.einsum('bk,kba->ba', expert_probs, pi_all_experts)
    return jnp.log(marginal_pi + 1e-10)


# === 2. The MoE GFlowNet Training Class ===

class MoEDAGGFlowNet(DAGGFlowNet):
    def __init__(self, config, **kwargs):
        self.config = config

        # --- START FIX ---
        # Create a function that has the signature (adjacency, mask)
        # by partially applying the config.
        model_fn = partial(moe_gflownet, config)

        # Now pass this raw, untransformed function to the base class.
        # The base class's __init__ will correctly call hk.transform(model_fn).
        super().__init__(model=model_fn, **kwargs)
        # --- END FIX ---


    def init(self, key, optimizer, *dummy_inputs):
        """Override to use the MoE model."""
        # The base class init now works correctly with the *dummy_inputs (adjacency, mask)
        return super().init(key, optimizer, *dummy_inputs)

    @partial(jit, static_argnums=(0,))
    def act(self, params, key, observations, epsilon, **kwargs):
        """MoE-specific act method for replay buffer."""
        masks = observations['mask'].astype(jnp.float32)
        adjacencies = observations['adjacency'].astype(jnp.float32)
        batch_size = adjacencies.shape[0]
        # Need 4 keys: 1 for uniform mix, 1 for sampling, 1 for gate, 1 to return
        key, subkey1, subkey2, gate_key = random.split(key, 4)

        vmodel = vmap(self.model.apply, in_axes=(None, 0, 0))
        logits_dict = vmodel(params, adjacencies, masks)

        # Sample expert k based on *full* softmax (no top-k for exploration)
        gating_logits = logits_dict['gating_logits']
        expert_k = jax.random.categorical(gate_key, gating_logits) # [B,]

        # === FIX: Correctly gather expert logits ===
        k_gather_indices_actions = jnp.expand_dims(expert_k, axis=(1, 2))
        k_gather_indices_stop = jnp.expand_dims(expert_k, axis=(1, 2))


        # logits_dict['action_logits'] has shape (B, K, N*N)
        batch_action_logits = jnp.take_along_axis(
            logits_dict['action_logits'],
            k_gather_indices_actions,
            axis=1).squeeze(axis=1) # Squeeze the K dimension (size 1)

        # logits_dict['stop_logits'] has shape (B, K, 1)
        batch_stop_logits = jnp.take_along_axis(
            logits_dict['stop_logits'],
            k_gather_indices_stop,
            axis=1).squeeze(axis=1) # Squeeze the K dimension (size 1)

        log_pi = _vmapped_log_policy(batch_action_logits, batch_stop_logits, masks)

        # Get uniform policy
        log_uniform = uniform_log_policy(masks)

        # Mixture of expert policy and uniform policy
        is_exploration = random.bernoulli(
            subkey1, p=epsilon, shape=(batch_size, 1)) # <-- CORRECTED
        log_pi = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        actions = batch_random_choice(subkey2, jnp.exp(log_pi), masks)

        logs = {
            'is_exploration': is_exploration.astype(jnp.int32),
        }
        return (actions, key, logs)

    @partial(jit, static_argnums=(0,))
    def step(self, params, state, samples):
        """MoE-specific update step with JAX-compatible hybrid loss."""

        @jax.jit
        def loss_fn(online_params):

            # === 1. L_GFN (Detailed Balance Loss) ===
            vmodel = vmap(self.model.apply, in_axes=(None, 0, 0))

            logits_dict_t = vmodel(
                online_params, samples['adjacency'], samples['mask']
            )
            log_pi_t = marginal_log_policy(logits_dict_t, samples['mask'])

            logits_dict_tp1 = vmodel(
                params.target, samples['next_adjacency'], samples['next_mask']
            )

            log_pi_tp1 = marginal_log_policy(logits_dict_tp1, samples['next_mask'])

            l_gfn, logs = detailed_balance_loss(
                log_pi_t, log_pi_tp1, samples['actions'],
                samples['delta_scores'], samples['num_edges'], self.delta
            )

            total_loss = l_gfn
            logs['l_gfn'] = l_gfn

            # === 2. L_diversity (Diversity Loss) ===
            l_diversity = 0.0
            if self.config.use_diversity:
                # --- UNCOMMENTED DIVERSITY LOSS ---
                weights = []
                for k in range(self.config.num_experts):
                    try:
                        # --- THE FINAL FIX ---
                        # Use the key confirmed by the debug print
                        w_key = f'expert_{k}_logits_mlp/~/linear_0'
                        w = online_params[w_key]['w'].flatten()
                        weights.append(w / (jnp.linalg.norm(w) + 1e-8))
                        # --- END FIX ---
                    except KeyError as e:
                        # This error should no longer happen
                        raise e

                sims = jnp.abs(jnp.corrcoef(jnp.stack(weights)))
                l_diversity = jnp.mean(jnp.triu(sims, k=1)) # Penalize high correlation
                total_loss += self.config.diversity_beta * l_diversity
                # --- END UNCOMMENTED SECTION ---

            logs['l_diversity'] = l_diversity

            # === 3. L_balance (JAX-Compatible Version) ===
            l_balance = 0.0
            if self.config.use_load_balancing:
                # Use the gating logits from L_GFN, no extra model call needed
                gating_logits = logits_dict_t['gating_logits'] # [B, K]

                # P_k: Average router probability for expert k over the batch
                expert_probs = jax.nn.softmax(gating_logits, axis=-1)
                P_k = jnp.mean(expert_probs, axis=0) # [K,]

                # f_k: Fraction of items in the batch routed to expert k
                # We use Top-1 routing for this calculation
                chosen_experts = jnp.argmax(gating_logits, axis=-1) # [B,]
                f_k = jnp.mean(
                    jax.nn.one_hot(chosen_experts, self.config.num_experts),
                    axis=0
                ) # [K,]

                # Calculate loss: alpha * K * sum(f_k * P_k)
                l_balance = self.config.load_balance_alpha * self.config.num_experts * jnp.sum(f_k * P_k)
                total_loss += l_balance
            logs['l_balance'] = l_balance

            return total_loss, logs

        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params.online)
        logs['loss'] = loss # Add total loss to logs

        # --- REPLICATED LOGIC FROM BASE CLASS 'step' ---
        # Update the online params
        updates, opt_state = self.optimizer.update(
            grads,
            state.optimizer,
            params.online
        )
        state = DAGGFlowNetState(optimizer=opt_state, steps=state.steps + 1)
        online_params = optax.apply_updates(params.online, updates)

        # Update the target params periodically
        params = DAGGFlowNetParameters(
            online=online_params,
            target=optax.periodic_update(
                online_params,
                params.target,
                state.steps,
                self.update_target_every
            ),
        )
        # --- END REPLICATED LOGIC ---

        return (params, state, logs)
