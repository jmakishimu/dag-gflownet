# In dag_gflownet/nets/ahr.py
import jax
import jax.numpy as jnp
import haiku as hk

class AHR_GatingNetwork(hk.Module):
    """Astrocyte-Hierarchical Routing (AHR) Gate in Haiku."""
    def __init__(self, num_experts, context_dim, name=None):
        super().__init__(name=name)
        self.num_experts = num_experts
        self.context_dim = context_dim

    def __call__(self, state_embedding):
        """
        Input: state_embedding, shape [B, D]
        Output: modulated_logits, shape [B, num_experts]
        """
        # Local (state-dependent) path
        local_net = hk.Sequential([
            hk.Linear(128), jax.nn.relu,
            hk.Linear(self.num_experts)
        ], name="ahr_local_net")
        local_logits = local_net(state_embedding)

        # Global (context-dependent) path
        global_context = hk.get_parameter(
            "global_context",
            shape=[1, self.context_dim],
            init=hk.initializers.RandomNormal()
        )

        global_net = hk.Sequential([
            hk.Linear(128), jax.nn.relu,
            hk.Linear(self.num_experts)
        ], name="ahr_global_net")

        # global_bias shape is [num_experts]
        global_bias = global_net(jnp.squeeze(global_context, axis=0))

        # Modulate local logits [B, num_experts] with global bias [num_experts]
        modulated_logits = local_logits + global_bias
        return modulated_logits
