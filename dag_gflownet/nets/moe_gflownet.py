# In dag_gflownet/nets/moe_gflownet.py
import jax
import jax.numpy as jnp
import haiku as hk
from dag_gflownet.nets.transformers import TransformerBlock
from dag_gflownet.nets.ahr import AHR_GatingNetwork # Assumes ahr.py is in this directory

# Re-define the expert heads (from original gflownet.py)
# This allows each expert to have its own parameters
def expert_logits_head(embeddings, adjacency, num_layers, name):
    head_name = f"{name}_logits"
    for i in range(2):
        embeddings = TransformerBlock(
            num_heads=4, key_size=64, embedding_size=128,
            init_scale=2. / num_layers, widening_factor=2,
            name=f'{head_name}_{i+1}'
        )(embeddings, adjacency)
    logits = hk.nets.MLP([256, 128, 1], name=f"{head_name}_mlp")(embeddings)
    return jnp.squeeze(logits, axis=-1)

def expert_stop_head(embeddings, adjacency, num_layers, name):
    head_name = f"{name}_stop"
    for i in range(2):
        embeddings = TransformerBlock(
            num_heads=4, key_size=64, embedding_size=128,
            init_scale=2. / num_layers, widening_factor=2,
            name=f'{head_name}_{i+1}'
        )(embeddings, adjacency)

    mean_embeddings = jnp.mean(embeddings, axis=-2)
    stop_logit = hk.nets.MLP([256, 128, 1], name=f"{head_name}_mlp")(mean_embeddings)
    return stop_logit, mean_embeddings

# Main MoE GFlowNet Haiku function
def moe_gflownet(config, adjacency, mask):
    num_experts = config.num_experts
    context_dim = config.context_dim
    num_variables = adjacency.shape[0]

    # === 1. Common Backbone (from original nets.gflownet.py) ===
    indices = jnp.arange(num_variables ** 2)
    sources, targets = jnp.divmod(indices, num_variables)
    edges = jnp.stack((sources, num_variables + targets), axis=1)
    embeddings = hk.Embed(2 * num_variables, embed_dim=128)(edges)
    embeddings = embeddings.reshape(num_variables ** 2, -1)
    adjacency_flat = adjacency.reshape(num_variables ** 2, 1)

    num_layers = 5 # As in original
    for i in range(3): # Common body
        embeddings = TransformerBlock(
            num_heads=4, key_size=64, embedding_size=128,
            init_scale=2. / num_layers, widening_factor=2,
            name=f'body_{i+1}'
        )(embeddings, adjacency_flat)

    shared_embeddings = embeddings
    # Use mean embeddings for gating
    mean_shared_embeddings = jnp.mean(shared_embeddings, axis=-2, keepdims=True)

    # === 2. Gating Network (Ablation-aware) ===
    if config.use_ahr:
        gating_net = AHR_GatingNetwork(num_experts, context_dim, name="ahr_gate")
        gating_logits = gating_net(mean_shared_embeddings)
    else:
        # Standard MoE (local-only)
        gating_net = hk.Sequential([
            hk.Linear(128), jax.nn.relu,
            hk.Linear(num_experts)
        ], name="standard_gate")
        gating_logits = gating_net(mean_shared_embeddings)

    # === 3. Expert Heads ===
    all_action_logits = []
    all_stop_logits = []
    for k in range(num_experts):
        expert_name = f"expert_{k}"
        action_logits_k = expert_logits_head(
            shared_embeddings, adjacency_flat, num_layers, name=expert_name)
        stop_logit_k, _ = expert_stop_head(
            shared_embeddings, adjacency_flat, num_layers, name=expert_name)
        all_action_logits.append(action_logits_k)
        all_stop_logits.append(stop_logit_k)

    # Return all logits
    return {
        "gating_logits": jnp.squeeze(gating_logits, axis=0), # [B, K]
        "action_logits": jnp.stack(all_action_logits, axis=0), # [K, B, N*N]
        "stop_logits": jnp.stack(all_stop_logits, axis=0) # [K, B, 1]
    }
