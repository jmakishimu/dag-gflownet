# In dag_gflownet/utils/sampling.py
import numpy as np
import pandas as pd
import networkx as nx

from numpy.random import default_rng
from pgmpy.models import LinearGaussianBayesianNetwork, BayesianNetwork
from pgmpy.sampling import BayesianModelSampling


def sample_from_linear_gaussian(model, num_samples, rng=default_rng()):
    """Sample from a linear-Gaussian model using ancestral sampling."""
    if not isinstance(model, LinearGaussianBayesianNetwork):
        raise ValueError('The model must be an instance '
                         'of LinearGaussianBayesianNetwork')

    samples = pd.DataFrame(columns=list(model.nodes()))
    for node in nx.topological_sort(model):
        cpd = model.get_cpds(node)

        # The 'beta' attribute is a list: [intercept, coeff1, coeff2, ...]
        if cpd.evidence:
            values = np.vstack([samples[parent] for parent in cpd.evidence])
            # Use beta[0] for intercept, beta[1:] for coefficients
            intercept = cpd.beta[0]
            coefficients = cpd.beta[1:]
            mean = intercept + np.dot(coefficients, values)
            # --- FIX 3: Replaced cpd.variance with cpd.std ---
            samples[node] = rng.normal(mean, cpd.std)
        else:
            # Use beta[0] for intercept when there's no evidence
            intercept = cpd.beta[0]
            # --- FIX 3: Replaced cpd.variance with cpd.std ---
            samples[node] = rng.normal(intercept, cpd.std, size=(num_samples,))

    return samples


def sample_from_discrete(model, num_samples, rng=default_rng(), **kwargs):
    """Sample from a discrete model using ancestral sampling."""
    if not isinstance(model, BayesianNetwork):
        raise ValueError('The model must be an instance of BayesianNetwork')
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_samples, show_progress=False, **kwargs)

    # Convert values to pd.Categorical for faster operations
    for node in samples.columns:
        cpd = model.get_cpds(node)
        samples[node] = pd.Categorical(samples[node], categories=cpd.state_names[node])

    return samples
