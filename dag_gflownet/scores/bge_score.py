import math
import numpy as np

from scipy.special import gammaln

from dag_gflownet.scores.base import BaseScore, LocalScore


def logdet(array):
    # FIX: Make logdet robust to singular/non-positive definite matrices
        # --- BEGIN CORRECTION ---
    # Add diagonal jitter for numerical stability
    array = array + 1e-3 * np.eye(array.shape[0])
    # --- END CORRECTION ---
    sign, logdet_val = np.linalg.slogdet(array)
    if sign <= 0:
        # If the matrix is singular (det=0) or not positive definite (det<0),
        # the log-likelihood is invalid (log(det) -> -inf), making the score
        # extremely low and preventing crashes.
        return -np.inf
    return logdet_val


class BGeScore(BaseScore):
    r"""BGe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (continuous) dataset D. Each column
        corresponds to one variable. The dataset D is assumed to only
        contain observational data (a `INT` column will be treated as
        a continuous variable like any other).

    prior : `BasePrior` instance
        The prior over graphs p(G).

    mean_obs : np.ndarray (optional)
        Mean parameter of the Normal prior over the mean $\mu$. This array must
        have size `(N,)`, where `N` is the number of variables. By default,
        the mean parameter is 0.

    alpha_mu : float (default: 1.)
        Parameter $\alpha_{\mu}$ corresponding to the precision parameter
        of the Normal prior over the mean $\mu$.

    alpha_w : float (optional)
        Parameter $\alpha_{w}$ corresponding to the number of degrees of
        freedom of the Wishart prior of the precision matrix $W$. This
        parameter must satisfy `alpha_w > N - 1`, where `N` is the number
        of varaibles. By default, `alpha_w = N + 2`.
    """
    def __init__(
            self,
            data,
            prior,
            mean_obs=None,
            alpha_mu=1.,
            alpha_w=None
        ):
        num_variables = len(data.columns)
        if mean_obs is None:
            mean_obs = np.zeros((num_variables,))
        if alpha_w is None:
            alpha_w = num_variables + 2.

        super().__init__(data, prior)
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w

        self.num_samples = self.data.shape[0]
        self.t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)

        T = self.t * np.eye(self.num_variables)
        data = np.asarray(self.data)
        data_mean = np.mean(data, axis=0, keepdims=True)
        data_centered = data - data_mean

        self.R = (T + np.dot(data_centered.T, data_centered)
            + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
            * np.dot((data_mean - self.mean_obs).T, data_mean - self.mean_obs)
        )
        all_parents = np.arange(self.num_variables)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + gammaln(0.5 * (self.num_samples + self.alpha_w - self.num_variables + all_parents + 1))
            - gammaln(0.5 * (self.alpha_w - self.num_variables + all_parents + 1))
            - 0.5 * self.num_samples * math.log(math.pi)
            + 0.5 * (self.alpha_w - self.num_variables + 2 * all_parents + 1) * math.log(self.t)
        )

    def local_score(self, target, indices):
            num_parents = len(indices)

            if indices:
                variables = [target] + list(indices)

                # --- BEGIN MODIFICATION ---
                # Calculate log-determinants separately to catch -inf
                logdet_parents = logdet(self.R[np.ix_(indices, indices)])
                logdet_vars = logdet(self.R[np.ix_(variables, variables)])

                # Check for invalid log-determinants before subtraction to avoid nan
                if logdet_parents == -np.inf or logdet_vars == -np.inf:
                    log_term_r = -np.inf
                else:
                    # Original calculation (now safe)
                    log_term_r = (
                        0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents)
                        * logdet_parents
                        - 0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents + 1)
                        * logdet_vars
                    )
                # --- END MODIFICATION ---
            else:
                log_term_r = (-0.5 * (self.num_samples + self.alpha_w - self.num_variables + 1)
                    * np.log(np.abs(self.R[target, target])))

            return LocalScore(
                key=(target, tuple(indices)),
                score=self.log_gamma_term[num_parents] + log_term_r,
                prior=self.prior(num_parents)
            )

    def get_local_scores(self, target, indices, indices_after=None):
        all_indices = indices if (indices_after is None) else indices_after
        local_score_after = self.local_score(target, all_indices)
        if indices_after is not None:
            local_score_before = self.local_score(target, indices)
        else:
            local_score_before = None
        return (local_score_before, local_score_after)
