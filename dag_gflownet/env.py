# In dag_gflownet/env.py
import numpy as np
import gymnasium as gym
import bisect

from multiprocessing import get_context
from copy import deepcopy
from gymnasium.spaces import Dict, Box, Discrete
from dag_gflownet.utils.cache import LRUCache

# REMOVAL OF INHERITANCE: The class no longer inherits from gym.vector.VectorEnv
class GFlowNetDAGEnv:
    def __init__(
            self,
            num_envs,
            scorer,
            max_parents=None,
            num_workers=4,
            context=None,
            cache_max_size=10_000
        ):
        """GFlowNet environment for learning a distribution over DAGs.
        (Docstring omitted for brevity)
        """
        self.scorer = scorer
        self.num_workers = num_workers
        self.num_envs = num_envs  # Keep num_envs as an explicit attribute
        self.is_vector_env = True # Keep for external compatibility checks

        self.num_variables = scorer.num_variables
        self.local_scores = LRUCache(max_size=cache_max_size)
        #self.local_scores = {}
        self._state = None
        self.max_parents = max_parents or self.num_variables

        # FIX 1 (Part A): Store stop_action as a native Python int for stability
        self.stop_action = self.num_variables ** 2

        # --- Define spaces (used primarily for self-documentation/metadata) ---
        shape = (self.num_variables, self.num_variables)
        max_edges = self.num_variables * (self.num_variables - 1) // 2

        # We keep space definitions, but they are no longer passed to a parent __init__
        self.observation_space = Dict({
            # FIX: Ensure mask is an int type in the space definition
            'adjacency': Box(low=0., high=1., shape=shape, dtype=np.int32),
            'mask': Box(low=0., high=1., shape=shape, dtype=np.int32),
            'num_edges': Discrete(max_edges),
            'score': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
            'order': Box(low=-1, high=max_edges, shape=shape, dtype=np.int_)
        })
        self.action_space = Discrete(self.stop_action + 1)

        # --- Multiprocessing setup ---
        if num_workers > 0:
            ctx = get_context(context)

            self.in_queue = ctx.Queue()
            self.out_queue = ctx.Queue()
            self.error_queue = ctx.Queue()

            self.processes = []
            for index in range(num_workers):
                process = ctx.Process(
                    target=self.scorer,
                    args=(index, self.in_queue, self.out_queue, self.error_queue),
                    daemon=True
                )
                process.start()
                self.processes.append(process)

    def reset(self, *, seed=None, options=None): # Gymnasium signature
        shape = (self.num_envs, self.num_variables, self.num_variables)
        closure_T = np.eye(self.num_variables, dtype=np.bool_)
        self._closure_T = np.tile(closure_T, (self.num_envs, 1, 1))
        self._state = {
            'adjacency': np.zeros(shape, dtype=np.int32), # Changed to np.int32
            # FIX: Ensure mask is np.int32
            'mask': (1 - self._closure_T).astype(np.int32),
            'num_edges': np.zeros((self.num_envs,), dtype=np.int_),
            'score': np.zeros((self.num_envs,), dtype=np.float64), # Use float64
            'order': np.full(shape, -1, dtype=np.int_)
        }
        # Mimic Gymnasium reset return (obs, info)
        return deepcopy(self._state), {}

    def step(self, actions):
        # *** FIX for "ValueError: assignment destination is read-only" ***
        # Ensure the incoming NumPy array is writeable before modification.
        actions = actions.copy()
        # ***************************************************************

        # FIX 2: Implement robust action validation and correction
        sources, targets = np.divmod(actions, self.num_variables) # Use np.divmod for numpy arrays
        is_stop_action = (actions == self.stop_action)

        # 1. Calculate the mask for the current state s_t
        cycle_prevention_matrix = self._closure_T.transpose((0, 2, 1))

        # Calculate invalid mask (adjacency | redundancy | cycle)
        invalid_mask = np.clip(
            self._state['adjacency'] + self._closure_T + cycle_prevention_matrix,
            0, 1
        )
        mask_base = (1 - invalid_mask).astype(np.int32)

        # Calculate max parents mask (num_parents < max_parents)
        num_parents = np.sum(self._state['adjacency'], axis=1)
        max_parents_validity = (num_parents < self.max_parents).astype(np.int32)
        max_parents_mask = np.expand_dims(max_parents_validity, axis=1)

        # The full mask for non-stop actions (shape [B, N, N])
        full_mask = mask_base * max_parents_mask

        # 2. Check validity of sampled actions (only for non-stop actions)
        actions_non_stop_mask = ~is_stop_action # Boolean mask for non-stop actions
        if np.any(actions_non_stop_mask): # Only check if there are non-stop actions
            sources_non_stop = sources[actions_non_stop_mask]
            targets_non_stop = targets[actions_non_stop_mask]

            # Check if the mask is 1 at the sampled index (is_valid_in_env is shape [~dones])
            # Need to index full_mask correctly for the boolean mask
            is_valid_in_env = full_mask[actions_non_stop_mask, sources_non_stop, targets_non_stop]

            # 3. Replace invalid actions with the stop action (Action Code: self.stop_action)
            corrected_actions = np.where(is_valid_in_env, actions[actions_non_stop_mask], self.stop_action)

            # Reconstruct the full actions array: replace non-stop actions with corrected ones
            actions[actions_non_stop_mask] = corrected_actions

            # Re-calculate sources/targets based on the corrected actions
            sources, targets = np.divmod(actions, self.num_variables) # Use np.divmod

        # ----------------------------------------------------------------------

        # --- MODIFICATION: Removed local_cache ---
        # Note: sources/targets now reflect potentially corrected actions
        keys, _, data = self.local_scores_async(sources, targets)
        dones = (actions == self.stop_action) # Use corrected actions to determine dones

        # Keep track of the full sources/targets before filtering out 'dones' (for debug)
        # all_sources = sources
        # all_targets = targets

        active_envs_mask = ~dones # Boolean mask for active environments
        sources_active, targets_active = sources[active_envs_mask], targets[active_envs_mask]

        # ----------------------------------------------------------------------

        # FINAL STATE UPDATES

        # Update the adjacency matrices (only valid non-done actions are here)
        if np.any(active_envs_mask):
             self._state['adjacency'][active_envs_mask, sources_active, targets_active] = 1
        self._state['adjacency'][dones] = 0

        # Update transitive closure of transpose (closure must be updated first)
        if np.any(active_envs_mask):
            source_rows = np.expand_dims(self._closure_T[active_envs_mask, sources_active, :], axis=1)
            target_cols = np.expand_dims(self._closure_T[active_envs_mask, :, targets_active], axis=2)
            self._closure_T[active_envs_mask] |= np.logical_and(source_rows, target_cols)  # Outer product
        self._closure_T[dones] = np.eye(self.num_variables, dtype=np.bool_)

        # Update the masks (using the new, corrected mask base logic)
        # Recompute mask_base based on potentially updated adjacency and closure_T
        cycle_prevention_matrix = self._closure_T.transpose((0, 2, 1))
        invalid_mask = np.clip(
            self._state['adjacency'] + self._closure_T + cycle_prevention_matrix,
            0, 1
        )
        mask_base = (1 - invalid_mask).astype(np.int32)

        # Recompute max parents constraint based on updated adjacency
        num_parents = np.sum(self._state['adjacency'], axis=1)
        max_parents_validity = (num_parents < self.max_parents).astype(np.int32)

        self._state['mask'] = mask_base * np.expand_dims(max_parents_validity, axis=1)

        # Update the order
        if np.any(active_envs_mask):
            self._state['order'][active_envs_mask, sources_active, targets_active] = self._state['num_edges'][active_envs_mask]
        self._state['order'][dones] = -1

        # Update the number of edges
        self._state['num_edges'][active_envs_mask] += 1
        self._state['num_edges'][dones] = 0

        # Get the difference of log-rewards.
        # --- MODIFICATION: Removed local_cache ---
        delta_scores = self.local_scores_wait(keys, data) # data is queued_data from async

        # --- CRITICAL CHECK ---
        # Ensure delta_scores has the correct shape BEFORE updating state
        if delta_scores.shape != (self.num_envs,):
            # This should ideally not happen with the fixes in local_scores_wait
            # but serves as a safeguard.
            raise ValueError(f"CRITICAL ERROR: delta_scores returned by local_scores_wait has incorrect shape {delta_scores.shape}. Expected ({self.num_envs},). Keys length: {len(keys)}")

        # Update the scores.
        self._state['score'] += delta_scores # This line caused the previous ValueError
        self._state['score'][dones] = 0

        # Mimic the Gymnasium VectorEnv.step return (obs, reward, terminated, truncated, info) - 5 values.
        truncations = np.zeros_like(dones, dtype=np.bool_)
        infos = {}

        return (deepcopy(self._state), delta_scores, dones, truncations, infos)


    def local_scores_async(self, sources, targets):
        keys, queued_data = [], []
        for i, (source, target) in enumerate(zip(sources, targets)):
            # FIX 1 (Part B): Use self.stop_action value and check source against num_variables
            # The stop action is now identified if the *action value* == self.stop_action
            # or equivalently if source == num_variables after divmod.
            if source == self.num_variables: # Stop action check
                key = (None, None, None) # Use None tuple for stop action key
                # Do not queue anything for stop action
            else:
                target_int = int(target) # Ensure stable hash key
                source_int = int(source)

                adjacency = self._state['adjacency'][i]
                # Ensure indices extraction is correct and uses Python ints
                indices = tuple(int(idx) for idx, is_parent
                                in enumerate(adjacency[:, target_int]) if is_parent) # Use target_int here

                # Create indices_after correctly
                indices_after_list = list(indices)
                bisect.insort(indices_after_list, source_int)
                indices_after = tuple(indices_after_list)

                # Key identifies the transition (target_int, parents_before, parents_after)
                key = (target_int, indices, indices_after)

                # --- ROBUSTNESS FIX ---
                # Always queue the calculation request.
                data_to_queue = (target_int, indices, indices_after) # Use ints and tuples
                if self.num_workers > 0:
                    self.in_queue.put(data_to_queue) # Send calculation request
                else:
                    queued_data.append(data_to_queue) # Store for local calculation
                # --- END ROBUSTNESS FIX ---

            keys.append(key)

        return keys, None, queued_data # queued_data is the 'data' passed to local_scores_wait

    def local_scores_wait(self, keys, data):
        # data (which was queued_data) contains the list of jobs potentially sent to workers.
        num_jobs_queued = len(data) # Number of non-stop actions that need calculation
        num_results_expected = 2 * num_jobs_queued # Each job yields 2 results (before/after)

        if self.num_workers > 0 and num_results_expected > 0:
            # We wait for exactly the number of results we will receive.
            for _ in range(num_results_expected):
                try:
                    is_success, key_from_worker, value, prior = self.out_queue.get(timeout=30) # Add timeout
                    if is_success:
                        # Ensure key_from_worker is in the correct format (int, tuple)
                        cache_key = (int(key_from_worker[0]), tuple(key_from_worker[1]))
                        # Always set the item. This updates its position in the LRUCache.
                        self.local_scores[cache_key] = value + prior
                    else:
                        _, exctype, value = self.error_queue.get()
                        raise exctype(f"Worker process failed for key {key_from_worker}: {value}")
                except queue.Empty:
                     raise TimeoutError("Timeout waiting for results from worker processes.")
                except Exception as e:
                     # Catch potential errors during queue processing
                     print(f"ERROR processing result from worker queue: {e}")
                     # Decide how to handle - maybe raise or try to continue

        elif num_jobs_queued > 0: # num_workers is 0, perform locally
            # data contains the calls that need to be made: (target_int, indices, indices_after)
            for target_int, indices, indices_after in data:
                try:
                    local_score_before, local_score_after = self.scorer.get_local_scores(
                        target_int, indices, indices_after=indices_after) # Pass int/tuples

                    # Always set/refresh the 'after' score
                    if local_score_after is not None:
                        cache_key_after = (target_int, tuple(local_score_after.key[1])) # Ensure tuple
                        self.local_scores[cache_key_after] = (
                            local_score_after.score + local_score_after.prior)

                    # Always set/refresh the 'before' score
                    if local_score_before is not None:
                         cache_key_before = (target_int, tuple(local_score_before.key[1])) # Ensure tuple
                         self.local_scores[cache_key_before] = (
                            local_score_before.score + local_score_before.prior)
                except Exception as e:
                    print(f"ERROR during local score calculation for target={target_int}, parents={indices}: {e}")
                    # Handle error, maybe continue with default scores?

        # Calculate delta_scores based on the original keys list (length num_envs)
        delta_scores_list = []
        for key_tuple in keys: # Iterate through the original keys list
            target, key_tm1, key_t = key_tuple

            if target is None: # Stop action
                delta_scores_list.append(0.)
            else:
                target_int = target # target should already be int from async
                delta_value = 0.0 # Default value in case of errors below

                # --- ROBUST APPEND LOGIC ---
                try:
                    # Construct cache keys ensuring tuple format for parents
                    cache_key_t = (target_int, tuple(key_t))
                    cache_key_tm1 = (target_int, tuple(key_tm1))

                    try:
                        # Attempt to retrieve scores from cache
                        score_t = self.local_scores[cache_key_t]
                        score_tm1 = self.local_scores[cache_key_tm1]
                        delta_value = score_t - score_tm1

                    except KeyError:
                        # Fallback calculation if key(s) not in cache
                        print(f"Warning: Score missing for key_t={cache_key_t} or key_tm1={cache_key_tm1}. Recalculating.")
                        print("Attempting synchronous recalculation as fallback...")
                        # print(f"[DEBUG] Cache miss details: target={target_int}, key_tm1={key_tm1}, key_t={key_t}")

                        try:
                            # Recalculate using target_int and original parent tuples (key_tm1, key_t)
                            fb_score, fa_score = self.scorer.get_local_scores(target_int, key_tm1, indices_after=key_t)

                            # Store recalculated scores back into cache
                            score_t_fallback = -np.inf # Use -inf to indicate failure
                            score_tm1_fallback = -np.inf

                            if fa_score is not None:
                                # Re-derive cache key from scorer result to be sure
                                cache_key_t_fb = (target_int, tuple(fa_score.key[1]))
                                score_t_fallback = fa_score.score + fa_score.prior
                                self.local_scores[cache_key_t_fb] = score_t_fallback
                                # print(f"[DEBUG] Stored fallback score_t for key: {cache_key_t_fb}")

                            if fb_score is not None:
                                # Re-derive cache key from scorer result to be sure
                                cache_key_tm1_fb = (target_int, tuple(fb_score.key[1]))
                                score_tm1_fallback = fb_score.score + fb_score.prior
                                self.local_scores[cache_key_tm1_fb] = score_tm1_fallback
                                # print(f"[DEBUG] Stored fallback score_tm1 for key: {cache_key_tm1_fb}")

                            # Use fallback scores if valid, otherwise keep delta_value = 0.0
                            if score_t_fallback > -np.inf and score_tm1_fallback > -np.inf:
                                delta_value = score_t_fallback - score_tm1_fallback
                                print("Recalculation successful.")
                            else:
                                 # This case happens if scorer returned None or inf/nan
                                 print(f"Recalculation failed (scorer returned invalid score?) for {cache_key_t}/{cache_key_tm1}. Falling back to delta=0.")
                                 delta_value = 0.0 # Explicitly set fallback delta

                        except Exception as e:
                            # Catch potential errors during scorer.get_local_scores
                            print(f"ERROR during fallback recalculation for target={target_int}, parents={key_tm1}->{key_t}: {e}")
                            print("Falling back to delta=0.")
                            delta_value = 0.0 # Ensure delta_value is set even on exception

                except Exception as e:
                     # Catch any other unexpected error during the processing of this key
                     print(f"ERROR processing delta score for key tuple {key_tuple}: {e}")
                     delta_value = 0.0 # Default to 0 on unexpected error


                # --- ENSURE APPEND HAPPENS ---
                delta_scores_list.append(delta_value)
                # --- END CHANGE ---


        # Convert the list to a NumPy array at the very end
        delta_scores_array = np.array(delta_scores_list, dtype=np.float64)

        # Final safety check (optional but good for debugging)
        if delta_scores_array.shape != (self.num_envs,):
             print(f"[WARN] Final delta_scores shape mismatch: {delta_scores_array.shape}. Expected ({self.num_envs},). List length: {len(delta_scores_list)}")

        return delta_scores_array


    def _is_in_cache(self, key):
        """Checks if key is in cache and refreshes it."""
        # Ensure key format matches storage: (int, tuple)
        cache_key = (int(key[0]), tuple(key[1]))
        if cache_key in self.local_scores:
            # Accessing via brackets refreshes LRU cache (LRUCache.__getitem__)
            _ = self.local_scores[cache_key]
            return True
        return False

    def close(self, **kwargs): # Gymnasium uses close()
        if self.num_workers > 0:
            # Check if processes list exists and has processes
            if hasattr(self, 'processes') and self.processes:
                 print("Closing worker processes...")
                 for _ in range(self.num_workers):
                      # Use try-except for robustness during shutdown
                      try:
                           self.in_queue.put(None) # Signal workers to exit
                      except Exception as e:
                           print(f"Error sending None to in_queue: {e}")

                 for process in self.processes:
                      try:
                           process.join(timeout=5) # Add timeout
                           if process.is_alive():
                                print(f"Warning: Worker process {process.pid} did not terminate gracefully. Terminating.")
                                process.terminate()
                                process.join() # Ensure termination
                      except Exception as e:
                           print(f"Error joining process {process.pid}: {e}")
                 print("Worker processes closed.")
            # Close queues safely
            if hasattr(self, 'in_queue'): self.in_queue.close()
            if hasattr(self, 'out_queue'): self.out_queue.close()
            if hasattr(self, 'error_queue'): self.error_queue.close()

    # It's good practice to implement __del__ for cleanup,
    # though close() is the standard Gymnasium method.
    def __del__(self):
        self.close()
