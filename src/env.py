import numpy as np
from scipy.special import softmax
import gymnasium as gym

class Env(gym.Env):
    def __init__(
        self,
        pref_matrix: np.ndarray,
        w: np.ndarray,
        k: int,
        n_categories: int,
        target_category: int = 0,
        max_steps: int = 25,
    ):
        '''
        Args: 
            pref_matrix:    The preference matrix of shape (N, n_categories).
            w:              The weight matrix of shape (n_categories, N, N).
            k:              The number of seed nodes to select per episode.
            n_categories:   The number of categories.
            target_category: The category that seed nodes are forced to adopt.
            max_steps:       Maximum propagation steps per episode (truncation).
        '''
        super().__init__()

        self.N = pref_matrix.shape[0]
        self.pref_matrix = pref_matrix
        self.w = w
        self.n_categories = n_categories
        self.k = k
        self.target_category = target_category
        self.max_steps = max_steps

        self.s0 = np.argmax(pref_matrix, axis=1)
        self.status = self.s0.copy()
        self.current_step = 0

        self.action_space = gym.spaces.MultiDiscrete([self.N] * self.k)
        self.observation_space = gym.spaces.MultiDiscrete([self.n_categories] * self.N)
        self.history = [np.bincount(self.status, minlength=self.n_categories)]
        
        self.s0_target = self.history[-1][self.target_category]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.status = self.s0.copy()
        self.current_step = 0
        self.history = [np.bincount(self.status, minlength=self.n_categories)]
        return self.status, {}

    def _propagate(self, status: np.ndarray, seed_set: np.ndarray) -> np.ndarray:
        z = np.zeros((self.N, self.n_categories))
        for n in range(self.n_categories):
            incoming = self.w[n].T
            indicator = (status == n)
            z[:, n] = incoming @ indicator

        z_hat = softmax(np.where(z > 0, z, -np.inf), axis=1)
        z_hat = np.nan_to_num(z_hat, nan=0.0)

        utility = self.pref_matrix + z_hat
        status = np.argmax(utility, axis=1)
        # status[seed_set] = self.target_category
        return status

    def step(self, seed_set):
        self.status[seed_set] = self.target_category
        self.status = self._propagate(self.status, seed_set)
        self.history.append(np.bincount(self.status, minlength=self.n_categories))

        self.current_step += 1

        reward = (self.history[-1][self.target_category] - self.history[-2][self.target_category]) / self.N + (self.history[-1][self.target_category] - self.s0_target) / self.N
        # reward = self.history[-1][self.target_category]/self.N

        terminated = False
        truncated = self.current_step >= self.max_steps

        return self.status, reward, terminated, truncated, {}

    def evaluate(self, seed_set) -> float:
        """
        Args:
            seed_set: [k] 节点索引 (numpy / list)

        Returns:
            reward: float
        """
        seed_set = np.asarray(seed_set, dtype=np.int64)

        status_copy = self.status.copy()
        pre_counts = np.bincount(status_copy, minlength=self.n_categories)

        status_copy[seed_set] = self.target_category
        new_status = self._propagate(status_copy, seed_set)
        post_counts = np.bincount(new_status, minlength=self.n_categories)

        return (post_counts[self.target_category] - pre_counts[self.target_category]) / self.N

    def evaluate_batch(self, seed_sets) -> np.ndarray:
        seed_sets = np.asarray(seed_sets)
        return np.array([self.evaluate(s) for s in seed_sets])

    def available_mask(self) -> np.ndarray:
        return self.status == self.target_category