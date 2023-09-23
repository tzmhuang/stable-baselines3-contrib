import torch




class EmaExplorer():
    def __init__(self, n_envs, dim, rew_weight, p=1e-5):
        self.p = p
        self.rew_weight = rew_weight
        self._means = None  # n_envs, dim
        self._vars = None

        self.initialized = False
        self.eps = 1e-6
        return

    def step(self, state):
        # state: (B, n_envs, dim) B=1
        # 

        self._means, self._vars = self._get_update_means_and_vars(state)
        rew = -1 * self._get_gaussian_rew(state, self._means, torch.sqrt(self._vars))   # actually a punishment (1, n_envs, dim)

        return rew.mean(-1).squeeze()   # (n_envs)

    def _get_gaussian_rew(self, x, mu, sig):
        # x: (1, n_envs, dim)
        sig_eps = (sig + self.eps).unsqueeze(1)
        log_val = (x-mu.unsqueeze(1))/sig_eps
        return self.rew_weight * torch.exp(-0.5 * log_val ** 2) / sig_eps


    def _get_update_means_and_vars(self, data):
        # data: (B, n_envs, dim)
        # polyak averaging
        if not self.initialized:
            mu = data.mean(1)
            self.initialized = True
        else:
            mu = (1-self.p)*self._means + self.p * data.mean(1)

        var = torch.var(data, dim=1)
        return mu, var

