import numpy as np
import torch
import torch.nn as nn




def valid_mean(tensor, valid=None, dim=None):
    """Mean of ``tensor``, accounting for optional mask ``valid``,
    optionally along a dimension."""
    dim = () if dim is None else dim
    if valid is None:
        return tensor.mean(dim=dim)
    valid = valid.type(tensor.dtype)  # Convert as needed.
    return (tensor * valid).sum(dim=dim) / valid.sum(dim=dim)

    
class ResBlock(nn.Module):
    def __init__(self,
                 feature_size,
                 action_size):
        super(ResBlock, self).__init__()

        self.lin_1 = nn.Linear(feature_size + action_size, feature_size)
        self.lin_2 = nn.Linear(feature_size + action_size, feature_size)

    def forward(self, x, action):
        res = nn.functional.leaky_relu(self.lin_1(torch.cat([x, action], -1)))
        res = self.lin_2(torch.cat([res, action], -1))
        return res + x

class ResForward(nn.Module):
    def __init__(self,
                 feature_size,
                 action_size):
        super(ResForward, self).__init__()

        self.lin_1 = nn.Linear(feature_size + action_size, feature_size)
        self.res_block_1 = ResBlock(feature_size, action_size)
        self.res_block_2 = ResBlock(feature_size, action_size)
        self.res_block_3 = ResBlock(feature_size, action_size)
        self.res_block_4 = ResBlock(feature_size, action_size)
        self.lin_last = nn.Linear(feature_size + action_size, feature_size)

    def forward(self, phi1, action):
        x = nn.functional.leaky_relu(self.lin_1(torch.cat([phi1, action], -1)))
        x = self.res_block_1(x, action)
        x = self.res_block_2(x, action)
        x = self.res_block_3(x, action)
        x = self.res_block_4(x, action)
        x = self.lin_last(torch.cat([x, action], -1))
        return x


class MlpEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        feature = [
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        ]
        self.feature = nn.Sequential(*feature)
    def forward(self, obs):
        return self.feature(obs)


class Disagreement(nn.Module):
    """Curiosity model for intrinsically motivated agents: similar to ICM
    except there is an ensemble of forward models that each make predictions.
    The intrinsic reward is defined as the variance between these predictions.
    """

    def __init__(
            self, 
            obs_dim, 
            action_dim,
            ensemble_size=5,
            prediction_beta=1.0,
            forward_loss_wt=0.2,
            # device="cpu",
            ):
        super(Disagreement, self).__init__()

        self.ensemble_size = ensemble_size
        self.prediction_beta = prediction_beta
        # self.device = device

        self.forward_loss_wt = forward_loss_wt

        self.feature_size = 512

        self.encoder = MlpEncoder(obs_dim[0], self.feature_size) #TODO: use simple encoder like model

        self.forward_model_1 = ResForward(feature_size=self.feature_size, action_size=action_dim)#.to(self.device)
        self.forward_model_2 = ResForward(feature_size=self.feature_size, action_size=action_dim)#.to(self.device)
        self.forward_model_3 = ResForward(feature_size=self.feature_size, action_size=action_dim)#.to(self.device)
        self.forward_model_4 = ResForward(feature_size=self.feature_size, action_size=action_dim)#.to(self.device)

    def forward(self, obs, next_obs, action):


        # if self.feature_encoding != 'none':
        #     phi1 = self.encoder(img1.view(T * B, *img_shape))
        #     phi1 = phi1.view(T, B, -1) # make sure you're not mixing data up here

        # obs: n_envs, *self.obs_shape
        # clipped_actions: n_envs, action_dim
        
        phi1 = self.encoder(obs)            # n_envs, feature_size
        phi2 = self.encoder(next_obs)       # n_envs, feature_size

        predicted_phi2 = []

        # NOTE: gradient not flowing to encoder
        predicted_phi2.append(self.forward_model_1(phi1.detach(), action.detach()))      # n_envs, feature_size
        predicted_phi2.append(self.forward_model_2(phi1.detach(), action.detach()))
        predicted_phi2.append(self.forward_model_3(phi1.detach(), action.detach()))
        predicted_phi2.append(self.forward_model_4(phi1.detach(), action.detach()))

        predicted_phi2_stacked = torch.stack(predicted_phi2)    # 5, n_envs, feature_size

        return phi1, phi2, predicted_phi2, predicted_phi2_stacked


    def compute_bonus(self, observations, next_observations, actions):
        phi1, phi2, predicted_phi2, predicted_phi2_stacked = self.forward(observations, next_observations, actions)
        feature_var = torch.var(predicted_phi2_stacked, dim=0) # feature variance across forward models             # variance between k, n_envs, feature_size
        reward = torch.mean(feature_var, axis=-1) # mean over feature       # n_envs
        return self.prediction_beta * reward

    def compute_loss(self, observations, next_observations, actions, valid):

        # observations: n_seq * max_length, obs_dim
        # actions = n_seq * max_length, action_dim
        # mask = n_seq * max_length

        #------------------------------------------------------------#
        # hacky dimension add for when you have only one environment (debugging)
        # if actions.dim() == 2: 
        #     actions = actions.unsqueeze(1)
        #------------------------------------------------------------#
        phi1, phi2, predicted_phi2, predicted_phi2_stacked = self.forward(observations, next_observations, actions)
        # actions = torch.max(actions.view(-1, *actions.shape[2:]), 1)[1] # conver action to (T * B, action_size), then get target indexes
        
        forward_loss_1 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[0], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss = torch.mean(forward_loss_1[valid])
        # forward_loss += valid_mean(forward_loss_1, valid)

        forward_loss_2 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[1], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += torch.mean(forward_loss_2[valid])
        # forward_loss += valid_mean(forward_loss_2, valid)

        forward_loss_3 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[2], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += torch.mean(forward_loss_3[valid])
        # forward_loss += valid_mean(forward_loss_3, valid)

        forward_loss_4 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[3], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += torch.mean(forward_loss_4[valid])
        # forward_loss += valid_mean(forward_loss_4, valid)

        return self.forward_loss_wt*forward_loss



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



class AtGoalPredictor(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim):
        super(AtGoalPredictor, self).__init__()

        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            MlpEncoder(feat_dim, hidden_dim),
            nn.ELU(),
            MlpEncoder(hidden_dim, output_dim),
            # nn.Sigmoid()
        )
    
    def forward(self, feat):
        at_goal = self.model(feat)  # nenv, 1
        return at_goal
        