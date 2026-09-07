import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .FeatureExtractor import GNNBackbone


class Actor(nn.Module):
    def __init__(self, D: int, hidden_dim: int, k: int = 10):
        super(Actor, self).__init__()

        self.state_dim = 2 * D
        self.virtual_dim = D
        self.k = k

        self.l1 = nn.Linear(self.state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, self.virtual_dim)

    def forward(self, state: torch.Tensor, node_features: torch.Tensor):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        virtual_rep = self.l3(x)
        return torch.matmul(virtual_rep, node_features.T)

    def get_logits(self, state: torch.Tensor, node_features: torch.Tensor, mask: torch.Tensor):
        logits = self.forward(state, node_features)
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))
        return logits


class Rollout:
    def __init__(self, k: int, D: int):
        self.D = D
        self.k = k
        self.virtual_dim = D

    def _step_state(self, state: torch.Tensor, graph_rep: torch.Tensor,
                    chosen_feat: torch.Tensor, step: int) -> torch.Tensor:
        old_seed = state[:, self.virtual_dim:]
        new_seed = (old_seed * step + chosen_feat) / (step + 1)
        return torch.cat([graph_rep, new_seed], dim=-1)

    # @torch.compile
    def degree_sample(self, graph_rep, degree, mask):
        G = graph_rep.shape[0]
        N = degree.shape[0]
        logits = degree.unsqueeze(0).repeat(graph_rep.shape[0], 1)

        all_indices = []
        all_log_probs = []

        for step in range(self.k):
            if mask is not None:
                logits = logits.masked_fill(mask, float('-inf'))

            log_probs = F.log_softmax(logits, dim=-1)
            action = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
            log_p = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)

            all_indices.append(action)
            all_log_probs.append(log_p)

            mask = mask.scatter(1, action.unsqueeze(1), True)

        indices = torch.stack(all_indices, dim=1)
        log_probs = torch.stack(all_log_probs, dim=1)

        return indices, log_probs

    # @torch.compile
    def sample(self, actor: Actor, graph_rep: torch.Tensor, node_features: torch.Tensor, mask):
        G = graph_rep.shape[0]
        N = node_features.shape[0]

        state = torch.cat([graph_rep, torch.zeros_like(graph_rep)], dim=-1)

        all_indices = []
        all_log_probs = []

        for step in range(self.k):
            logits = actor.get_logits(state, node_features, mask)
            log_probs = F.log_softmax(logits, dim=-1)
            action = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
            log_p = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)

            all_indices.append(action)
            all_log_probs.append(log_p)

            chosen_feat = node_features[action]
            state = self._step_state(state, graph_rep, chosen_feat, step)

            mask = mask.scatter(1, action.unsqueeze(1), True)

        indices = torch.stack(all_indices, dim=1)
        log_probs = torch.stack(all_log_probs, dim=1)
        return indices, log_probs

    # @torch.compile
    def select_deterministic(self, actor: Actor, graph_rep: torch.Tensor, node_features: torch.Tensor, mask):
        G = graph_rep.shape[0]
        N = node_features.shape[0]

        mask = mask.view(1, -1)

        state = torch.cat([graph_rep, torch.zeros_like(graph_rep)], dim=-1)  # [G, 2D]

        all_indices = []

        for step in range(self.k):
            logits = actor.get_logits(state, node_features, mask)
            action = logits.argmax(dim=-1)

            all_indices.append(action)

            chosen_feat = node_features[action]
            state = self._step_state(state, graph_rep, chosen_feat, step)

            mask = mask.scatter(1, action.unsqueeze(1), True)

        indices = torch.stack(all_indices, dim=1)
        return indices

    # @torch.compile
    def evaluate_log_probs(self, actor: Actor, graph_rep: torch.Tensor, node_features: torch.Tensor, mask, selected_indices: torch.Tensor, degree=None):
        G = graph_rep.shape[0]
        N = node_features.shape[0]

        state = torch.cat([graph_rep, torch.zeros_like(graph_rep)], dim=-1)  # [G, 2D]

        all_log_probs = []
        all_log_degree_probs = []
        total_entropy = 0.0

        for step in range(self.k):
            logits = actor.get_logits(state, node_features, mask)
            log_probs = F.log_softmax(logits, dim=-1)
            action = selected_indices[:, step]  # [G]
            log_p = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
            all_log_probs.append(log_p)

            if degree is not None:
                degree = degree.masked_fill(mask, float('-inf'))
                log_degree_probs = F.log_softmax(degree, dim=-1)
                log_degree_p = log_degree_probs.gather(1, action.unsqueeze(1)).squeeze(1)
                all_log_degree_probs.append(log_degree_p)

            safe_log_probs = log_probs.masked_fill(mask, 0.0)
            probs = safe_log_probs.exp()

            entropy = -(probs * safe_log_probs).sum(dim=-1)
            total_entropy += entropy.mean()

            chosen_feat = node_features[action]  # [G, D]
            state = self._step_state(state, graph_rep, chosen_feat, step)

            mask = mask.scatter(1, action.unsqueeze(1), True)

        all_log_probs = torch.stack(all_log_probs, dim=1)
        all_log_degree_probs = torch.stack(all_log_degree_probs, dim=1) if degree is not None else None
        return all_log_probs, all_log_degree_probs, total_entropy / self.k


class SPRL(object):
    def __init__(self, D: int, hidden_dim: int, k: int = 10, feature_extractor: GNNBackbone = None,
                 lr: float = 3e-4, clip_epsilon: float = 0.2, group_size: int = 8, beta_kl: float = 0.01,
                 beta_ent: float = 0.01, device="cpu"):
        self.device = device
        self.actor = Actor(D, hidden_dim, k).to(self.device)
        self.actor_ref = None
        self.k = k
        self.rollout = Rollout(k, D)
        self.feature_extractor = feature_extractor
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.group_size = group_size
        self.beta_kl = beta_kl
        self.beta_ent = beta_ent
        self.total_it = 0
        
        self.lr_changed = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.gnn_optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=self.lr)

    def init_degree(self, degree):
        self.degree = torch.FloatTensor(degree).to(self.device)

    def select_action(self, graph_rep: torch.Tensor, node_features: torch.Tensor, mask=None):
        graph_t = graph_rep.unsqueeze(0)
        nf_t = node_features

        with torch.no_grad():
            indices = self.rollout.select_deterministic(self.actor, graph_t, nf_t, mask)

        return indices.squeeze(0)

    def sample_trajectories(self, graph_rep: torch.Tensor, node_features: torch.Tensor, mask, G: int = None, init=False):
        if G is None:
            G = self.group_size

        graph_t = graph_rep
        nf_t = node_features

        if init:
            indices, log_probs = self.rollout.degree_sample(graph_t, self.degree, mask)
        else:
            with torch.no_grad():
                indices, log_probs = self.rollout.sample(self.actor, graph_t, nf_t, mask)

        return indices, log_probs

    def _compute_advantages(self, rewards: torch.Tensor):
        return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    def _compute_grpo_loss(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, ref_log_probs, advantages: torch.Tensor, entropy: torch.Tensor):
        ratio = torch.exp(new_log_probs - old_log_probs)
        importance_ratio = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        loss = -torch.min(importance_ratio, clipped_ratio).mean()

        log_ratio = ref_log_probs - new_log_probs

        loss += (log_ratio.exp() - log_ratio - 1).mean() * self.beta_kl
        loss -= self.beta_ent * entropy.mean()

        return loss

    def train(self, graph_rep, node_rep, mask, env, num_traj, training_epoch, warmup=False, sv=None):
        total_loss = 0.

        graph_rep = graph_rep.unsqueeze(0).repeat(num_traj, 1)  # (G, D)
        mask = mask.unsqueeze(0).repeat(num_traj, 1)

        indices, old_log_probs = self.sample_trajectories(graph_rep, node_rep, mask, num_traj, init=warmup)  # (G, K), (G, K)
        rewards = env.evaluate_batch(indices.cpu().numpy())
        rewards = torch.FloatTensor(rewards).to(self.device)
        advantages = self._compute_advantages(rewards).unsqueeze(1)
        seed_set = indices[rewards.argmax()]

        if warmup:
            actor_log_probs, _, _ = self.rollout.evaluate_log_probs(self.actor, graph_rep, node_rep, mask, indices)
            bc_loss = -actor_log_probs.mean()
            self.actor_optimizer.zero_grad()
            self.gnn_optimizer.zero_grad()

            bc_loss.backward()

            self.actor_optimizer.step()
            self.gnn_optimizer.step()

            return bc_loss.item(), seed_set
        else:
            if not self.lr_changed:
               self.set_lr(self.lr*0.1)
               self.lr_changed = True
                
            for epoch in range(training_epoch):
                self.total_it += 1
                node_rep = self.feature_extractor({"states": sv})
                graph_rep = node_rep.mean(dim=0)
                graph_rep = graph_rep.unsqueeze(0).repeat(num_traj, 1)
                new_log_probs, ref_log_probs, entropy = self.rollout.evaluate_log_probs(self.actor, graph_rep, node_rep, mask, indices, self.degree)
                # ref_log_probs = self.log_degree_prob.expand(num_traj, -1).gather(dim=1, index=indices)
                loss = self._compute_grpo_loss(old_log_probs, new_log_probs, ref_log_probs, advantages, entropy)
                self.gnn_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.gnn_optimizer.step()
                total_loss += loss.item()

        return total_loss, seed_set

    def set_lr(self, lr):
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = lr

        for param_group in self.gnn_optimizer.param_groups:
            param_group["lr"] = lr
            
    def save(self, filename):
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        if self.feature_extractor is not None:
            torch.save(self.feature_extractor.state_dict(),
                       filename + "_feature_extractor")
            torch.save(self.gnn_optimizer.state_dict(),
                       filename + "_gnn_optimizer")

    def load(self, filename, lr=None):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        if self.feature_extractor is not None:
            feat_path = filename + "_feature_extractor"
            if os.path.exists(feat_path):
                self.feature_extractor.load_state_dict(torch.load(feat_path))
            gnn_opt_path = filename + "_gnn_optimizer"
            if os.path.exists(gnn_opt_path):
                self.gnn_optimizer.load_state_dict(torch.load(gnn_opt_path))
        if lr is not None:
            for pg in self.actor_optimizer.param_groups:
                pg['lr'] = lr
            if self.gnn_optimizer is not None:
                for pg in self.gnn_optimizer.param_groups:
                    pg['lr'] = lr