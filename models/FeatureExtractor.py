import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_mean

class GNNBackbone(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_topics: int, D: int, n_layers: int, 
                 norm_adj_ui, edge_index_uu, item_to_topic_map,
                 use_lightgcn: bool = True, use_sage: bool = True):
        super().__init__()

        self.use_lightgcn = use_lightgcn
        self.use_sage = use_sage

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, D)
        self.item_embedding = nn.Embedding(n_items, D)
        
        # Buffers
        self.register_buffer('norm_adj_ui', norm_adj_ui)
        self.register_buffer('edge_index_uu', edge_index_uu)
        
        self.n_users, self.n_items, self.n_topics = n_users, n_items, n_topics
        self.n_layers = n_layers
        self.register_buffer('item_topic_map', item_to_topic_map)
        
        self.l1 = nn.Linear(D * 2, D)
        if self.use_sage:
            self.sage_conv1 = SAGEConv(D, D, aggr='mean')
            self.sage_conv2 = SAGEConv(D, D, aggr='mean')
        # else: use_sage=False 时邻居聚合退化为简单平均池化(无参数, 见 _mean_pool)

    def forward(self, observations=None):
        ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        if self.use_lightgcn:
            all_feat = [ego]
            for _ in range(self.n_layers):
                ego = torch.sparse.mm(self.norm_adj_ui, ego)
                all_feat.append(ego)
            light_feat = torch.stack(all_feat, dim=1).mean(dim=1)  # [N, D]
        else:
            light_feat = ego
        u_feat, i_feat = light_feat.split([self.n_users, self.n_items])
        
        t_feat = scatter_mean(i_feat, self.item_topic_map, dim=0, dim_size=self.n_topics)
        
        if observations is None:
            return u_feat, i_feat
        
        status = observations["states"].long().clamp(0, self.n_topics - 1)
        
        curr_t = t_feat[status]                          # [N, D]
        static_u = u_feat
        x = torch.cat([static_u, curr_t], dim=-1)         # [N, 2D]

        ei = self.edge_index_uu
        
        if self.use_sage:
            h = self.l1(x)
            h = F.relu(self.sage_conv1(h, ei))
            h = self.sage_conv2(h, ei)
        else:
            h = self.l1(x)
            h = self._mean_pool(h)
        
        return h                                        # [N, D]

    def _mean_pool(self, h: torch.Tensor) -> torch.Tensor:
        ei = self.edge_index_uu
        src = torch.cat([ei[0], torch.arange(self.n_users, device=h.device)])
        dst = torch.cat([ei[1], torch.arange(self.n_users, device=h.device)])
        h = scatter_mean(h[src], dst, dim=0, dim_size=self.n_users)
        return h

    def compute_bpr_loss(self, user_ids, pos_ids, neg_ids):
        u_feat, i_feat = self.forward(observations=None)
        
        pos_score = (u_feat[user_ids] * i_feat[pos_ids]).sum(dim=1)
        neg_score = (u_feat[user_ids] * i_feat[neg_ids]).sum(dim=1)
        
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss

    def freeze_embeddings(self):
        self.user_embedding.weight.requires_grad = False
        self.item_embedding.weight.requires_grad = False
        print("[Freeze] user_embedding and item_embedding are freezing")

    def unfreeze_embeddings(self):
        for param in self.user_embedding.parameters():
            param.requires_grad = True
        for param in self.item_embedding.parameters():
            param.requires_grad = True
        print("[Unfreeze] user_embedding and item_embedding are unfreezing")