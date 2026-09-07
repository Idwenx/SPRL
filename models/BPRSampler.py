"""BPR 训练三元组采样器：正样本取自用户真实交互历史。"""
import numpy as np
import pandas as pd
import torch


class BPRSampler:
    """从真实交互历史采样 BPR 训练三元组 (user, pos_item, neg_item).

    正样本必须取自用户在保留节点集合内的真实交互物品,
    否则正负样本同分布, BPR loss 只会震荡在 ln2 附近无法下降.
    负样本从全部物品中均匀采样.

    Args:
        df:      DataFrame, 必须包含 userId、itemId 列 (过滤孤立节点前的完整交互).
        keep:    filterIsolatedNodes 返回的保留掩码 [n_users] bool.
        n_items: 物品总数.
    """

    def __init__(self, df: pd.DataFrame, keep: np.ndarray, n_items: int, seed):
        keep_arr = np.asarray(keep)
        old_to_new = np.full(keep_arr.shape[0], -1, dtype=np.int64)
        old_to_new[keep_arr] = np.arange(int(keep_arr.sum()))

        df_keep = df[keep_arr[df['userId'].values]]
        uid_new = old_to_new[df_keep['userId'].values]
        item_ids = df_keep['itemId'].values

        self.n_items = int(n_items)
        self.n_users = int(keep_arr.sum())

        per_user: dict[int, list[int]] = {}
        for u, it in zip(uid_new.tolist(), item_ids.tolist()):
            per_user.setdefault(u, []).append(it)
        self.user_items = {u: np.array(v, dtype=np.int64) for u, v in per_user.items()}
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int, device: str | torch.device = "cpu"):
        """返回 (user_ids, pos_ids, neg_ids), 均为 [batch_size] long tensor."""
        rng = self.rng
        users = rng.integers(0, self.n_users, size=batch_size)
        pos = np.empty(batch_size, dtype=np.int64)
        for i, u in enumerate(users):
            items_u = self.user_items[int(u)]
            pos[i] = items_u[rng.integers(0, len(items_u))]
        neg = rng.integers(0, self.n_items, size=batch_size)

        return (
            torch.from_numpy(users).to(device),
            torch.from_numpy(pos).to(device),
            torch.from_numpy(neg).to(device),
        )
