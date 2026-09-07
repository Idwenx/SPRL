import os

import numpy as np
import torch

import src.utils as utils
from src.env import Env
from models.FeatureExtractor import GNNBackbone
from models.SPRL import SPRL
from parser import parse_args

args = parse_args()

CKPT_DIR = f"checkpoints/{args.dataset}"

VARIANTS = [
    ("Full Model",   f"{CKPT_DIR}/SPRL_{args.k}_full",        True,  True),
    ("w/o BPR",      f"{CKPT_DIR}/SPRL_{args.k}_wo_bpr",      True,  True),
    ("w/o LightGCN", f"{CKPT_DIR}/SPRL_{args.k}_wo_LightGCN", False, True),
    ("w/o SAGE",     f"{CKPT_DIR}/SPRL_{args.k}_wo_SAGE",     True,  False),
]


def loadDataFrame(dataset: str):
    if dataset == "MovieLens":
        return utils.loadMovieLens('datasets/MovieLens', min_interactions=10)
    if dataset == "Amazon":
        return utils.loadAmazonBeauty('datasets/Amazon', min_interactions=5)
    if dataset == "LastFM":
        return utils.loadLastFM('datasets/LastFM', min_interactions=35)
    raise ValueError(f"Unknown dataset: {dataset}")


def rolloutHistory(env: Env, gnn: GNNBackbone, model: SPRL, device: str) -> np.ndarray:
    status, _ = env.reset()
    for _ in range(env.max_steps):
        sv = torch.LongTensor(status).to(device)
        with torch.no_grad():
            nf = gnn.forward({"states": sv})
        graph_rep = nf.mean(dim=0)
        mask = torch.BoolTensor(env.available_mask()).to(device)
        if mask.all():
            mask = torch.zeros_like(mask)
        actual = model.select_action(graph_rep, nf, mask)
        status, _, _, truncated, _ = env.step(actual.cpu().numpy())
        if truncated:
            break
    return np.array(env.history)


def main():
    device = args.device
    df, _, n_categories = loadDataFrame(args.dataset)

    cache = utils.loadCache(args.dataset)
    edge_matrix = cache["edge_matrix"]
    category_jaccard = cache["category_jaccard"]
    preference_matrix = cache["preference_matrix"]
    keep = cache.get("keep")

    s0 = np.argmax(preference_matrix, axis=1).astype(np.int32)
    target = int(np.argmin(np.bincount(s0, minlength=n_categories)))

    w = utils.getInfluenceWeight(edge_matrix, category_jaccard, preference_matrix)
    gnn_inputs = utils.buildGNNInputs(df, edge_matrix, keep)
    n_users = preference_matrix.shape[0]

    histories = {}
    for name, ckpt, use_lg, use_sage in VARIANTS:
        if not os.path.exists(ckpt + "_actor"):
            print(f"[Skip] checkpoint not found: {ckpt}")
            continue

        env = Env(pref_matrix=preference_matrix, w=w, k=args.k,
                  n_categories=n_categories, target_category=target,
                  max_steps=args.timesteps)

        norm_adj_ui, edge_index_uu, item_to_topic_map, n_users_kept, n_items = gnn_inputs
        gnn = GNNBackbone(n_users_kept, n_items, n_categories, args.embedding_dim,
                          args.lightgcn_layers, norm_adj_ui, edge_index_uu,
                          item_to_topic_map,
                          use_lightgcn=use_lg, use_sage=use_sage).to(device)

        model = SPRL(D=args.embedding_dim, hidden_dim=4 * args.embedding_dim, k=args.k,
                     feature_extractor=gnn, lr=args.main_lr,
                     group_size=args.num_traj, beta_kl=args.kl_coef,
                     beta_ent=args.entropy_coef, device=device)
        model.load(ckpt)

        hist = rolloutHistory(env, gnn, model, device)
        histories[name] = hist

        total_reward = (hist[-1, target] - hist[0, target]) / n_users
        final = hist[-1, target]
        print(f"{name:<12}: reward={total_reward:.4f}  "
              f"final target={final}/{n_users} ({final / n_users * 100:.2f}%)")

    if not histories:
        print("No checkpoint found, nothing to plot.")
        return

    utils.plotHistory(histories, n_categories,
                      target_category=target, dataset_name=f"{args.dataset}_ablation")


if __name__ == "__main__":
    main()
