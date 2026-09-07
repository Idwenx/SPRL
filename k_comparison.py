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
K_LIST = [10, 20, 30, 40, 50]


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

    histories = {}
    for k in K_LIST:
        history = None
        for path in (f"results/{args.dataset}/SPRL_{k}_{args.ablation}.npy",
                     f"results/{args.dataset}/SPRL_{k}.npy"):
            if os.path.exists(path):
                history = np.load(path)
                print(f"[Load] {path}  shape={history.shape}")
                break

        if history is None:
            ckpt = f"{CKPT_DIR}/SPRL_{k}_full"
            if not os.path.exists(ckpt + "_actor"):
                print(f"[Skip] checkpoint not found: {ckpt}")
                continue

            env = Env(pref_matrix=preference_matrix, w=w, k=k,
                      n_categories=n_categories, target_category=target,
                      max_steps=args.timesteps)

            norm_adj_ui, edge_index_uu, item_to_topic_map, n_users_kept, n_items = gnn_inputs
            gnn = GNNBackbone(n_users_kept, n_items, n_categories, args.embedding_dim,
                              args.lightgcn_layers, norm_adj_ui, edge_index_uu,
                              item_to_topic_map).to(device)

            model = SPRL(D=args.embedding_dim, hidden_dim=4 * args.embedding_dim, k=k,
                         feature_extractor=gnn, lr=args.main_lr,
                         group_size=args.num_traj, beta_kl=args.kl_coef,
                         beta_ent=args.entropy_coef, device=device)
            model.load(ckpt)

            history = rolloutHistory(env, gnn, model, device)
            utils.saveResults(args.dataset, cache_root="results", model_name="SPRL",
                              results=history, k=k, ablation=args.ablation)
            print(f"K={k}: history shape {history.shape}")

        histories[f"K={k}"] = history

    if not histories:
        print("No checkpoint found, nothing to plot.")
        return

    utils.plotHistory(histories, n_categories,
                      target_category=target, dataset_name=f"{args.dataset}_k")


if __name__ == "__main__":
    main()
