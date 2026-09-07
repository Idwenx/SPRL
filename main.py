import sys
import random
import numpy as np
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F
import src.utils as utils
from src.env import Env
from models.FeatureExtractor import GNNBackbone
from models.BPRSampler import BPRSampler
from models.SPRL import SPRL
from models.PPO import PPO
from models.DQN import DQN
from models.Degree import Degree
from models.NoIM import NoIM

from parser import parse_args

args = parse_args()
print(args)
utils.setSeed(args.seed)
device = args.device
D = args.embedding_dim
K = args.k
G = args.num_traj
M = args.num_training
lgn_layers = args.lightgcn_layers
N_EPISODES = args.episodes
MAX_STEPS = args.timesteps
BPR_EPOCHS = args.bpr_epoches
BPR_BATCH = args.batch_size
BPR_LR = args.bpr_lr
ABLATION = args.ablation
EVAL_EPISODES = 1
EVAL_INTERVAL = 1
LOAD_CHECKPOINT = f"checkpoints/{args.dataset}/{args.model}_{args.k}_{args.ablation}" 
EVAL_ONLY = args.eval
target_category = -1
n_categories = 0

def run():
    if args.dataset == "MovieLens":
        df, n_users, n_categories = utils.loadMovieLens('datasets/MovieLens', min_interactions=10)
        cache = utils.loadCache("MovieLens")
    elif args.dataset == "Amazon":
        df, n_users, n_categories = utils.loadAmazonBeauty('datasets/Amazon', min_interactions=5)
        cache = utils.loadCache("Amazon")
    elif args.dataset == "LastFM":
        df, n_users, n_categories = utils.loadLastFM('datasets/LastFM', min_interactions=35)
        cache = utils.loadCache("LastFM")

    edge_matrix = cache["edge_matrix"]
    category_jaccard = cache["category_jaccard"]
    preference_matrix = cache["preference_matrix"]

    print(preference_matrix[0])
    keep = cache["keep"]
    s0 = np.argmax(preference_matrix, axis=1).astype(np.int32)

    print("Number of edges:", np.sum(edge_matrix > 0))

    w = utils.getInfluenceWeight(edge_matrix, category_jaccard, preference_matrix)

    target_category = int(np.argmin(np.bincount(s0, minlength=n_categories)))
    print("Target category:", target_category)

    env = Env(pref_matrix=preference_matrix, w=w, k=K, n_categories=n_categories, target_category=target_category, max_steps=MAX_STEPS)

    norm_adj_ui, edge_index_uu, item_to_topic_map, n_users_kept, n_items = utils.buildGNNInputs(df, edge_matrix, keep)
    print("Number of items:", n_items)

    gnn = GNNBackbone(n_users_kept, n_items, n_categories, D, lgn_layers, norm_adj_ui, edge_index_uu, item_to_topic_map,
                      use_lightgcn=(ABLATION != "wo_LightGCN"),
                      use_sage=(ABLATION != "wo_SAGE")).to(device)

    if not EVAL_ONLY:
        if ABLATION == "full" or ABLATION == "wo_SAGE":
            bpr_sampler = BPRSampler(df, keep, n_items, args.seed)
            bpr_optimizer = torch.optim.Adam(gnn.parameters(), lr=BPR_LR)
            for epoch in range(BPR_EPOCHS):
                user_ids, pos_ids, neg_ids = bpr_sampler.sample(BPR_BATCH, device)
                bpr_optimizer.zero_grad()
                bpr_loss = gnn.compute_bpr_loss(user_ids, pos_ids, neg_ids)
                bpr_loss.backward()
                bpr_optimizer.step()
                if epoch % 100 == 0:
                    print(f"BPR Loss ({epoch}):", bpr_loss.item())
            print("BPR pretraining done")
            # gnn.freeze_embeddings()

    if args.model == "SPRL":
        model = SPRL(D=D, hidden_dim=4 * D, k=K, feature_extractor=gnn, lr=args.main_lr, group_size=G, beta_kl=args.kl_coef, beta_ent=args.entropy_coef, device=device)
        out_degree = edge_matrix.sum(axis=1)
        model.init_degree(out_degree)
    elif args.model == "PPO":
        model = PPO(D=D, hidden_dim=4 * D, k=K, feature_extractor=gnn, lr=args.main_lr, device=device)
        out_degree = edge_matrix.sum(axis=1)
        model.init_degree(out_degree)
    elif args.model == "DQN":
        n_nodes = preference_matrix.shape[0]
        model = DQN(D=D, hidden_dim=4 * D, k=K, feature_extractor=gnn, N=n_nodes, target_category=target_category, lr=args.main_lr, device=device)
        out_degree = edge_matrix.sum(axis=1)
        model.init_degree(out_degree)
    elif args.model == "Degree":
        model = Degree(k=K)
        out_degree = edge_matrix.sum(axis=1)
        model.init_degree(out_degree)
    elif args.model == "NoIM":
        model = NoIM()
        

    if EVAL_ONLY:
        print("Eval-only mode...")
        model.load(LOAD_CHECKPOINT)
        status, _ = env.reset()
        total_r = 0.0
        for t in range(env.max_steps):
            sv = torch.LongTensor(status).to(device)
            with torch.no_grad():
                nf = gnn.forward({"states": sv})
            graph_rep = nf.mean(dim=0)
            mask = torch.BoolTensor(env.available_mask()).to(device)
            actual = model.select_action(graph_rep, nf, mask)
            status, step_r, _, truncated, _ = env.step(actual.cpu().numpy())
            total_r += step_r
            print(f"  step {t:2d}: reward={step_r:+.4f}  categories={env.history[-1]}")
            if truncated:
                break
        print(f"Eval reward: {total_r:.4f}")
        print(f"Final categories: {env.history[-1]}")

        history_arr = np.array(env.history)   # [T, n_categories]
        utils.saveResults(args.dataset, cache_root="results", model_name=args.model, results=history_arr, k=K, ablation=ABLATION)
        print(f"History saved: results/{args.dataset}/history_{args.model}.npy  shape={history_arr.shape}")

        sys.exit(0)

    episodes_reward = 0.0
    best_eval_reward = -float("inf")
    for episode in range(N_EPISODES):
        status, _ = env.reset()
        episode_reward = 0.0

        for t in range(env.max_steps):
            sv = torch.LongTensor(status).to(device)
            nf = gnn.forward({"states": sv})       # [N, D]
            graph_rep = nf.mean(dim=0)                      # [D]
            mask = torch.BoolTensor(env.available_mask()).to(device)
            
            
            warmup = True if episode < args.warmup else False
            
            if args.model=="SPRL":
                loss, actual = model.train(graph_rep, nf, mask, env, num_traj=G, training_epoch=M, warmup=warmup, sv=sv)
            
            elif args.model == "PPO":
                loss, actual = model.train(graph_rep, nf, mask, env, num_traj=G, training_epoch=M, warmup=warmup, sv=sv)
                        
            elif args.model == "DQN":
                loss, actual = model.train(graph_rep, nf, mask, env, num_traj=G, warmup=warmup)
                
            status, step_reward, _, truncated, _ = env.step(actual.cpu().numpy())
                
            episode_reward += step_reward
            if truncated:
                break
                
        if episode % 1 == 0:
            print(f"episode {episode:4d}  loss={loss:.4f}  reward={episode_reward:.4f}")

        if episode > 0 and episode % EVAL_INTERVAL == 0:
            eval_rs, counts = evaluate(env, gnn, model)
            mean_r = float(eval_rs.mean())
            # mean_r = float(counts.mean())
            if mean_r > best_eval_reward:
                best_eval_reward = mean_r
                model.save(f"checkpoints/{args.dataset}/{args.model}_{K}_{ABLATION}")
                print(f"[Best] episode {episode}: eval reward {mean_r:.4f} -> saved")

    print("Evaluating...")
    eval_rewards, final_target_counts = evaluate(env, gnn, model)
    n = preference_matrix.shape[0]
    print(f"Eval reward:        {eval_rewards.mean():.4f} ± {eval_rewards.std():.4f}")
    print(f"Final target count: {final_target_counts.mean():.1f} ± {final_target_counts.std():.1f} / {n}"
        f"  ({final_target_counts.mean() / n * 100:.2f}%)")
    print(f"Best eval reward during training: {best_eval_reward:.4f}")

    print("Training done")

def evaluate(env, gnn, model):
    rewards = []
    final_counts = []
    for _ in range(EVAL_EPISODES):
        status, _ = env.reset()
        total_r = 0.0
        for _ in range(env.max_steps):
            sv = torch.LongTensor(status).to(device)
            with torch.no_grad():
                nf = gnn.forward({"states": sv})
            graph_rep = nf.mean(dim=0)
            mask = torch.BoolTensor(env.available_mask()).to(device)
            actual = model.select_action(graph_rep, nf, mask)
            status, step_r, _, truncated, _ = env.step(actual.cpu().numpy())
            total_r += step_r
            if truncated:
                break
            final_counts.append(np.bincount(status, minlength=n_categories)[target_category])
        rewards.append(total_r)
        
    return np.array(rewards), np.array(final_counts)

if __name__ == "__main__":
    run()