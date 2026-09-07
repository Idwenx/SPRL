import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--seed', type=int, default=4869, help='Seed init.')
    parser.add_argument('--model', default='SPRL', help='Choose a model from {SPRL, PPO, DQN, Degree, NoIM}')
    parser.add_argument('--device', default='cuda:0', help='Choose device, cpu or cuda')
    parser.add_argument('--dataset', nargs='?', default='MovieLens', help='Choose a dataset from {MovieLens, Amazon, LastFM}')
    parser.add_argument('--lightgcn_layers', type=int, default=2, help='Layers of LightGCN')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Layers of GNN')
    parser.add_argument('--k', type=int, default=20, help='Size of the seed set.')
    parser.add_argument('--warmup', type=int, default=20, help='Number of warmup episodes.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes.')
    parser.add_argument('--timesteps', type=int, default=25, help='Number of time steps per episode.')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Latent dimension.')
    parser.add_argument('--bpr_lr', type=float, default=1e-2, help='Learning rate for pretraining.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for BPR loss.')
    parser.add_argument('--bpr_epoches', type=int, default=1000, help='Number of epoches for pretraining.')
    parser.add_argument('--main_lr', type=float, default=1e-3, help='Learning rate for main model.')
    parser.add_argument('--num_traj', type=int, default=8, help='Number of sampled trajectories.')
    parser.add_argument('--num_training', type=int, default=4, help='Number of epoches for training SPRL')
    parser.add_argument('--kl_coef', type=float, default=1e-2, help='Coefficient for the KL term.')
    parser.add_argument('--entropy_coef', type=float, default=1e-2, help='Coefficient for the entropy term.')
    parser.add_argument('--eval', action='store_true', help='Evaluation.')
    parser.add_argument('--ablation', default='full',
                        choices=['wo_bpr', 'wo_LightGCN', 'wo_SAGE'],
                        help='Ablation setting.')
    return parser.parse_args()
