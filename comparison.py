import numpy as np

import src.utils as utils
from src.env import Env
from parser import parse_args

args = parse_args()

RESULTS_DIR = f"results/{args.dataset}"


def main():
    cache = utils.loadCache(args.dataset)
    edge_matrix = cache["edge_matrix"]
    category_jaccard = cache["category_jaccard"]
    preference_matrix = cache["preference_matrix"]

    n_categories = preference_matrix.shape[1]
    s0 = np.argmax(preference_matrix, axis=1).astype(np.int32)
    target = int(np.argmin(np.bincount(s0, minlength=n_categories)))

    w = utils.getInfluenceWeight(edge_matrix, category_jaccard, preference_matrix)
    env = Env(pref_matrix=preference_matrix, w=w, k=args.k,
              n_categories=n_categories, target_category=target)

    sprl = np.load(f"{RESULTS_DIR}/SPRL_{args.k}_full.npy")
    degree = np.load(f"{RESULTS_DIR}/Degree_{args.k}.npy")
    imm = np.load(f"{RESULTS_DIR}/IMM_{args.k}.npy")
    dqn = np.load(f"{RESULTS_DIR}/DQN_{args.k}.npy")
    ppo = np.load(f"{RESULTS_DIR}/PPO_{args.k}.npy")
    noim = np.load(f"{RESULTS_DIR}/NoIM_{args.k}.npy")


    histories = {
        "SPRL": sprl,
        "Degree": degree,
        "IMM": imm,
        "No IM": noim,
        "DQN": dqn,
        "PPO": ppo,
    }

    utils.plotHistory(histories, n_categories,
                      target_category=target, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
