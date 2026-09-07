import numpy as np
import pandas as pd
import src.utils as utils
from parser import parse_args

args = parse_args()
'''
if args.dataset == "MovieLens":
    processing_MovieLens()
elif args.dataset == "LastFM":
    processing_LastFM()
elif args.dataset == "Amazon":
    processing_Amazon()

'''
def processing_MovieLens():
    df, n_users, n_categories = utils.loadMovieLens('datasets/MovieLens', min_interactions=10, top_k=4)
    ls_threshold = 0.25

    LinkSrengthMatrix = utils.JaccardSimilarity(df)
    np.fill_diagonal(LinkSrengthMatrix, 0)

    preference_matrix, s0 = utils.getPreference(df, n_users, n_categories)
    edge_matrix = utils.filterDirection(df, LinkSrengthMatrix, ls_threshold)
    category_jaccard = utils.categoryJaccard(df, n_users, n_categories)

    edge_matrix, preference_matrix, s0, keep = utils.filterIsolatedNodes(edge_matrix, preference_matrix, s0)

    category_jaccard = category_jaccard[:, keep, :][:, :, keep]

    print("Number of edges:", np.sum(edge_matrix > 0))

    w = utils.getInfluenceWeight(edge_matrix, category_jaccard, preference_matrix)

    utils.saveCache("MovieLens", edge_matrix=edge_matrix, category_jaccard=category_jaccard, preference_matrix=preference_matrix, keep=keep)
    
def processing_LastFM():
    df = pd.read_pickle('cache/LastFM(NonFilter)/df.pkl')
    cache = utils.loadCache("LastFM(NonFilter)")
    LinkSrengthMatrix = cache["LinkSrengthMatrix"]
    preference_matrix = cache["preference_matrix"]
    print(len(df))
    ls_threshold = 0.02

    n_users = preference_matrix.shape[0]
    n_categories = preference_matrix.shape[1]
    s0 = np.argmax(preference_matrix, axis=1).astype(np.int32)
    category_jaccard = cache["category_jaccard"]

    flat = LinkSrengthMatrix.ravel()
    print("LinkSrengthMatrix distribution:")
    print(f"  mean: {flat.mean():.6f}")
    print(f"  p25 : {np.percentile(flat, 25):.6f}")
    print(f"  p75 : {np.percentile(flat, 75):.6f}")
    print(f"  max : {flat.max():.6f}")
    print(f"  min : {flat.min():.6f}")

    edge_matrix = utils.filterDirection(df, LinkSrengthMatrix, ls_threshold)
    edge_matrix, preference_matrix, s0, keep = utils.filterIsolatedNodes(edge_matrix, preference_matrix, s0)
    print("Number of edges:", np.sum(edge_matrix > 0))

    category_jaccard = category_jaccard[:, keep, :][:, :, keep]

    print("Number of edges:", np.sum(edge_matrix > 0))

    w = utils.getInfluenceWeight(edge_matrix, category_jaccard, preference_matrix)

    utils.saveCache("LastFM",edge_matrix=edge_matrix, category_jaccard=category_jaccard, preference_matrix=preference_matrix, keep=keep)

    
def processing_Amazon():
    df, n_users, n_categories = utils.loadAmazonBeauty('datasets/Amazon', min_interactions=5, top_k=4)
    LinkSrengthMatrix = utils.JaccardSimilarity(df)
    np.fill_diagonal(LinkSrengthMatrix, 0)
    threshold = 0.95

    preference_matrix, s0 = utils.getPreference(df, n_users, n_categories)
    edge_matrix = utils.filterDirection(df, LinkSrengthMatrix, threshold)
    category_jaccard = utils.categoryJaccard(df, n_users, n_categories)
    print("Number of items:", len(df['itemId'].unique()))

    edge_matrix, preference_matrix, s0, keep = utils.filterIsolatedNodes(edge_matrix, preference_matrix, s0)

    category_jaccard = category_jaccard[:, keep, :][:, :, keep]

    print("Number of edges:", np.sum(edge_matrix > 0))

    w = utils.getInfluenceWeight(edge_matrix, category_jaccard, preference_matrix)

    utils.saveCache("Amazon",edge_matrix=edge_matrix, category_jaccard=category_jaccard, preference_matrix=preference_matrix, keep=keep)
    
processing_Amazon()
processing_LastFM()
processing_MovieLens()