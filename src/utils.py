import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.special import softmax
from sklearn.metrics import pairwise_distances

def loadMovieLens(
    data_dir: str = "datasets/MovieLens",
    min_interactions: int = 10,
    top_k: int = 4,
) -> tuple[pd.DataFrame, int, int]:
    """
    加载 MovieLens 数据集并预处理.

    Args:
        data_dir:         数据集目录.
        min_interactions: 最小交互数过滤阈值.
        top_k:            保留物品数最多的 top_k 个 category.

    Returns:
        df:           DataFrame, 列 ['userId','itemId','rating','timestamp','category'].
        n_users:      用户数.
        n_categories: 类别数.
    """
    rating_columns = ['userId', 'movieId', 'rating', 'timestamp']
    ratings = pd.read_csv(f'{data_dir}/ratings.dat', sep='::', names=rating_columns,
                          engine='python', encoding='ISO-8859-1')
    movie_columns = ['movieId', 'title', 'genres']
    movies = pd.read_csv(f'{data_dir}/movies.dat', sep='::', names=movie_columns,
                         engine='python', encoding='ISO-8859-1')
    movies.drop('title', axis=1, inplace=True)
    data = pd.merge(ratings, movies, on='movieId', how='left')
    data['genres'] = data['genres'].str.split('|').str[0]

    df = data.copy()
    df.rename(columns={'movieId': 'itemId'}, inplace=True)
    df['rating'] = pd.to_numeric(df['rating'])

    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    new_df = df[df['userId'].isin(valid_users)]
    df = new_df

    top_genres = df.groupby('genres')['itemId'].nunique().nlargest(top_k).index.tolist()
    df = df[df['genres'].isin(top_genres)]
    genre_names = sorted(df['genres'].unique())

    user_id_map = {uid: i for i, uid in enumerate(sorted(df['userId'].unique()))}
    item_id_map = {iid: i for i, iid in enumerate(sorted(df['itemId'].unique()))}
    genre_id_map = {g: i for i, g in enumerate(genre_names)}

    df['userId'] = df['userId'].map(user_id_map)
    df['itemId'] = df['itemId'].map(item_id_map)
    df['genres'] = df['genres'].map(genre_id_map)
    df.columns = ['userId', 'itemId', 'rating', 'timestamp', 'category']

    n_users = len(user_id_map)
    n_categories = len(genre_id_map)

    return df, n_users, n_categories


def loadAmazonBaby(
    data_dir: str = "datasets/Amazon",
    min_interactions: int = 3,
    top_k: int = 4,
) -> tuple[pd.DataFrame, int, int]:
    """
    加载 Amazon 数据集并预处理.

    Args:
        data_dir:         数据集目录.
        min_interactions: 最小交互数过滤阈值.
        top_k:            保留物品数最多的 top_k 个 category.

    Returns:
        df:           DataFrame, 列 ['userId','itemId','rating','timestamp','category'].
        n_users:      用户数.
        n_categories: 类别数.
    """
    # 1. 读取 categories.txt: productId → category_name
    cat_map: dict[str, str] = {}
    current_pid: str | None = None
    with open(os.path.join(data_dir, 'categories.txt'), 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line:
                continue
            if not line.startswith(' '):
                current_pid = line
            elif current_pid is not None and current_pid not in cat_map:
                parts = [p.strip() for p in line.split(',')]
                if parts[0] == 'Baby Products' and len(parts) > 1:
                    cat_map[current_pid] = parts[1]
                else:
                    cat_map[current_pid] = parts[0]

    # 2. 解析 Baby.txt (key: value 格式, 空行分隔)
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    with open(os.path.join(data_dir, 'Baby.txt'), 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    pid = current.get('product/productId', '')
                    current['category'] = cat_map.get(pid, 'unknown')
                    records.append(current)
                    current = {}
            else:
                sep = line.find(': ')
                if sep != -1:
                    current[line[:sep]] = line[sep + 2:]
                elif current:
                    last_key = list(current.keys())[-1]
                    current[last_key] += ' ' + line
        if current:
            pid = current.get('product/productId', '')
            current['category'] = cat_map.get(pid, 'unknown')
            records.append(current)

    df = pd.DataFrame(records)
    df = df[['review/userId', 'product/productId', 'review/score', 'review/time', 'category']]
    df.columns = ['userId', 'itemId', 'rating', 'timestamp', 'category']
    df['rating'] = pd.to_numeric(df['rating'])

    print("The number of unique users before filtering: ", df['userId'].nunique())
    print("The number of unique items before filtering: ", df['itemId'].nunique())

    '''
    while True:
        item_counts = df['itemId'].value_counts()
        user_counts = df['userId'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        valid_users = user_counts[user_counts >= min_interactions].index
        new_df = df[df['itemId'].isin(valid_items) & df['userId'].isin(valid_users)]
        if len(new_df) == len(df):
            break
        df = new_df
    '''
    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    new_df = df[df['userId'].isin(valid_users)]
    df = new_df
    print("The number of unique users after filtering: ", df['userId'].nunique())
    print("The number of unique items after filtering: ", df['itemId'].nunique())

    top_genres = df.groupby('category')['itemId'].nunique().nlargest(top_k).index.tolist()
    df = df[df['category'].isin(top_genres)]
    genre_names = sorted(df['category'].unique())

    user_id_map = {uid: i for i, uid in enumerate(sorted(df['userId'].unique()))}
    item_id_map = {iid: i for i, iid in enumerate(sorted(df['itemId'].unique()))}
    genre_id_map = {g: i for i, g in enumerate(genre_names)}

    df['userId'] = df['userId'].map(user_id_map)
    df['itemId'] = df['itemId'].map(item_id_map)
    df['category'] = df['category'].map(genre_id_map)
    df = df[['userId', 'itemId', 'rating', 'timestamp', 'category']]

    n_users = len(user_id_map)
    n_categories = len(genre_id_map)

    return df, n_users, n_categories


def loadAmazonBeauty(
    data_dir: str = "datasets/Amazon",
    min_interactions: int = 3,
    top_k: int = 4,
) -> tuple[pd.DataFrame, int, int]:
    """
    加载 Amazon 数据集并预处理.

    Args:
        data_dir:         数据集目录.
        min_interactions: 最小交互数过滤阈值.
        top_k:            保留物品数最多的 top_k 个 category.

    Returns:
        df:           DataFrame, 列 ['userId','itemId','rating','timestamp','category'].
        n_users:      用户数.
        n_categories: 类别数.
    """
    # 1. 读取 categories.txt: productId → category_name
    cat_map: dict[str, str] = {}
    current_pid: str | None = None
    with open(os.path.join(data_dir, 'categories.txt'), 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line:
                continue
            if not line.startswith(' '):
                current_pid = line
            elif current_pid is not None and current_pid not in cat_map:
                parts = [p.strip() for p in line.split(',')]
                if parts[0] == 'Beauty' and len(parts) > 1:
                    cat_map[current_pid] = parts[1]
                else:
                    cat_map[current_pid] = parts[0]

    # 2. 解析 Baby.txt (key: value 格式, 空行分隔)
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    with open(os.path.join(data_dir, 'Beauty.txt'), 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    pid = current.get('product/productId', '')
                    current['category'] = cat_map.get(pid, 'unknown')
                    records.append(current)
                    current = {}
            else:
                sep = line.find(': ')
                if sep != -1:
                    current[line[:sep]] = line[sep + 2:]
                elif current:
                    last_key = list(current.keys())[-1]
                    current[last_key] += ' ' + line
        if current:
            pid = current.get('product/productId', '')
            current['category'] = cat_map.get(pid, 'unknown')
            records.append(current)

    df = pd.DataFrame(records)
    df = df[['review/userId', 'product/productId', 'review/score', 'review/time', 'category']]
    df.columns = ['userId', 'itemId', 'rating', 'timestamp', 'category']
    df['rating'] = pd.to_numeric(df['rating'])

    print("The number of unique users before filtering: ", df['userId'].nunique())
    print("The number of unique items before filtering: ", df['itemId'].nunique())

    '''
    while True:
        item_counts = df['itemId'].value_counts()
        user_counts = df['userId'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        valid_users = user_counts[user_counts >= min_interactions].index
        new_df = df[df['itemId'].isin(valid_items) & df['userId'].isin(valid_users)]
        if len(new_df) == len(df):
            break
        df = new_df
    '''
    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    new_df = df[df['userId'].isin(valid_users)]
    df = new_df
    print("The number of unique users after filtering: ", df['userId'].nunique())
    print("The number of unique items after filtering: ", df['itemId'].nunique())

    top_genres = df.groupby('category')['itemId'].nunique().nlargest(top_k).index.tolist()
    df = df[df['category'].isin(top_genres)]
    genre_names = sorted(df['category'].unique())

    user_id_map = {uid: i for i, uid in enumerate(sorted(df['userId'].unique()))}
    item_id_map = {iid: i for i, iid in enumerate(sorted(df['itemId'].unique()))}
    genre_id_map = {g: i for i, g in enumerate(genre_names)}

    df['userId'] = df['userId'].map(user_id_map)
    df['itemId'] = df['itemId'].map(item_id_map)
    df['category'] = df['category'].map(genre_id_map)
    df = df[['userId', 'itemId', 'rating', 'timestamp', 'category']]

    n_users = len(user_id_map)
    n_categories = len(genre_id_map)

    return df, n_users, n_categories


def loadLastFM(
    data_dir: str = "datasets/lastfm",
    min_interactions: int = 10,
    top_k: int = 4,
) -> tuple[pd.DataFrame, int, int]:
    """
    加载 LastFM (RecBole 格式) 数据集并预处理.

    输入文件:
        lastfm_recbole.inter: user_id:token  track_id:token  rating:float  timestamp:float
        lastfm_recbole.item:  track_id:token  tags:token_seq  artist_id:token  ...

    Args:
        data_dir:         数据集目录.
        min_interactions: 最小交互数过滤阈值.
        top_k:            保留物品数最多的 top_k 个 category (标签).

    Returns:
        df:           DataFrame, 列 ['userId','itemId','rating','timestamp','category'].
        n_users:      用户数.
        n_categories: 类别数 (标签数).
    """
    data_dir = 'datasets/lastfm'
    inter = pd.read_csv(f'{data_dir}/lastfm_recbole.inter', sep='\t')
    item = pd.read_csv(f'{data_dir}/lastfm_recbole.item', sep='\t')

    # 直接用 tags (token_seq 原文) 作为物品类别
    item = item[['track_id:token', 'tags:token_seq']].rename(
        columns={'tags:token_seq': 'category'})

    data = pd.merge(inter, item, on='track_id:token', how='left')
    data.columns = ['userId', 'itemId', 'rating', 'timestamp', 'category']

    df = data.copy()
        
    df['rating'] = pd.to_numeric(df['rating'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df[df['timestamp'].notna()]      # 过滤掉缺失 timestamp 的交互

    tag_list = ['2step', 'acidhouse', 'acidjazz', 'acidrock', 'acidtechno', 'acidtrance', 'acousticblues', 'afrobeat', 'afrobeats', 'afrocubanjazz', 'aggrotech', 'altcountry', 'alternative', 'alternativerock', 'amapiano', 'ambient', 'americana', 'anison', 'antifolk', 'artpop', 'artrock', 'atmosphericblackmetal', 'avantgarde', 'avantgardejazz', 'avantgardemetal', 'bachata', 'baltimoreclub', 'banda', 'baroque', 'baroquepop', 'basshouse', 'bassline', 'beatdown', 'bebop', 'bhangra', 'bigband', 'bigbeat', 'bigroom', 'blackeneddeathmetal', 'blackgaze', 'blackmetal', 'bluegrass', 'blues', 'bolero', 'bollywood', 'bongoflava', 'boogaloo', 'boogiewoogie', 'boombap', 'bossanova', 'breakbeat', 'breakcore', 'breaks', 'brega', 'britpop', 'brostep', 'brutaldeathmetal', 'cajun', 'calypso', 'cantopop', 'carnatic', 'celtic', 'celticpunk', 'chaabi', 'chachacha', 'chalga', 'chamberpop', 'chicagoblues', 'chicagohouse', 'chillout', 'chillstep', 'chillwave', 'choppedandscrewed', 'citypop', 'classical', 'cloudrap', 'coldwave', 'conscioushiphop', 'contemporaryclassical', 'cooljazz', 'corridos', 'country', 'countryblues', 'countrypop', 'countryrock', 'coupedecale', 'cpop', 'crossoverthrash', 'crunk', 'crustpunk', 'cumbia', 'dance', 'dancehall', 'dancepop', 'dangdut', 'darkambient', 'darkwave', 'deathcore', 'deathgrind', 'deathmetal', 'deephouse', 'deltablues', 'depressiveblackmetal', 'desertrock', 'detroittechno', 'disco', 'dixieland', 'djent', 'doommetal', 'doowop', 'downtempo', 'dreampop', 'drill', 'drone', 'dronemetal', 'drumandbass', 'dub', 'dubstep', 'dubtechno', 'easylistening', 'ebm', 'edm', 'electricblues', 'electro', 'electroacoustic', 'electroclash', 'electrohouse', 'electronic', 'electropop', 'electropunk', 'emo', 'emocore', 'emorap', 'enka', 'eurobeat', 'eurodance', 'exotica', 'experimental', 'fado', 'flamenco', 'folk', 'folkmetal', 'folkpop', 'folkpunk', 'folkrock', 'footwork', 'forro', 'freakfolk', 'freejazz', 'frenchhouse', 'funeraldoom', 'funk', 'funkmetal', 'funkyhouse', 'futurebass', 'futurehouse', 'futurepop', 'gabber', 'gamelan', 'gangstarap', 'garagerock', 'gfunk', 'ghazal', 'glammetal', 'glamrock', 'glitch', 'glitchhop', 'gnawa', 'goatrance', 'goregrind', 'gospel', 'gothicmetal', 'gothicrock', 'gqom', 'gregorian', 'grime', 'grindcore', 'grunge', 'gypsyjazz', 'happyhardcore', 'hardbop', 'hardcore', 'hardcorepunk', 'hardcoretechno', 'hardrock', 'hardstyle', 'hardtechno', 'hardtrance', 'harshnoise', 'heavymetal', 'highlife', 'hindustani', 'hiphop', 'honkytonk', 'house', 'hyphy', 'idm', 'impressionist', 'indie', 'indiefolk', 'indiepop', 'indierock', 'indipop', 'industrial', 'industrialmetal', 'irishfolk', 'italodisco', 'jackinhouse', 'janglepop', 'jazz', 'jazzfunk', 'jazzfusion', 'jerseyclub', 'jpop', 'jrock', 'juju', 'jumpblues', 'jumpup', 'jungle', 'kayokyoku', 'klezmer', 'kpop', 'krautrock', 'kwaito', 'laiko', 'lambada', 'latin', 'latinjazz', 'liquidfunk', 'lofi', 'lofihiphop', 'lofihouse', 'lounge', 'lukthung', 'makossa', 'mambo', 'mandopop', 'mariachi', 'mathcore', 'mathrock', 'mbalax', 'melodicdeathmetal', 'melodicdubstep', 'melodichardcore', 'merengue', 'metal', 'metalcore', 'microhouse', 'milonga', 'minimaltechno', 'modaljazz', 'moombahcore', 'moombahton', 'morna', 'motown', 'mpb', 'mumblerap', 'musiqueconcrete', 'neoclassical', 'neofolk', 'neosoul', 'neurofunk', 'newage', 'newwave', 'noise', 'norteño', 'northernsoul', 'nowave', 'nudisco', 'nujazz', 'numetal', 'nwobhm', 'opera', 'opm', 'outlawcountry', 'pagode', 'pfunk', 'phonk', 'piedmontblues', 'pinoypop', 'polka', 'pop', 'poppunk', 'poprock', 'postdubstep', 'posthardcore', 'postmetal', 'postpunk', 'postrock', 'powerelectronics', 'powermetal', 'powerpop', 'powerviolence', 'progressivehouse', 'progressivemetal', 'progressiverock', 'progressivetrance', 'protopunk', 'psychedelicfolk', 'psychedelicrock', 'psychobilly', 'psytrance', 'punk', 'punkrock', 'qawwali', 'quietstorm', 'raga', 'ragtime', 'rai', 'ranchera', 'rap', 'rapmetal', 'rebetiko', 'reggae', 'reggaeton', 'renaissance', 'retrowave', 'rhythmandblues', 'riddim', 'riotgrrrl', 'rock', 'rockabilly', 'rocknroll', 'rocksteady', 'romantic', 'rumba', 'salsa', 'samba', 'schranz', 'screamo', 'seashanty', 'sertanejo', 'sevdalinka', 'shoegaze', 'ska', 'skapunk', 'skramz', 'sludgemetal', 'smoothjazz', 'soca', 'softrock', 'soncubano', 'soukous', 'soul', 'soulfulhouse', 'souljazz', 'speedcore', 'speedgarage', 'speedmetal', 'spirituals', 'stonermetal', 'stonerrock', 'streetpunk', 'surfrock', 'swing', 'symphonicmetal', 'symphonicrock', 'synthpop', 'synthpunk', 'synthwave', 'tango', 'techhouse', 'technicaldeathmetal', 'techno', 'techstep', 'techtrance', 'tejano', 'texasblues', 'thrashmetal', 'timba', 'trance', 'trap', 'triphop', 'tropicalhouse', 'tropicalia', 'trot', 'turbofolk', 'twee', 'ukgarage', 'upliftingtrance', 'vallenato', 'vaporwave', 'vikingmetal', 'visualkei', 'vocaltrance', 'westernswing', 'witchhouse', 'wonky', 'worldbeat', 'worldmusic', 'zouk', 'zydeco']

    # 只保留 tag_list 中的流派标签, 并按首次出现顺序去重
    tag_set = set(tag_list)
    df['category'] = (
        df['category'].fillna('')
        .str.split()
        .map(lambda tags: ' '.join(dict.fromkeys(t for t in tags if t in tag_set)))
    )
    df = df[df['category'] != '']      # 去掉过滤后没有任何流派标签的行
    df['category'] = df['category'].str.split().str[0]   # 每个物品只保留第一个流派标签

    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['userId'].isin(valid_users)]

    top_tags = df.groupby('category')['itemId'].nunique().nlargest(top_k).index.tolist()
    df = df[df['category'].isin(top_tags)]
    tag_names = sorted(df['category'].unique())

    user_id_map = {uid: i for i, uid in enumerate(sorted(df['userId'].unique()))}
    item_id_map = {iid: i for i, iid in enumerate(sorted(df['itemId'].unique()))}
    tag_id_map = {t: i for i, t in enumerate(tag_names)}

    df['userId'] = df['userId'].map(user_id_map)
    df['itemId'] = df['itemId'].map(item_id_map)
    df['category'] = df['category'].map(tag_id_map)
    df = df[['userId', 'itemId', 'rating', 'timestamp', 'category']]

    n_users = len(user_id_map)
    n_categories = len(tag_id_map)

    return df, n_users, n_categories


def JaccardSimilarity(df: pd.DataFrame) -> np.ndarray:
    """
    计算用户间的 Jaccard 相似度矩阵(这里的实现使用了sklearn中的Jaccard距离).

    Jaccard(u, v) = |I_u ∩ I_v| / |I_u ∪ I_v|, 其中 I_u 为用户 u 交互过的物品集合.

    Args:
        df: DataFrame, 必须包含 userId 和 itemId 列.

    Returns:
        sim: np.ndarray, 形状 (n_users, n_users), sim[i][j] 为用户 i 与用户 j 的 Jaccard 相似度.
    """
    # 构建用户-物品二值矩阵
    user_item_matrix = pd.crosstab(df['userId'], df['itemId']).clip(upper=1).values

    # sklearn 的 jaccard 距离: d(u,v) = 1 - |I_u ∩ I_v| / |I_u ∪ I_v|
    # 相似度 = 1 - 距离

    dist = pairwise_distances(user_item_matrix.astype(bool), metric='jaccard')
    sim = 1.0 - dist

    return sim

def getPreference(
    df: pd.DataFrame, n_users: int, n_categories: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算用户对类别的偏好矩阵及初始状态.

    Args:
        df:           DataFrame, 必须包含 userId、category 和 rating 列.
        n_users:      用户数.
        n_categories: 类别数.

    Returns:
        preference_matrix: np.ndarray, 形状 (n_users, n_categories),
                           每行是用户对各类别的偏好概率分布.
        s0:                np.ndarray, 形状 (n_users,),
                           每个用户的初始状态 = 偏好最高的类别索引.
    """
    grouped = df.groupby(['userId', 'category']).agg(
        sum_rating=('rating', 'sum'),
        count=('rating', 'count'),
    ).reset_index()

    raw_pref = np.zeros((n_users, n_categories), dtype=np.float64)
    count_mat = np.zeros((n_users, n_categories), dtype=np.float64)

    raw_pref[grouped['userId'].values, grouped['category'].values] = grouped['sum_rating'].values
    count_mat[grouped['userId'].values, grouped['category'].values] = grouped['count'].values

    max_count = count_mat.max(axis=1, keepdims=True)
    max_count[max_count == 0] = 1.0

    raw_pref = raw_pref / max_count

    raw_pref_max = raw_pref.max(axis=1, keepdims=True)
    exp_pref = np.exp(raw_pref - raw_pref_max)
    preference_matrix = exp_pref / exp_pref.sum(axis=1, keepdims=True)

    s0 = np.argmax(preference_matrix, axis=1).astype(np.int32)

    return preference_matrix, s0

def filterDirection(
    df: pd.DataFrame,
    linkStrengthMatrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    将无向边转为有向边: 对每条 linkStrength > threshold 的边 (u, v),
    统计共同交互物品中谁先看的多, 先看多的为影响源.

    判断规则 (参考):
        count_u = |{ item ∈ I_u ∩ I_v : t_u(item) ≤ t_v(item) }|
        count_v = |{ item ∈ I_u ∩ I_v : t_v(item) <  t_u(item) }|
        若 count_u ≥ count_v → 边方向 u→v, 否则 v→u.

    Args:
        df:                 DataFrame, 必须包含 userId, itemId, timestamp 列.
        linkStrengthMatrix: 无向链接强度矩阵, 形状 (n_users, n_users).
        threshold:          边保留阈值.

    Returns:
        directed_adj: 有向邻接矩阵 (n_users, n_users), int8 类型,
                      directed_adj[u][v] = 1 表示存在有向边 u → v.
    """
    n_users = linkStrengthMatrix.shape[0]

    # 1. 构建 user → {item: timestamp}
    user_item_time: dict[int, dict[int, int]] = {}
    for uid, group in df.groupby('userId'):
        user_item_time[uid] = dict(zip(group['itemId'], group['timestamp']))

    # 2. 只取上三角且超过阈值的边 (无向, 每条边只判一次)
    triu_rows, triu_cols = np.triu_indices(n_users, k=1)
    mask = linkStrengthMatrix[triu_rows, triu_cols] > threshold
    edge_pairs = np.column_stack([triu_rows[mask], triu_cols[mask]])  # (E, 2)

    # 3. 逐边判断方向
    directed_adj = np.zeros((n_users, n_users), dtype=np.int8)

    for u, v in edge_pairs:
        u = int(u)
        v = int(v)

        u_dict = user_item_time.get(u, {})
        v_dict = user_item_time.get(v, {})

        common_items = set(u_dict.keys()) & set(v_dict.keys())

        count_u = 0
        count_v = 0
        for item in common_items:
            if u_dict[item] <= v_dict[item]:
                count_u += 1
            else:
                count_v += 1

        if count_u >= count_v:
            directed_adj[u, v] = 1
        else:
            directed_adj[v, u] = 1

    return directed_adj


def filterIsolatedNodes(
    directed_adj: np.ndarray,
    preference_matrix: np.ndarray | None = None,
    s0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    过滤掉既无出边也无入边的孤立节点.

    Args:
        directed_adj:      有向邻接矩阵, 形状 (n_users, n_users).
        preference_matrix: 偏好矩阵, 形状 (n_users, n_categories), 可选.
        s0:                种子节点向量, 形状 (n_users,), 可选.

    Returns:
        directed_adj_filtered:   过滤后的邻接矩阵, 形状 (n_connected, n_connected).
        preference_matrix_filtered: 过滤后的偏好矩阵 (或 None).
        s0_filtered:               过滤后的种子向量 (或 None).
        connected_mask:            bool 数组, 形状 (n_users,), 标记保留的节点.
    """
    has_out = np.any(directed_adj > 0, axis=1)  # 有出边
    has_in = np.any(directed_adj > 0, axis=0)   # 有入边
    connected_mask = has_out | has_in

    n_total = len(connected_mask)
    n_kept = connected_mask.sum()
    print(f"  filterIsolatedNodes: {n_total} → {n_kept} "
          f"(removed {n_total - n_kept} isolated nodes)")

    directed_adj_filtered = directed_adj[np.ix_(connected_mask, connected_mask)]

    preference_matrix_filtered = None
    if preference_matrix is not None:
        preference_matrix_filtered = preference_matrix[connected_mask]

    s0_filtered = None
    if s0 is not None:
        s0_filtered = s0[connected_mask]
    
    

    return directed_adj_filtered, preference_matrix_filtered, s0_filtered, connected_mask

def categoryJaccard(df: pd.DataFrame, n_users: int, n_categories: int) -> np.ndarray:
    """
    计算每个类别下的用户间 Jaccard 相似度矩阵.

    对每个 category g, 筛选该类别下的交互记录, 构建用户-物品二值矩阵,
    复用 JaccardSimilarity 的逻辑计算 sim^g[u][v].

    Args:
        df:           DataFrame, 必须包含 userId、itemId 和 category 列.
        n_users:      用户数.
        n_categories: 类别数.

    Returns:
        sim_mats: np.ndarray, 形状 (n_categories, n_users, n_users),
                  sim_mats[g] 为类别 g 下的用户 Jaccard 相似度矩阵.
    """
    sim_mats = np.zeros((n_categories, n_users, n_users))

    for c in range(n_categories):
        df_c = df[df['category'] == c]
        if len(df_c) == 0:
            continue

        # 构建用户-物品二值矩阵, 补全所有用户行
        user_item = (
            pd.crosstab(df_c['userId'], df_c['itemId'])
            .clip(upper=1)
            .reindex(index=range(n_users), fill_value=0)
        )

        dist = pairwise_distances(user_item.values.astype(bool), metric='jaccard')
        sim = 1.0 - dist

        # 全零行 (无交互用户) 会导致 NaN, 统一置 0
        sim[np.isnan(sim)] = 0.0
        sim_mats[c] = sim

    return sim_mats

def getInfluenceWeight(
    edge_matrix: np.ndarray,
    category_jaccard: np.ndarray,
    preference_matrix: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """
    计算有向边上的社会影响力权重

    Args:
        edge_matrix:       有向边掩码矩阵, 形状 (n_users, n_users), 0/1.
        category_jaccard:  各类别 Jaccard 相似度 CIS^n, 形状 (n_categories, n_users, n_users).
        preference_matrix: 用户偏好矩阵 P, 形状 (n_users, n_categories).
        beta:              平衡超参数 (默认 0.5).

    Returns:
        W: np.ndarray, 形状 (n_categories, n_users, n_users), float32,
           W[n][i][j] = w^n_{ij}, 无边处为 0.
    """
    n_users = edge_matrix.shape[0]
    n_categories = category_jaccard.shape[0]

    cos_dist = pairwise_distances(preference_matrix, metric='cosine')
    ps = 1.0 - cos_dist  # (n_users, n_users)

    w = np.zeros((n_categories, n_users, n_users))

    for n in range(n_categories):
        w[n] = edge_matrix * (
            beta * ps + (1.0 - beta) * category_jaccard[n]
        )

    return w


def propagation(
    w: np.ndarray,
    preference_matrix: np.ndarray,
    s0: np.ndarray,
    time_steps: int = 25,
) -> np.ndarray:
    """
    传播过程

    Args:
        w:                影响力权重矩阵, 形状 (n_categories, n_users, n_users),
                           w[n][u][v] = w^n_{uv}.
        preference_matrix: 偏好矩阵 P, 形状 (n_users, n_categories).
        s0:               初始状态, 形状 (n_users,).
        time_steps:       传播轮数.

    Returns:
        history: np.ndarray, 形状 (time_steps, n_categories), int32,
                 history[t][c] = 时刻 t 属于类别 c 的用户数.
    """
    n_categories, n_users, _ = w.shape
    status = s0.copy().astype(np.int32)
    history = np.zeros((time_steps, n_categories), dtype=np.int32)

    for t in range(time_steps):
        z = np.zeros((n_users, n_categories))
        for n in range(n_categories):
            incoming = w[n].T
            indicator = (status == n)
            z[:, n] = incoming @ indicator

        # 只在有正影响力的类别上做 softmax, 其余类别置 -inf → 概率为 0
        z_hat = softmax(np.where(z > 0, z, float('-inf')), axis=1)
        z_hat = np.nan_to_num(z_hat, nan=0.0)

        utility = preference_matrix + z_hat
        status = np.argmax(utility, axis=1).astype(np.int32)
        history[t] = np.bincount(status, minlength=n_categories)

    return history

def plotHistory(
    history: np.ndarray | dict[str, np.ndarray],
    n_categories: int,
    target_category: int | None = None,
    dataset_name: str = "",
    total_users: int | None = None,
):
    """
    绘制传播曲线.

    两种模式:
      1) 单次传播: history 为 (time_steps, n_categories) 的 np.ndarray,
         绘制各类别用户数随时间变化.
      2) 方法对比: history 为 {"Method": ndarray, ...} 的 dict,
         绘制 target_category 在该类别下的用户比例对比.

    Args:
        history:         (time_steps, n_categories) 或 {"name": (time_steps, n_categories)}.
        n_categories:    类别数.
        target_category: 对比模式下关注的目标类别索引.
        dataset_name:    图表标题/文件名前缀.
        total_users:     对比模式下的分母, None 则自动推断.
    """
    plt.style.use('seaborn-v0_8-dark')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'lines.markersize': 6,
    })

    markers = {"Degree": "o", "IMM": "s", "GRPO": "D", "No IM": "v",
               "Random": "^", "SPRL": "D", "DQN": "^", "PPO": "v"}

    # ---- 模式 2: 多方法对比 ----
    if isinstance(history, dict):
        if target_category is None:
            raise ValueError("dict 模式下必须指定 target_category")

        if total_users is None:
            total_users = max(
                int(h[0].sum()) for h in history.values()
            )

        plt.figure()
        for method_name, his in history.items():
            values = his[:, target_category].astype(np.float64) / total_users
            mk = markers.get(method_name, "o")
            plt.plot(range(len(values)), values, marker=mk,
                     label=method_name, markersize=6, linewidth=2.5)

        plt.xlabel("Time Step")
        plt.ylabel("Proportion of Total Users")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if dataset_name:
            plt.savefig(f"comparison_{dataset_name}.pdf", dpi=300, bbox_inches='tight')
        plt.show()
        return

    time_steps = history.shape[0]

    plt.figure()
    for c in range(n_categories):
        plt.plot(range(time_steps), history[:, c], marker='o', label=c)

    plt.xlabel("Time Step")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def saveResults(dataset_name: str, cache_root: str, model_name: str, results, k, ablation) -> None:
    cache_dir = os.path.join(cache_root, dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    path = os.path.join(cache_dir, f"{model_name}_{k}_{ablation}.npy")
    np.save(path, results)
    print(f"  saved: {path}  {results.shape}  {results.dtype}")
        
def saveCache(dataset_name: str, cache_root: str = "cache", **arrays: np.ndarray) -> None:
    """
    将 numpy 数组存入 cache/{dataset_name}/ 目录.

    Args:
        dataset_name: 数据集名称, 如 "MovieLens", "Amazon".
        cache_root:   缓存根目录.
        **arrays:     键值对, 如 edge_matrix=edge_matrix.
    """
    cache_dir = os.path.join(cache_root, dataset_name)
    os.makedirs(cache_dir, exist_ok=True)

    for name, arr in arrays.items():
        path = os.path.join(cache_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"  saved: {path}  {arr.shape}  {arr.dtype}")


def loadCache(
    dataset_name: str, cache_root: str = "cache", *keys: str
) -> dict[str, np.ndarray]:
    """
    从 cache/{dataset_name}/ 加载 numpy 数组.

    Args:
        dataset_name: 数据集名称.
        cache_root:   缓存根目录.
        *keys:        要加载的变量名; 不传则加载目录下所有 .npy 文件.

    Returns:
        data: dict[str, np.ndarray].
    """
    cache_dir = os.path.join(cache_root, dataset_name)

    if keys:
        paths = {k: os.path.join(cache_dir, f"{k}.npy") for k in keys}
    else:
        paths = {
            os.path.splitext(f)[0]: os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".npy")
        }

    data = {}
    for name, path in paths.items():
        if os.path.exists(path):
            data[name] = np.load(path)
            print(f"  loaded: {path}  {data[name].shape}")
        else:
            print(f"  [WARNING] not found: {path}")

    return data

def buildGNNInputs(
    df: pd.DataFrame,
    edge_matrix: np.ndarray,
    keep: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    构建 GNNBackbone 所需的输入。

    Args:
        df:          DataFrame, 必须包含 userId、itemId、category 列.
        edge_matrix: 过滤孤立节点后的有向用户-用户邻接矩阵 [N, N].
        keep:        filterIsolatedNodes 返回的保留掩码 [n_users] bool, None 则不过滤.

    Returns:
        norm_adj_ui:      LightGCN 归一化 user-item 邻接稀疏 tensor [(N+n_items) x 同].
        edge_index_uu:    [2, E] long tensor.
        item_to_topic_map: [n_items] long tensor.
        n_users_kept:     int
        n_items:          int
    """
    n_items = int(df['itemId'].max()) + 1

    item_to_topic_map = torch.LongTensor(
        np.array(df.groupby('itemId')['category'].first().sort_index().values, copy=True)
    )

    if keep is not None:
        keep_arr = np.asarray(keep)
        df_u = df[keep_arr[df['userId'].values]].copy()
        old_to_new = np.full(keep_arr.shape[0], -1, dtype=np.int64)
        old_to_new[keep_arr] = np.arange(int(keep_arr.sum()))
        df_u['userId'] = old_to_new[df_u['userId'].values]
        n_users_kept = int(keep_arr.sum())
    else:
        df_u = df
        n_users_kept = int(df['userId'].max()) + 1

    R = sp.coo_matrix(
        (
            np.ones(len(df_u)),
            (df_u['userId'].values, df_u['itemId'].values),
        ),
        shape=(n_users_kept, n_items),
    )

    adj = sp.bmat([[None, R], [R.T, None]])
    deg = np.asarray(adj.sum(axis=1)).squeeze()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1)), 0.0)
    norm_adj = sp.diags(d_inv_sqrt) @ adj @ sp.diags(d_inv_sqrt)
    norm_adj_coo = norm_adj.tocoo()

    norm_adj_ui = torch.sparse_coo_tensor(
        torch.LongTensor(np.stack([norm_adj_coo.row, norm_adj_coo.col], axis=0)),
        torch.FloatTensor(norm_adj_coo.data),
        size=(n_users_kept + n_items, n_users_kept + n_items),
    ).coalesce()

    src, dst = np.nonzero(edge_matrix > 0)
    edge_index_uu = torch.LongTensor(np.stack([src, dst], axis=0))

    return norm_adj_ui, edge_index_uu, item_to_topic_map, n_users_kept, n_items


def setSeed(seed: int) -> None:
    """
    固定所有随机源 (Python / numpy / torch CPU+GPU), 保证实验可复现.

    Args:
        seed: 随机种子 (非负整数).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


