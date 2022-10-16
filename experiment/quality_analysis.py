"""
@file: quality_analysis.py
@time: 2022/10/10
    This python script is used to analyze the quality of the embeddings.
"""
import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from tqdm import trange

from experiment.evaluator import Evaluator


def predict(test_path, baseline_embeddings, raw_labels, object_count, bins, repeat=1, land=True):
    with open(test_path, 'rb') as f:
        raw_embeddings = pkl.load(f)

    raw_embeddings = {k: v for k, v in raw_embeddings.items() if k in baseline_embeddings}
    embeddings = np.zeros((len(raw_embeddings), raw_embeddings[1].shape[0]), dtype=np.float32)
    labels = np.zeros((len(raw_embeddings), len(raw_labels[0]['land_use'])), dtype=np.float32)
    id2key = {}
    for i, (key, value) in enumerate(raw_embeddings.items()):
        embeddings[i] = value
        labels[i] = np.array(raw_labels[key]['land_use'])
        id2key[i] = key

    index = np.arange(len(embeddings))
    l1_dict = {}
    with trange(repeat) as t:
        t.set_description(f'Testing: {test_path}')
        for i in t:
            # KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=i)
            for train_idx, test_idx in kf.split(index):
                train_embeddings = embeddings[train_idx]
                test_embeddings = embeddings[test_idx]
                train_labels = labels[train_idx]
                test_labels = labels[test_idx]
                if land:
                    evaluator = Evaluator(train_embeddings, test_embeddings, train_labels, test_labels, seed=i)
                    _, best_l1_array = evaluator.predict(150)
                else:
                    rf = RandomForestRegressor(n_estimators=100, random_state=i, n_jobs=32)
                    rf.fit(train_embeddings, train_labels)
                    pred_labels = rf.predict(test_embeddings)
                    best_l1_array = np.abs(pred_labels - test_labels)
                for j, l1 in enumerate(best_l1_array):
                    if test_idx[j] not in l1_dict:
                        l1_dict[test_idx[j]] = []
                    l1_dict[test_idx[j]].append(l1)
    intervals = {}
    for key, value in l1_dict.items():
        if key not in object_count:
            continue
        objects = object_count[key]
        interval = -1
        for i in range(len(bins)):
            if objects < bins[i]:
                interval = i
                break
        if interval == -1:
            interval = len(bins) - 1
        if interval not in intervals:
            intervals[interval] = []
        intervals[interval].append(np.mean(value))
    points = []
    for key, value in intervals.items():
        points.append((key, np.mean(value)))
    points = sorted(points, key=lambda x: x[0])
    # use sns to plot bars
    return [p[1] for p in points]


def parse_args():
    parser = argparse.ArgumentParser(description='Quality Analysis')
    parser.add_argument('--city', type=str, default='Singapore', help='City, can be Singapore or NYC')
    parser.add_argument('--task', type=str, default='land', help='Task, can be land (land use inference) or pop ('
                                                                 'population density inference)')
    parser.add_argument('--repeat', type=int, default=5, help='Repeat')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.chdir('..')
    args = parse_args()
    city = args.city
    task = args.task
    repeat = args.repeat
    assert task in ['land', 'pop']
    assert city in ['Singapore', 'NYC']

    test_data = {
        'GAE': 'baselines/{}_gae.pkl'.format(city.lower()),
        'Urban2Vec': 'baselines/{}_urban2vec.pkl'.format(city.lower()),
        'Doc2Vec': 'baselines/{}_doc2vec.pkl'.format(city.lower()),
        'Place2Vec': 'baselines/{}_place2vec.pkl'.format(city.lower()),
        'DGI': 'baselines/{}_dgi.pkl'.format(city.lower()),
        'Transformer': 'baselines/{}_transformer.pkl'.format(city.lower()),
        'RegionDCL-no random': 'embeddings/{}/RegionDCL_no_random.pkl'.format(city),
        'RegionDCL': 'embeddings/{}/RegionDCL.pkl'.format(city)
    }

    baseline_path = 'baselines/{}_doc2vec.pkl'.format(city.lower())
    with open(baseline_path, 'rb') as f:
        baseline_embeddings = pkl.load(f)
    with open('data/processed/' + city + '/downstream_region.pkl', 'rb') as f:
        raw_labels = pkl.load(f)
    object_count = {}
    for key, value in enumerate(raw_labels):
        if key not in baseline_embeddings:
            continue
        object_count[key] = value['building_count'] + value['poi_count']
    quartile = np.percentile(list(object_count.values()), [25, 50, 75, 100])
    result = {}
    if task == 'land':
        for name, path in test_data.items():
            result[name] = predict(path, baseline_embeddings, raw_labels, object_count, bins=quartile, repeat=repeat)
    else:
        for name, path in test_data.items():
            result[name] = predict(path, baseline_embeddings, raw_labels, object_count, bins=quartile, repeat=repeat)
    #
    result['Building and POI amount within each region'] = [f'1-{int(quartile[0]) - 1}',
                                                            f'{int(quartile[0])}-{int(quartile[1]) - 1}',
                                                            f'{int(quartile[1])}-{int(quartile[2]) - 1}',
                                                            f'>={int(quartile[2])}']

    metric = 'L1-distance - Land Use Distribution' if task == 'land' else 'MAE - Population Density'
    cluster_standard = 'Building and POI amount within each region'
    df = pd.DataFrame(result)
    df = df.melt(id_vars=cluster_standard, var_name='Model',
                 value_name=metric)
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(x=cluster_standard, y=metric,
                     hue="Model", data=df)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    # save picture
    out_path = 'visualization/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig(f'visualization/{city.lower()}_data_sparsity_'
                + 'land_use' if task == 'land' else 'population' + '.jpg')
    plt.show()
