"""
@file: urban_func.py
@time: 2022/09/16
"""
import argparse
import os
import pickle as pkl
from collections import OrderedDict

from experiment.evaluator import land_use_inference, population_density_inference

"""
    Evaluate the embeddings on land use and population
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Singapore', help='City name, can be Singapore or NYC')
    # Task
    parser.add_argument('--task', type=str, default='land', help='Task to evaluate, can be land or pop (=Land Use '
                                                                 'Inference / Population Density Inference')
    parser.add_argument('--partition', type=str, default='default', help='Region partition, can be default(i.e., '
                                                                         'Singapore Subzones / NYC Census Tracts) or '
                                                                         'grid')

    # How many times to repeat the experiment
    parser.add_argument('--repeat', type=int, default=30)
    # split ratio
    parser.add_argument('--split', type=str, default='0.6,0.2,0.2')
    # verbose or not
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir('..')
    args = parse_args()
    city = args.city
    task = args.task
    partition = args.partition
    assert city in ['Singapore', 'NYC']
    assert task in ['land', 'pop']
    assert partition in ['default', 'grid']
    repeat = args.repeat
    split = [float(x) for x in args.split.split(',')]
    assert len(split) == 3
    verbose = args.verbose

    baseline_path = 'baselines/{}_doc2vec.pkl'.format(city.lower())

    # In paper, we evaluate 30 different embeddings from each baseline, therefore it tests 300 times land use and
    # 900 times population density for each baseline.
    # However, it will be too long for you to wait, so we randomly pick 1 embedding from each baseline.
    # We didn't cherry-pick, so our performance should be very close to the results in the paper, or even better.

    test_paths = {
        'Urban2Vec': 'baselines/{}_urban2vec.pkl'.format(city.lower()),
        'Place2Vec': 'baselines/{}_place2vec.pkl'.format(city.lower()),
        'Doc2Vec': 'baselines/{}_doc2vec.pkl'.format(city.lower()),
        'GAE': 'baselines/{}_gae.pkl'.format(city.lower()),
        'DGI': 'baselines/{}_dgi.pkl'.format(city.lower()),
        'Transformer': 'baselines/{}_transformer.pkl'.format(city.lower()),
    }
    if partition != 'default':
        if city == 'NYC':
            raise NotImplementedError('Grid partition is only available for Singapore data')
        baseline_path = 'baselines/{}_doc2vec_grid.pkl'.format(city.lower())
        test_paths['RegionDCL'] = 'embeddings/{}/RegionDCL.pkl'.format(city)
        for key in test_paths:
            test_paths[key] = test_paths[key][:-4] + '_grid.pkl'
    else:
        test_paths['RegionDCL-no random'] = 'embeddings/{}/RegionDCL_no_random.pkl'.format(city)
        test_paths['RegionDCL-fixed margin'] = 'embeddings/{}/RegionDCL_fixed_margin.pkl'.format(city)
        test_paths['RegionDCL'] = 'embeddings/{}/RegionDCL.pkl'.format(city)



    result = OrderedDict()
    with open(baseline_path, 'rb') as f:
        baseline_embeddings = pkl.load(f)
    if partition == 'default':
        with open('data/processed/' + city + '/downstream_region.pkl', 'rb') as f:
            raw_labels = pkl.load(f)
    else:
        with open('data/processed/' + city + '/downstream_grid.pkl', 'rb') as f:
            raw_labels = pkl.load(f)

    for key, value in test_paths.items():
        result[key] = land_use_inference(value, baseline_embeddings, raw_labels, split, repeat) if task == 'land' else \
            population_density_inference(value, baseline_embeddings, raw_labels, split, repeat)
    if task == 'land':
        print('=========================== Land Use Inference in Singapore ===========================')
        print(
            'Baseline'.ljust(30) + 'L1'.ljust(10) + 'std'.ljust(10) + 'KL-Div'.ljust(10) + 'std'.ljust(10) + 'Cosine'.ljust(
                10) + 'std'.ljust(10))
    else:
        print('=========================== Population Density Inference in Singapore ===========================')
        print(
            'Baseline'.ljust(30) + 'MAE'.ljust(10) + 'std'.ljust(10) + 'RMSE'.ljust(10) + 'std'.ljust(10) + 'R-Square'.ljust(
                10) + 'std'.ljust(10))
    for key, value in result.items():
        # keep 3 digits after decimal point
        value = [f'{x:.3f}' for x in value]
        # use ljust to align the output
        print(key.ljust(30) + value[0].ljust(10) + value[1].ljust(10) + value[2].ljust(10) + value[3].ljust(10) + value[
            4].ljust(10) + value[5].ljust(10))
