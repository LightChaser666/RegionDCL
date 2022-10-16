"""
@file: main.py
@time: 2022/09/21
"""
import argparse

import numpy as np
import torch

from data_util.dataset import CityData
from model.regiondcl import PatternEncoder, RegionEncoder
from model.trainer import PatternTrainer, RegionTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Singapore', help='City name, can be Singapore or NYC')
    parser.add_argument('--no_random', action='store_true', help='Whether to disable random points')
    parser.add_argument('--fixed', action='store_true', help='Whether to disable adaptive margin')
    parser.add_argument('--dim', type=int, default=64, help='Dimension of output representation')
    parser.add_argument('--d_feedforward', type=int, default=1024)
    parser.add_argument('--building_head', type=int, default=8)
    parser.add_argument('--building_layers', type=int, default=2)
    parser.add_argument('--building_dropout', type=float, default=0.2)
    parser.add_argument('--building_activation', type=str, default='relu')
    parser.add_argument('--bottleneck_head', type=int, default=8)
    parser.add_argument('--bottleneck_layers', type=int, default=2)
    parser.add_argument('--bottleneck_dropout', type=float, default=0.2)
    parser.add_argument('--bottleneck_activation', type=str, default='relu')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--save_name', type=str, default='pattern_embedding')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    city_data = CityData(args.city, with_random=not args.no_random)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pattern_encoder = PatternEncoder(d_building=city_data.building_feature_dim,
                                     d_poi=city_data.poi_feature_dim,
                                     d_hidden=args.dim,
                                     d_feedforward=args.d_feedforward,
                                     building_head=args.building_head,
                                     building_layers=args.building_layers,
                                     building_dropout=args.building_dropout,
                                     building_distance_penalty=1,
                                     building_activation=args.building_activation,
                                     bottleneck_head=args.bottleneck_head,
                                     bottleneck_layers=args.bottleneck_layers,
                                     bottleneck_dropout=args.bottleneck_dropout,
                                     bottleneck_activation=args.bottleneck_activation).to(device)
    # Encode building pattern
    pattern_optimizer = torch.optim.Adam(pattern_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pattern_scheduler = torch.optim.lr_scheduler.StepLR(pattern_optimizer, step_size=1, gamma=args.gamma)
    pattern_trainer = PatternTrainer(city_data, pattern_encoder, pattern_optimizer, pattern_scheduler)
    pattern_trainer.train_pattern_contrastive(epochs=20, save_name=args.save_name)
    region_aggregator = RegionEncoder(d_hidden=args.dim, d_head=8)
    region_aggregator.to(device)
    region_optimizer = torch.optim.Adam(region_aggregator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    region_scheduler = torch.optim.lr_scheduler.StepLR(region_optimizer, step_size=1, gamma=args.gamma)
    region_trainer = RegionTrainer(city_data, pattern_encoder, pattern_optimizer, pattern_scheduler, region_aggregator,
                                   region_optimizer, region_scheduler)
    # embeddings = pattern_trainer.get_embeddings()
    # Alternatively, you can load the trained pattern embedding
    embeddings = np.load(f'embeddings/{args.city}/{args.save_name}_20.npy')
    region_trainer.train_region_triplet_freeze(epochs=20, embeddings=embeddings, adaptive=not args.fixed, save_name='RegionDCL_',
                                               window_sizes=[1000, 2000, 3000])
    print('Training finished. Embeddings have been saved in embeddings/ directory.')
