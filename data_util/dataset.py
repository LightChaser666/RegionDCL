"""
@file: dataset.py
@time: 2022/09/14
This file load all necessary data from the disk.
"""
import os
import pickle as pkl
import random

import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree
from torch.utils.data import Dataset
from tqdm import tqdm


class CityData(object):
    def __init__(self, city, random_radius=100, with_type=True, with_random=True, cached_region_path=None,
                 cached_grid_path=None):
        assert city in ['NYC', 'Singapore']
        self.city = city
        # Try to load cached data
        in_path = 'data/processed/{}/'.format(city)
        # Create cache path
        cache_dir = 'cache/'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cached_pattern_path = 'cache/pattern_{}_{}_'.format(city, random_radius) + (
            'with_type' if with_type else '') + ('_no_random' if not with_random else '') + '.pkl'
        if os.path.exists(cached_pattern_path):
            with open(cached_pattern_path, 'rb') as f:
                self.patterns = pkl.load(f)
                self.building_feature_dim = self.patterns[0]['building_feature'].shape[1]
                for pattern in self.patterns:
                    if pattern['poi_feature'] is not None:
                        self.poi_feature_dim = pattern['poi_feature'].shape[1]
                        break
        else:
            # process pattern data
            with open(in_path + 'building.pkl', 'rb') as f:
                buildings = pkl.load(f)
            self.building_shape_feature = np.load(in_path + 'building_features.npy')
            self.building_rotation = np.load(in_path + 'building_rotation.npz')['arr_0']
            with open(in_path + 'random_point_' + str(random_radius) + 'm.pkl', 'rb') as f:
                self.random_points = pkl.load(f)
            # Poi outside buildings
            with open(in_path + 'poi.pkl', 'rb') as f:
                pois = pkl.load(f)
            self.building_feature_dim = 1 + 2 + self.building_shape_feature.shape[1] + len(buildings[0]['poi'])
            if with_type:
                self.building_feature_dim += len(buildings[0]['onehot'])
                random_path = 'data/processed/{}/random_point_with_type.npy'.format(self.city)
            else:
                random_path = 'data/processed/{}/random_point.npy'.format(self.city)
            if os.path.exists(random_path):
                self.random_feature = np.load(random_path)
            if not os.path.exists(random_path) or self.random_feature.shape[1] != self.building_feature_dim:
                self.random_feature = np.random.randn(1, self.building_feature_dim)
                np.save(random_path, self.random_feature)

            self.poi_feature_dim = len(pois[0]['onehot'])

            self.patterns = []
            # Road network segmentation
            print('Pre-calculating pattern features...')
            with open(in_path + f'segmentation_{random_radius}.pkl', 'rb') as f:
                raw_patterns = pkl.load(f)
            for pattern in tqdm(raw_patterns):
                if with_random:
                    building_num = len(pattern['building']) + len(pattern['random_point'])
                else:
                    building_num = len(pattern['building'])
                building_feature = np.zeros((building_num, self.building_feature_dim))
                for row, idx in enumerate(pattern['building']):
                    building_feature[row, 0] = buildings[idx]['shape'].area
                    building_feature[row, 1:3] = self.building_rotation[idx]
                    building_feature[row, 3:3 + self.building_shape_feature.shape[1]] = \
                        self.building_shape_feature[idx]
                    building_feature[row,
                    3 + self.building_shape_feature.shape[1]: 3 + self.building_shape_feature.shape[1] + len(
                        buildings[0]['poi'])] = \
                        buildings[idx]['poi']
                    if with_type:
                        building_feature[row, 3 + self.building_shape_feature.shape[1] + len(buildings[0]['poi']):] = \
                            buildings[idx]['onehot']

                if len(pattern['poi']) > 0:
                    poi_feature = np.zeros((len(pattern['poi']), self.poi_feature_dim))
                    for row, idx in enumerate(pattern['poi']):
                        poi_feature[row, :] = pois[idx]['onehot']
                else:
                    poi_feature = None
                xy = np.zeros((building_num, 2))
                for row, idx in enumerate(pattern['building']):
                    xy[row, 0] = buildings[idx]['shape'].centroid.x
                    xy[row, 1] = buildings[idx]['shape'].centroid.y

                if with_random:
                    for row, idx in enumerate(pattern['random_point']):
                        building_feature[len(pattern['building']) + row, :] = self.random_feature
                    for row, idx in enumerate(pattern['random_point']):
                        xy[len(pattern['building']) + row, :] = self.random_points[idx]

                self.patterns.append({
                    'building_feature': building_feature,
                    'poi_feature': poi_feature,
                    'xy': xy,
                })
            with open(cached_pattern_path, 'wb') as f:
                pkl.dump(self.patterns, f)
        if cached_region_path is None:
            cached_region_path = 'cache/region_{}.pkl'.format(city)
        if os.path.exists(cached_region_path):
            with open(cached_region_path, 'rb') as f:
                self.regions = pkl.load(f)
        else:
            with open(in_path + f'segmentation_{random_radius}.pkl', 'rb') as f:
                raw_patterns = pkl.load(f)
            with open(in_path + 'downstream_region.pkl', 'rb') as f:
                downstream_regions = pkl.load(f)
            self.regions = {}
            pattern_tree = KDTree(
                np.array([[pattern['shape'].centroid.x, pattern['shape'].centroid.y] for pattern in raw_patterns]))
            for idx, region in tqdm(enumerate(downstream_regions)):
                # calculate the diameter of the region
                bounds = region['shape'].bounds
                diameter = np.sqrt((bounds[2] - bounds[0]) ** 2 + (bounds[3] - bounds[1]) ** 2)
                # find all pattern centers within the diameter
                pattern_idx = pattern_tree.query_ball_point([region['shape'].centroid.x, region['shape'].centroid.y],
                                                            diameter)
                if len(pattern_idx) == 0:
                    print('No pattern found for region {}'.format(idx))
                    continue
                # find all patterns that are intersected with the region
                pattern_idx = [i for i in pattern_idx if raw_patterns[i]['shape'].intersects(region['shape'])]
                if len(pattern_idx) == 0:
                    print('No pattern found for region {}'.format(idx))
                    continue
                self.regions[idx] = pattern_idx
            with open(cached_region_path, 'wb') as f:
                pkl.dump(self.regions, f)
        if cached_grid_path is None:
            cached_grid_path = 'cache/grid_{}.pkl'.format(city)

        if os.path.exists(cached_grid_path):
            with open(cached_grid_path, 'rb') as f:
                self.grids = pkl.load(f)
        else:
            grid_path = 'data/raw/{}/grid/Singapore_tessellation_2km_square_projected.shp'.format(city)
            if os.path.exists(grid_path):
                grid_shapefile = gpd.read_file(grid_path)
                self.grids = {}
                with open(in_path + 'segmentation.pkl', 'rb') as f:
                    raw_patterns = pkl.load(f)
                pattern_tree = KDTree(
                    np.array([[pattern['shape'].centroid.x, pattern['shape'].centroid.y] for pattern in raw_patterns]))
                for idx, grid in tqdm(enumerate(grid_shapefile['geometry'])):
                    bounds = grid.bounds
                    diameter = np.sqrt((bounds[2] - bounds[0]) ** 2 + (bounds[3] - bounds[1]) ** 2)
                    pattern_idx = pattern_tree.query_ball_point([grid.centroid.x, grid.centroid.y], diameter)
                    if len(pattern_idx) == 0:
                        print('No pattern found for grid {}'.format(idx))
                        continue
                    pattern_idx = [i for i in pattern_idx if raw_patterns[i]['shape'].intersects(grid)]
                    if len(pattern_idx) == 0:
                        print('No pattern found for grid {}'.format(idx))
                        continue
                    self.grids[idx] = pattern_idx
                with open(cached_grid_path, 'wb') as f:
                    pkl.dump(self.grids, f)



class UnsupervisedPatternDataset(Dataset):
    def __init__(self, city_data: CityData):
        self.patterns = city_data.patterns

    def __getitem__(self, index):
        return self.patterns[index]

    def __len__(self):
        return len(self.patterns)

    @staticmethod
    def collate_fn_dropout(batch):
        """
                building_feature: [max_seq_len, batch_size, feature_dim]
                building_density: [max_seq_len, batch_size, density_dim]
                building_location: [max_seq_len, batch_size, location_dim]
                poi_feature: [max_seq_len, batch_size, feature_dim]
                poi_mask: [batch_size, max_seq_len]
                xy: [max_seq_len, batch_size, 2]
        """
        # duplicate the batch
        new_batch = []
        for pattern in batch:
            new_batch.append(pattern)
            new_batch.append(pattern)
        batch = new_batch
        return UnsupervisedPatternDataset.collate_fn(batch, 0.2)

    @staticmethod
    def collate_fn(batch, dropout=0.0, max_seq_len_limit=256):
        batch_size = len(batch)
        max_building_seq_len = 0
        building_feature_list = []
        xy_list = []
        positive = 0
        for pattern in batch:
            building_seq_len = pattern['building_feature'].shape[0]
            if building_seq_len > max_seq_len_limit:
                idx = np.random.choice(building_seq_len, max_seq_len_limit, replace=False)
                building_feature_list.append(pattern['building_feature'][idx, :])
                xy_list.append(pattern['xy'][idx, :])
                building_seq_len = max_seq_len_limit
            elif positive % 2 == 0 and dropout > 0 and int(building_seq_len * (1 - dropout)) > 2:
                idx = np.random.choice(building_seq_len, int(building_seq_len * (1 - dropout)), replace=False)
                building_feature_list.append(pattern['building_feature'][idx, :])
                accurate = pattern['xy'][idx, :]
                xy_list.append(accurate)
                building_seq_len = int(building_seq_len * (1 - dropout))
            else:
                building_feature_list.append(pattern['building_feature'])
                xy_list.append(pattern['xy'])
            positive += 1
            max_building_seq_len = max(max_building_seq_len, building_seq_len)
        building_feature = np.zeros((max_building_seq_len, batch_size, building_feature_list[0].shape[1]),
                                    dtype=np.float32)
        building_mask = np.ones((batch_size, max_building_seq_len), dtype=np.bool)
        xy = np.zeros((max_building_seq_len, batch_size, 2), dtype=np.float32)
        for i in range(batch_size):
            building_seq_len = building_feature_list[i].shape[0]
            building_feature[:building_seq_len, i, :] = building_feature_list[i]
            building_mask[i, :building_seq_len] = False
            xy[:building_seq_len, i, :] = xy_list[i]

        max_poi_seq_len = 0
        poi_feature_dim = 0
        for pattern in batch:
            if pattern['poi_feature'] is not None:
                max_poi_seq_len = max(max_poi_seq_len, pattern['poi_feature'].shape[0])
                poi_feature_dim = pattern['poi_feature'].shape[1]
        if max_poi_seq_len == 0:
            poi_feature = None
            poi_mask = None
        else:
            if max_poi_seq_len > max_seq_len_limit:
                max_poi_seq_len = max_seq_len_limit
            poi_feature = np.zeros((max_poi_seq_len, batch_size, poi_feature_dim), dtype=np.float32)
            poi_mask = np.ones((batch_size, max_poi_seq_len), dtype=np.bool)
            for i in range(batch_size):
                if batch[i]['poi_feature'] is not None:
                    poi_seq_len = batch[i]['poi_feature'].shape[0]
                    if poi_seq_len > max_seq_len_limit:
                        idx = np.random.choice(poi_seq_len, max_seq_len_limit, replace=False)
                        poi_feature[:max_seq_len_limit, i, :] = batch[i]['poi_feature'][idx, :]
                        poi_mask[i, :max_seq_len_limit] = 0
                    else:
                        poi_feature[:poi_seq_len, i, :] = batch[i]['poi_feature']
                        poi_mask[i, :poi_seq_len] = 0
        return building_feature, building_mask, xy, poi_feature, poi_mask


class FreezePatternPretrainDataset(Dataset):
    def __init__(self, patterns, city_data, window_size=2000):
        self.patterns = patterns
        self.city = city_data.city
        self.window_size = window_size
        raw_pattern_path = 'data/processed/{}/segmentation_100.pkl'.format(self.city)
        with open(raw_pattern_path, 'rb') as f:
            self.raw_patterns = pkl.load(f)
        self.centroids = [[pattern['shape'].centroid.x, pattern['shape'].centroid.y] for pattern in self.raw_patterns]
        # get the max_x, max_y, min_x, min_y of centroids
        self.max_x = max([centroid[0] for centroid in self.centroids])
        self.max_y = max([centroid[1] for centroid in self.centroids])
        self.min_x = min([centroid[0] for centroid in self.centroids])
        self.min_y = min([centroid[1] for centroid in self.centroids])
        # divide the city into grids
        self.grid_size = self.window_size / 2
        self.num_grid_x = int((self.max_x - self.min_x) / self.grid_size) + 1
        self.num_grid_y = int((self.max_y - self.min_y) / self.grid_size) + 1
        self.grid = [[[] for _ in range(self.num_grid_y)] for _ in range(self.num_grid_x)]
        for idx, centroid in enumerate(self.centroids):
            grid_x = int((centroid[0] - self.min_x) / self.grid_size)
            grid_y = int((centroid[1] - self.min_y) / self.grid_size)
            self.grid[grid_x][grid_y].append(idx)
        self.windows = {}
        for i in range(self.num_grid_x - 1):
            for j in range(self.num_grid_y - 1):
                current = []
                current.extend(self.grid[i][j])
                current.extend(self.grid[i + 1][j])
                current.extend(self.grid[i][j + 1])
                current.extend(self.grid[i + 1][j + 1])
                if len(current) > 0:
                    self.windows[(i, j)] = current
        self.anchors = {}
        self.overlaps = {}
        self.neighbors = {}

        for key, value in self.windows.items():
            idx = len(self.anchors)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    if (key[0] + i, key[1] + j) in self.windows:
                        if idx not in self.anchors:
                            self.anchors[idx] = key
                            self.overlaps[idx] = []
                            self.neighbors[idx] = []
                        self.overlaps[idx].append((key[0] + i, key[1] + j))
            if idx in self.anchors:
                neighbor_candidate = [
                    (key[0] - 2, key[1] - 2),
                    (key[0] - 2, key[1]),
                    (key[0] - 2, key[1] + 2),
                    (key[0], key[1] - 2),
                    (key[0], key[1] + 2),
                    (key[0] + 2, key[1] - 2),
                    (key[0] + 2, key[1]),
                    (key[0] + 2, key[1] + 2),
                ]
                for neighbor in neighbor_candidate:
                    if neighbor in self.windows:
                        self.neighbors[idx].append(neighbor)

        self.negative_keys = list(self.windows.keys())

    def __getitem__(self, index):
        anchor = self.windows[self.anchors[index]]
        positive = self.windows[random.choice(self.overlaps[index])]
        # easy_negative = random.random() < 0.5
        easy_negative = True
        if easy_negative or index not in self.neighbors or len(self.neighbors[index]) == 0:
            negative = random.choice(self.negative_keys)
            while negative in self.overlaps[index]:
                negative = random.choice(self.negative_keys)
        else:
            negative = random.choice(self.neighbors[index])
            while negative in self.overlaps[index]:
                negative = random.choice(self.negative_keys)
        negative = self.windows[negative]
        anchor = [self.patterns[i] for i in anchor]
        positive = [self.patterns[i] for i in positive]
        negative = [self.patterns[i] for i in negative]
        return anchor, positive, negative

    def __len__(self):
        return len(self.anchors)

    @staticmethod
    def collate_fn(batch):
        return batch[0]


class FreezePatternForwardDataset(Dataset):
    def __init__(self, patterns, city_data):
        self.patterns = patterns
        self.regions = city_data.regions
        self.idx2key = {idx: key for idx, key in enumerate(self.regions.keys())}

    def __getitem__(self, index):
        key = self.idx2key[index]
        pattern = [self.patterns[i] for i in self.regions[key]]
        return key, pattern

    def __len__(self):
        return len(self.regions)

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)
        regions = []
        for i in range(batch_size):
            key, patterns = batch[i]
            regions.append([key, patterns])
        return regions


if __name__ == '__main__':
    os.chdir('..')
    city_data = CityData('NYC')
