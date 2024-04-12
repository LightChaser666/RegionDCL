"""
@file: NYCCensusTract.py
@time: 2022/09/13
This file preprocesses the NYC census tract data.
[
    {
        'shape': subzone shape, shapely.geometry.Polygon
        'land_use': 5-class land use ratio, numpy vector
        'population': population number, integer
    },
    ...
]
"""
import math
import os
import pickle as pkl

import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree, Delaunay
from shapely.geometry import Point
from tqdm import tqdm


class NYCCensusTract:
    def __init__(self):
        self.region_path = 'data/projected/NYC/region/region.shp'
        self.landuse_in_path = 'data/projected/NYC/landuse/landuse.shp'
        self.population_in_path = 'data/raw/NYC/population/worldpop_2020_un.csv'
        self.out_path = 'data/processed/NYC/downstream_region.pkl'
        self.merge_dict = {
            '01': 0,  # One &Two Family Buildings
            '02': 0,  # Multi-Family Walk-Up Buildings
            '03': 0,  # Multi-Family Elevator Buildings
            '04': 1,  # Mixed Residential & Commercial Buildings
            '05': 1,  # Commercial & Office Buildings
            '06': 2,  # Industrial & Manufacturing
            '07': 3,  # Transportation & Utility
            '08': 3,  # Public Facilities & Institutions
            '09': 4,  # Open Space & Outdoor Recreation
            '10': 3,  # Parking Facilities
            '11': 4,  # Vacant Land
        }

        # self.merge_dict = {
        #     '01': 0,  # One &Two Family Buildings
        #     '02': 1,  # Multi-Family Walk-Up Buildings
        #     '03': 2,  # Multi-Family Elevator Buildings
        #     '04': 3,  # Mixed Residential & Commercial Buildings
        #     '05': 4,  # Commercial & Office Buildings
        #     '06': 5,  # Industrial & Manufacturing
        #     '07': 6,  # Transportation & Utility
        #     '08': 7,  # Public Facilities & Institutions
        #     '09': 8,  # Open Space & Outdoor Recreation
        #     '10': 9,  # Parking Facilities
        #     '11': 10,  # Vacant Land
        # }




    def get(self, force=False):
        if os.path.exists(self.out_path) and not force:
            print('Loading NYC census tract data from disk...')
            with open(self.out_path, 'rb') as f:
                return pkl.load(f)
        # load region shapefile
        region_shapefile = gpd.read_file(self.region_path)
        regions = []
        region_dict = {}
        for index, row in region_shapefile.iterrows():
            name = row['BoroCT2020']
            region = {
                'name': name,
                'shape': row['geometry'],
                'land_use': [0.0] * 5,
                'population': 0
            }
            region_dict[name] = region
            regions.append(region)
        # load population data
        with open(self.population_in_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip().split(',')
                region_id = int(line[0])
                regions[region_id]['population'] = float(line[1])
        # load land use data
        print('Loading land use...')
        landuse_shapefile = gpd.read_file(self.landuse_in_path)
        print('Aggregating land use...')
        for index, row in tqdm(landuse_shapefile.iterrows(), total=len(landuse_shapefile)):
            BCT2020 = row['BCT2020']
            if BCT2020 not in region_dict:
                print('Census tract {} not found for landuse {}'.format(BCT2020, index))
                continue
            label = row['LandUse']
            if label is None:
                continue
            region_dict[BCT2020]['land_use'][self.merge_dict[label]] += row['geometry'].area
        # normalize land use
        for idx, region in enumerate(regions):
            region['population'] = region_dict[region['name']]['population']
            total_area = sum(region['land_use'])
            if total_area == 0:
                print('Land use area is 0 for census tract {}'.format(idx))
                continue
            region['land_use'] = [x / total_area for x in region['land_use']]
        # save
        with open(self.out_path, 'wb') as f:
            pkl.dump(regions, f)
        return regions

    def pack_poi_data(self):
        with open(self.out_path, 'rb') as f:
            regions = pkl.load(f)
        # load poi
        poi_in_path = 'data/projected/NYC/poi/poi.shp'
        pois_shapefile = gpd.read_file(poi_in_path)
        pois = []
        for index, poi_row in tqdm(pois_shapefile.iterrows(), total=pois_shapefile.shape[0]):
            output = {}
            # process point
            output['x'] = poi_row['geometry'].x
            output['y'] = poi_row['geometry'].y
            output['code'] = poi_row['code']
        code_dict = {}
        fclass_dict = {}
        for poi in pois:
            if poi['code'] not in code_dict:
                code_dict[poi['code']] = len(code_dict)
            if poi['fclass'] not in fclass_dict:
                fclass_dict[poi['fclass']] = len(fclass_dict)
        for poi in pois:
            poi['code'] = code_dict[poi['code']]
            poi['fclass'] = fclass_dict[poi['fclass']]
        # aggregate poi data
        print('Poi number:', len(pois))
        poi_loc = [[poi['x'], poi['y']] for poi in pois]
        poi_tree = KDTree(poi_loc)
        print('Aggregating poi...')
        count_no_point = 0
        for idx, region in tqdm(enumerate(regions)):
            region_shape = region['shape']
            region.pop('shape')
            # calculate region diameter
            dx = region_shape.bounds[2] - region_shape.bounds[0]
            dy = region_shape.bounds[3] - region_shape.bounds[1]
            diameter = math.sqrt(dx * dx + dy * dy) / 2
            # find poi in the region
            poi_index = poi_tree.query_ball_point(region_shape.centroid, diameter)
            for index in poi_index:
                if region_shape.contains(Point(poi_loc[index])):
                    pois[index]['region'] = idx
        print('Region without poi:', count_no_point)
        poi_within_region = [poi for poi in pois if 'region' in poi]
        # save to csv
        with open('data/processed/NYC/poi_for_baselines.csv', 'w') as f:
            f.write(' ,index,X,Y,FirstLevel,SecondLevel,PoiID,ZoneID')
            for idx, poi in enumerate(poi_within_region):
                f.write('\n')
                f.write(','.join([str(idx), str(idx), str(poi['x']), str(poi['y']), str(poi['code']), str(poi['fclass']), str(idx), str(poi['region'])]))

    def normalize_edge_weight(self, D,d):
        return np.log((1+np.power(D,1.5))/(np.power(d,1.5)))

    def pack_graph_data(self):
        """
            Process graph data for DGI & MVGRL
        """
        with open(self.out_path, 'rb') as f:
            regions = pkl.load(f)
        # load buildings & pois
        with open('data/processed/NYC/building.pkl', 'rb') as f:
            buildings = pkl.load(f)
        with open('data/processed/NYC/poi.pkl', 'rb') as f:
            pois = pkl.load(f)
        building_features = np.load('data/processed/NYC/building_resnet.npy')
        print('Building features shape:', building_features.shape)
        feature_dim = 1 + building_features.shape[1] + len(buildings[0]['onehot']) + len(buildings[0]['poi'])
        rows = len(buildings) + len(pois)
        overall_features = np.zeros((rows, feature_dim), dtype=np.float32)
        overall_locations = np.zeros((rows, 2), dtype=np.float32)
        # process buildings
        print('Processing buildings...')
        for idx, building in tqdm(enumerate(buildings), total=len(buildings)):
            overall_features[idx][0] = building['shape'].area
            overall_features[idx][1:1 + building_features.shape[1]] = building_features[idx]
            overall_features[idx][1 + building_features.shape[1]:1 + building_features.shape[1] + len(building['onehot'])] = building['onehot']
            overall_features[idx][1 + building_features.shape[1] + len(building['onehot']):] = building['poi']
            overall_locations[idx] = np.array([building['shape'].centroid.x, building['shape'].centroid.y])
        # process pois
        print('Processing pois...')
        for idx, poi in tqdm(enumerate(pois)):
            overall_features[len(buildings) + idx][feature_dim - len(poi['onehot']):] = poi['onehot']
            overall_locations[len(buildings) + idx] = np.array([poi['x'], poi['y']])
        region_graphs = {}
        tree = KDTree(overall_locations)
        print('Processing graphs...')
        for idx, region in tqdm(enumerate(regions)):
            region_shape = region['shape']
            # calculate region diameter
            dx = region_shape.bounds[2] - region_shape.bounds[0]
            dy = region_shape.bounds[3] - region_shape.bounds[1]
            diameter = math.sqrt(dx * dx + dy * dy) / 2
            # find poi in the region
            current_region = {}
            raw_index = tree.query_ball_point(region_shape.centroid, diameter)
            inside_index = []
            for index in raw_index:
                if region_shape.contains(Point(overall_locations[index])):
                    inside_index.append(index)
            if len(inside_index) < 4:
                continue
            # construct graph via Delaunay Triangulation
            current_region['features'] = overall_features[inside_index]
            locations = overall_locations[inside_index]
            graph = Delaunay(locations)
            # turn graph into edge list
            edge_list = np.zeros((graph.simplices.shape[0] * 6, 2), dtype=np.uint8)
            edge_weight = np.zeros((graph.simplices.shape[0] * 6, 1), dtype=np.float32)
            for i, (x, y, z) in enumerate(graph.simplices):
                edge_list[i * 6] = [x, y]
                edge_list[i * 6 + 1] = [x, z]
                edge_list[i * 6 + 2] = [y, x]
                edge_list[i * 6 + 3] = [y, z]
                edge_list[i * 6 + 4] = [z, x]
                edge_list[i * 6 + 5] = [z, y]
                xy = np.linalg.norm(locations[x] - locations[y])
                xz = np.linalg.norm(locations[x] - locations[z])
                yz = np.linalg.norm(locations[y] - locations[z])
                edge_weight[i * 6] = edge_weight[i * 6 + 2] = self.normalize_edge_weight(diameter*2, xy)
                edge_weight[i * 6 + 1] = edge_weight[i * 6 + 4] = self.normalize_edge_weight(diameter*2, xz)
                edge_weight[i * 6 + 3] = edge_weight[i * 6 + 5] = self.normalize_edge_weight(diameter*2, yz)
            current_region['edge_list'] = edge_list
            current_region['edge_weight'] = edge_weight
            region_graphs[idx] = current_region
        with open('data/processed/NYC/graph.pkl', 'wb') as f:
            pkl.dump(region_graphs, f, protocol=4)

if __name__ == '__main__':
    os.chdir('../../')
    NYCCensusTract().get(force=True)
