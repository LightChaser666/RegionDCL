"""
@file: SingaporeSubzone.py
@time: 2022/09/13
This file generates the downstream land use and population data for Singapore.
output format:
[
    {
        'name': subzone name, string
        'shape': subzone shape, shapely.geometry.Polygon
        'land_use': 5-class land use ratio, numpy vector
        'population': population number, integer
    },
    ...
]
"""
import math
import os
import re
import pickle as pkl
import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree, Delaunay, cKDTree
from shapely.geometry import Point
from tqdm import tqdm

class SingaporeSubzone:
    def __init__(self):
        self.region_path = 'data/projected/Singapore/region/subzone.shp'
        self.landuse_in_path = 'data/projected/Singapore/landuse/landuse.shp'
        self.population_in_path = 'data/raw/Singapore/population/worldpop_2020_un.csv'
        self.out_path = 'data/processed/Singapore/downstream_region.pkl'
        # merge land use to 5-class
        self.merge_dict = {
            # Residential Area (largest in singapore)
            'RESIDENTIAL': 0,
            'RESIDENTIAL WITH COMMERCIAL AT 1ST STOREY': 0,
            # Commercial Area
            'COMMERCIAL': 1,
            'COMMERCIAL & RESIDENTIAL': 1,
            'HOTEL': 1,
            # All 'BUSINESS' labels in Singapore actually refer to the industrial areas
            'BUSINESS 2': 2,
            'BUSINESS 1': 2,
            'BUSINESS PARK': 2,
            'BUSINESS 1 - WHITE': 2,
            'BUSINESS PARK - WHITE': 2,
            'BUSINESS 2 - WHITE': 2,
            # Public Service
            # 'ROAD': 3,
            # 'PORT / AIRPORT': 3,
            # 'TRANSPORT FACILITIES': 3,
            # 'MASS RAPID TRANSIT': 3,
            # 'LIGHT RAPID TRANSIT': 3,
            # Institution
            'EDUCATIONAL INSTITUTION': 3,
            'CIVIC & COMMUNITY INSTITUTION': 3,
            'COMMERCIAL / INSTITUTION': 3,
            'RESIDENTIAL / INSTITUTION': 3,
            'HEALTH & MEDICAL CARE': 3,
            'CEMETERY': 3,
            'PLACE OF WORSHIP': 3,
            'UTILITY': 3,
            'TRANSPORT FACILITIES': 3,
            'MASS RAPID TRANSIT': 3,
            'LIGHT RAPID TRANSIT': 3,
            'ROAD': 3,
            'PARK': 3,
            'PORT / AIRPORT': 3,
            # Open Space mainly includes open space, agriculture, and recreation
            # We merge other large empty areas into this category
            'BEACH AREA': 4,
            'RESERVE SITE': 4,
            'WATERBODY': 4,
            'SPECIAL USE': 4,
            'WHITE': 4,

            'OPEN SPACE': 4,
            'AGRICULTURE': 4,
            'RECREATION': 4,
            'SPORTS & RECREATION': 4,
            # TODO: Support and verify the merging strategy
        }

    def get(self, region_path=None, out_path=None, force=False):
        if out_path is None:
            out_path = self.out_path
        if not force and os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                return pkl.load(f)
        # load region
        if region_path is None:
            region_path = self.region_path
        region_shapefile = gpd.read_file(region_path)
        regions = []
        region_dict = {}
        for index, row in region_shapefile.iterrows():
            region = {}
            subzone_description = row['PopupInfo'].replace('\n', '').replace('\r', '')
            try:
                subzone_name = re.search(r'<th>SUBZONE_N</th><td>([0-9A-Z\s\'()\-]+)</td>', subzone_description).group(
                    1)
            except AttributeError:
                print('Error:', subzone_description)
                continue
            region['name'] = subzone_name
            region['shape'] = row['geometry']
            regions.append(region)
            region_dict[subzone_name] = region
        # load population
        with open(self.population_in_path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                line = line.strip().split(',')
                regions[int(line[0])]['population'] = line[1]
        # load land use
        landuses = self.init_land_use()
        # aggregate land use data
        landuse_loc = [[landuse['shape'].centroid.x, landuse['shape'].centroid.y] for landuse in landuses]
        landuse_tree = KDTree(landuse_loc)
        print('Aggregating land use...')
        for region in tqdm(regions):
            region_shape = region['shape']
            region['land_use'] = [0.0] * 5
            # calculate region diameter
            dx = region_shape.bounds[2] - region_shape.bounds[0]
            dy = region_shape.bounds[3] - region_shape.bounds[1]
            diameter = math.sqrt(dx * dx + dy * dy)
            # find land use in the region
            landuse_index = landuse_tree.query_ball_point(region_shape.centroid, diameter)
            for index in landuse_index:
                if region_shape.intersects(landuses[index]['shape']):
                    current_land_use = landuses[index]
                    region['land_use'][current_land_use['land_use']] += region_shape.buffer(0).intersection(
                        current_land_use['shape'].buffer(0)).area
            # normalize
            total_area = sum(region['land_use'])
            if total_area > 0:
                region['land_use'] = [x / total_area for x in region['land_use']]
        # save
        with open(out_path, 'wb') as f:
            pkl.dump(regions, f)
        return regions

    def get_grid(self, region_path=None, out_path=None, force=False):
        if out_path is None:
            out_path = self.out_path
        if not force and os.path.exists(out_path):
            with open(out_path, 'rb') as f:
                return pkl.load(f)
        # load region
        if region_path is None:
            region_path = self.region_path
        region_shapefile = gpd.read_file(region_path)
        regions = []
        for index, row in region_shapefile.iterrows():
            region = {}
            region['shape'] = row['geometry']
            regions.append(region)
        # load population
        # load land use
        landuses = self.init_land_use()
        # aggregate land use data
        landuse_loc = [[landuse['shape'].centroid.x, landuse['shape'].centroid.y] for landuse in landuses]
        landuse_tree = KDTree(landuse_loc)
        print('Aggregating land use...')
        for region in tqdm(regions):
            region_shape = region['shape']
            region['land_use'] = [0.0] * 5
            # calculate region diameter
            dx = region_shape.bounds[2] - region_shape.bounds[0]
            dy = region_shape.bounds[3] - region_shape.bounds[1]
            diameter = math.sqrt(dx * dx + dy * dy) * 2
            # find land use in the region
            landuse_index = landuse_tree.query_ball_point(region_shape.centroid, diameter)
            for index in landuse_index:
                if region_shape.intersects(landuses[index]['shape']):
                    current_land_use = landuses[index]
                    region['land_use'][current_land_use['land_use']] += region_shape.intersection(
                        current_land_use['shape'].buffer(0)).area
            # normalize
            total_area = sum(region['land_use'])
            if total_area > 0:
                region['land_use'] = [x / total_area for x in region['land_use']]
        # save
        with open(out_path, 'wb') as f:
            pkl.dump(regions, f)
        return regions

    def init_land_use(self):
        landuse_shapefile = gpd.read_file(self.landuse_in_path)
        landuses = []
        print('Loading land use...')
        for index, row in tqdm(landuse_shapefile.iterrows(), total=len(landuse_shapefile)):
            landuse_description = row['PopupInfo'].replace('\n', '').replace('\r', '')
            try:
                label = re.search('<th>LU_DESC</th><td>([0-9A-Z\s\'()\-&/]+)</td>', landuse_description).group(1)
            except AttributeError:
                print('Error:', landuse_description)
                continue
            landuse = {
                'shape': row['geometry'],
                'land_use': self.merge_dict[label]
            }
            landuses.append(landuse)
        # save the land use shapefile
        return landuses

    def show_land_use(self):
        landuses = self.init_land_use()
        print('Showing land use...')
        import matplotlib.pyplot as plt
        color_dict = {
            0: 'green',
            1: 'blue',
            2: 'red',
            3: 'yellow',
            4: 'black'
        }
        plt.figure(figsize=(50, 30))
        for landuse in tqdm(landuses):
            if landuse['shape'].centroid.x > 65000:
                continue
            plt.fill(*landuse['shape'].exterior.xy, color=color_dict[landuse['land_use']],alpha=0.8)
        plt.show()

    def pack_poi_data(self, regions=None, save_path='data/processed/Singapore/poi_for_baselines.csv'):
        if regions is None:
            with open(self.out_path, 'rb') as f:
                regions = pkl.load(f)
        # load poi
        poi_in_path = 'data/projected/Singapore/poi/poi.shp'
        pois_shapefile = gpd.read_file(poi_in_path)
        pois = []
        for index, poi_row in tqdm(pois_shapefile.iterrows(), total=pois_shapefile.shape[0]):
            output = {}
            # process point
            output['x'] = poi_row['geometry'].x
            output['y'] = poi_row['geometry'].y
            output['code'] = poi_row['code']
            output['fclass'] = poi_row['fclass']
            pois.append(output)
        # turn code & fclass into numbers
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
        with open(save_path, 'w') as f:
            f.write(' ,index,X,Y,FirstLevel,SecondLevel,PoiID,ZoneID')
            for idx, poi in enumerate(poi_within_region):
                f.write('\n')
                f.write(','.join([str(idx), str(idx), str(poi['x']), str(poi['y']), str(poi['code']), str(poi['fclass']), str(idx), str(poi['region'])]))




    def pack_poi_grid_data(self):
        region_shapefile = gpd.read_file('data/raw/Singapore/grid/Singapore_tessellation_2km_square_projected.shp')
        regions = []
        for index, row in tqdm(region_shapefile.iterrows(), total=len(region_shapefile)):
            region = {
                'shape': row['geometry']
            }
            regions.append(region)
        self.pack_poi_data(regions, 'data/processed/Singapore/poi_grid_for_baselines.csv')


    def normalize_edge_weight(self, D,d):
        return np.log((1+np.power(D,1.5))/(np.power(d,1.5)))

    def pack_graph_data(self, regions=None, save_path='data/processed/Singapore/graph_1.pkl'):
        """
            Process graph data for DGI & MVGRL
        """
        if regions is None:
            with open(self.out_path, 'rb') as f:
                regions = pkl.load(f)
        # load buildings & pois
        with open('data/processed/Singapore/building.pkl', 'rb') as f:
            buildings = pkl.load(f)
        with open('data/processed/Singapore/poi.pkl', 'rb') as f:
            pois = pkl.load(f)
        building_features = np.load('data/processed/Singapore/raw_building_features.npz')['arr_0']
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
        with open(save_path, 'wb') as f:
            pkl.dump(region_graphs, f, protocol=4)

    def pack_graph_grid_data(self):
        region_shapefile = gpd.read_file('data/raw/Singapore/grid/Singapore_tessellation_2km_square_projected.shp')
        regions = []
        for index, row in tqdm(region_shapefile.iterrows(), total=len(region_shapefile)):
            region = {
                'shape': row['geometry']
            }
            regions.append(region)
        self.pack_graph_data(regions, 'data/processed/Singapore/graph_grid.pkl')


if __name__ == '__main__':
    os.chdir('../../')
    SingaporeSubzone().get(force=True)
    # SingaporeSubzone().get_grid(region_path='data/raw/Singapore/grid/Singapore_tessellation_2km_square_projected.shp', out_path='data/processed/Singapore/downstream_grid2.pkl')
    # regions = SingaporeSubzone().pack_poi_data()
    # SingaporeSubzone().pack_graph_data()
    # SingaporeSubzone().show_land_use()
