"""
@file: SingaporePattern.py
@time: 2022/09/14
This file generates the downstream land use data for Singapore road segmentation patterns.
It simply calculates the land use max label
"""

import math
import os
import re
import pickle as pkl
import numpy as np
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import Point
from tqdm import tqdm


class SingaporePattern:
    def __init__(self):
        with open('data/processed/Singapore/segmentation_100.pkl', 'rb') as f:
            self.patterns = pkl.load(f)
        self.landuse_in_path = 'data/projected/Singapore/landuse/landuse.shp'
        self.out_path = 'data/processed/Singapore/downstream_pattern2.pkl'
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
            'ROAD': 3,
            'TRANSPORT FACILITIES': 3,
            'MASS RAPID TRANSIT': 3,
            'LIGHT RAPID TRANSIT': 3,
            # Institution
            'EDUCATIONAL INSTITUTION': 3,
            'CIVIC & COMMUNITY INSTITUTION': 3,
            'COMMERCIAL / INSTITUTION': 3,
            'RESIDENTIAL / INSTITUTION': 3,
            'HEALTH & MEDICAL CARE': 3,
            'CEMETERY': 3,
            'PLACE OF WORSHIP': 3,
            # Other
            'PORT / AIRPORT': 4,
            'PARK': 4,
            'UTILITY': 4,
            'BEACH AREA': 4,
            'RESERVE SITE': 4,
            'WATERBODY': 4,
            'SPECIAL USE': 4,
            'WHITE': 4,
            # Open Space including open space, agriculture, and recreation
            'OPEN SPACE': 4,
            'AGRICULTURE': 4,
            'RECREATION': 4,
            'SPORTS & RECREATION': 4,
            # TODO: Support and verify the merging strategy
        }

    def get(self, force=False):
        if not force and os.path.exists(self.out_path):
            with open(self.out_path, 'rb') as f:
                return pkl.load(f)
        label = {}
        # load land use
        landuses = self.init_land_use()
        # aggregate land use data
        landuse_loc = [[landuse['shape'].centroid.x, landuse['shape'].centroid.y] for landuse in landuses]
        landuse_tree = KDTree(landuse_loc)
        print('Aggregating land use...')
        for idx, region in enumerate(tqdm(self.patterns)):
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
                    region['land_use'][current_land_use['land_use']] += region_shape.intersection(landuses[index]['shape'].buffer(0)).area
            # argmax
            total_area = sum(region['land_use'])
            if total_area > 0:
                label[idx] = np.argmax(region['land_use'])
        # save
        with open(self.out_path, 'wb') as f:
            pkl.dump(label, f)
        return label
    
    def get_ratio(self, force=False):
        if not force and os.path.exists(self.out_path):
            with open(self.out_path, 'rb') as f:
                return pkl.load(f)
        # load land use
        landuses = self.init_land_use()
        # save it to a separate file
        with open('data/processed/Singapore/landuse.pkl', 'wb') as f:
            pkl.dump(landuses, f)
        # aggregate land use data
        # landuse_loc = [[landuse['shape'].centroid.x, landuse['shape'].centroid.y] for landuse in landuses]
        # landuse_tree = KDTree(landuse_loc)
        # print('Aggregating land use...')
        # for idx, region in enumerate(tqdm(self.patterns)):
        #     region_shape = region['shape']
        #     region['land_use'] = [0.0] * 5
        #     # calculate region diameter
        #     dx = region_shape.bounds[2] - region_shape.bounds[0]
        #     dy = region_shape.bounds[3] - region_shape.bounds[1]
        #     diameter = math.sqrt(dx * dx + dy * dy)
        #     # find land use in the region
        #     landuse_index = landuse_tree.query_ball_point([region_shape.centroid.x, region_shape.centroid.y], diameter)
        #     for index in landuse_index:
        #         if region_shape.intersects(landuses[index]['shape']):
        #             current_land_use = landuses[index]
        #             region['land_use'][current_land_use['land_use']] += region_shape.intersection(landuses[index]['shape'].buffer(0)).area
        #     # argmax
        #     total_area = sum(region['land_use'])
        #     if total_area > 0:
        #         region['land_use'] = [x / total_area for x in region['land_use']]
        # # save
        # with open(self.out_path, 'wb') as f:
        #     pkl.dump(self.patterns, f)

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


if __name__ == '__main__':
    #os.chdir('../../')
    labels = SingaporePattern().get_ratio(force=True)