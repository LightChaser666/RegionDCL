"""
@file: NYCCensusTract.py
@time: 2022/09/13
This file generates the downstream land use data for NYC road segmentation patterns.
It simply calculates the land use max label
"""
import math
import os
import pickle as pkl
import numpy as np
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import Point
from tqdm import tqdm


class NYCPattern(object):
    def __init__(self):
        with open('data/processed/NYC/segmentation.pkl', 'rb') as f:
            self.patterns = pkl.load(f)
        self.landuse_in_path = 'data/projected/NYC/landuse/landuse.shp'
        self.out_path = 'data/processed/NYC/downstream_pattern.pkl'
        self.merge_dict = {
            '01': 0,  # One &Two Family Buildings
            '02': 0,  # Multi-Family Walk-Up Buildings
            '03': 0,  # Multi-Family Elevator Buildings
            '04': 0,  # Mixed Residential & Commercial Buildings
            '05': 1,  # Commercial & Office Buildings
            '06': 2,  # Industrial & Manufacturing
            '07': 3,  # Transportation & Utility
            '08': 3,  # Public Facilities & Institutions
            '09': 4,  # Open Space & Outdoor Recreation
            '10': 4,  # Parking Facilities
            '11': 4,  # Vacant Land
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
            region['land_use'] = [0.0] * 11
            # calculate region diameter
            dx = region_shape.bounds[2] - region_shape.bounds[0]
            dy = region_shape.bounds[3] - region_shape.bounds[1]
            diameter = math.sqrt(dx * dx + dy * dy) / 2
            # find land use in the region
            landuse_index = landuse_tree.query_ball_point(region_shape.centroid, diameter)
            for index in landuse_index:
                if region_shape.contains(Point(landuse_loc[index])):
                    current_land_use = landuses[index]
                    region['land_use'][current_land_use['land_use']] += current_land_use['shape'].area
            # argmax
            total_area = sum(region['land_use'])
            if total_area > 0:
                label[idx] = np.argmax(region['land_use'])
        # save
        with open(self.out_path, 'wb') as f:
            pkl.dump(label, f)
        return label

    def init_land_use(self):
        landuse_shapefile = gpd.read_file(self.landuse_in_path)
        landuses = []
        print('Loading land use...')
        for index, row in tqdm(landuse_shapefile.iterrows(), total=len(landuse_shapefile)):
            label = row['LandUse']
            if label is None:
                continue
            landuse = {
                'shape': row['geometry'],
                'land_use': self.merge_dict[label]
            }
            landuses.append(landuse)
        # save the land use shapefile
        return landuses


if __name__ == '__main__':
    os.chdir('../../')
    NYCPattern().get(force=True)
