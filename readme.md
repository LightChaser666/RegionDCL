# RegionDCL - Code Instructions

- This repository includes essential scripts of the proposed framework RegionDCL in our paper:  **Urban Region Representation Learning ｗith OpenStreetMap Building Footprints**

## Whats' new

- We released all raw & projected dataset and the processing script that generates the ground truth data.
- We fixed the population task reproduction problem. 
  - The data/processed/Singapore/downstream_region.pkl wrongly uses the Singapore citizens data, which ignores permanent residents and foreigners that are common in Singapore.
  - In our paper, we use more widely accepted WorldPop population data as ground truth. Now we correct the file all our results on Singapore can be exactly reproduced.
  - You can see from data_util/downstream/SingaporeSubzone to see how the ground truth is generated.


## Quick Start

- With this repository, you can
  - Quickly reproduce the results in the paper
  - Quickly train the model and get the region embeddings of Singapore Subzones, and evaluate its performance.

### Environment

Required for evaluation:

- Please use **Miniconda or Anaconda**
- python 3.7 or 3.8 are acceptable. Higher version is not tested.
- pytorch >= 1.5.1 with GPU (for train only). We use pytorch == 1.8.2
- geopandas == 0.9.0 
  - Can be easily installed with **conda install geopandas**
- sklearn == 1.1.2 (For evaluation purpose only. Other versions may be also acceptable)
- tqdm (For progress bar only. You can use any version)
- seaborn (For visualizing the bar in quality_analysis.py only)

For training, you need to add the following package:

- pot == 0.8.2 (Python Optimal Transport Library for adaptive margin calculation)

- And ensure you have roughly 16 GB GPU available memory (If not, you need to low down the batch size in the trainer.py)

### Experiment

- For your convenience, we randomly picked one embedding for each baseline into the /baselines/ folder.
- You can quickly evaluate all baselines versus our method. 
- We strictly conduct experiments without any cherry-picking, so the result should be very close to that reported in our paper.

#### Main Experiment (Land Use / Population Inference in Singapore and NYC)

- Simply execute /evaluator/main_experiment.py. You should be able to see something like this:

```
=========================== Land Use Inference in Singapore ===========================
Baseline                      L1        std       KL-Div    std       Cosine    std       
Urban2Vec                     0.676     0.050     0.547     0.132     0.785     0.033     
Place2Vec                     0.639     0.045     0.480     0.074     0.808     0.025     
Doc2Vec                       0.692     0.043     0.513     0.073     0.773     0.027     
GAE                           0.748     0.034     0.543     0.048     0.767     0.022     
DGI                           0.597     0.039     0.368     0.048     0.846     0.020     
Transformer                   0.542     0.040     0.331     0.052     0.859     0.021     
RegionDCL-no random           0.522     0.045     0.306     0.046     0.869     0.021     
RegionDCL-fixed margin        0.509     0.031     0.298     0.039     0.874     0.014     
RegionDCL                     0.495     0.036     0.281     0.039     0.882     0.017     
```

- Try --task pop, you can get:

=========================== Population Density Inference in Singapore ===========================
Baseline                      MAE       std       RMSE      std       R-Square  std       
Urban2Vec                     6487.310  562.799   8674.112  989.720   0.331     0.104     
Place2Vec                     6843.648  600.236   9501.109  853.687   0.141     0.128     
Doc2Vec                       6818.389  416.066   9135.961  562.911   0.209     0.059     
GAE                           7006.395  818.177   9467.563  1141.608  0.144     0.133     
DGI                           6282.033  567.734   8233.419  892.947   0.346     0.127     
Transformer                   6556.571  915.317   8667.137  1190.557  0.253     0.092     
RegionDCL-no random           5896.042  733.596   8165.986  1240.533  0.336     0.103     
RegionDCL-fixed margin        5863.208  612.696   7982.024  817.545   0.357     0.114     
RegionDCL                     5591.122  570.922   7502.129  551.069   0.427     0.109     

- The main_experiment.py has the following options:
  - --city: can be **Singapore** or **NYC**
  - --task: can be **land** or **pop**
  - --partition: can be **default** (in Singapore this will be subzones, in NYC will be census tract) or **grid**(2km * 2km)
    - *Note that grid partition only works in Singapore*

#### Embedding Quality Analysis

- Simply execute /evaluator/quality_analysis.py. You should be able to see the picture in the paper like this:![singapore_data_sparsity_land_use](../master/visualization/singapore_data_sparsity_land_use.png?raw=true)

- The following options are available:
  - --city: can be **Singapore** or **NYC**
  - --task: can be **land** or **pop**

## Training on Singapore dataset

- Unzip the building_feature.7z in the data/processed/Singapore.
  - This is the feature extracted by **model/resnet.py**
  - So alternatively, you can use the script to train the resnet and generate new features. **Timm** library is needed.
- Run main.py. The following options are available:
  - --city: can be **Singapore** or **NYC** (The preprocessed NYC data will be available very soon)
  - --no_random: **use this to disable random point sampling.**
  - --fixed: **use this to disable the proposed adaptive margin.**
  - --save_name: the name of the building pattern embedding file. By default it is pattern_embedding
  - *For more options, please refer to the main.py.*
- After trained, you will get a new embedding file in the /embeddings/Singapore/
- The newly generated embeddings will be named RegionDCL_20.pkl, You can run experiment/evaluator.py with proper test_path to evaluate it. please change the filename in the script accordingly.

## Pre-processing (required on NYC dataset)

- We offered the preprocessing code for input shapefile. In principle it can be used for any OSM data.
  - You need **rasterio** to rasterize the building polygons. 
  - Use this if something wrong: **conda install -c https://conda.anaconda.org/ioos rasterio**
- The raw and projected dataset (both NYC and Singapore, projected via ArcGIS): [Google Drive](https://drive.google.com/file/d/1WcIBcGDude5Q3_sZk_wXYpTQIPPAd6Ho/view?usp=sharing)
- The preprocess code will automatically convert buildings to images (in forms of numpy 0-1) for you, but please extract the NYC building polygons features to data/processed/NYC/building_features.npy with resnet.py on your own, as NYC has too many buildings inconvenient for us to upload.

## Content

```
RegionDCL
├── baselines # Randomly picked embeddings from other baselines for comparison
├── data
| ├── raw  === Raw data of Sinagpore and NYC. Please download and decomporess the file from the Google Drive above.
| ├── projected  ===  Projected data of Sinagpore and NYC. 
| └── processed  === Store the processed data for model training.
├── data_util
|	├── dataset.py  ===  Pytorch dataset for the model
|	├── grid.py  === Poisson Disk Sampling algorithm
|	└── preprocess.py === Turn the input data into model accepted data.
├── embeddings  === Save the generated embeddings
├── experiment
|	├── evaluator.py  === MLP and RandomForest for land use population density inference
|	└── main_experiment.py  === Generate the main experiment result in default partition (Singapore Subzone, NYC Census Tracts) and grids.
|	└── quality_analysis.py  === Analyze the embedding quality with respect to the building and POI amount.
├── model
|	├── adaptive_triplet.py  === The proposed adaptive margin in triplet loss
|	├── biased_attention.py  === The proposed distance-biased self-attention
|	├── regiondcl.py  === Model, regular transformers.
|	├── resnet.py  === Use rasterized buildings to get building features.
|	└── trainer.py === Train the proposed model.
└── main.py === run and get region embeddings!

```

## Thanks for Reading!
