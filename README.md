# DBSCAN_cluster_analyzer

DBSCAN_cluster_ananlyzer loads Picasso generated linked .hdf5 files and performs a DBSCAN analysis on the dataset. The properties of the clusters are calculated and filtered based on user defined settings. 
The inverse dark time of cluster tracetories are calculated and the cluster properties and dbscan cluster files are saved as new .hdf5 files. 
- loads picasso generated .hdf5 files (localized and linked, columns: "x", "y", "groups")
- optionally, a nerarest neighbor analysis is performed to identify DBSCAN input parameter (config.ini file nearest_neighbor_analysis = True)
- DBSCAN analysis is performed and cluster area and properties (mean + std. values) are calculated
- The inverse dark time of cluster events is calculated
- cluster properties are filtered based on user defined settings (config.ini file)
- cluster properties and dbscan_cluster files are saved with corresponding .yaml file (allows opening and editing in Picasso)
- cluster properties from all files are merged and saved as a new cluster property file

Data preparation: 
Save Picasso linked files in a seperate folder in ascending order together with corresponding .yaml file (e.g. File1.hdf5 File1.yaml, File2.hdf5 File2.yaml). The script will load all .hdf5 & yaml files in file_path and analyse the datasets. New .hdf5 files that contain analysis results (cluster properties and dbscan_cluster) will be saved individually and as a concatenated version (all_cluster_properties) for further external analysis. 

Requirements: python 3.7, os, configparser, h5py, numpy, matplotlib, yaml, numba, , pandas, sklearn, scipy

Input files: Picasso[1] .hdf5 (picasso linked file)
Ouput files: new .hdf5 files (cluster properties and dbscan_cluster)

Execution: DBSCAN_cluster_analyzer.py

Config file:

[INPUT_FILES] path: path to picasso linked .hdf5 files (name.hdf5)

[PARAMETERS] 
epsilon: DBSCAN distance parameter (in pixel)
min_sample: DBSCAN density parameter (int)
frame_time: camera integration time (s)
imager_concentration: imager strand concentration (nM)
n_frames: # frames acquired
create_cluster: Boolean parameter (set True to artificially connect 2 dbscan_cluster)
filter_traces: Boolean parameter (set True to apply rolling window based filtering process of traces in cluster (removes cluster with unspecific DNA-PAINT signal)
show_unspecific_traces: Boolean parameter (set True to display every trace that does not pass filtering process)
nearest_neighbor_analysis: Nearest neighbor analysis of dataset, used to identify DBSCAN input parameter
NN_pts: list of number of points for nearest neighbor analysis
k_on: association rate of DNA-PAINT imager strand
min_cluster_events: minimum # of events per cluster required in filtering process (cluster with events < min_cluster events will be removed)
locs: list containing min and max number of events for filtering cluster
meanframe: list containing min and max number of mean_frame values allowed for filtering cluster
stdframe: list containing min and max number of std_frame values allowed for filtering cluster
meandark: list containing min and max number of mean_dark values allowed for filtering cluster
stdark  list containing min and max number of std_dark values allowed for filtering cluster
path_results: path to folder for saving analysis results
filename: name of new generated .hdf5 files from analysis


links: [1] https://github.com/jungmannlab/picasso
