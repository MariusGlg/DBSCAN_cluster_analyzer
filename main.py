import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import yaml as _yaml
from scipy.spatial import ConvexHull
from numba import njit
from configparser import ConfigParser

#  config.ini file
config = ConfigParser()
file = "config.ini"
config.read(file)
#  config parameter
epsilon = float(config["PARAMETERS"]["epsilon"])
min_sample = int(config["PARAMETERS"]["min_sample"])
frame_time = float(config["PARAMETERS"]["frame_time"])
min_cluster_events = int(config["PARAMETERS"]["min_cluster_events"])
create_cluster = config["PARAMETERS"]["create_cluster"]
# filter parameter
locs = config["PARAMETERS"]["locs"]
locs = locs.split(",")
meanframe = config["PARAMETERS"]["meanframe"]
meanframe = meanframe.split(",")
stdframe = config["PARAMETERS"]["stdframe"]
stdframe = stdframe.split(",")
meandark = config["PARAMETERS"]["meandark"]
meandark = meandark.split(",")
stddark = config["PARAMETERS"]["stddark"]
stddark = stddark.split(",")

# result path parameter
path_results = config["PARAMETERS"]["path_results"]
filename = config["PARAMETERS"]["filename"]
path_yaml = config["PARAMETERS"]["path_yaml"]
# load _dblucster.hdf5 files
path = config["INPUT_FILES"]["path"]


sdf=1

class ClusterAnalyzer(object):
    """ loads .hdf5 files from path.
        :return: lists containing individual dbscan_cluster information.
        param:
        min_cluster_events: minimum events per cluster
        epsilon:
        min_sample:
        frame_time:
        ..."""

    def __init__(self, file_path, locs, meanframe, stdframe, meandark, stddark, epsilon, min_sample, filename, frame_time, min_cluster_events, create_cluster):  # path to data
        self.file_path = file_path
        self.locs = locs
        self.meanframe = meanframe
        self.stdframe = stdframe
        self.meandark = meandark
        self.stddark = stddark
        self.epsilon = epsilon
        self.min_sample = min_sample
        self.filename = filename
        self.frame_time = frame_time
        self.min_cluster_events = min_cluster_events
        self.create_cluster = create_cluster

    def load_file(self):
        """load .hdf5_file"""
        with h5py.File(self.file_path, "r") as locs_file:
            key = list(locs_file.keys())[0]  # get key name
            locs = locs_file[str(key)][...]
        self.data_pd = pd.DataFrame(locs)

    def nearest_neighbors(self):
        """calc nearest neighbor to estimate DBSCAN input parameter. Plot average k-distance on k-distance graph.
        n = MinPts for dataset.
        Optimal epsilon = point of maximum curvature (greatest slope)
        """
        n = [4, 8, 12, 20, 80]  # n_neighbors
        distance_list = []
        for i in n:  # loop through list
            neighbors = NearestNeighbors(n_neighbors=i)
            neighbors_fit = neighbors.fit(self.data_pd[["x", "y"]])
            distances, indices = neighbors_fit.kneighbors(self.data_pd[["x", "y"]])

            distances = np.mean(distances[:, 1:], axis=1)
            distances = np.sort(distances)
            distance_list.append(distances)

        for i in range(len(distance_list)):
            plt.plot(distance_list[i], label=n[i])
            plt.xlabel("Points n (sorted by distance)")
            plt.ylabel("e (NN-distance)")
            plt.title("k-distance graph")
            plt.legend()
        plt.show()

    def dbscan(self):
        """
        DBSCAN analysis of dataset
        epsilon: distance parameter
        MinPts: number of points/cluster (usually > 2*Dimension of dataset)
        larger datasets: use larger MinPts value
        noisier datasets: choose larger MinPts value
        epsilon: use nearest neighbor analysis where k_NN = MinPts
        """
        db = DBSCAN(eps=self.epsilon, min_samples=self.min_sample).fit(self.data_pd[["x", "y"]].to_numpy())  # dbscan
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # number of clusters
        # print(n_clusters_)
        n_noise_ = list(labels).count(-1)  # number of noise locs
        self.data_pd["group"] = labels  # add group column
        unique_labels = set(labels)  # get unique cluster label
        core_samples_mask = np.zeros_like(labels, dtype=bool)  # preallocate core_sample_mask
        core_samples_mask[db.core_sample_indices_] = True  # core_sample mask
        s = pd.Series(core_samples_mask, name='bools')  # generate series for masking
        self.dbscan_cluster = self.data_pd[s.values]  # mask localization file to extract dbscan cluster
        self.groups = np.sort(self.dbscan_cluster["group"].unique())  # cluster goups indices
        # filter dbscan cluster based on min_cluster events
        counts = self.dbscan_cluster['group'].value_counts() > self.min_cluster_events  # count events/cluster
        rev_counts = counts[counts == False]  # reverse series
        rev_counts_index = rev_counts.index  # get index
        self.dbscan_cluster = self.dbscan_cluster[~self.dbscan_cluster['group'].isin(rev_counts_index)]  # bool filter
        self.groups = np.sort(self.dbscan_cluster["group"].unique())  # update groups list after filtering


    def artificial_cluster(self):
        """generate larger cluster by assinging odd groups to even groups"""
        if self.create_cluster == True:
            even_num = self.dbscan_cluster.loc[self.dbscan_cluster["group"] % 2 == 0]
            odd_num = self.dbscan_cluster.loc[self.dbscan_cluster["group"] % 2 == 1]
            odd_num.eval("group = group - 1", inplace=True)
            self.dbscan_cluster = pd.concat([even_num, odd_num], axis=0)
            self.groups = np.sort(self.dbscan_cluster["group"].unique())  # update groups list after filtering

    def cluster_area(self):
        self.conv_hull_area = []
        self.centroids = []
        """ calculate area (convex hull) of dbscan cluster"""
        for i in self.groups:  # append mean/std data to dataframe
            temp = self.dbscan_cluster.loc[self.dbscan_cluster['group'] == i]
            hull = ConvexHull(np.stack((temp["x"].to_numpy(), temp["y"].to_numpy()), axis=1))
            self.conv_hull_area.append(hull.area)  # append area
            cx, cy = self.cluster_center(hull)  # calculate centroid of cluster
            self.centroids.append([cx, cy])  # append centroids
        return self.centroids
    def cluster_center(self, hull):
        # Get centroid of cluster
        return np.mean(hull.points[hull.vertices, 0]), np.mean(hull.points[hull.vertices, 1])

    def get_formats(self):
        self.cluster_props_header = self.dbscan_cluster.columns.values.tolist()
        mean_lst = ['{0}_mean'.format(words) for words in self.cluster_props_header]
        std_lst = ['{0}_std'.format(words) for words in self.cluster_props_header]
        self.column_lst = [sub[item] for item in range(len(std_lst))
                           for sub in [mean_lst, std_lst]]  # merge lists alternatively
        return self.column_lst

    def rolling_win_filter(self, x, k):
        E_dark = np.max(self.dbscan_cluster["frame"]) / x.size  # expected average dark time for every trace
        x = x.dropna()  # drop NaN
        rolling_signal = x.rolling(window=5).mean()  # rolling window analysis (Window size?)
        threshold = rolling_signal.loc[lambda x: x < E_dark/4]  # threshold 1/4 expected average dark time
        if not threshold.empty:
            if len(threshold) > len(rolling_signal)/5:  # clear if >10% of signal < 1/4 Expected dark time of dataset
                # plt.plot(rolling_signal, "o")
                # plt.hlines(y=E_dark, xmin=0, xmax=600000, colors="green")
                # plt.hlines(y=E_dark/4, xmin=0, xmax=600000, colors="red")
                # plt.show()
                self.dbscan_cluster = self.dbscan_cluster.drop(self.dbscan_cluster[self.dbscan_cluster['group']
                                                                                   == k].index)  # drop rows
    def calc_clusterprops(self):
        """calculate properties (mean/std, dark_times) for every cluster in dataset"""
        dark_mean, dark_std, row, events = [], [], [], []  # preallocate
        arr = []
        for i, k in enumerate(self.groups):  # append mean/std data to dataframe
            temp = self.dbscan_cluster.loc[self.dbscan_cluster['group'] == k]
            temp = temp.sort_values(by=['frame'])  # only for oligomers important
            dark = temp["frame"].diff() - temp["len"]
            self.rolling_win_filter(dark, k)
            merged = np.array([[i, j] for i, j in zip(temp.mean(), temp.std())]).ravel()
            arr.append(merged)
            x = dark[~np.isnan(dark)]
            dark_mean.append(self.calc_mean(x.to_numpy()))  # calc mean and std using numba njit
            dark_std.append(self.calc_std(x.to_numpy()))
            events.append(len(temp))
        self.cluster_props = pd.DataFrame(columns=self.column_lst, data=arr)
        self.cluster_props["dark_mean"] = dark_mean
        self.cluster_props["dark_mean"] = self.cluster_props["dark_mean"].astype(float)
        self.cluster_props["dark_std"] = dark_std
        self.cluster_props["dark_std"] = self.cluster_props["dark_std"].astype(float)
        self.cluster_props["n_events"] = events
        self.cluster_props["n_events"] = self.cluster_props["n_events"].astype(int)
        self.cluster_props["area"] = self.conv_hull_area
        self.cluster_props["area"] = self.cluster_props["area"].astype(float)


    def calc_mean(self, x):
        return _calc_mean(x)

    def calc_std(self, x):
        return _calc_std(x)

    def calc_inverse_darktime(self):
        self.cluster_props["inverse_dark [s]"] = (1 / self.cluster_props["dark_mean"]) * self.frame_time
        return self.cluster_props

    def filter(self):
        """filter dataset"""
        # self.cluster_props = self.cluster_props.loc[(self.cluster_props['inverse_dark [s]'] < 0.5)]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['n_events'] > int(self.locs[0])) & (self.cluster_props['n_events'] < int(self.locs[1]))]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['frame_mean'] > int(self.meanframe[0])) & (self.cluster_props['frame_mean'] < int(self.meanframe[1]))]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['frame_std'] > int(self.stdframe[0])) & (self.cluster_props['frame_std'] < int(self.stdframe[1]))]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['dark_mean'] > int(self.meandark[0])) & (self.cluster_props['dark_mean'] < int(self.meandark[1]))]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['dark_std'] > int(self.stddark[0])) & (self.cluster_props['dark_std'] < int(self.stddark[1]))]
        return self.cluster_props

    def testplot(self, concat_data, i):
        concat_data.hist(column="frame_mean", bins=60)
        concat_data.hist(column="frame_std", bins=20)
        concat_data.hist(column="dark_mean", bins=20)
        concat_data.hist(column="dark_std", bins=20)
        concat_data.hist(column="n_mean", bins=120)
        plt.title(self.file_path + i)
        plt.show()

    def combine_data(self, all_cluster_props, cluster_props):
        """concatenate data"""
        all_cluster_props = pd.concat([all_cluster_props, cluster_props], axis=0)
        return all_cluster_props


class Save(object):
    """ Saves new .hdf5 files, corresponding .yaml file and .xlsm containing ROI coordinates"""
    def __init__(self, path, filename, data, epsilon, min_sample, column_lst, i):
        self.path = path
        self.filename = filename
        self.data = data
        self.epsilon = epsilon
        self.min_sample = min_sample
        self.column_lst = column_lst
        self.i = i

    def getformats(self):
        # get formats of hdf5 files
        formats = ([np.float32] * (len(self.data.columns)))
        return formats

    def save_pd_to_hdf5(self, formats):
        data_lst = self.data.values.tolist()  # convert to list of lists
        data_lst = [tuple(x) for x in data_lst]  # convert to list of tuples

        name = self.filename + "_DBSCAN_" + str(self.epsilon) + "_" + str(self.min_sample) + "_properties" +".hdf5"
        with h5py.File(os.path.join(self.path, str(name)), "w") as locs_file:
            ds_dt = np.dtype({'names': self.data.columns.values, 'formats': formats})
            locs_file.create_dataset("locs", data=data_lst, dtype=ds_dt)

    def save_yaml(self):
        name = self.filename + "_DBSCAN_" + str(self.epsilon) + "_" + str(self.min_sample) + "_properties" +".yaml"
        content = []
        self.yaml_param()
        # if self.i == 1:
            # save yaml file to reopen modified dbscan file
        with open(os.path.join(path_yaml), 'r') as yaml_file:
            text = _yaml.load_all(yaml_file, _yaml.FullLoader)
            with open(os.path.join(self.path, name), 'w') as outfile:
                for doc in text:
                    content.append(doc)
                content.append(self.yaml_content)
                _yaml.dump_all(content, outfile)

    def yaml_param(self):
        self.yaml_content = dict(
            frame_time=frame_time,
            min_cluster_events=min_cluster_events,
            create_cluster=create_cluster,
            DBSCAN=dict(
                epsilon=epsilon,
                min_sample=min_sample
            ),
            filter_param=dict(
                locs=locs,
                meanframe=meanframe,
                stdframe=stdframe,
                meandark=meandark,
                stddark=stddark,
            )
        )

    def main(self):
        formats = self.getformats()
        self.save_pd_to_hdf5(formats)
        self.save_yaml()

@njit
def _calc_mean(dark):
    return np.array(dark.mean())
@njit
def _calc_std(dark):
    return np.array(dark.std())

def main(path, locs, meanframe, stdframe, meandark, stddark, epsilon, min_sample, path_results, filename, frame_time, min_cluster_events, create_cluster):

    all_cluster_props = pd.DataFrame() # init emtpy dataframe
    for i in os.listdir(path):
        # loop through dir_list and open files
        # print(i)
        if i.endswith(".hdf5"):  #== "ROI1.hdf5": #
            obj = ClusterAnalyzer(path + "\\" + i, locs, meanframe, stdframe, meandark, stddark,
                                  epsilon, min_sample, filename, frame_time, min_cluster_events, create_cluster)
            obj.load_file()
            # obj.nearest_neighbors()
            obj.dbscan()
            obj.artificial_cluster()
            centroids = obj.cluster_area()
            column_lst = obj.get_formats()
            obj.calc_clusterprops()
            #  dbscan_cluster = []
            cluster_props = obj.calc_inverse_darktime()
            cluster_props = obj.filter()
            print("n cluster: ", len(cluster_props))
            print("density: ", cluster_props["n_events"].mean())
            all_cluster_props = obj.combine_data(all_cluster_props, cluster_props)  # concat all files

    # obj.testplot(all_cluster_props, i)
        save_obs = Save(path_results, filename, all_cluster_props, epsilon, min_sample, column_lst, i)
        save_obs.main()
    return all_cluster_props, centroids


Cluster_props, centroids = main(path, locs, meanframe, stdframe,
                      meandark, stddark, epsilon, min_sample, path_results,
                      filename, frame_time, min_cluster_events, create_cluster)




# fig, axes = plt.subplots(figsize=(15, 6), nrows=2, ncols=2)
#
# FcyR2b_ROI.hist(column="inverse_dark", bins=50, ax=axes[0][0], alpha=0.6, color="red", label="02_5")
# FcyR2b_BG.hist(column="inverse_dark", bins=50, ax=axes[0][0], alpha=0.6, color="darkblue", label="02_5")
# FcyR2b_ROI.hist(column="n_events", bins=35, ax=axes[0][1], alpha=0.6, color="red", label="02_5")
# FcyR2b_BG.hist(column="n_events", bins=25, ax=axes[0][1], alpha=0.6, color="darkblue", label="02_5")
#
# FcyR1a_ROI.hist(column="inverse_dark", bins=50, ax=axes[1][0], alpha=0.6, color="red", label="02_5")
# FcyR1a_BG.hist(column="inverse_dark", bins=50, ax=axes[1][0], alpha=0.6, color="darkblue", label="02_5")
# FcyR1a_ROI.hist(column="n_events", bins=35, ax=axes[1][1], alpha=0.6, color="red", label="02_5")
# FcyR1a_BG.hist(column="n_events", bins=25, ax=axes[1][1], alpha=0.6, color="darkblue", label="02_5")
#
# plt.show()
#
# plt.savefig(path_results + "cluter_analysis.png", dpi=300) #  os.path.dirname(root)

