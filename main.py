import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import lognorm
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class ClusterAnalyzer(object):
    """ loads .hdf5 files from path.
        :return: lists containing individual dbscan_cluster information."""

    def __init__(self, file_path, locs, meanframe, stdframe, meandark, stddark, epsilon, min_sample):  # path to data
        self.file_path = file_path
        self.locs = locs
        self.meanframe = meanframe
        self.stdframe = stdframe
        self.meandark = meandark
        self.stddark = stddark
        self.epsilon = epsilon
        self.min_sample = min_sample

    def load_file(self):
        """load .hdf5_file"""
        with h5py.File(self.file_path, "r") as locs_file:
            key = list(locs_file.keys())[0]  # get key name
            locs = locs_file[str(key)][...]
        self.data_pd = pd.DataFrame(locs)

    def nearest_neighbors(self):
        n = [4, 8, 12, 16, 24, 28, 36, 48]
        distance_list = []
        for i in (n):
            print(i)
            neighbors = NearestNeighbors(n_neighbors=i)
            neighbors_fit = neighbors.fit(self.data_pd[["x", "y"]])
            distances, indices = neighbors_fit.kneighbors(self.data_pd[["x", "y"]])

            distances = np.mean(distances[:, 1:], axis=1)
            distances = np.sort(distances)
            distance_list.append(distances)

        for i in range(len(distance_list)):
            plt.plot(distance_list[i], label=n[i])
            plt.xlabel("n")
            plt.ylabel("e")
            plt.legend()
        plt.show()
        sdf=1

    def dbscan(self):
        db = DBSCAN(eps=self.epsilon, min_samples=self.min_sample).fit(self.data_pd[["x", "y"]].to_numpy())  # dbscan
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # number of clusters
        print(n_clusters_)
        n_noise_ = list(labels).count(-1)  # number of noise locs
        self.data_pd["group"] = labels  # add group column
        unique_labels = set(labels)  # get unique cluster label
        core_samples_mask = np.zeros_like(labels, dtype=bool)  # preallocate core_sample_mask
        core_samples_mask[db.core_sample_indices_] = True  # core_sample mask

        s = pd.Series(core_samples_mask, name='bools')  # generate series for masking
        self.dbscan_cluster = self.data_pd[s.values]  # mask localization file to extract dbscan cluster
        return self.dbscan_cluster

    def calc_clusterprops(self):
        # create emtpy dataframe
        dark_mean, dark_std, frame_mean, frame_std, n_events = [], [], [], [], []
        self.cluster_props = pd.DataFrame(columns=['groups', 'n_events', 'frame_mean', 'frame_std', 'dark_mean', 'dark_std'])
        # get group values from cluster
        groups = np.sort(self.dbscan_cluster["group"].unique())
        # add groups to dataframe
        self.cluster_props["groups"] = groups
        for i in groups:  # append mean/std data to dataframe
            temp = self.dbscan_cluster.loc[self.dbscan_cluster['group'] == i]
            dark = temp["frame"].diff() - temp[
                "len"]  # calculate darktimes per group and correct for length of binding events
            dark_mean.append(dark.mean())
            dark_std.append(dark.std())
            frame_mean.append(temp["frame"].mean())
            frame_std.append(temp["frame"].std())
            n_events.append(len(temp))
        (self.cluster_props["dark_mean"], self.cluster_props["dark_std"], self.cluster_props["frame_mean"],
         self.cluster_props["frame_std"], self.cluster_props["n_events"]) = dark_mean, dark_std, frame_mean, frame_std, n_events

    def calc_inverse_darktime(self):
        self.cluster_props["inverse_dark"] = (1 / (self.cluster_props["dark_mean"] * 0.05))
        return self.cluster_props

    def filter(self):
        self.cluster_props = self.cluster_props.loc[(self.cluster_props['inverse_dark'] < 0.1)]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['n_events'] > self.locs[0]) & (self.cluster_props['n_events'] < self.locs[1])]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['frame_mean'] > self.meanframe[0]) & (self.cluster_props['frame_mean'] < self.meanframe[1])]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['frame_std'] > self.stdframe[0]) & (self.cluster_props['frame_std'] < self.stdframe[1])]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['dark_mean'] > self.meandark[0]) & (self.cluster_props['dark_mean'] < self.meandark[1])]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['dark_std'] > self.stddark[0]) & (self.cluster_props['dark_std'] < self.stddark[1])]
        return self.cluster_props

    def testplot(self, concat_data, i):
        concat_data.hist(column="frame_mean", bins=60)
        concat_data.hist(column="frame_mean", bins=60)
        concat_data.hist(column="frame_std", bins=20)
        concat_data.hist(column="dark_mean", bins=20)
        concat_data.hist(column="dark_std", bins=20)
        concat_data.hist(column="n_events", bins=120)
        plt.title(self.file_path + i)
        plt.show()

    def getscatter_data(self):
        pass

    def combine_data(self):
        pass



def main(path, locs, meanframe, stdframe, meandark, stddark, epsilon, min_sample):
    all_cluster_props = pd.DataFrame(columns=['groups', 'n_events', 'frame_mean', 'frame_std', 'dark_mean', 'dark_std'])
    all_cluster_props['groups'] = all_cluster_props['groups'].astype(int)
    all_cluster_props['n_events'] = all_cluster_props['n_events'].astype(int)   # improve

    for i in os.listdir(path):
        # loop through dir_list and open files
        print(i)
        if i.endswith(".hdf5"):  #== "ROI1.hdf5": #
            obj = ClusterAnalyzer(path + "\\" + i, locs, meanframe, stdframe, meandark, stddark, epsilon, min_sample)
            obj.load_file()
            #obj.nearest_neighbors()
            dbscan_cluster = obj.dbscan()
            obj.calc_clusterprops()
            obj.calc_inverse_darktime()
            cluster_props = obj.filter()
            # concat all files
            all_cluster_props = pd.concat([all_cluster_props, cluster_props], axis=0)
    obj.testplot(all_cluster_props, i)
    return all_cluster_props, dbscan_cluster


# load file
path2 = 'C:\\Users\\mglogger\\Desktop\\test\\FcyRIIb\\Imager1_4nM'  # path
path = 'C:\\Users\\mglogger\\Desktop\\test\\FcyRIIb\\Imager1_4nM'  # path

all_cluster_props_BG, dbscan_cluster_BG = main(path2, [6, 200],[5000, 15000], [3000, 20000], [200, 1200], [250, 1500], 0.5, 20)
all_cluster_props_ROI, dbscan_cluster_ROI = main(path, [4, 200],[5000, 15000], [3000, 20000], [200, 2400], [250, 1500], 0.2, 4)

fig, axes = plt.subplots(figsize=(15, 6), nrows=1, ncols=2)

all_cluster_props_BG.hist(column="inverse_dark", bins=50, ax=axes[0], alpha=0.6, color="red", label="05_20")
all_cluster_props_ROI.hist(column="inverse_dark", bins=150, ax=axes[0], alpha=0.6, color="darkblue", label="02_6")
#scatter = dbscan_cluster.plot.scatter(x=["x"], y=["y"], s=5, c=dbscan_cluster["group"], cmap='tab20b', ax=axes[1])
#plt.legend()
plt.show()

# # plot
# fig, axes = plt.subplots(figsize=(15,6), nrows=1, ncols=2)
# all_cluster_props.hist(column="inverse_dark", bins=40, ax=axes[0], alpha=0.6, color="darkblue", label="cluster")
# scatter = dbscan_cluster.plot.scatter(x=["x"], y=["y"], s=5, c=dbscan_cluster["group"], cmap='tab20b', ax=axes[1])
# plt.show()
# sf=1