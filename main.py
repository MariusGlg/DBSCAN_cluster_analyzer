"""
@author. Marius Glogger
Optical Imaging Competence Centre, FAU Erlangen

Perform DBSCAN analysis on linked .hdf5 files generated with Picasso (@Jungmann group).
- load multiple files from folder
- apply DBSCAN analysis
- Filter DNA-PAINT trajectories
- Calculate Properties of cluster
- Save DBSCAN files (open in Picasso render) and assicuated property files
"""


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
imager_concentration = float(config["PARAMETERS"]["imager_concentration"])
n_frames = float(config["PARAMETERS"]["n_frames"])
min_cluster_events = int(config["PARAMETERS"]["min_cluster_events"])
create_cluster = config["PARAMETERS"]["create_cluster"]
filter_traces = config["PARAMETERS"]["filter_traces"]
show_unspecific_traces = config["PARAMETERS"]["show_unspecific_traces"]
nearest_neighbor_ananlysis = config["PARAMETERS"]["nearest_neighbor_analysis"]
NN_pts = config["PARAMETERS"]["NN_pts"].split(",")
k_on = config["PARAMETERS"]["k_on"]

# filter parameter
locs = config["PARAMETERS"]["locs"].split(",")
meanframe = config["PARAMETERS"]["meanframe"].split(",")
stdframe = config["PARAMETERS"]["stdframe"].split(",")
meandark = config["PARAMETERS"]["meandark"].split(",")
stddark = config["PARAMETERS"]["stddark"].split(",")

# result path parameter
path_results = config["PARAMETERS"]["path_results"]
filename = config["PARAMETERS"]["filename"]
# load _dblucster.hdf5 files
path = config["INPUT_FILES"]["path"]

# calc theoretical dark time
k_on = int(k_on)*10**6
c = imager_concentration*10**-9
t = int(n_frames)*frame_time
e = k_on*c*t
tau_d_theo = n_frames/e

class ClusterAnalyzer(object):
    """ loads .hdf5 files from path numbered in ascending order. path must contain .yaml files identically named as
        .hdf5 files. Calculates
        param str file_path: path to folder containing linked .hdf5 files
        param list locs: min/max # localizations per cluster
        param list meanframe/stdframe: mean/std of binding event in cluster
        param list meandark/stddark: mean/std of dark times between binding events in cluster
        param float/int epsilon/min_sample: DBSCAN distance and density parameter (px & points)
        param str filename: name of files to be saved
        param float frame_time: camera integration time in s
        param int min_cluster_events: minimum events per cluster allowed
        param bool create_cluster: bool expression (True = create artifical cluster, False = ignore function)
        return: hdf5-file (all_cluster_props) that contains all cluster properties
        """

    def __init__(self, file_path, locs, meanframe, stdframe, meandark, stddark, epsilon, min_sample, filename,
                 frame_time, min_cluster_events, create_cluster):  # path to data
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
        """load .hdf5_files"""
        with h5py.File(self.file_path, "r") as locs_file:
            key = list(locs_file.keys())[0]  # get key name
            locs = locs_file[str(key)][...]
        self.data_pd = pd.DataFrame(locs)

    def nearest_neighbors(self):
        """Unsupervised learner for implementing neighbor searches to estimate DBSCAN input parameter.
        Plots average k-distance on k-distance graph.
        n = MinPts for dataset.
        Optimal epsilon == point of maximum curvature (greatest slope)
        """
        if nearest_neighbor_ananlysis == "True":
            distance_list = []
            for i in NN_pts:  # loop through list
                neighbors = NearestNeighbors(n_neighbors=int(i))
                neighbors_fit = neighbors.fit(self.data_pd[["x", "y"]])
                distances, indices = neighbors_fit.kneighbors(self.data_pd[["x", "y"]])

                distances = np.mean(distances[:, 1:], axis=1)
                distances = np.sort(distances)
                distance_list.append(distances)

            for i in range(len(distance_list)):
                plt.plot(distance_list[i], label=int(i))
                plt.xlabel("Points (sorted by distance)")
                plt.ylabel("NN-distance [px]")
                plt.title("k-distance graph")
                plt.legend()
            plt.show()

    def dbscan(self):
        """
        DBSCAN analysis of dataset
        epsilon: distance parameter
        MinPts: number of points/cluster (usually > 2*Dimension of dataset)
        larger datasets -> use larger MinPts value
        noisier datasets -> choose larger MinPts value
        epsilon -> use nearest neighbor analysis where k_NN = MinPts
        """
        db = DBSCAN(eps=self.epsilon, min_samples=self.min_sample).fit(self.data_pd[["x", "y"]].to_numpy())  # dbscan
        labels = db.labels_
        self.data_pd["group"] = labels  # add group column
        core_samples_mask = np.zeros_like(labels, dtype=bool)  # preallocate core_sample_mask
        core_samples_mask[db.core_sample_indices_] = True  # core_sample mask
        s = pd.Series(core_samples_mask, name='bools')  # generate series for masking
        self.dbscan_cluster = self.data_pd[s.values]  # mask localization file to extract dbscan cluster
        self.groups = np.sort(self.dbscan_cluster["group"].unique())  # cluster groups indices
        # filter dbscan cluster based on min_cluster events
        counts = self.dbscan_cluster['group'].value_counts() > self.min_cluster_events  # count events/cluster + filter
        rev_counts = counts[counts == False]  # reverse series
        rev_counts_index = rev_counts.index  # get index
        self.dbscan_cluster = self.dbscan_cluster[~self.dbscan_cluster['group'].isin(rev_counts_index)]  # bool filter
        self.groups = np.sort(self.dbscan_cluster["group"].unique())  # update groups list after filtering


    def artificial_cluster(self):
        """generate larger cluster by assigning odd groups to even groups"""
        if self.create_cluster == "True":
            even_num = self.dbscan_cluster.loc[self.dbscan_cluster["group"] % 2 == 0]
            odd_num = self.dbscan_cluster.loc[self.dbscan_cluster["group"] % 2 == 1]
            odd_num.eval("group = group - 1", inplace=True)
            self.dbscan_cluster = pd.concat([even_num, odd_num], axis=0)
            self.groups = np.sort(self.dbscan_cluster["group"].unique())  # update groups list after filtering

    def cluster_area(self):
        """ calc area and centroids of clusters from convex hull"""
        self.conv_hull_area = []
        self.centroids = []
        #  calculate area (convex hull) of dbscan cluster
        for i in self.groups:  # append mean/std data to dataframe
            temp = self.dbscan_cluster.loc[self.dbscan_cluster['group'] == i]
            hull = ConvexHull(np.stack((temp["x"].to_numpy(), temp["y"].to_numpy()), axis=1))
            self.conv_hull_area.append(hull.area)  # append area
            cx, cy = self.calc_cluster_center(hull)  # calculate centroid of cluster
            self.centroids.append([cx, cy])  # append centroids
        return self.centroids

    def calc_cluster_center(self, hull):
        """calculate centroids of clusters"""
        return np.mean(hull.points[hull.vertices, 0]), np.mean(hull.points[hull.vertices, 1])

    def get_header_property_file(self):
        """get list of column header for property file"""
        self.cluster_props_header = self.dbscan_cluster.columns.values.tolist()
        mean_lst = ['{0}_mean'.format(words) for words in self.cluster_props_header]
        std_lst = ['{0}_std'.format(words) for words in self.cluster_props_header]
        self.column_lst = [sub[item] for item in range(len(std_lst))
                           for sub in [mean_lst, std_lst]]  # merge lists alternatively
        return self.column_lst

    def rolling_win_filter(self, dark, k, temp):
        """Filters DNA-PAINT signal in DBSCAN cluster for temporal clustering (sticky, beads etc.) using rolling window
        calculations on dark times.
        E_dark = expected average dark time for every trace in DBSCAN cluster
        window = window size (# of events per trace)
        tau_d_theo = theoretical darktime (E = k_on * c * t)
        threshold = 1/4 expected average dark time (can be set optionally)
        clusters will be removed if >10% of rolling window signal < 1/4 of expected dark times
        """
        E_dark_homo = np.max(self.dbscan_cluster["frame"]) / dark.size  # expected average dark time for every trace
        dark = dark.dropna()  # drop NaN
        rolling_signal = dark.rolling(window=5).mean()  # rolling window analysis
        rolling_signal = rolling_signal.dropna()
        threshold = rolling_signal.loc[lambda dark: dark < E_dark_homo/4]  # threshold 1/4 expected average dark time
        if not threshold.empty:
            if len(threshold) > len(rolling_signal)/5:  # clear if > 20% of signal < 1/4 expected dark time of dataset
                if show_unspecific_traces == "True":  # show sticky traces
                    fig, (ax1, ax2) = plt.subplots(ncols=2)
                    # plot dark times from rolling window analysis
                    ax1.plot(range(len(rolling_signal)), rolling_signal.tolist(), "o")
                    ax1.set_xlabel('rolling window events')
                    ax1.set_ylabel('rolling window dark time')
                    ax1.hlines(y=E_dark_homo, xmin=0, xmax=len(rolling_signal), colors="green")
                    ax1.text(1, E_dark_homo, r"E_$\tau$_d_homogen", va="top")
                    ax1.hlines(y=E_dark_homo/4, xmin=0, xmax=len(rolling_signal), colors="red")
                    ax1.text(1, E_dark_homo/4, r"1/4 $\tau$_d", va="top")
                    ax1.hlines(y=tau_d_theo, xmin=0, xmax=len(rolling_signal), colors="black")
                    ax1.text(1, tau_d_theo, r"$\tau$_d_theo", va="top")
                    ax2.plot(temp["frame"].tolist(), np.ones(len(temp["frame"].tolist())), 'o')  # plot traces
                    ax2.set_xlim([0, n_frames])
                    ax2.vlines(temp["frame"].tolist(), ymin=0.8, ymax=1, colors="black")
                    ax2.set_xlabel('frame')
                    ax2.set_ylabel('on')
                    ax2.set_ylim([0.8, 1.2])
                    ax2.set_yticks([])
                    plt.show()
                self.dbscan_cluster = self.dbscan_cluster.drop(self.dbscan_cluster[self.dbscan_cluster['group']
                                                                                   == k].index)  # drop rows
    def calc_clusterprops(self):
        """calculate properties (mean/std, dark_times) for every cluster in dataset. Filter cluster if filter_traces
        is set to True in config.ini file. Cluster properties are stored in new dataframe (cluster_props)."""
        dark_mean, dark_std, row, events = [], [], [], []  # preallocate
        arr = []
        for i, k in enumerate(self.groups):  # append mean/std data to dataframe
            temp = self.dbscan_cluster.loc[self.dbscan_cluster['group'] == k]
            temp = temp.sort_values(by=['frame'])  # only for oligomers important
            dark = temp["frame"].diff() - temp["len"]  # dark times between binding events in dbscan cluster
            if filter_traces == "True":  # apply rolling window filtering on traces if filter_traces == True
                self.rolling_win_filter(dark, k, temp)
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
        # calls numba njit function to calc mean
        return _calc_mean(x)

    def calc_std(self, x):
        # calls numba njit function to calc std

        return _calc_std(x)

    def calc_inverse_darktime(self):
        # calculates the inverse dark time
        self.cluster_props["inverse_dark [s]"] = 1 / (self.cluster_props["dark_mean"] * self.frame_time)
        return self.cluster_props

    def filter(self):
        """filter dataset based on user defined parameter in config.ini file."""
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['n_events'] > int(self.locs[0])) &
            (self.cluster_props['n_events'] < int(self.locs[1]))]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['frame_mean'] > int(self.meanframe[0])) &
            (self.cluster_props['frame_mean'] < int(self.meanframe[1]))]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['frame_std'] > int(self.stdframe[0])) &
            (self.cluster_props['frame_std'] < int(self.stdframe[1]))]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['dark_mean'] > int(self.meandark[0])) &
            (self.cluster_props['dark_mean'] < int(self.meandark[1]))]
        self.cluster_props = self.cluster_props.loc[
            (self.cluster_props['dark_std'] > int(self.stddark[0])) &
            (self.cluster_props['dark_std'] < int(self.stddark[1]))]
        return self.cluster_props

    def save_dbscan_file(self, i):
        """save filtered dbscan file in results folder. DBSCAN files can be opened in Picasso and further analyzed."""
        self.groups = np.sort(self.cluster_props["group_mean"].unique())
        self.groups = self.groups.astype(np.int32)
        self.dbscan_filtered = self.dbscan_cluster.loc[self.dbscan_cluster['group'].isin(self.groups)]
        header = self.dbscan_filtered.head()  # get column name of dbscan_file
        header = list(header)  # list
        obs_temp = Save(path_results, path, filename, self.dbscan_filtered, epsilon, min_sample, header,
                        "cluster", i)
        obs_temp.main()
        return self.dbscan_filtered

    def testplot(self, concat_data, i):
        # plots results (histograms) - currently not activated
        concat_data.hist(column="frame_mean", bins=60)
        concat_data.hist(column="frame_std", bins=20)
        concat_data.hist(column="dark_mean", bins=20)
        concat_data.hist(column="dark_std", bins=20)
        concat_data.hist(column="n_mean", bins=120)
        plt.title(self.file_path + i)
        plt.show()

    def combine_data(self, all_cluster_props, cluster_props):
        """concatenates data"""
        all_cluster_props = pd.concat([all_cluster_props, cluster_props], axis=0)
        return all_cluster_props


class Save(object):
    """ Saves new .hdf5 files containing cluster properties & corresponding .yaml files.
        param str path_results: path to results folder
        param str path_files: path to folder containing data to be analyzed
        param str filename: name of new files
        param dataframe data: df containing analyzed data
        param float/int epsilon/min_sample: DBSCAN distance and density parameter (px & points)
        param list column_lst: list containing column names
        param str appendix: appendix files, e.g. "properties"
    """
    def __init__(self, path_results, path_files, filename, data, epsilon, min_sample, column_lst, appendix, i):
        self.path_results = path_results
        self.path_files = path_files
        self.filename = filename
        self.data = data
        self.epsilon = epsilon
        self.min_sample = min_sample
        self.column_lst = column_lst
        self.appendix = appendix
        self.i = i

    def getformats(self):
        """get and adapt column formats"""
        formats = ([np.float32] * (len(self.data.columns)))
        formats[0] = np.uint32
        formats[11] = np.uint32
        formats[12] = np.uint32
        formats[14] = np.uint32
        return formats

    def save_pd_to_hdf5(self, formats):
        """transform dataframe to hdf5 and save files in path."""
        data_lst = self.data.values.tolist()  # convert to list of lists
        data_lst = [tuple(x) for x in data_lst]  # convert to list of tuples
        self.fn = self.i.split(".")  # split name
        name = (str(self.fn[0]) + "_DBSCAN_" + str(self.epsilon) + "_" + str(self.min_sample) +
                "_" + str(self.appendix) + ".hdf5")
        if self.appendix == "properties_all":
            self.fn = self.i.split(".")  # split name
            self.fn = self.fn[0][:-1]  # remove last char
            name = (str(self.fn) + "_DBSCAN_" + str(self.epsilon) + "_" + str(self.min_sample) +
                    "_" + str(self.appendix) + ".hdf5")
            with h5py.File(os.path.join(self.path_results, str(name)), "w") as locs_file:
                ds_dt = np.dtype({'names': self.data.columns.values, 'formats': formats})
                locs_file.create_dataset("locs", data=data_lst, dtype=ds_dt)
        else:
            with h5py.File(os.path.join(self.path_results, str(name)), "w") as locs_file:
                ds_dt = np.dtype({'names': self.data.columns.values, 'formats': formats})
                locs_file.create_dataset("locs", data=data_lst, dtype=ds_dt)

    def save_yaml(self):
        """opens existing yaml file, adds analysis parameters and saves file in new path"""
        name = (str(self.fn[0]) + "_DBSCAN_" + str(self.epsilon) + "_" + str(self.min_sample) +
                "_" + str(self.appendix) + ".yaml")
        content = []
        self.yaml_param()  # save parameter in yaml file
        if self.appendix == "properties_all":  # save yaml file for all cluster prop
            self.fn2 = self.i.split(".")  # split name
            self.fn2 = self.fn2[0]
            with open(os.path.join(self.path_files, str(self.fn2) + ".yaml"), 'r') as yaml_file:
                text = _yaml.load_all(yaml_file, _yaml.FullLoader)
                print(self.fn2[:-1])
                name = (self.fn2[:-1] + "_DBSCAN_" + str(self.epsilon) + "_" + str(self.min_sample) +
                        "_" + str(self.appendix) + ".yaml")
                with open(os.path.join(self.path_results, name), 'w') as outfile:
                    for line in text:
                        content.append(line)
                    content.append(self.yaml_content)
                    _yaml.dump_all(content, outfile)
        else:
            with open(os.path.join(self.path_files, str(self.fn[0]) + ".yaml"), 'r') as yaml_file:
                text = _yaml.load_all(yaml_file, _yaml.FullLoader)
                with open(os.path.join(self.path_results, name), 'w') as outfile:
                    for line in text:
                        content.append(line)
                    content.append(self.yaml_content)
                    _yaml.dump_all(content, outfile)

    def yaml_param(self):
        """yaml file context"""
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
        """ main file: calls getformats, save_pd_to_hdf5 and save_yaml"""
        formats = self.getformats()
        self.save_pd_to_hdf5(formats)
        self.save_yaml()

@njit
def _calc_mean(dark):
    return np.array(dark.mean())
@njit
def _calc_std(dark):
    return np.array(dark.std())

def main(path, locs, meanframe, stdframe, meandark, stddark, epsilon, min_sample, path_results, filename,
         frame_time, min_cluster_events, create_cluster):
    """ main function. Loops through data in filepath, opens data, performs DBSCAN analysis, filters data
        calculates cluster properties, saves properties and dbscan files in result path."""
    all_cluster_props = pd.DataFrame()  # init emtpy dataframe
    for i in os.listdir(path):
        print(i)  # print name of file
        # loop through dir_list and open files
        if i.endswith(".hdf5"):  # = "ROI1.hdf5":#
            obj = ClusterAnalyzer(path + "\\" + i, locs, meanframe, stdframe, meandark, stddark,
                                  epsilon, min_sample, filename, frame_time, min_cluster_events, create_cluster)
            obj.load_file()  # load files
            obj.nearest_neighbors()  # NN analysis to determine DBSCAN input
            obj.dbscan()  # DBSCAN cluster
            column_lst = obj.get_header_property_file()
            centroids = obj.cluster_area()
            obj.calc_clusterprops()
            cluster_props = obj.filter()
            obj.calc_inverse_darktime()
            dbscan_filtered = obj.save_dbscan_file(i)
            all_cluster_props = obj.combine_data(all_cluster_props, cluster_props)  # concat all files
        save_obs = Save(path_results, path, filename, cluster_props, epsilon, min_sample, column_lst,
                        "properties", i)
        save_obs.main()
    save_obs = Save(path_results, path, filename, all_cluster_props, epsilon, min_sample, column_lst,
                    "properties_all", i)
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



# with open(os.path.join(path_yaml), 'r') as yaml_file:
        #     text = _yaml.load_all(yaml_file, _yaml.FullLoader)
        #     with open(os.path.join(self.path, name), 'w') as outfile:
        #         for doc in text:
        #             content.append(doc)
        #         content.append(self.yaml_content)
        #         _yaml.dump_all(content, outfile)