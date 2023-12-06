import numpy as np
from os.path import join as oj
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

from distutils.spawn import find_executable
if find_executable('latex'): 
    print('latex installed')
    plt.rcParams['text.usetex'] = True

LABEL_FONTSIZE = 20
MARKER_SIZE = 10
AXIS_FONTSIZE = 26
TITLE_FONTSIZE= 26
LINEWIDTH = 4

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('figure', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
plt.rc('axes', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXIS_FONTSIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LABEL_FONTSIZE)    # legend fontsize
plt.rc('lines', markersize=MARKER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=LINEWIDTH)  # fontsize of the figure title

from scipy.stats import rankdata

results_dir = oj('MNIST_results', 'precision', 'N50')

initial_distances_C = np.loadtxt(oj(results_dir, 'initial_distances_C'))
initial_distances_H = np.loadtxt(oj(results_dir, 'initial_distances_H'))

SVs_C = np.loadtxt(oj(results_dir, 'SVs_C'))
ranks = rankdata(SVs_C) # rank of 1 is the smallest SVs_C

nonneg_SVs_C = SVs_C - (min(SVs_C) ) + 1e-10 
normed_SVs_C = nonneg_SVs_C / sum(nonneg_SVs_C)
sizes = (normed_SVs_C * (300 / max(normed_SVs_C)))

initial_distances_C = np.clip(initial_distances_C, a_min=1e-8, a_max=np.max(initial_distances_C))


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


X = initial_distances_C
labels_true = [0] * 50 + [1] *50 + [2] *50

# Compute DBSCAN
db = DBSCAN(eps=1e-3, min_samples=10, metric='precomputed').fit(X)
# clustering = DBSCAN(eps=3, min_samples=5, metric='precomputed').fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))


# Black removed and is used for noise instead.
unique_labels = set(labels)

colors = ['C'+str(i) for i in range(len(unique_labels))] + ['k']

architectures = ['CNN', 'MLP', 'LR']

print(unique_labels)
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
        continue

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=col, # tuple(col),
        markeredgecolor="k",
        markersize=14,
        label=architectures[k],
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=col, #tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.legend()
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()
