from sys import call_tracing
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, SpectralEmbedding, TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap
import mpl_toolkits.mplot3d.axes3d as p3
import os

# 3D scatter plot
def scatter3d(data3d, color_map=None, azimuth=-74, elevation=54, figsize=(10, 10)):

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    ax.view_init(elevation, azimuth)

    if color_map is None:
        ax.scatter(data3d[:,0], data3d[:,1], data3d[:,2],
                    s=30, edgecolor='k')
    else:
        ax.scatter(data3d[:,0], data3d[:,1], data3d[:,2],
                    color=plt.cm.jet(color_map),
                    s=30, edgecolor='k')
    return ax

def stdscaler (X, use_std=True):
    if use_std:
        return StandardScaler().fit_transform(X)
    else:
        return X

#########################################################################################################
win_size = 10
use_log = True
use_std = True
# ignore the first day
ignored = 1440

# group_func = 'mean'
group_func = 'mean_and_median'

nn = 50

reducers = {
    'pca': PCA(n_components=3),
    # 'umap': umap.UMAP(n_components=3, metric='cosine', n_neighbors=5, random_state=5566),
    'umap': umap.UMAP(n_components=3, metric='cosine', n_neighbors=nn, random_state=5566),
    # 'umap': umap.UMAP(n_components=3, n_neighbors=15, random_state=5566),
}
# reducer_name = 'pca'
reducer_name = 'umap'
reducer = reducers[reducer_name]
# custom_profile = 'nocustom'
custom_profile = 'cosine-nn%d' % nn
profile = '_'.join([reducer_name, 'log' if use_log else 'nolog', group_func, 'std' if use_std else 'nostd', 'ignore%d' % ignored, custom_profile])
print(profile)
#########################################################################################################
conv_size = 3
txts = [f for f in os.listdir('./aligned') if f.endswith('.txt')]
monitor_cleaned_smooths = []
act_digests = []
act_origs = []
metas = []

one_meta = pd.read_csv('./meta/one_meta.csv', sep='\t')
# txt = txts[0]
for txt in txts:

    # try:
    #     meta = pd.read_csv('./meta/%s' % txt, sep='\t')
    # except:
    #     continue

    meta = one_meta[one_meta.File_Name == os.path.splitext(txt)[0]]
    #meta = one_meta[one_meta.File_Name == 'sdfghjkl;']
    if len(meta) == 0:
        continue

    monitor = pd.read_csv('./aligned/%s' % txt, sep='\t', header=None)
    monitor = monitor.rename({1:'date', 2:'time'}, axis=1)
    monitor_cleaned = pd.concat([monitor.iloc[:,1:3], monitor.iloc[:,10:]], axis=1)
    monitor_cleaned = monitor_cleaned.iloc[ignored:,:]
    # monitor_cleaned_smooth = monitor_cleaned.iloc[:,2:].apply(np.convolve, v=np.array([1,1,1,1,1]), mode='valid')
    monitor_cleaned_smooth = monitor_cleaned.iloc[:,2:].apply(np.convolve, v=np.ones(conv_size), mode='valid')
    
    if use_log:
        monitor_cleaned_smooth = np.log(monitor_cleaned_smooth + 1)

    monitor_cleaned_smooth = pd.concat([monitor_cleaned.iloc[(conv_size-1):,:2].reset_index(drop=True), monitor_cleaned_smooth], axis=1)

    hms = np.array([t.replace(' ', ':').split(':') for t in monitor_cleaned_smooth.time], dtype=int)
    monitor_cleaned_smooth['h'] = hms[:,0]
    monitor_cleaned_smooth['mNcell'] = hms[:,1] // win_size
    #monitor_cleaned_smooth = monitor_cleaned_smooth[monitor_cleaned_smooth.h.isin([18,19,20,21,22,23,0,1,2,3,4,5])]
    #monitor_cleaned_smooth['m'] = hms[:,1]
    
    if group_func == 'mean':
        act_digest = pd.concat([monitor_cleaned_smooth.groupby(['h', 'mNcell']).mean().T, monitor_cleaned_smooth.groupby(['h', 'mNcell']).std().T], axis=1)
    else:
        # act_digest = pd.concat([monitor_cleaned_smooth.groupby(['h', 'mNcell']).median().T, monitor_cleaned_smooth.groupby(['h', 'mNcell']).std().T], axis=1)
        q1 = monitor_cleaned_smooth.groupby(['h', 'mNcell']).apply(pd.DataFrame.quantile, q=.25).T.iloc[:-2]
        q3 = monitor_cleaned_smooth.groupby(['h', 'mNcell']).apply(pd.DataFrame.quantile, q=.75).T.iloc[:-2]

        monitor_cleaned_smooth_min = monitor_cleaned_smooth.groupby(['h', 'mNcell']).min().T.iloc[2:]
        monitor_cleaned_smooth_max = monitor_cleaned_smooth.groupby(['h', 'mNcell']).max().T.iloc[2:]
        IQR = q3 - q1
        monitor_cleaned_smooth_whisker_min = q1 - 1.5 * IQR
        monitor_cleaned_smooth_whisker_max = q3 + 1.5 * IQR
        
        whisker_min_oob = (monitor_cleaned_smooth_whisker_min < monitor_cleaned_smooth_min)
        whisker_max_oob = (monitor_cleaned_smooth_whisker_max > monitor_cleaned_smooth_max)
        monitor_cleaned_smooth_whisker_min[whisker_min_oob] = monitor_cleaned_smooth_min[whisker_min_oob]
        monitor_cleaned_smooth_whisker_max[whisker_max_oob] = monitor_cleaned_smooth_max[whisker_max_oob]
        
        act_digest = pd.concat([
            monitor_cleaned_smooth.groupby(['h', 'mNcell']).mean().T, 
            monitor_cleaned_smooth.groupby(['h', 'mNcell']).std().T,
            monitor_cleaned_smooth.groupby(['h', 'mNcell']).median().T, 
            q1,
            q3,
            monitor_cleaned_smooth_whisker_min,
            monitor_cleaned_smooth_whisker_max,
            ], axis=1)
    
    act_orig = monitor_cleaned_smooth.iloc[:,2:34].T
    #act_digest = monitor_cleaned_smooth.groupby(['h', 'mNcell']).mean().T / monitor_cleaned_smooth.groupby(['h', 'mNcell']).std().T
    #act_digest = monitor_cleaned_smooth.groupby(['h', 'mNcell']).mean().T
    act_digests.append(act_digest)
    act_origs.append(act_orig)

    metas.append(meta)
    monitor_cleaned_smooths.append(monitor_cleaned_smooth)

meta_union = pd.concat(metas).reset_index(drop=True)
np.sum(meta_union.Instar.isna())
act_digests_npy = np.concatenate(act_digests)
act_origs_npy = np.concatenate(act_origs)
monitor_cleaned_smooths_union = pd.concat(monitor_cleaned_smooths).reset_index(drop=True)

from sklearn.cluster import OPTICS, DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
# clusterer = OPTICS()
# clusterer = KMeans(n_clusters=3)

reducer = umap.UMAP(n_components=3, metric='cosine', n_neighbors=nn, random_state=5566)

try:
    pwdists = pairwise_distances(dr)
except:
    act_digests_npy = np.concatenate(act_digests)
    act_digests_npy_transformed = stdscaler(act_digests_npy, use_std)
    dr = reducer.fit_transform(act_digests_npy_transformed)
    act_digests_npy_transformed.shape
    dr.shape
    # reducer.fit(act_digests_npy_transformed)
    # reducer.transform(act_digests_npy_transformed)
    pwdists = pairwise_distances(dr)

pwdists.shape


min_samples = 20

shortest_dists_mean = np.take_along_axis(pwdists, np.argsort(pwdists)[:,1:(1+min_samples)], axis=1).mean(axis=1)
shortest_dists_mean_std = shortest_dists_mean.std()
eps = shortest_dists_mean.mean() + 2 * shortest_dists_mean_std

clusterer = DBSCAN(eps=eps, min_samples=min_samples)
group_idxs_ = clusterer.fit_predict(dr) + 1
group_idxs = group_idxs_[group_idxs_ > 0]
dr_wg = dr[group_idxs_ > 0]
gcolor_map = group_idxs / (group_idxs.max() + 1)


fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs[0][0].scatter(dr_wg[:,0], dr_wg[:,1], c=gcolor_map)
axs[0][1].scatter(dr_wg[:,2], dr_wg[:,1], c=gcolor_map)
axs[1][0].scatter(dr_wg[:,0], dr_wg[:,2], c=gcolor_map)
for tid in range(dr.shape[0]):
    axs[0][0].text(dr_wg[tid,0], dr_wg[tid,1], group_idxs[tid])
    axs[0][1].text(dr_wg[tid,2], dr_wg[tid,1], group_idxs[tid])
    axs[1][0].text(dr_wg[tid,0], dr_wg[tid,2], group_idxs[tid])
plt.show()

instar_values = meta_union.Instar.values
instar_values[meta_union.Instar.isna()] = -1
instar_values = instar_values.astype(int)

gcolor_map_ = (instar_values+1) / (instar_values.max() + 1)
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs[0][0].scatter(dr[:,0], dr[:,1], c=gcolor_map_)
axs[0][1].scatter(dr[:,2], dr[:,1], c=gcolor_map_)
axs[1][0].scatter(dr[:,0], dr[:,2], c=gcolor_map_)
for tid in range(dr.shape[0]):
    axs[0][0].text(dr[tid,0], dr[tid,1], instar_values[tid]+1)
    axs[0][1].text(dr[tid,2], dr[tid,1], instar_values[tid]+1)
    axs[1][0].text(dr[tid,0], dr[tid,2], instar_values[tid]+1)
plt.show()


y_ = meta_union.Instar.values
y_[meta_union.Instar.isna()] = -1
y = y_[y_ != -1]
len(y)
y.sum()

x_ = meta_union[['Source', 'Gen', 'Sex']].copy()
x_['group'] = group_idxs_
x = x_[y_ != -1].copy()


meta_union[x_.group==3]
meta_union.columns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

rfc = RandomForestClassifier(random_state=42, max_depth=4, n_estimators=5000, criterion='entropy')
rfc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
rfc = SVC(kernel='linear')
rfc = LogisticRegression()

enc = OneHotEncoder(sparse=False).fit(x)
new_x = enc.transform(x)
new_x.shape
x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=.5, random_state=42)
# rfc.fit(x_train, y_train)
# rfc.score(x_test, y_test)
# rfc.score(x_train, y_train)
# rfc.feature_importances_
enc.categories_


rfc.fit(x_train, y_train)
accuracy_score(y_train, rfc.predict(x_train))
rfc.score(x_test, y_test)
accuracy_score(y_test, rfc.predict(x_test))
rfc.feature_importances_
np.argsort(np.abs(rfc.coef_))
np.argsort(rfc.feature_importances_)

np.concatenate(enc.categories_, axis=0)[np.argsort(np.abs(rfc.coef_))][:,-5:]
np.concatenate(enc.categories_, axis=0)[np.argsort(rfc.feature_importances_)][-5:]

meta_union[meta_union.Source=='OK']

x_test[:,2].astype(int)
y_test

np.sum(x_test[:,3] == y_test) / len(y_test)
np.sum(x_test[:,3] != y_test) / len(y_test)

np.sum(new_x[:,3] == y) / len(y)

meta_union.Instar

enc.categories_



def packing(df, bug_col = 2):
    window_acts_per_bug.append(df.iloc[:,bug_col].values)
    return None

# for bug_col in range(2, 34):
bug_col=33
window_acts_per_bug = []
dum = monitor_cleaned_smooths_union.groupby(['h', 'mNcell']).apply(packing, bug_col=bug_col)
plt.boxplot(window_acts_per_bug)
plt.show()

###########################################################################################################################

# cand_cols = ['LD_cycle', 'Nest', 'Source', 'Location', 'Elevation', 'Gen', 'Sex', 'Photo']
# cand_cols = ['Nest', 'Source', 'Elevation', 'Gen', 'Sex', 'Photo']
cand_cols = ['Source', 'Gen']
cand_cols = ['Source']
# cand_cols = ['LD_cycle', 'Source', 'Photo']
# np.array(np.meshgrid(cand_cols, cand_cols)).T.reshape(-1, 2)
for i in range(len(cand_cols)):
    for j in range(i, len(cand_cols)):
        cat_col = list(np.unique([cand_cols[i], cand_cols[j]]))
        print(cat_col)

        meta_union = pd.concat(metas).reset_index(drop=True)
        act_digests_npy = np.concatenate(act_digests)

        # filtered_idx = meta_union.Source.isin(['BF', 'MF'])
        # # filtered_idx = meta_union.Source.isin(['WL'])
        # # filtered_idx = meta_union.Source.isin(['WL'])
        # meta_union = meta_union[filtered_idx]
        # act_digests_npy = act_digests_npy[filtered_idx]

        # cat_col = ['Source', 'Photo', 'LD_cycle']
        # cat_col = ['Source']
        # cat_col = ['Source', 'Photo']
        cat_col_str = '_x_'.join(cat_col)
        meta_union[cat_col_str] = meta_union[cat_col].replace(np.nan, 'NaN').astype(str).apply('_x_'.join, axis=1)

        # meta_union.groupby(cat_col).size()

        cat_list, cat_idxs = np.unique(meta_union[cat_col_str].values, return_inverse=True)
        color_map = cat_idxs / (cat_idxs.max() + 1)
        act_digests_npy_transformed = stdscaler(act_digests_npy, use_std)

        dr = reducer.fit_transform(act_digests_npy_transformed)
        # dr = dr[group_idxs_ > 0]
        # color_map = color_map[group_idxs_ > 0]
        # cat_idxs = cat_idxs[group_idxs_ > 0]

        # ax = scatter3d(dr, color_map)
        # for tid in range(dr.shape[0]):
        #     ax.text3D(dr[tid, 0], dr[tid, 1], dr[tid, 2], cat_idxs[tid])
        # plt.show()
        # plt.savefig('./explore/%s.png' % cat_col_str)
        # plt.close()

        if dr.shape[1] >= 3:
            print(cat_list)
            
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            # print(axs)
            
            axs[0][0].scatter(dr[:,0], dr[:,1], c=color_map)
            axs[0][1].scatter(dr[:,2], dr[:,1], c=color_map)
            axs[1][0].scatter(dr[:,0], dr[:,2], c=color_map)
            for tid in range(dr.shape[0]):
                axs[0][0].text(dr[tid,0], dr[tid,1], cat_idxs[tid])
                axs[0][1].text(dr[tid,2], dr[tid,1], cat_idxs[tid])
                axs[1][0].text(dr[tid,0], dr[tid,2], cat_idxs[tid])
            try:
                os.makedirs('./explore/%s/' % profile,)
            except:
                pass

            plt.savefig('./explore/%s/%s.png' % (profile, cat_col_str))
            plt.close()
            # plt.show()


# pd.DataFrame.from_dict({'gid': group_idxs, 'char': meta_union.Photo.values}).groupby(['gid', 'char']).size().to_frame('size').reset_index(['gid', 'char']).pivot_table()

from scipy.stats import chisquare

group_sizes = pd.DataFrame({'gid':group_idxs}).groupby('gid').size().values
f_exp = group_sizes / group_sizes.sum()

# cand_cols = ['Nest', 'Source', 'Elevation', 'Gen', 'Sex', 'Photo']
cand_cols = ['Source', 'Gen', 'Sex', 'Photo']
cand_cols = ['Instar']
cand_cols = ['Source']
# cand_cols = ['LD_cycle', 'Source', 'Photo']
# np.array(np.meshgrid(cand_cols, cand_cols)).T.reshape(-1, 2)
biased_chars_all = np.array([])
chi2_all = np.array([])
pvalue_all = np.array([])
char_sample_size_all = np.array([])
cat_col_str_all = np.array([])
for i in range(len(cand_cols)):
    for j in range(i, len(cand_cols)):
        cat_col = list(np.unique([cand_cols[i], cand_cols[j]]))

        meta_union = pd.concat(metas).reset_index(drop=True)
        act_digests_npy = np.concatenate(act_digests)

        cat_col_str = '_x_'.join(cat_col)

        print(cat_col_str)
        meta_union[cat_col_str] = meta_union[cat_col].replace(np.nan, 'NaN').astype(str).apply('_x_'.join, axis=1)

        char_to_group = pd.DataFrame.from_dict({'gid': group_idxs, 'char': meta_union[cat_col_str].values[group_idxs_ > 0]}).pivot_table(index='gid', columns='char', aggfunc=len)
        char_to_group = char_to_group.replace(np.nan, 0)

        f_exp_weighted = char_to_group.sum().values * np.repeat(np.expand_dims(f_exp, axis=0), char_to_group.shape[1], axis=0).T
        # pd.concat([char_to_group.reset_index(), pd.DataFrame(f_exp)], axis=1)
        chi2test = chisquare(char_to_group, f_exp=f_exp_weighted)
        
        # group_sizesT = meta_union[[cat_col_str]].groupby(cat_col_str).size().values
        # f_expT = group_sizesT / group_sizesT.sum()
        # chi2testT = chisquare(char_to_group.T, f_exp=np.repeat(np.expand_dims(f_expT, axis=0), char_to_group.shape[0], axis=0).T)
        # if (chi2testT.pvalue < 1e-4).all():
        #     print(cat_col_str)
        
        pvalue_thres_idxs = (chi2test.pvalue < 0.05)
        biased_chars = char_to_group.columns.values[pvalue_thres_idxs]
        biased_chars_all = np.append(biased_chars_all, biased_chars)
        chi2 = chi2test.statistic[pvalue_thres_idxs]
        chi2_all = np.append(chi2_all, chi2)
        pvalue_all = np.append(pvalue_all, chi2test.pvalue[pvalue_thres_idxs])
        char_sample_size_all = np.append(char_sample_size_all, char_to_group.sum()[pvalue_thres_idxs].values)
        cat_col_str_all = np.append(cat_col_str_all, np.repeat(cat_col_str, chi2.shape[0]))
        print(biased_chars)
        # print([d for d in char_to_group.columns.values if d not in biased_chars])

biased_chars_all
biased_chars_all[np.argsort(chi2_all)[-5:]]
pvalue_all[np.argsort(chi2_all)[-5:]]
char_sample_size_all

pd.DataFrame({
    'char_name': np.array(cat_col_str_all)[char_sample_size_all > 20],
    'char_val': biased_chars_all[char_sample_size_all > 20],
    'pvalue': pvalue_all[char_sample_size_all > 20],
    'size': char_sample_size_all[char_sample_size_all > 20],
})

cat_col = ['Photo', 'Source']
meta_union = pd.concat(metas).reset_index(drop=True)
act_digests_npy = np.concatenate(act_digests)

cat_col_str = '_x_'.join(cat_col)
print(cat_col_str)
meta_union[cat_col_str] = meta_union[cat_col].replace(np.nan, 'NaN').astype(str).apply('_x_'.join, axis=1)
char_to_group = pd.DataFrame.from_dict({'gid': group_idxs, 'char': meta_union[cat_col_str].values}).pivot_table(index='gid', columns='char', aggfunc=len)
char_to_group = char_to_group.replace(np.nan, 0)
char_to_group
f_exp

monitor_cleaned_smooth
meta_union