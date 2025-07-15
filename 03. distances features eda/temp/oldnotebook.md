### 0. Setup


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

os.chdir("../")
from scripts import utils
from pathlib import Path
import matplotlib.gridspec as gridspec
from tqdm.auto import tqdm
```


```python
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
```


```python
from mlxtend.evaluate import feature_importance_permutation
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.utils.estimator_checks import check_estimator
from mlxtend.feature_selection import (
    SequentialFeatureSelector,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import matplotlib.ticker as ticker
import distclassipy as dcpy
```


```python
epsilon = np.finfo(np.float32).eps
```


```python
with open("settings.txt") as f:
    settings_dict = json.load(f)
seed_val = settings_dict["seed_choice"]
np.random.seed(seed_val)
sns_dict = settings_dict["sns_dict"]
sns.set_theme(**sns_dict)
```


```python
unique_metrics = ['euclidean',
 'braycurtis',
 'canberra',
 'cityblock',
 'chebyshev',
 'clark',
 'correlation',
 'cosine',
 'hellinger',
 'jaccard',
 'lorentzian',
 'meehl',
 'motyka',
 'soergel',
 'wave_hedges',
 'kulczynski',
                 ]


final_features = [
    "SPM_A_Y",
    "Multiband_period",
    "r-i",
    "Harmonics_phase_4_i",
    "Harmonics_phase_2_r",
    "Power_rate_4",
]
```


```python
features = pd.read_parquet("data/reduced_balancedfeatures_LATEST.parquet")
features["class"].value_counts()
```




    class
    CEP     683
    DSCT    683
    EB      683
    RRL     683
    Name: count, dtype: int64




```python
features = features.sample(frac=1)
```


```python
y_normal_df = features["class"]
# X_df = features.drop(["Coordinate_x", "Coordinate_y", "Coordinate_z", "class"], axis=1)
X_normal_df = features.loc[:, final_features]
```


```python
other_features_df = pd.read_parquet("data/otherclassobjs_features.parquet")
other_features_df.index.name = "snid"
other_lc_df = pd.read_parquet("data/otherclassobjs.parquet")
other_lc_df.index.name = "snid"

other_lc_df=other_lc_df[~other_lc_df["class"].isin(['d-Sct', 'Cepheid', 'EB', 'RRL'])]
other_features_df=other_features_df.loc[other_lc_df.index]


other_lc_df = other_lc_df.loc[other_features_df.index]

assert (other_lc_df.index == other_features_df.index).all()
```


```python
X_anom_df = other_features_df.loc[:, X_normal_df.columns].dropna()
X_anom_df = X_anom_df.drop(np.intersect1d(X_anom_df.index, X_normal_df.index))
y_anom_df = other_lc_df.loc[X_anom_df.index]["class"]
X_anom_df=X_anom_df.loc[y_anom_df.index]
```


```python
X_df = pd.concat([X_normal_df, X_anom_df])
y_df = pd.concat([y_normal_df, y_anom_df])
```


```python
y_normal_df.value_counts()
```




    class
    EB      683
    DSCT    683
    RRL     683
    CEP     683
    Name: count, dtype: int64




```python
y_anom_df.value_counts()
```




    class
    uLens-Single_PyLIMA       86
    uLens-Binary              79
    uLens-Single-GenLens      77
    SLSN-I+host               74
    PISN-MOSFIT               72
    PISN-STELLA_HYDROGENIC    66
    dwarf-nova                65
    PISN-STELLA_HECORE        64
    ILOT                      63
    AGN                       62
    SNIb+HostXT_V19           60
    SLSN-I_no_host            60
    SNIa-SALT3                59
    SNIc+HostXT_V19           53
    SNIc-Templates            52
    SNIIn+HostXT_V19          51
    SNIax                     51
    SNIa-91bg                 51
    SNIIn-MOSFIT              50
    SNIIb+HostXT_V19          50
    SNIcBL+HostXT_V19         50
    SNIb-Templates            49
    CART                      46
    SNII-Templates            46
    SNII+HostXT_V19           40
    TDE                       38
    SNII-NMF                  19
    SL-SNII                   16
    SL-SN1a                   15
    Mdwarf-flare               8
    KN_K17                     6
    KN_B19                     4
    SL-SNIb                    3
    SL-SNIc                    1
    Name: count, dtype: int64




```python
assert (X_df.index == y_df.index).all()
```

### 1. Isolation Forest


```python
from sklearn.ensemble import IsolationForest
from scipy import stats
```


```python
def runif(random_state, X_traindf, X_testdf):
    clf = IsolationForest(
        max_samples=10, contamination="auto", random_state=random_state
    )
    clf.fit(X_traindf)  # , sample_weight=1.0/(Xfixed[:,2,:])**2)
    scores_pred_Norm = clf.decision_function(X_testdf)
    # scores_pred_Norm = clf.score_samples(X)
    # .sample_score(X)

    threshold_Norm = stats.scoreatpercentile(scores_pred_Norm, 100 * 0.1)
    #y_predNorm = clf.predict(X)
    abornmality = -scores_pred_Norm
    iforest_df = pd.DataFrame(
        abornmality, columns=["abnormality"], index=X_testdf.index,
    )  # higher is more abnormal
    # iforest_df.index.name = "sample_num"
    # iforest_df = iforest_df.sort_values(by="abnormality", ascending=False)
    return iforest_df.index, iforest_df
```


```python
ranksIF = []
ranksIDdfs = []
ifseeds = np.random.randint(100, 1000, 3)
for i in ifseeds:
    _ = runif(i, X_traindf=X_normal_df, X_testdf=X_df)
    ranksIF.append(_[0])
    ranksIDdfs.append(_[1])
ranksIF
```




    [Index([ 93852051, 133953454, 156704754,  94869465, 143158599,  97209193,
            118286557, 145870285, 107197896,  34324163,
            ...
             85155628,  35432717,  89686555, 132925610, 135470168,  69018019,
             97601275,  97897233, 130127663,  90121628],
           dtype='int64', name='snid', length=4318),
     Index([ 93852051, 133953454, 156704754,  94869465, 143158599,  97209193,
            118286557, 145870285, 107197896,  34324163,
            ...
             85155628,  35432717,  89686555, 132925610, 135470168,  69018019,
             97601275,  97897233, 130127663,  90121628],
           dtype='int64', name='snid', length=4318),
     Index([ 93852051, 133953454, 156704754,  94869465, 143158599,  97209193,
            118286557, 145870285, 107197896,  34324163,
            ...
             85155628,  35432717,  89686555, 132925610, 135470168,  69018019,
             97601275,  97897233, 130127663,  90121628],
           dtype='int64', name='snid', length=4318)]




```python
df=pd.concat(ranksIDdfs,axis=1)
df.columns = ["if1", "if2", "if3"]
df = df.sort_values(by="if2",ascending=False)
for c in df.columns:
    df["quartile_" + c] = pd.qcut(df[c], q=4, labels=False)  # Divides into 4 quantiles
```


```python
df["class"] = y_df.loc[df.index]
```

### Distances


```python
lcdc = dcpy.DistanceMetricClassifier(
    scale=True,
    central_stat="median",
    dispersion_stat="std",
)
lcdc.fit(X_normal_df, y_normal_df)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DistanceMetricClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>DistanceMetricClassifier</div></div><div><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>DistanceMetricClassifier()</pre></div> </div></div></div></div>




```python
dist_df_dict = {}
for metric in tqdm(unique_metrics, desc="Metric", leave=False):
    metric_str = utils.get_metric_name(metric)
    _ = lcdc.predict_and_analyse(X_df, metric=metric)

    dist_df = lcdc.centroid_dist_df_

    dist_df["minimum_distance"] = dist_df.min(axis=1)
    dist_df["median_distance"] = dist_df.median(axis=1)

    dist_df_dict[metric_str] = dist_df
```


    Metric:   0%|          | 0/16 [00:00<?, ?it/s]



```python
n_metrics = len(unique_metrics)
clusters = np.unique(y_normal_df)
n_clusters = len(clusters)
n_objects = dist_df.shape[0]


print(f"n: {n_objects=}")
print(f"m: {n_metrics=}")
print(f"k: {n_clusters=}")
```

    n: n_objects=4318
    m: n_metrics=16
    k: n_clusters=4



```python
dist_df_arr = np.zeros((n_objects, n_clusters, n_metrics))
for i in range(n_objects):
    for j, cluster in enumerate(clusters):
        for k, metric in enumerate(unique_metrics):
            metric_str = utils.get_metric_name(metric)
            dist_df_arr[i, j, k] = dist_df_dict[metric_str].loc[i, f"{cluster}_dist"]
```


```python
from sklearn.preprocessing import minmax_scale
scaled_data = np.empty_like(dist_df_arr)

# Min-Max Scaling
for i in range(len(unique_metrics)):
  scaled_data[:,:,i] = minmax_scale(dist_df_arr[:,:,i], feature_range=(epsilon, 1), axis=1)

scaled_dist_df_dict = {}
for i,m in enumerate(unique_metrics):
  m = utils.get_metric_name(m)
  scaled_dist_df_dict[m] = pd.DataFrame(scaled_data[:,:,i], columns = dist_df_dict[m].columns[:-2], index=dist_df_dict[m].index)
```


```python
scaled_dist_df_arr = np.zeros((n_objects, n_clusters, n_metrics))
for i in range(n_objects):
    for j, cluster in enumerate(clusters):
        for k, metric in enumerate(unique_metrics):
            metric_str = utils.get_metric_name(metric)
            scaled_dist_df_arr[i, j, k] = scaled_dist_df_dict[metric_str].loc[i,f"{cluster}_dist"]
```


```python
for metric in tqdm(unique_metrics, desc="Metric", leave=False):
    metric_str = utils.get_metric_name(metric)
    dist_df = scaled_dist_df_dict[metric_str]

    dist_df['minimum_distance'] = dist_df.min(axis=1)
    dist_df['median_distance'] = dist_df.median(axis=1)

    scaled_dist_df_dict[metric_str] = dist_df
```


    Metric:   0%|          | 0/16 [00:00<?, ?it/s]



```python
cluster_aggregates = ["min", "median"]
metric_aggregates = ["min", "p25", "median"]
```


```python
dist_aggregates = np.zeros((len(dist_df_dict["Euclidean"]), 3, 2))
```


```python
dist_minK_by_M = np.array([dist_df_dict[utils.get_metric_name(metric)]["minimum_distance"]
                           for i,metric in enumerate(unique_metrics)])
dist_medK_by_M = np.array([dist_df_dict[utils.get_metric_name(metric)]["median_distance"]
                           for i,metric in enumerate(unique_metrics)])
scaled_dist_minK_by_M = np.array([scaled_dist_df_dict[utils.get_metric_name(metric)]["minimum_distance"]
                           for i,metric in enumerate(unique_metrics)])
scaled_dist_medK_by_M = np.array([scaled_dist_df_dict[utils.get_metric_name(metric)]["median_distance"]
                           for i,metric in enumerate(unique_metrics)])
```


```python
assert (ranksIDdfs[0].index == X_df.index).all()

## THIS IS THE REASON WHY I REMOVED SORTING FROM IF FUNCTION
```


```python
dist_aggregates = pd.DataFrame(np.concatenate(
    [np.quantile(dist_minK_by_M, [0,0.25,0.5], axis=0), #metric percentiles on the min of clusters
    np.quantile(dist_medK_by_M, [0,0.25,0.5], axis=0), #metric percentiles on the median of clusters
     np.quantile(scaled_dist_minK_by_M, [0,0.25,0.5], axis=0), #metric percentiles on the scaled min of clusters
     np.quantile(scaled_dist_medK_by_M, [0,0.25,0.5], axis=0), #metric percentiles on the scaled med of clusters
    ]).T,
                               columns=[k+m for k in ["min", "med"] for m in ["min", "25th", "med"]] +
     [k+m for k in ["scaledmin", "scaledmed"] for m in ["min", "25th", "med"]],
                              index=X_df.index)

dist_aggregates = pd.concat([dist_aggregates,df],axis=1).drop(["quartile_if1", "quartile_if2", "quartile_if3"],axis=1)
```


```python
df = dist_aggregates

normal_classes = ["CEP", "DSCT", "EB", "RRL"]
df['status'] = df['class'].apply(lambda x: 'normal' if x in normal_classes else 'anomalous')
```


```python
# df.loc[:,["if2","class"]].sort_values("if2",ascending=False).iloc[:10]
```


```python
df.loc[:,["minmin","class","status"]].sort_values("minmin",ascending=False).iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>minmin</th>
      <th>class</th>
      <th>status</th>
    </tr>
    <tr>
      <th>snid</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>88253880</th>
      <td>0.772094</td>
      <td>SNIa-91bg</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>141445943</th>
      <td>0.708816</td>
      <td>ILOT</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>37734415</th>
      <td>0.703729</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>140819459</th>
      <td>0.700631</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>118315745</th>
      <td>0.695961</td>
      <td>uLens-Single-GenLens</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>114522598</th>
      <td>0.683073</td>
      <td>uLens-Single-GenLens</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>103313063</th>
      <td>0.679649</td>
      <td>SNIa-91bg</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>99141506</th>
      <td>0.675586</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>43265685</th>
      <td>0.671533</td>
      <td>PISN-STELLA_HYDROGENIC</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>46466259</th>
      <td>0.666938</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[:,["med25th","class","status"]].sort_values("med25th",ascending=False).iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>med25th</th>
      <th>class</th>
      <th>status</th>
    </tr>
    <tr>
      <th>snid</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>92717442</th>
      <td>0.990457</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>96837672</th>
      <td>0.989393</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>130307557</th>
      <td>0.987811</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>125103424</th>
      <td>0.986368</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>83406315</th>
      <td>0.983402</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>50610308</th>
      <td>0.981675</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>74295693</th>
      <td>0.978249</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>155481312</th>
      <td>0.975153</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>115790147</th>
      <td>0.974630</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>94602724</th>
      <td>0.973455</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[:,["minmed","class","status"]].sort_values("minmed",ascending=False).iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>minmed</th>
      <th>class</th>
      <th>status</th>
    </tr>
    <tr>
      <th>snid</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10771496</th>
      <td>4.176774</td>
      <td>KN_K17</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>88253880</th>
      <td>3.986333</td>
      <td>SNIa-91bg</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>46466259</th>
      <td>3.929110</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>141445943</th>
      <td>3.859973</td>
      <td>ILOT</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>96837672</th>
      <td>3.821193</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>37734415</th>
      <td>3.738783</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>37691298</th>
      <td>3.662364</td>
      <td>SNII-Templates</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>75421831</th>
      <td>3.656606</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>99141506</th>
      <td>3.492102</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>118315745</th>
      <td>3.474106</td>
      <td>uLens-Single-GenLens</td>
      <td>anomalous</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[:,["medmed","class","status"]].sort_values("medmed",ascending=False).iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>medmed</th>
      <th>class</th>
      <th>status</th>
    </tr>
    <tr>
      <th>snid</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73799575</th>
      <td>5.329046</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>103308635</th>
      <td>5.162816</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>59245585</th>
      <td>5.139324</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>108559610</th>
      <td>4.901718</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>17960110</th>
      <td>4.836763</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>4183976</th>
      <td>4.736280</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>130723508</th>
      <td>4.727224</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>125103424</th>
      <td>4.699312</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>34882116</th>
      <td>4.655074</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>96837672</th>
      <td>4.640638</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
  </tbody>
</table>
</div>


