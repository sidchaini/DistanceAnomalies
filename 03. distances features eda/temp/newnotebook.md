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
 # 'marylandbridge',
 'meehl',
 'motyka',
 'soergel',
 'wave_hedges',
 'kulczynski',
 # 'add_chisq'
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

### 1. Data Prep
- Make a knowns/inlier dataset with 4 classes (CEP, RR, DSCT and EB): ```X_knowns_df``` and ```y_knowns_df```
- Make an unknowns/outlier/anomaly dataset with other classes: ```X_anom_df``` and ```y_anom_df```

An ```X_all_df``` and ```y_all_df``` contains both of these.


```python
knowns = pd.read_parquet("data/reduced_balancedfeatures_LATEST.parquet")
knowns = knowns.sample(frac=1) #shuffle

y_knowns_df = knowns["class"]
X_knowns_df = knowns.loc[:, final_features]


unknowns = pd.read_parquet("data/otherclassobjs_features.parquet")
unknowns.index.name = "snid"
unknowns_lc_df = pd.read_parquet("data/otherclassobjs.parquet")
unknowns_lc_df.index.name = "snid"

unknowns_lc_df=unknowns_lc_df[~unknowns_lc_df["class"].isin(['d-Sct', 'Cepheid', 'EB', 'RRL'])]
unknowns=unknowns.loc[unknowns_lc_df.index]


unknowns_lc_df = unknowns_lc_df.loc[unknowns.index]

assert (unknowns.index == unknowns_lc_df.index).all()


X_anom_df = unknowns.loc[:, X_knowns_df.columns].dropna()
X_anom_df = X_anom_df.drop(np.intersect1d(X_anom_df.index, X_knowns_df.index))
y_anom_df = unknowns_lc_df.loc[X_anom_df.index]["class"]
X_anom_df=X_anom_df.loc[y_anom_df.index]
```


```python
X_all_df = pd.concat([X_knowns_df, X_anom_df]).sample(frac=1)
y_all_df = pd.concat([y_knowns_df, y_anom_df]).loc[X_all_df.index]
```

### 2. Distance-Anomalies


```python
from distclassipy.anomaly import DistanceAnomaly
```


```python
def return_anomcandidates(ad_model):
    ad_model.fit(X_knowns_df.values, y_knowns_df.values)
    
    anomaly_scores = ad_model.decision_function(X_all_df.values)
    
    results_df = pd.DataFrame(index=X_all_df.index)
    results_df['dcpy_score'] = anomaly_scores
    results_df['class'] = y_all_df
    results_df['status'] = results_df['class'].apply(
        lambda x: 'normal' if x in ["CEP", "DSCT", "EB", "RRL"] else 'anomalous'
    )
    
    print("Top anomalies")
    return results_df.sort_values("dcpy_score", ascending=False).iloc[:10]
```


```python
minmin_model = DistanceAnomaly(
    cluster_agg='min', # options:
    metric_agg='min',  # 'min' is equivalent to a 0th percentile
    normalize_scores=True,
    metrics=unique_metrics
)

return_anomcandidates(minmin_model)
```

    Top anomalies





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
      <th>dcpy_score</th>
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
      <th>75421831</th>
      <td>0.606569</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>2857614</th>
      <td>0.594080</td>
      <td>uLens-Single_PyLIMA</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>103344580</th>
      <td>0.285702</td>
      <td>uLens-Binary</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>58287001</th>
      <td>0.268365</td>
      <td>SNIIn-MOSFIT</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>132288802</th>
      <td>0.188947</td>
      <td>uLens-Single-GenLens</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>10771496</th>
      <td>0.187548</td>
      <td>KN_K17</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>69330474</th>
      <td>0.182256</td>
      <td>EB</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>76558149</th>
      <td>0.163304</td>
      <td>SNIc-Templates</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>88253880</th>
      <td>0.155072</td>
      <td>SNIa-91bg</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>143000840</th>
      <td>0.135581</td>
      <td>EB</td>
      <td>normal</td>
    </tr>
  </tbody>
</table>
</div>




```python
med25th_model = DistanceAnomaly(
    cluster_agg='median', # options:
    metric_agg='percentile_25',  # 'min' is equivalent to a 0th percentile
    normalize_scores=True,
    metrics=unique_metrics
)

return_anomcandidates(med25th_model)
```

    Top anomalies





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
      <th>dcpy_score</th>
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
      <th>96837672</th>
      <td>0.892216</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>92717442</th>
      <td>0.838555</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>130307557</th>
      <td>0.742600</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>74295693</th>
      <td>0.722970</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>17824796</th>
      <td>0.707974</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>125103424</th>
      <td>0.655375</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>34882116</th>
      <td>0.641512</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>94602724</th>
      <td>0.620149</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>50610308</th>
      <td>0.571292</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>115790147</th>
      <td>0.568397</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
  </tbody>
</table>
</div>




```python
minmed_model = DistanceAnomaly(
    cluster_agg='min', # options:
    metric_agg='median',  # 'min' is equivalent to a 0th percentile
    normalize_scores=True,
    metrics=unique_metrics
)

return_anomcandidates(minmed_model)
```

    Top anomalies





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
      <th>dcpy_score</th>
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
      <td>0.892393</td>
      <td>SNIa-91bg</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>75421831</th>
      <td>0.890495</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>2857614</th>
      <td>0.867602</td>
      <td>uLens-Single_PyLIMA</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>141445943</th>
      <td>0.847133</td>
      <td>ILOT</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>140819459</th>
      <td>0.838644</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>37734415</th>
      <td>0.836380</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>46466259</th>
      <td>0.820010</td>
      <td>dwarf-nova</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>43265685</th>
      <td>0.807233</td>
      <td>PISN-STELLA_HYDROGENIC</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>56888325</th>
      <td>0.793123</td>
      <td>SNIcBL+HostXT_V19</td>
      <td>anomalous</td>
    </tr>
    <tr>
      <th>118315745</th>
      <td>0.791458</td>
      <td>uLens-Single-GenLens</td>
      <td>anomalous</td>
    </tr>
  </tbody>
</table>
</div>




```python
medmed_model = DistanceAnomaly(
    cluster_agg='median', # options:
    metric_agg='median',  # 'min' is equivalent to a 0th percentile
    normalize_scores=True,
    metrics=unique_metrics
)

return_anomcandidates(medmed_model)
```

    Top anomalies





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
      <th>dcpy_score</th>
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
      <th>96837672</th>
      <td>1.000000</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>92717442</th>
      <td>0.935164</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>74295693</th>
      <td>0.882628</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>125103424</th>
      <td>0.839870</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>130307557</th>
      <td>0.817785</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>17824796</th>
      <td>0.808731</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>4183976</th>
      <td>0.796567</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>137874275</th>
      <td>0.784675</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>108559610</th>
      <td>0.784628</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>34882116</th>
      <td>0.778570</td>
      <td>CEP</td>
      <td>normal</td>
    </tr>
  </tbody>
</table>
</div>


