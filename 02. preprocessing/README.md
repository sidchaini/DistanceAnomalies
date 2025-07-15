### README: Feature Selection Workflow

Disclaimer, this is an LLM summary, but still accurate from my reading of it:
---

Here's how I went from a massive feature table to the final, robust set of 6 features. The whole process is a "funnel" designed to narrow down the options intelligently.

**1. Preprocessing & De-correlation (`02a. preprocessing.ipynb`)**

*   **Goal:** Clean up the raw feature set and get rid of redundant features.
*   **Actions:**
    *   Loaded the full `allfeatures.parquet` file.
    *   Dropped any stars missing the basic `g-r`, `r-i`, `i-z` colors.
    *   Calculated a correlation matrix for all remaining features.
    *   Found groups of features with a correlation > 0.9.
    *   From each highly correlated group, I programmatically kept only **one** feature—the one with the fewest `NaN` values. This is a simple but effective way to reduce redundancy.
*   **Result:** A smaller, de-correlated feature set (`reduced_features_LATEST.parquet`) and a class-balanced version for the next steps.

**2. Broad Search with SFS (`02b. sfs.ipynb`)**

*   **Goal:** See which features are generally useful across many different ways of measuring distance.
*   **Actions:**
    *   Used Sequential Feature Selection (SFS), a fast "greedy" method.
    *   I ran SFS for **18 different distance metrics** (Euclidean, Canberra, Cosine, etc.). I didn't want to bet on just one metric.
    *   For each metric, I found the best-performing feature subset and saved the results (plots, scores, etc.).
*   **Result:** 18 different "best" feature sets. I noticed some features appeared consistently across many of the lists.

**3. Creating a Superset & Deep Dive with EFS (`02c. efs on sfs common subset.ipynb`)**

*   **Goal:** Take the "all-star" features from the SFS runs and find the absolute best combination among them.
*   **Actions:**
    *   I collected all the features from the 18 "best" sets and counted how many times each appeared.
    *   I created a **superset of the top 12 most frequent features**. These are the clear MVPs.
    *   I then ran Exhaustive Feature Selection (EFS) on this 12-feature superset. EFS is computationally intensive because it tries **every single possible combination**, but it guarantees finding the true best subset.
    *   Again, I did this for all 18 distance metrics.
*   **Result:** A CSV file for each metric containing the performance score for all 4,095 possible subsets of the 12 "all-star" features.

**4. Final Selection (`02d. final feature set.ipynb`)**

*   **Goal:** Pick the single best, most robust feature set from the EFS results.
*   **Actions:**
    *   I aggregated all the EFS results into one big DataFrame. Rows were the feature subsets, and columns were their F1 scores for each of the 18 metrics.
    *   I calculated the **average F1 score** and **standard deviation** for each subset across all metrics.
    *   The winning subset wasn't just the one with the highest average score, but one that was small, had a high average, and was stable (low std dev).
*   **Final Result:** The 6-feature set `['SPM_A_Y', 'Multiband_period', 'r-i', 'Harmonics_phase_4_i', 'Harmonics_phase_2_r', 'Power_rate_4']` was chosen as the best trade-off between high performance and simplicity.