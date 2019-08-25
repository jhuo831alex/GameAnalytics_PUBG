# About the Project
The main goal of this project was to identify cheaters in PlayerUnknown’s Battlegrounds (PUBG), one of the most popular games in the world, using anomaly detection. The analysis was based on the assumption that cheaters are players who have absurdly above-average performance in games.

# Methodology 
* Data: 4 million players' match summaries collected through *PUBG Developer API* with 29 features
* Identified top features that were correlated with winning the game using LightGBM
* Detected outliers via data profiling
  * Visualized top features using Plotly
  * Looked at extreme values via calculating quantiles of each features
* Treated the outliers found by data profiling as the ground truth
* Used One-Class SVM and Isolation Forest to automatically detect outliers
* Tuned and evaluated models through computing the true positive rates (what proportion of outliers found by data profiling are also identified as anomaly by models)
  * SVM: 98.87%
  * Isolation Forest: 91.27%
* Integrated the results of two models
  * Computed overlap coefficient of the outliers by One-Class SVM and Isolation Forest: 64.39%
  * Regarded the overlapping population to be highly likely to be cheaters

# Further Details
For more information, check out: 
<p>- <a href="Presentation_Markdown.html" target="_blank">Markdown</a>
<p>- <a href="Project_Report.pdf" target="_blank">Project Report</a>
