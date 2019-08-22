<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a>
    <img src="https://steamcdn-a.akamaihd.net/steam/apps/578080/header.jpg?t=1564606217" alt="Logo" width="230" height="107">
  </a>
  <h2 align="center">Data Analytics: PlayerUnknown’s Battlegrounds </h2>

  <p align="center">
    Identify Cheaters in game PlayerUnknown’s Battlegrounds (PUBG)
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About the project
The main goal of this project was to catch cheaters in PlayerUnknown’s Battlegrounds (PUBG), one of the most popular games in the world, using anomaly detection. Our analysis is based on the assumption that cheaters are players who have absurdly above-average performance in games.

* Data: 4 million players' match summaries with 29 features
* Identified top features that are correlated with winning the game using LightGBM
* Detected outliers via data profiling
  * Visualized top features using Plotly and looked for outliers on graphs and through calculating quantiles
* Treated the outliers found by data profiling as ground truth
* Used One-Class SVM and Isolation Forest to automatically detect outliers
* Tuned and evaluated models through computing the true positive rates (what proportion of outliers found by data profiling are also identified as anomaly by models)
  * SVM: 98.87%
  * Isolation Forest: 91.27%
* Integrated the results of two models
  * Computed overlap coefficient: 64.39%
  * Regarded the overlapping population to be highly likely to be cheaters

For more information: 
- [Markdown](https://github.com/jhuo831alex/DataAnalytics_PUBG/blob/master/Presentation%20Markdown_files/Presentation%20Markdown.md) 
- [Report](https://github.com/jhuo831alex/DataAnalytics_PUBG/blob/master/Final%20Report.pdf)

<!-- CONTACT -->
## Contact
Alex (Jiahao) Huo: 
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Email][email-shield]][email-url]


<!-- MARKDOWN LINKS & IMAGES -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/jiahaohuo/
[email-shield]: https://img.shields.io/badge/-Gmail-black.svg?style=flat-square&logo=gmail&colorB=555
[email-url]: mailto:jiahao.h@columbia.edu
