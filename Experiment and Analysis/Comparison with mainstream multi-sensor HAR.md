# Comparison with mainstream multi-sensor HAR


| Dataset | Model | NA | CA |
| :--- | :--- | :--- | :--- |
| UCI-HAR [1] | CNN [2] | 6 | 95.00 |
| Berkeley-MHAD [3] | PSKD [4] | 11 | 94.76 |
| PAMAP 2 [5] | BiLSTM [6] | 18 | 99.27 |
| CWNU-RDM | Model7(RSSI,DF,Phase) | 21 | 98.59 |
| CWNU-RDM | Model8(DF,RSSI,Phase) | 21 | 99.27 |
| CWNU-RDM | Model9(DF,Phase,RSSI) | 21 | 98.82 |


Comparison with Mainstream Multi-sensor HARs: To comprehensively evaluate the performance of our proposed models, we also compared them with state-of-the-art mainstream multi-sensor HAR benchmarks. As summarized in Table VIII, our models (Model17, Model18, and Model19) achieve highly competitive results on CWNU-RDM: Model18 reaches an accuracy of 99.27% for 21 activities. This performance is comparable to that of the leading methods on widely recognized public datasets, and even superior in most cases. For instance, on the UCI-HAR [1], a classical CNN [2] reports 95.0% accuracy for 6 activities. Similarly, on the dataset Berkeley-MHAD [3], PSKD [4] achieves 94.76% accuracy for 11 activities. Even on the more challenging dataset PAMAP2 [5], which contains 18 activities, a recent BiLSTM-based approach [6] attains 99.27%. Notably, our model achieves such high performance by only utilizing the tag response feature data (RSSI, DF, and Phase), without relying on additional sensors such as IMUs or cameras. These results indicate that our HAR not only outperforms existing bound-RFID HARs but also exhibits strong competitiveness compared with mainstream multi-sensor-based HARs, while our HAR retains the advantages of low cost, easy deployment, and privacy protection.



[1] D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, “Human activity recognition on smartphones using a multiclass hardware-friendly support vector machine,” International workshop on ambient assisted living, pp. 216–223, 2012. 

[2] M. M. Hossain, T. A. Han, S. S. Ara, and Z. U. Shamszaman, “Benchmarking classical, deep, and generative models for human activity recognition,” arXiv preprint arXiv:2501.08471, 2025. 

[3] F. Ofli, R. Chaudhry, G. Kurillo, R. Vidal, and R. Bajcsy, “Berkeley mhad: A comprehensive multimodal human action database,” IEEE Workshop on Applications of Computer Vision (WACV), pp. 53–60, 2013. 

[4] J. Ni, A. H. Ngu, and Y. Yan, “Progressive cross-modal knowledge distillation for human action recognition,” Proceedings of the 30th ACM International Conference on Multimedia, pp. 5903–5912, 2022. 

[5] A. Reiss, “PAMAP2 Physical Activity Monitoring,” UCI Machine Learning Repository, 2012. DOI: https://doi.org/10.24432/C5NW2H. 

[6] A. Bollampally, J. Kavitha, P. Sumanya, D. Rajesh, A. Y. Jaffar, W. N. Eid, H. M. Albarakati, F. M. Aldosari, and A. A. Alharbi, “Optimizing edge computing for activity recognition: A bidirectional lstm approach on the pamap2 dataset,” Engineering, Technology & Applied Science Research, vol. 14, no. 6, pp. 18086–18093, 2024.