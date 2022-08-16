# Distribution-Free Federated Learning with Conformal Predictions
---
> Federated learning has attracted considerable interest for collaborative machine learning in healthcare to leverage separate institutional datasets while maintaining patient privacy. However, additional challenges such as poor calibration and lack of interpretability may also hamper widespread deployment of federated models into clinical practice, leading to user distrust or misuse of ML tools in high-stakes clinical decision-making. In this paper, we propose to address these challenges by incorporating an adaptive conformal framework into federated learning to ensure distribution-free prediction sets that provide coverage guarantees. Importantly, these uncertainty estimates can be obtained without requiring any additional modifications to the model. Empirical results on the MedMNIST medical imaging benchmark demonstrate our federated method provides tighter coverage over local conformal predictions on 6 different medical imaging datasets for 2D and 3D multi-class classification tasks. Furthermore, we correlate class entropy with prediction set size to assess task uncertainty.

* [Paper](https://arxiv.org/abs/2110.07661)
* [Dataset](https://medmnist.com/)

Accepted as poster presentation at [FL-AAAI22](https://federated-learning.org/fl-aaai-2022/)

Please cite this work as
```
@article{DBLP:journals/corr/abs-2110-07661,
  author    = {Charles Lu and
               Jayashree Kalpathy{-}Cramer},
  title     = {Distribution-Free Federated Learning with Conformal Predictions},
  journal   = {CoRR},
  volume    = {abs/2110.07661},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.07661},
  eprinttype = {arXiv},
  eprint    = {2110.07661},
  timestamp = {Fri, 22 Oct 2021 13:33:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-07661.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
