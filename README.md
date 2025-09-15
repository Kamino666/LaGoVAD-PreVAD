# Language-guided Open-world Video Anomaly Detection under Weak Supervision

This repository contains the code and dataset for the paper "Language-guided Open-world Video Anomaly Detection under Weak Supervision"

ArXiv: https://arxiv.org/abs/2503.13160

Video anomaly detection (VAD) aims to detect anomalies that deviate from what is expected.
In open-world scenarios, the expected events may change as requirements change. 
For example, not wearing a mask may be considered abnormal during a flu outbreak but normal otherwise.
However, existing methods assume that the definition of anomalies is invariable, and thus are not applicable to the open world.
To address this, we propose a novel open-world VAD paradigm with variable definitions, allowing guided detection through user-provided natural language at inference time. 
This paradigm necessitates establishing a robust mapping from video and textual definition to anomaly scores.
Therefore, we propose LaGoVAD (Language-guided Open-world Video Anomaly Detector), a model that dynamically adapts anomaly definitions with two regularization strategies: diversifying the relative durations of anomalies via dynamic video synthesis, and enhancing feature robustness through contrastive learning with negative mining.
Training such adaptable models requires diverse anomaly definitions, but existing datasets typically provide labels without semantic descriptions.
To bridge this gap, we collect PreVAD (Pre-training Video Anomaly Dataset), the largest and most diverse video anomaly dataset to date, featuring 35,279 annotated videos with multi-level category labels and descriptions that explicitly define anomalies.
Zero-shot experiments on seven datasets demonstrate SOTA performance.

## TODO
We are continuously building this repo!

- [x] Release PreVAD annotations and features
- [ ] Release PreVAD raw videos
- [x] Release PreVAD Data Preparation Toolkit
- [ ] Release LaGoVAD code
- [ ] Release LaGoVAD weights

## Dataset: PreVAD

The dataset is available at https://www.modelscope.cn/datasets/Kamino/PreVAD

If you are interested in the code we used to build this dataset, please refer to the `data_download` folder.

## Model: LaGoVAD

Coming soon...
