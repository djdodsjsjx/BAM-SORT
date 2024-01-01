# BAM-SORT

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![test](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)


### Abstract
Multiple Object Tracking (MOT) aims to estimate object bounding boxes and identities in videos. Most tracking methods jointly detect and Kalman filter (KF), using pairwise IoU distance features to match previous trajectories with current detections. Those scenes with congestion and frequent occlusions usually lead to problems of trajectory ID switching and trajectory breakage. To solve this problem, this paper proposes a simple, effective and identical correlation method. First, the bottom border cost matrix (BEC) is introduced to use depth information to improve data association and increase robustness in occlusion situations. Secondly, the activation trajectory matching strategy is adopted to reduce the interference of noise and instantaneous targets on tracking. Finally, the number of trajectory state switching times and the state stability threshold are introduced to effectively reduce the frequency of erroneous trajectories and trajectory breaks caused by false high-scoring detection frames by dynamically maintaining and deleting trajectories. These innovations have achieved excellent performance in various benchmark tests, including MOT17, MOT20, KITTI, especially on DanceTrack where interaction and occlusion phenomena are frequent and severe. Code and models are available at https://github.com/djdodsjsjx/BAM-SORT/.

### Highlights

- BAM-SORT is a **SOTA** heuristic trackers on DanceTrack and performs excellently on MOT17/MOT20 datasets.
- Maintains **Simple, Online and Real-Time (SORT)** characteristics.
- **Training-free** and **plug-and-play** manner.
- **Strong generalization** for diverse trackers and scenarios

### Pipeline
<center>
<img src="assets/Pipeline.jpg" width="1000"/>
</center>



## News

## Tracking performance

### Results on DanceTrack test set

| Tracker          | HOTA | MOTA | IDF1 |
| :--------------- | :--: | :--: | :--: |
| OC-SORT          | 54.6 | 89.6 | 54.6 |
| BAM-SORT         | 64.3 | 91.3 | 68.7 | 

### Results on MOT17 challenge test set

| Tracker          | HOTA | MOTA | IDF1 |
| :--------------- | :--: | :--: | :--: |
| OC-SORT          | 63.2 | 78.0 | 77.5 |
| BAM-SORT         | 64.5 | 80.5 | 79.9 |

### Results on MOT20 challenge test set

| Tracker          | HOTA | MOTA | IDF1 |
| :--------------- | :--: | :--: | :--: |
| OC-SORT          | 62.1 | 75.5 | 75.9 |
| BAM-SORT         | 62.0 | 74.0 | 76.2 |



## Get Started
* See [INSTALL.md](./docs/INSTALL.md) for instructions of installing required components.

* See [GET_STARTED.md](./docs/GET_STARTED.md) for how to get started with BAM-SORT.

* See [MODEL_ZOO.md](./docs/MODEL_ZOO.md) for available YOLOX weights.

<!-- * See [DEPLOY.md](./docs/DEPLOY.md) for deployment support over ONNX, TensorRT and ncnn. -->


## Demo
```shell
python tools/bamsort_demo.py --path dancetrack0052 --det_type yolox_x --dataset dancetrack --dataset_type train
```

<center>
<img src="assets/dancetrack0052.gif" width="600"/>
</center>



## Citation

If you find this work useful, please consider to cite our paper:
```

```
Also see OC-SORT, which we base our work upon:
```
@article{cao2022observation,
  title={Observation-centric sort: Rethinking sort for robust multi-object tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```
## Acknowledgement
A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [OC-SORT](https://github.com/noahcao/OC_SORT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [HybridSORT](https://github.com/ymzis69/HybridSORT), [BoT-SORT](https://github.com/NirAharon/BOT-SORT), [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT) and [FastReID](https://github.com/JDAI-CV/fast-reid). Many thanks for their wonderful works.

