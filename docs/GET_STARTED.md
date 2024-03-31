# Get Started 
**Our data structure is the same as [OC-SORT](https://github.com/noahcao/OC_SORT).** 

## Data preparation
### Datasets
Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [DanceTrack](https://github.com/DanceTrack/DanceTrack) and put them under <OCSORT_HOME>/datasets in the following structure:
```
datasets
|——————MOT17
|        └——————train
|        └——————test
└——————crowdhuman
|        └——————Crowdhuman_train
|        └——————Crowdhuman_val
|        └——————annotation_train.odgt
|        └——————annotation_val.odgt
└——————MOT20
|        └——————train
|        └——————test
└——————Cityscapes
|        └——————images
|        └——————labels_with_ids
└——————ETHZ
|        └——————eth01
|        └——————...
|        └——————eth07
└——————dancetrack        
        └——————train
        └——————val
        └——————test
```
### 2. YOLOX-Detection-Output
Download [MOT17](https://drive.google.com/drive/folders/18c4Zj95PQu6KBsrY-I1ub0KpY-z1ZVAZ?usp=sharing), [MOT20](https://drive.google.com/drive/folders/18c4Zj95PQu6KBsrY-I1ub0KpY-z1ZVAZ?usp=sharing), [DanceTrack](https://drive.google.com/drive/folders/18c4Zj95PQu6KBsrY-I1ub0KpY-z1ZVAZ?usp=sharing) preprocessing detection frame file under detector YOLOX and put them under <BAM_HOME>/exps in the following structure:
```
exps
|——————dancetrack
|        └——————yolox_x
|           └——————val
|           └——————test
|——————MOT17
|        └——————yolox_x
|           └——————test
|        └——————ablation
|           └——————val
|——————MOT20
|        └——————yolox_x
|           └——————test
```

## Evaluation
* **on DanceTrack Val set**
    ```shell
    python tools/run_bamsort.py --det_type yolox_x --dataset dancetrack --dataset_type val --w_bec 0.3 --bec_num 4 --min_hits 7 --std_time_since_update 5 --std_switch_cnt 1 --std_max_hits 50 --fp16 --fuse --expn $exp_name 
    ```
    We follow the [TrackEval protocol](https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval) for evaluation on the officially released validation set. This gives HOTA = 55.5 ~ 55.9.

* **on DanceTrack Test set**
    ```shell
    python tools/run_bamsort.py --det_type yolox_x --dataset dancetrack --dataset_type test --w_bec 0.3 --bec_num 4 --min_hits 5 --std_time_since_update 5 --std_switch_cnt 1 --std_max_hits 50 --expn $exp_name 
    ```
    Submit the outputs to [the DanceTrack evaluation site](https://competitions.codalab.org/competitions/35786). This gives HOTA = 59.9 ~ 60.5.

* **on MOT17 half val**
    ```shell
    python tools/run_bamsort.py --det_type ablation --dataset MOT17 --dataset_type val --w_bec 0.4 --bec_num 4 --min_hits 1 --sort_with_std False --expn $exp_name 
    ```
    We follow the [TrackEval protocol](https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval) for evaluation on the self-splitted validation set. This gives you HOTA = 67.3 ~ 67.7.

* **on MOT17/MOT20 Test set**
    ```shell
    # MOT17
    python tools/run_bamsort.py --det_type yolox_x --dataset MOT17 --dataset_type test --w_bec 0.4 --bec_num 4 --min_hits 1 --sort_with_std False --expn $exp_name 

    # MOT20
    python tools/run_bamsort.py --det_type yolox_x --dataset MOT20 --dataset_type test --w_bec 0.3 --bec_num 6 --min_hits 1 --sort_with_std False --expn $exp_name 
    ```
    Submit the zipped output files to [MOTChallenge](https://motchallenge.net/) system. Following [the adaptive detection thresholds](https://github.com/ifzhang/ByteTrack/blob/d742a3321c14a7412f024f2218142c7441c1b699/yolox/evaluators/mot_evaluator.py#L139) by ByteTrack can further boost the performance. After interpolation (see below), this gives you HOTA = ~63.5 on MOT17 and HOTA = \~61.9 on MOT20.


*Note: We find the current implementation may show some randomness in different running trials. We are still inspecting this.*

## [Optional] Interpolation
I have utilized the interpolation features from the OC-SORT algorithm to post-process existing tracking results. The OC-SORT algorithm offers two interpolation methods: linear interpolation and Gaussian Process Regression interpolation.
```shell
    # optional offline post-processing
    python tools/interpolation.py $result_path $save_path
```
The OC-SORT algorithm also provides an attempt to use Gaussian Process Regression interpolation, which operates on existing linear interpolation results. To use Gaussian Process Regression interpolation, you can run the following command:
```shell
    python tools/gp_interpolation.py $raw_results_path $linear_interp_path $save_path
```
*Note: for the results in our paper on MOT17/MOT20 private settings and HeadTrack, we use linear interpolation by default.*

