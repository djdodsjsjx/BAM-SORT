# Get Started 
**Our data structure is the same as [OC-SORT](https://github.com/noahcao/OC_SORT).** 

## Data preparation
### 1. Datasets
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
### 3. YOLOX and ReID Output
Please download the per-frame detection results and corresponding ReID feature vectors for each bounding box of the [MOT17](https://pan.baidu.com/s/1vk1EfO-fqxhu81NombQzRg?pwd=obws)、[MOT20](https://pan.baidu.com/s/1vk1EfO-fqxhu81NombQzRg?pwd=obws)、[DanceTrack](https://pan.baidu.com/s/1vk1EfO-fqxhu81NombQzRg?pwd=obws), processed with the YOLOX detector. Place the files in the <BAM_HOME>/exps directory, maintaining the following structure:
```
exps
|——————reid
|        └——————dancetrack
|           └——————yolox_x
|               └——————test
|        └——————MOT17
|           └——————yolox_x
|               └——————test
|        └——————MOT20
|           └——————yolox_x
|               └——————test
```

## Evaluation
### online
* **on DanceTrack Val set**
    ```shell
    python tools/run_bamsort.py --det_type yolox_x --dataset dancetrack --dataset_type val --w_bec 0.3 --bec_num 4 --min_hits 7 --std_time_since_update 5 --std_switch_cnt 1 --std_max_hits 50 --fp16 --fuse --expn $exp_name 
    ```
    We follow the [TrackEval protocol](https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval) for evaluation on the officially released validation set. This gives HOTA = 56.5 ~ 56.9.

* **on DanceTrack Test set**
    ```shell
    python tools/run_bamsort.py --det_type yolox_x --dataset dancetrack --dataset_type test --w_bec 0.3 --bec_num 4 --min_hits 5 --std_time_since_update 5 --std_switch_cnt 1 --std_max_hits 50 --expn $exp_name 
    ```
* **on DanceTrack Test set with ReID**
    ```shell
    python tools/run_bamsort_reid.py --sort_with_reid --det_type yolox_x --dataset dancetrack --dataset_type test --w_bec 0.3 --bec_num 4 --min_hits 5 --std_time_since_update 5 --std_switch_cnt 1 --std_max_hits 50 --expn $exp_name 
    ```

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
    Following [the adaptive detection thresholds](https://github.com/ifzhang/ByteTrack/blob/d742a3321c14a7412f024f2218142c7441c1b699/yolox/evaluators/mot_evaluator.py#L139) by ByteTrack can further boost the performance.

* **on MOT17/MOT20 Test set with ReID**
    ```shell
    # MOT17
    python tools/run_bamsort_reid.py --sort_with_reid --det_type yolox_x --dataset MOT17 --dataset_type test --w_bec 0.4 --bec_num 4 --min_hits 1 --sort_with_std False --expn $exp_name 

    # MOT20
    python tools/run_bamsort_reid.py --sort_with_reid --det_type yolox_x --dataset MOT20 --dataset_type test --w_bec 0.3 --bec_num 6 --min_hits 1 --sort_with_std False --expn $exp_name 
    ```
    Following [the adaptive detection thresholds](https://github.com/ifzhang/ByteTrack/blob/d742a3321c14a7412f024f2218142c7441c1b699/yolox/evaluators/mot_evaluator.py#L139) by ByteTrack can further boost the performance.


*Note: We find the current implementation may show some randomness in different running trials. We are still inspecting this.*

### offline Interpolation
I have utilized the interpolation features from the OC-SORT algorithm to post-process existing tracking results. The OC-SORT algorithm offers two interpolation methods: linear interpolation and Gaussian Process Regression interpolation. To use linear interpolation on the results output by BAM-SORT online tracking, you can run the following command:
```shell
    # optional offline post-processing
    python tools/interpolation.py $result_path $save_path
```
The OC-SORT algorithm also provides an attempt to use Gaussian Process Regression interpolation, which operates on existing linear interpolation results. To use Gaussian Process Regression interpolation, you can run the following command:
```shell
    python tools/gp_interpolation.py $raw_results_path $linear_interp_path $save_path
```

For the benchmark results of DanceTrack, please submit your output files to [the DanceTrack evaluation site](https://competitions.codalab.org/competitions/35786). Following submission, you can expect a HOTA score ranging from 64.0 to 64.5. For the benchmark tests of MOT17 and MOT20, outputs should be submitted to [MOTChallenge](https://motchallenge.net/). In this case, the expected HOTA score for MOT17 is approximately 64.5, while for MOT20 it is around 62.0.
