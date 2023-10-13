'''
    This script makes tracking over the results of existing
    tracking algorithms. Namely, we run OC-SORT over theirdetections.
    Output in such a way is not strictly accurate because
    building tracks from existing tracking results causes loss
    of detections (usually initializing tracks requires a few
    continuous observations which are not recorded in the output
    tracking results by other methods). But this quick adaptation
    can provide a rough idea about OC-SORT's performance on
    more datasets. For more strict study, we encourage to implement 
    a specific detector on the target dataset and then run OC-SORT 
    over the raw detection results.
    NOTE: this script is not for the reported tracking with public
    detection on MOT17/MOT20 which requires the detection filtering
    following previous practice. See an example from centertrack for
    example: https://github.com/xingyizhou/CenterTrack/blob/d3d52145b71cb9797da2bfb78f0f1e88b286c871/src/lib/utils/tracker.py#L83
'''

from loguru import logger
import time

import sys
sys.path.insert(0, 'D:/Code/python/DeepLearning/track/BAM-SORT/')
# from trackers.myocsort_tracker.ocsort3_reid import OCSort
from trackers.bamsort_tracker.bamsort_bec_atm_std_reid import OCSort
# from trackers.dcocsort_tracker.ocsort import OCSort
from utils.utils import write_results, write_results_no_score, write_det_results
from utils.args import make_parser
from tools.mota import eval
import os
import cv2
import motmetrics as mm
import numpy as np
from pathlib import Path

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def get_all_files_in_directory(directory):
    file_list = []
    
    # 使用 os.walk() 遍历指定文件夹及其子文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            # # 使用 os.path.join() 构建完整的文件路径
            # file_path = os.path.join(root, file)
            # # 将文件路径添加到列表中
            file_list.append(file)
    
    return file_list

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(args):
    results_folder = args.out_path
    results_folder = str(increment_path(results_folder, exist_ok=False))  # 对已有的文件进行评估，需要注释
    os.makedirs(results_folder, exist_ok=True)

    # results_folder_tracker_num = args.tn_out_path
    # results_folder_tracker_num = str(increment_path(results_folder_tracker_num, exist_ok=False))  # 对已有的文件进行评估，需要注释
    # os.makedirs(results_folder_tracker_num, exist_ok=True)

    raw_path = "{}/reid/{}/{}/{}".format(args.raw_results_path, args.dataset, args.det_type, args.dataset_type)  # 检测路径
    dataset = args.dataset
    if args.dataset == "dancetrack":
        pic_raw_path = "datasets/{}/{}".format(args.dataset, args.dataset_type)
    else:
        pic_raw_path = "datasets/{}/{}".format(args.dataset, "test" if args.dataset_type == "test" else "train")
    total_time = 0
    total_frame = 0

    test_seqs = get_all_files_in_directory(raw_path)

    for seq_name in test_seqs:
        print("starting seq {}".format(seq_name))
        tracker = OCSort(args=args, det_thresh = args.track_thresh, iou_threshold=args.iou_thresh, asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia, use_byte=args.use_byte, min_hits=args.min_hits)

        # tracker = OCSort(det_thresh = args.track_thresh, iou_threshold=args.iou_thresh,
        #     asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia)

        results_filename = os.path.join(results_folder, seq_name)
        
        seq_file = os.path.join(raw_path, seq_name)
        seq_trks = np.loadtxt(seq_file, delimiter=',')
        min_frame = seq_trks[:,0].min()
        max_frame = seq_trks[:,0].max()
        results = []
        # results_tracker_num = []
        for frame_ind in range(int(min_frame), int(max_frame)+1):
            # print("{}:{}/{}".format(seq_name, frame_ind, max_frame))
            dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2:6]
            scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6]
            det_embs = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,10:]
            cur_dets = np.concatenate((dets, scores.reshape(-1, 1)), axis=1)
            # if args.dataset == "dancetrack":
            #     cur_img = cv2.imread(os.path.join(pic_seq_path, '{:08d}.jpg'.format(frame_ind + start_img_idx)))
            # else:
            #     cur_img = cv2.imread(os.path.join(pic_seq_path, '{:06d}.jpg'.format(frame_ind + start_img_idx)))
            t0 = time.time()

            # online_targets = tracker.update(cur_img, cur_dets)
            online_targets = tracker.update(cur_dets, det_embs)

            t1 = time.time()
            total_frame += 1
            total_time += t1 - t0

            online_tlwhs = []
            online_ids = []
            # online_scores = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                # score = t[5]
                # tlwh = t.tlwh  # SparseTracker
                # tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    # online_scores.append(score)
            # save results
            results.append((frame_ind, online_tlwhs, online_ids))  # 每一帧跟踪器信息: fid, x, y, w, h, tid
            # results_tracker_num.append(tracker.save_info(cur_dets))

        write_results_no_score(results_filename, results)  # 将results写入到result_filename, fid, tid, x, y, w, h
        # np.savetxt(results_filename_tracker_num, results_tracker_num, fmt="%d %d %d")
    if args.dataset == "test":
        return 

    # if args.dataset == "dancetrack":
    #     eval(results_folder, "datasets/{}/{}".format(args.dataset, args.dataset_type), args.gt_type) 
    # else:
    #     eval(results_folder, "datasets/{}/train".format(args.dataset), args.gt_type)  # "" | "_val_half" | "_train_half"


    if args.dataset == "dancetrack":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK dancetrack --SPLIT_TO_EVAL val --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity --TIME_PROGRESS False --TRACKER_SUB_FOLDER '' --GT_FOLDER datasets/dancetrack/ --USE_PARALLEL False --NUM_PARALLEL_CORES 8 --TRACKERS_FOLDER evaldata/trackers/DanceTrack/improve/val/baseline+bec+act --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       f"--BENCHMARK dancetrack " \
                       f"--SPLIT_TO_EVAL {args.dataset_type}  " \
                       "--METRICS HOTA CLEAR Identity " \
                       "--TRACKERS_TO_EVAL '' " \
                       "--TIME_PROGRESS False " \
                       "--TRACKER_SUB_FOLDER ''  " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 8 " \
                       "--GT_FOLDER datasets/dancetrack/ " \
                       "--TRACKERS_FOLDER " + results_folder + " "\
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
    elif args.dataset == "MOT17":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity VACE --TIME_PROGRESS False --GT_FOLDER datasets/MOT17/ --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --TRACKERS_FOLDER evaldata/trackers/MOT17/improve/val-half/baseline+bec+act+new --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT17 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL '' " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TIME_PROGRESS False " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/MOT17/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
                    #    "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(args.dataset_type)
    elif args.dataset == "MOT20":
        # python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT20 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL '' --METRICS HOTA CLEAR Identity VACE --TIME_PROGRESS False --GT_FOLDER datasets/MOT20/ --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --TRACKERS_FOLDER evaldata/trackers/MOT20/improve/val-half/baseline+bec+act --GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_val_half.txt
        hota_command = "python TrackEval/scripts/run_mot_challenge.py " \
                       "--BENCHMARK MOT20 " \
                       "--SPLIT_TO_EVAL train " \
                       "--TRACKERS_TO_EVAL '' " \
                       "--METRICS HOTA CLEAR Identity VACE " \
                       "--TIME_PROGRESS False " \
                       "--USE_PARALLEL False " \
                       "--NUM_PARALLEL_CORES 1  " \
                       "--GT_FOLDER datasets/MOT20/ " \
                       "--TRACKERS_FOLDER " + results_folder + " " \
                       "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt.txt"
                    #    "--GT_LOC_FORMAT {gt_folder}/{seq}/gt/gt_" + "{}_half.txt".format(args.dataset_type)
    else:
        assert args.dataset in ["dancetrack", "MOT17"]
    os.system(hota_command)

    logger.info('Completed')
    # print("Running over {} frames takes {}s. FPS={}".format(total_frame, total_time, total_frame / total_time))
    return 


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    # min_hits = [2, 3, 4, 5, 6]
    # for min_hit in min_hits:      
    #     filename = "{}_min_hit".format(min_hit)
    #     tmp_out_path, tmp_tn_out_path = args.out_path, args.tn_out_path
    #     args.out_path = os.path.join(args.out_path, filename)
    #     args.tn_out_path = os.path.join(args.out_path, filename)
    #     args.min_hits = min_hit
    #     main(args)
    #     args.out_path, args.tn_out_path = tmp_out_path, tmp_tn_out_path

