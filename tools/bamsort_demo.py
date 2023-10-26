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
# from trackers.bamsort_tracker.bamsort_bec import OCSort
from trackers.bamsort_tracker.bamsort_bec_atm_std import OCSort
from utils.utils import write_results, write_results_no_score, write_det_results
from yolox.utils import setup_logger
from utils.args import make_parser
from tools.mota import eval, eval_hota
import os
import motmetrics as mm
import numpy as np
from pathlib import Path
from yolox.utils.visualize import plot_tracking
from trackers.tracking_utils.timer import Timer
import cv2
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
    results_folder = os.path.join(args.demo_out, args.dataset, args.dataset_type)
    os.makedirs(results_folder, exist_ok=True)

    if args.dataset == "dancetrack":
        pic_raw_path = "datasets/{}/{}".format(args.dataset, args.dataset_type)
    else:
        pic_raw_path = "datasets/{}/{}".format(args.dataset, "test" if args.dataset_type == "test" else "train")

    raw_path = "{}/{}/{}/{}".format(args.raw_results_path, args.dataset, args.det_type, args.dataset_type)  # 检测路径
    dataset = args.dataset

    test_seqs = [f"{args.path}"]

    for seq_name in test_seqs:
        tracker = OCSort(args=args, det_thresh = args.track_thresh, iou_threshold=args.iou_thresh, asso_func=args.asso, delta_t=args.deltat, inertia=args.inertia, use_byte=args.use_byte, min_hits=args.min_hits)
        start_img_idx = 0
        pic_seq_path = os.path.join(pic_raw_path, seq_name, "img1")
        if args.dataset != "dancetrack" and args.dataset_type == "val":
            images = os.listdir(pic_seq_path)
            num_images = len([image for image in images if 'jpg' in image])
            start_img_idx = num_images // 2 + 1
        seq_file = "{}/{}.txt".format(raw_path, seq_name)
        seq_trks = np.loadtxt(seq_file, delimiter=',')
        min_frame = seq_trks[:,0].min()
        max_frame = seq_trks[:,0].max()
        tmp_pic = None
        if args.dataset == "dancetrack":
            tmp_pic = cv2.imread(os.path.join(pic_seq_path, '{:08d}.jpg'.format(1)))
        else:
            tmp_pic = cv2.imread(os.path.join(pic_seq_path, '{:06d}.jpg'.format(1)))
        height, width = tmp_pic.shape[:2]
        vid_writer = cv2.VideoWriter(
            f"{results_folder}/{seq_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (int(width), int(height))
        )
        timer_track = Timer()
        for frame_ind in range(int(min_frame), int(max_frame)+1):
            dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2:6]
            scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6]
            cur_dets = np.concatenate((dets, scores.reshape(-1, 1)), axis=1)

            if args.dataset == "dancetrack":
                cur_img = cv2.imread(os.path.join(pic_seq_path, '{:08d}.jpg'.format(frame_ind + start_img_idx)))
            else:
                cur_img = cv2.imread(os.path.join(pic_seq_path, '{:06d}.jpg'.format(frame_ind + start_img_idx)))
            
            timer_track.tic()
            online_targets = tracker.update(cur_dets)
            timer_track.toc()

            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            online_im = plot_tracking(
                cur_img, online_tlwhs, online_ids, frame_id=frame_ind + 1, fps=1. / timer_track.average_time
            )
            vid_writer.write(online_im)
            cv2.imshow("frame", online_im)
        logger.info(f"save results to {results_folder}/{seq_name}.mp4")
    return 


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
