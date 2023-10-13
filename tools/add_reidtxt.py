from loguru import logger
import time

import sys
sys.path.insert(0, 'D:/Code/python/DeepLearning/track/OC_SORT/')

from external.fast_reid.fast_reid_interfece import FastReIDInterface
from utils.utils import write_results, write_results_no_score, write_det_reid_results
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

    # results_folder_tracker_num = args.tn_out_path
    # results_folder_tracker_num = str(increment_path(results_folder_tracker_num, exist_ok=False))  # 对已有的文件进行评估，需要注释
    # os.makedirs(results_folder_tracker_num, exist_ok=True)
    results_folder = "{}/reid/{}/{}/{}".format(args.raw_results_path, args.dataset, args.det_type, args.dataset_type)
    # results_folder = str(increment_path(results_folder, exist_ok=False))  # 对已有的文件进行评估，需要注释
    os.makedirs(results_folder, exist_ok=True)
    
    raw_path = "{}/{}/{}/{}".format(args.raw_results_path, args.dataset, args.det_type, args.dataset_type)  # 检测路径
    dataset = args.dataset
    if args.dataset == "dancetrack":
        pic_raw_path = "datasets/{}/{}".format(args.dataset, args.dataset_type)
    else:
        pic_raw_path = "datasets/{}/{}".format(args.dataset, "test" if args.dataset_type == "test" else "train")
    total_time = 0
    total_frame = 0

    test_seqs = get_all_files_in_directory(raw_path)
    embedder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, 'cuda')
    for seq_name in test_seqs:
        print("starting seq {}".format(seq_name))
        results_filename = os.path.join(results_folder, seq_name)
        pic_seq_path = os.path.join(pic_raw_path, seq_name[:-4], "img1")
        img_cnt = len([f for f in os.listdir(pic_seq_path) if os.path.isfile(os.path.join(pic_seq_path, f))])
        start_img_idx = 0
        if (args.dataset == "MOT20" or args.dataset == "MOT17") and args.dataset_type == "val":
            start_img_idx = img_cnt // 2

        seq_file = os.path.join(raw_path, seq_name)
        seq_trks = np.loadtxt(seq_file, delimiter=',')
        min_frame = seq_trks[:,0].min()
        max_frame = seq_trks[:,0].max()
        det_results = []
        for frame_ind in range(int(min_frame), int(max_frame)+1):
            print("{}:{}/{}".format(seq_name, frame_ind, max_frame))
            dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2:6]
            scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6]
            cur_dets = np.concatenate((dets, scores.reshape(-1, 1)), axis=1)
            
            if args.dataset == "dancetrack":
                cur_img = cv2.imread(os.path.join(pic_seq_path, '{:08d}.jpg'.format(frame_ind + start_img_idx)))
            else:
                cur_img = cv2.imread(os.path.join(pic_seq_path, '{:06d}.jpg'.format(frame_ind + start_img_idx)))

            dets_embs = np.ones((dets.shape[0], 1))
            if dets.shape[0] != 0:
                dets_embs = embedder.inference(cur_img, dets[:, :4])

            dets_reid = np.hstack((cur_dets, dets_embs))

            det_x1y1x2y2 = []
            det_scores = []
            det_embs =  []
            for t in dets_reid:
                det_x1y1x2y2.append([t[0], t[1], t[2], t[3]])
                det_scores.append(t[4])
                det_embs.append(list(t[5:]))
            # save results
            det_results.append((frame_ind, det_x1y1x2y2, det_scores, det_embs))  # 每一帧检测信息: fid, x1, y1, x2, y2, s, embs
            if frame_ind % 500 == 0:
                write_det_reid_results(results_filename, det_results)
                det_results = []
        write_det_reid_results(results_filename, det_results)

    return 


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
