"""
    This script is to draw trajectory prediction as in Fig.6 of the paper
"""

import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.insert(0, 'D:/Code/python/DeepLearning/track/BAM-SORT/')
import numpy as np 
import os
from pathlib import Path
import math
from tools import gif



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

def get_tracker_num(improve2_file, improve3_file, gt_file):
    improve2_seq_trks = np.loadtxt(improve2_file, delimiter=" ")
    improve3_seq_trks = np.loadtxt(improve3_file, delimiter=" ")
    gt_seq_trks = np.loadtxt(gt_file, delimiter=",")
    min_frame = gt_seq_trks[:,0].min()
    max_frame = gt_seq_trks[:,0].max()
    improve2_tracker_num, improve3_tracker_num, gt_tracker_num, frame_ids = [], [], [], []
    for frame_id in range(int(min_frame), int(max_frame)+1, 1):

        trks = gt_seq_trks[np.logical_and(gt_seq_trks[:,0]==frame_id, gt_seq_trks[:,-1]>0.3)]
        improve2_tracker_num.append(improve2_seq_trks[np.where(improve2_seq_trks[:, 0]==frame_id)][0][2])
        improve3_tracker_num.append(improve3_seq_trks[np.where(improve3_seq_trks[:, 0]==frame_id)][0][2])
        gt_tracker_num.append(len(trks))
        frame_ids.append(frame_id)
    
    return frame_ids, improve2_tracker_num, improve3_tracker_num, gt_tracker_num



def plot_multi_tracker_num(improve2_dir_in, improve3_dir_in, gt_dir_in, gt_type, output):
    
    seqs = get_all_files_in_directory(improve2_dir_in)
    for seq_name in seqs:
        improve2_file_path = os.path.join(improve2_dir_in, seq_name)
        improve3_file_path = os.path.join(improve3_dir_in, seq_name)
        gt_file_path = os.path.join(gt_dir_in, seq_name[:-4], "gt", gt_type)
        save_path = os.path.join(output, "{}.png".format(seq_name[:-4]))
        
        frame_ids, improve2_tracker_num, improve3_tracker_num, gt_tracker_num = get_tracker_num(improve2_file_path, improve3_file_path, gt_file_path)

        fig = plt.figure()

        # 使用图形对象创建坐标轴对象
        ax = fig.add_subplot(111)  # 111表示1行1列的第1个子图，也可以根据需要调整子图的布局

        # 创建折线图
        ax.plot(frame_ids, gt_tracker_num, label='gt', color='blue')
        ax.plot(frame_ids, improve2_tracker_num, label='BEC+ATM', color='green')
        ax.plot(frame_ids, improve3_tracker_num, label='BEC+ATM+STD', color='darkgoldenrod')
        # 添加标题和标签
        ax.set_title(seq_name[:-4])
        ax.set_xlabel('frame')
        ax.set_ylabel('counts')
        ax.legend()
        ax.figure.savefig(save_path)
        # # 显示图像
        # plt.show()


if __name__ == "__main__":

    train_half_path = "exps/MOT20/yolox_x/train"
    val_half_path = "exps/MOT20/yolox_x/val"
    out_half_path = "exps/MOT20/yolox_x/all"
    os.makedirs(out_half_path, exist_ok=True)
    seqs = get_all_files_in_directory(train_half_path)
    for seq in seqs:
        train_half_filename = os.path.join(train_half_path, seq)
        val_half_filename = os.path.join(val_half_path, seq)
        out_filename = os.path.join(out_half_path, seq)
        train_half_trks = np.loadtxt(train_half_filename, delimiter=',')
        val_half_trks = np.loadtxt(val_half_filename, delimiter=',')
        max_frame = train_half_trks[:,0].max()
        val_half_trks[:,0] += max_frame
        all_trks = np.vstack((train_half_trks, val_half_trks))
        np.savetxt(out_filename, all_trks, fmt="%d,%d,%f,%f,%f,%f,%f,%d,%d,%d")

        # seq_trks = np.empty((0, 10))
        # seq_file = os.path.join(raw_path, seq_name)
        # seq_file = open(seq_file)
        # results_filename = os.path.join(results_folder, seq_name)
        # results_filename_tracker_num = os.path.join(results_folder_tracker_num, seq_name)
        # lines = seq_file.readlines()
        # line_count = 0 
        # for line in lines:
        #     # print("{}/{}".format(line_count,len(lines)))
        #     line_count+=1
        #     line = line.strip()
        #     tmps = line.strip().split(",")
        #     trk = np.array([float(d) for d in tmps])
        #     trk = np.expand_dims(trk, axis=0)
        #     seq_trks = np.concatenate([seq_trks, trk], axis=0)
        