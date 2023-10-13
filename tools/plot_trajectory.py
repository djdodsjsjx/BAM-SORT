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

def plot_traj(traj_file, name):
    trajs = np.loadtxt(traj_file, delimiter=",")
    track_ids = np.unique(trajs[:,1])  # 轨迹数
    for tid in track_ids:
        traj = trajs[np.where(trajs[:,1]==tid)]  # 获取当前轨迹  fid, tid, x, y, w, h
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # 轨迹图初始化
        frames = traj[:100, 0]  # 当前轨迹的所有帧数
        boxes = traj[:100, 2:6]  # 当前轨迹的所有box
        boxes_x = boxes[:,0]
        boxes_y = boxes[:,1]
        plt.plot(boxes_x, boxes_y, "ro")
        box_num = boxes_x.shape[0]
        for bind in range(0, box_num-1):
            frame_l = frames[bind]
            frame_r = frames[bind+1]
            box_l = boxes[bind]
            box_r = boxes[bind+1]
            if frame_r == frame_l + 1:
                l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="red")
                ax.add_line(l)
            else:
                l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="gray")
                ax.add_line(l)
        plt.savefig("traj_plots/{}/{}.png".format(name, int(tid)))

def plot_traj_marge3(traj_file, bam_result_src, oc_result_src, name, file_name):  # gt, res
    bam_has_tid = set()  # 已经被选过
    oc_has_tid = set()
    def get_simmlar(traj_a, traj_bs):
        tx, ty, tw, th = traj_a[2:6]
        mi, res = 100000000, -1
        for fid, tid, x, y, w, h in traj_bs[:,0:6]:
            diff = abs(x - tx) + abs(y - ty) + abs(w - tw) + abs(h - th)
            if diff < 100 and diff < mi:
                res = tid
                mi = diff
        return res

    bam_res_trajs = np.loadtxt(bam_result_src, delimiter=",")
    oc_res_trajs = np.loadtxt(oc_result_src, delimiter=",")
    trajs = np.loadtxt(traj_file, delimiter=",")
    track_ids = np.unique(trajs[:,1])  # 所有gt轨迹id
    for tid in track_ids:
        traj = trajs[np.where(trajs[:,1]==tid)]  # OCSORT轨迹  fid, tid, x, y, w, h
        bam_res_trajs_frame1 = bam_res_trajs[np.where(bam_res_trajs[:,0]==traj[0,0])]  # 获取tid轨迹的起始帧
        bam_res_id = get_simmlar(traj[0], bam_res_trajs_frame1)
        
        oc_res_trajs_frame1 = oc_res_trajs[np.where(oc_res_trajs[:,0]==traj[0,0])]
        oc_res_id = get_simmlar(traj[0], oc_res_trajs_frame1)
        
        if bam_res_id == -1 or bam_res_id in bam_has_tid:
            continue
        if oc_res_id == -1 or oc_res_id in oc_has_tid:
            continue

        bam_has_tid.add(bam_res_id)
        oc_has_tid.add(oc_res_id)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # 轨迹图初始化
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 1200)
        
        step, div = 1, 2
        max_frame = int(traj[:,0].max()) // div
        frames = traj[:max_frame, 0]
        boxes = traj[:max_frame, 2:6] 
        boxes_x = boxes[:,0]
        boxes_y = boxes[:,1]
        plt.plot(boxes_x, boxes_y, label='GT', color=(0, 0, 0))
        box_num = boxes_x.shape[0]
        for bind in range(0, box_num-step, step):
            frame_l = frames[bind]
            frame_r = frames[bind+step]
            box_l = boxes[bind]
            box_r = boxes[bind+step]
            l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color=(0, 0, 0))
            ax.add_line(l)

        traj = oc_res_trajs[np.where(oc_res_trajs[:, 1]==oc_res_id)]  # GT轨迹
        max_frame = int(traj[:,0].max()) // div
        frames = traj[:max_frame, 0]
        boxes = traj[:max_frame, 2:6]
        boxes_x = boxes[:,0]
        boxes_y = boxes[:,1]
        plt.plot(boxes_x, boxes_y, label='OCSORT', color=(1, 0, 0))
        box_num = boxes_x.shape[0]
        for bind in range(0, box_num-step, step):
            frame_l = frames[bind]
            frame_r = frames[bind+step]
            box_l = boxes[bind]
            box_r = boxes[bind+step]
            l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color=(1, 0, 0))
            ax.add_line(l)

        traj = bam_res_trajs[np.where(bam_res_trajs[:, 1]==bam_res_id)]  # GT轨迹
        max_frame = int(traj[:,0].max()) // div
        frames = traj[:max_frame, 0]
        boxes = traj[:max_frame, 2:6]
        boxes_x = boxes[:,0]
        boxes_y = boxes[:,1]
        plt.plot(boxes_x, boxes_y, label='BAMSORT', color=(0, 127/255, 0))
        box_num = boxes_x.shape[0]
        for bind in range(0, box_num-step, step):
            frame_l = frames[bind]
            frame_r = frames[bind+step]
            box_l = boxes[bind]
            box_r = boxes[bind+step]
            l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color=(0, 127/255, 0))
            ax.add_line(l)
        plt.savefig("{}/{}/{}.png".format(file_name, name, int(tid)))


def plot_traj_marge2(traj_file, result_src, name, file_name, color=(0, 0, 0)):  # gt, res
    has_tid = set()  # 已经被选过
    def get_simmlar(traj_a, traj_bs):
        tx, ty, tw, th = traj_a[2:6]
        mi, res = 100000000, -1
        for fid, tid, x, y, w, h in traj_bs[:,0:6]:
            diff = abs(x - tx) + abs(y - ty) + abs(w - tw) + abs(h - th)
            if diff < 100 and diff < mi:
                res = tid
                mi = diff
        return res

    res_trajs = np.loadtxt(result_src, delimiter=",")
    trajs = np.loadtxt(traj_file, delimiter=",")
    track_ids = np.unique(trajs[:,1])  # 所有gt轨迹id
    for tid in track_ids:
        traj = trajs[np.where(trajs[:,1]==tid)]  # OCSORT轨迹  fid, tid, x, y, w, h
        res_trajs_frame1 = res_trajs[np.where(res_trajs[:,0]==traj[0,0])]  # 获取tid轨迹的起始帧
        res_id = get_simmlar(traj[0], res_trajs_frame1)
        
        if res_id == -1 or res_id in has_tid:
            continue
        has_tid.add(res_id)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # 轨迹图初始化
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 1200)
        
        step, div = 1, 2
        max_frame = int(traj[:,0].max()) // div
        frames = traj[:max_frame, 0]
        boxes = traj[:max_frame, 2:6] 
        boxes_x = boxes[:,0]
        boxes_y = boxes[:,1]
        plt.plot(boxes_x, boxes_y, label='GT', color=(0, 0, 0))
        box_num = boxes_x.shape[0]
        for bind in range(0, box_num-step, step):
            frame_l = frames[bind]
            frame_r = frames[bind+step]
            box_l = boxes[bind]
            box_r = boxes[bind+step]
            l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color=(0, 0, 0))
            ax.add_line(l)

        traj = res_trajs[np.where(res_trajs[:, 1]==res_id)]  # GT轨迹
        max_frame = int(traj[:,0].max()) // div
        frames = traj[:max_frame, 0]
        boxes = traj[:max_frame, 2:6]
        boxes_x = boxes[:,0]
        boxes_y = boxes[:,1]
        plt.plot(boxes_x, boxes_y, label='BAMSORT', color=color)
        box_num = boxes_x.shape[0]
        for bind in range(0, box_num-step, step):
            frame_l = frames[bind]
            frame_r = frames[bind+step]
            box_l = boxes[bind]
            box_r = boxes[bind+step]
            l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color=color)
            ax.add_line(l)
        plt.savefig("{}/{}/{}.png".format(file_name, name, int(tid)))

def plot_traj_gif(traj_file, result_src, name, file_name):
    has_tid = set()  # 已经被选过
    def get_simmlar(traj_a, traj_bs):
        tx, ty, tw, th = traj_a[2:6]
        mi, res = 100000000, -1
        for fid, tid, x, y, w, h in traj_bs[:,0:6]:
            diff = abs(x - tx) + abs(y - ty) + abs(w - tw) + abs(h - th)
            if diff < 50 and diff < mi:
                res = tid
                mi = diff
        return res
    
    res_trajs = np.loadtxt(result_src, delimiter=",")
    res_trajs_frame1 = res_trajs[np.where(res_trajs[:,0]==1)]  # 获取第一帧
    trajs = np.loadtxt(traj_file, delimiter=",")
    track_ids = np.unique(trajs[:,1])  # 找出所有不同的轨迹
    for tid in track_ids:
        traj = trajs[np.where(trajs[:,1]==tid)]  # OCSORT轨迹  fid, tid, x, y, w, h
        res_id = get_simmlar(traj[0], res_trajs_frame1)
        if res_id == -1 or res_id in has_tid:
            continue
        has_tid.add(res_id)
        res_traj = res_trajs[np.where(res_trajs[:,1]==res_id)]

        boxes = traj[:100, 2:6] 
        res_boxes = res_traj[:100, 2:6]
        fs = []

        # @gif.frame
        # def plott(bind):  # 没有历史轨迹
        #     fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # 轨迹图初始化
        #     ax.set_xlim(0, 1000)
        #     ax.set_ylim(0, 1000)
        #     box_l = boxes[bind]
        #     box_r = boxes[bind+1]
        #     l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="blue")
        #     ax.add_line(l)
            
        #     res_box_l = res_boxes[bind]
        #     res_box_r = res_boxes[bind+1]
        #     l = matplotlib.lines.Line2D([res_box_l[0], res_box_r[0]], [res_box_l[1], res_box_r[1]], color="red")
        #     ax.add_line(l)

        @gif.frame
        def plott(bind):  # 存在历史轨迹
            fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # 轨迹图初始化
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 1000)
            for i in range(bind+1):
                box_l = boxes[i]
                box_r = boxes[i+1]
                l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="blue")
                ax.add_line(l)

                res_box_l = res_boxes[i]
                res_box_r = res_boxes[i+1]
                l = matplotlib.lines.Line2D([res_box_l[0], res_box_r[0]], [res_box_l[1], res_box_r[1]], color="red")
                ax.add_line(l)
        for bind in range(0, min(boxes.shape[0], res_boxes.shape[0])-1):
            f = plott(bind)
            fs.append(f)
        gif.save(fs, "{}/{}/{}.gif".format(file_name, name, int(tid)), duration=50)

if __name__ == "__main__":
    # name = sys.argv[1]
    # os.makedirs(os.path.join("traj_plots/{}".format(name)), exist_ok=True)
    OUT_SRC = "dancetrack_fic_bamsort_result"
    file_name = str(increment_path("traj_plots/{}".format(OUT_SRC), exist_ok=False))
    os.makedirs(file_name, exist_ok=True)

    gt_src = "datasets/dancetrack/val"
    bam_result_src = "evaldata/trackers/dancetrack/improve/val/baseline+bec+atm+std+reid"
    oc_result_src = "evaldata/trackers/dancetrack/improve/val/baseline"
    # ours = "path/to/pred/output" # preds
    # baseline = "path/to/baseline/output" # baseline outputs
    seqs = os.listdir(gt_src)
    for seq in seqs:
        name = "gt_{}".format(seq)
        os.makedirs(os.path.join(file_name, name), exist_ok=True)

        # plot_traj(os.path.join(gt_src, seq, "gt/gt.txt"), name)
        # plot_traj_gif(os.path.join(gt_src, seq, "gt/gt.txt"), "{}/{}.txt".format(result_src, seq), "{}/{}.txt".format(oc_result_src, seq), name, file_name)  # datasets/dancetrack/val/dancetrack004/gt/gt.txt
        # plot_traj_marge3(os.path.join(gt_src, seq, "gt/gt.txt"), "{}/{}.txt".format(bam_result_src, seq), "{}/{}.txt".format(oc_result_src, seq), name, file_name)
        # plot_traj_marge2(os.path.join(gt_src, seq, "gt/gt.txt"), "{}/{}.txt".format(oc_result_src, seq), name, file_name, color=(1, 0, 0))
        plot_traj_marge2(os.path.join(gt_src, seq, "gt/gt.txt"), "{}/{}.txt".format(bam_result_src, seq), name, file_name, color=(0, 127/255, 0))
        
        # name = "baseline_{}".format(seq)
        # os.makedirs(os.path.join("traj_plots/{}".format(name)), exist_ok=True)
        # plot_traj(os.path.join(baseline, "{}.txt".format(seq)), name)

        # name = "ours_{}".format(seq)
        # os.makedirs(os.path.join("traj_plots/{}".format(name)), exist_ok=True)
        # plot_traj(os.path.join(ours, "{}.txt".format(seq)), name)