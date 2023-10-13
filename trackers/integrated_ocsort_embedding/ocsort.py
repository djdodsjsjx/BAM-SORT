"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import pdb
import pickle

import cv2
import torch
import torchvision

import numpy as np
from .association import *
from .embedding import EmbeddingComputer
from external.fast_reid.fast_reid_interfece import FastReIDInterface
from .cmc import CMCComputer

# [cur_age - k, cur_age) || [0, cur_age)
def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i  # 3 2 1
        if cur_age - dt in observations:  # -3 -2 -1
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]

# [x1, y1, x2, y2] => [x, y, s, r]
def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_bbox_to_z_new(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox_new(x):
    x, y, w, h = x.reshape(-1)[:4]
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)

# [x, y, s, r] => [x1, y1, x2, y2]
def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))

# 计算两个检测框之间的运动速度
def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
    Q = np.diag(
        (
            (p * w) ** 2,
            (p * h) ** 2,
            (p * w) ** 2,
            (p * h) ** 2,
            (v * w) ** 2,
            (v * h) ** 2,
            (v * w) ** 2,
            (v * h) ** 2,
        )
    )
    return Q


def new_kf_measurement_noise(w, h, m=1 / 20):
    w_var = (m * w) ** 2
    h_var = (m * h) ** 2
    R = np.diag((w_var, h_var, w_var, h_var))
    return R


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, delta_t=3, orig=False, emb=None, alpha=0, new_kf=False):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:  # ORU
            from .kalmanfilter import KalmanFilterNew as KalmanFilter
        else:  # KF
            from filterpy.kalman import KalmanFilter

        self.new_kf = new_kf
        if new_kf:
            self.kf = KalmanFilter(dim_x=8, dim_z=4)
            self.kf.F = np.array(
                [
                    # x y w h x' y' w' h'
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                ]
            )
            _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
            self.kf.P = new_kf_process_noise(w, h)
            self.kf.P[:4, :4] *= 4
            self.kf.P[4:, 4:] *= 100
            # Process and measurement uncertainty happen in functions
            self.bbox_to_z_func = convert_bbox_to_z_new
            self.x_to_bbox_func = convert_x_to_bbox_new
        else:
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array(
                [
                    # x  y  s  r  x' y' s'
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ]
            )
            self.kf.R[2:, 2:] *= 10.0
            self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
            self.kf.P *= 10.0
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            self.bbox_to_z_func = convert_bbox_to_z
            self.x_to_bbox_func = convert_x_to_bbox

            # Attempt
            # self.kf.P[2, 2] = 10000
            # self.kf.R[2, 2] = 10000

        self.kf.x[:4] = self.bbox_to_z_func(bbox)  # 位置和速度

        self.time_since_update = 0  # 到当前帧，目标的丢失帧数
        self.id = KalmanBoxTracker.count  # 轨迹线ID号
        KalmanBoxTracker.count += 1
        self.history = []  # 跟踪器 仅存放丢失预测状态
        self.hits = 0  # 到当前帧，总匹配的帧数
        self.hit_streak = 0  # 到当前帧，连续匹配帧数
        self.age = 0  # 预测总帧数
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        # Used for OCR
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # 上一次观测状态
        # Used to output track after min_hits reached
        self.history_observations = []  # 跟踪器 历史观测状态
        # Used for velocity
        self.observations = dict()  # 每一帧对应检测框
        self.velocity = None  # 跟踪器的速度
        self.delta_t = delta_t  # 时间间隔

        self.emb = emb

        self.frozen = False

    def update(self, bbox):  # [x1, y1, x2, y2, conf], cls
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            self.frozen = False

            if self.last_observation.sum() >= 0:  # 寻找上一帧检测框
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]  # -3 -2 -1
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)  # 历史观测框(previous_box)与当前观测框(box)之间的速度, 用于下一时刻速度关联矩阵求解
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            if self.new_kf:
                R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
                self.kf.update(self.bbox_to_z_func(bbox), R=R, new_kf=True)
            else:
                self.kf.update(self.bbox_to_z_func(bbox))
        else:
            self.kf.update(bbox, new_kf=self.new_kf)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):  # DA
        self.emb = alpha * self.emb + (1 - alpha) * emb
        # self.emb = 0.9 * self.emb + 0.1 * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        # For OCR
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        # Apply to each box in the range of velocity computation
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        # Also need to change kf state, but might be frozen
        self.kf.apply_affine_correction(m, t, self.new_kf)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Don't allow negative bounding boxes
        if self.new_kf:
            if self.kf.x[2] + self.kf.x[6] <= 0:
                self.kf.x[6] = 0
            if self.kf.x[3] + self.kf.x[7] <= 0:
                self.kf.x[7] = 0

            # Stop velocity, will update in kf during OOS
            if self.frozen:
                self.kf.x[6] = self.kf.x[7] = 0
            Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
        else:
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0
            Q = None

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:  # 目标存在丢失
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))  # [x1, y1, x2, y2]
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}


class OCSort(object):
    def __init__(
        self,
        det_thresh,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        w_association_emb=0.75,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        new_kf_off=False,
        grid_off=False,
        result_folder=None,
        **kwargs,
    ):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age  # 最大存活帧数
        self.min_hits = min_hits  # 最小观测次数
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0  # 当前帧数
        self.det_thresh = det_thresh  # 高分阈值
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        self.result_folder = result_folder
        KalmanBoxTracker.count = 0

        # self.embedder = EmbeddingComputer(dataset=None, test_dataset=True, grid_off=grid_off)
        self.embedder = FastReIDInterface("external/fast_reid/configs/MOT20/sbs_S50.yml",
                                            "external/weights/mot20_sbs_S50.pth",
                                            "cuda")
        self.cmc = CMCComputer()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        self.new_kf_off = new_kf_off
        self.grid_off = grid_off

    def update(self, output_results, img_info, img_size, tag):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))
        if not isinstance(output_results, np.ndarray):
            output_results = output_results.cpu().numpy()
        self.frame_count += 1
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        
        video_name = img_info[4][0].split('/')[0]
        # Rescale
        curr_img = cv2.imread('datasets/MOT20T/train/' + img_info[4][0])
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / img_h, img_size[1] / img_w)
        dets[:, :4] /= scale

        # 特征向量获取
        dets_embs = np.ones((dets.shape[0], 1))
        if not self.embedding_off and dets.shape[0] != 0:
            # dets_embs = self.embedder.compute_embedding(curr_img, dets[:, :4], tag)  # size*ch*w*h(例：5*3*384*128) => size*ch(例：5*2048)
            dets_embs = self.embedder.inference(curr_img, dets[:, :4])

        # CMC
        if not self.cmc_off:
            transform = self.cmc.compute_affine(curr_img, dets[:, :4], tag)
            for trk in self.trackers:
                trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        dets_alpha = af + (1 - af) * (1 - trust)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))
        trk_embs = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]  # 预测
            hits = self.trackers[t].hit_streak
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, hits]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # Shape = (num_trackers, 3, 512) if grid
        trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])  # 当前跟踪器上一时刻的检测框
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])  # [cur_age - k, cur_age) || [0, cur_age)

        """
            First round of association
            matched: 轨迹线与检测框匹配
            unmatched_dets: 该检测框，没有与之匹配的跟踪线
            unmatched_trks: 该跟踪线，没有与之匹配的检测框
        """
        matched, unmatched_dets, unmatched_trks = associate(  # OCM
            dets,
            trks,
            dets_embs,
            trk_embs,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
            self.embedding_off,
            self.grid_off,
            self.result_folder+f'/{video_name}',
            self.frame_count
        )
        for m in matched:  # 更新轨迹线
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
        """
            Second round of associaton by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]

            # TODO: maybe use embeddings here
            iou_left = self.asso_func(left_dets, left_trks)  # [undet, 上一帧untrack]  iou_cost
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:  # iou匹配阈值
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)  # asso_func匹配

                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])  # 从待删除轨迹中去除
                    self.trackers[trk_ind].update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:  # 无检测框
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # 创建新轨迹
            trk = KalmanBoxTracker(
                dets[i, :], delta_t=self.delta_t, emb=dets_embs[i], alpha=dets_alpha[i], new_kf=not self.new_kf_off
            )
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # 连续匹配帧数 >= 最小阈值
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # 返回满足要求的跟踪框
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:  # 丢失的帧数超过阈值
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def dump_cache(self):
        self.cmc.dump_cache()
        self.embedder.dump_cache()
