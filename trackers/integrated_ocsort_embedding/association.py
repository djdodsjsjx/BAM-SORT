import numpy as np
import scipy.spatial as sp
import pandas as pd

def aspect_ratio_difference_batch(bboxes1, bboxes2):
    """
    Compute the aspect ratio difference between any two bounding boxes in the form [x1, y1, x2, y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # 计算任意两个矩形框的宽度和高度
    width1 = bboxes1[..., 2] - bboxes1[..., 0]
    height1 = bboxes1[..., 3] - bboxes1[..., 1]
    width2 = bboxes2[..., 2] - bboxes2[..., 0]
    height2 = bboxes2[..., 3] - bboxes2[..., 1]

    # 计算任意两个矩形框的长宽比值
    aspect_ratio1 = width1 / height1
    aspect_ratio2 = width2 / height2

    # 计算长宽比值的差
    aspect_ratio_difference = np.abs(aspect_ratio1 - aspect_ratio2)
    
    aspect_ratio_difference /= np.linalg.norm(aspect_ratio_difference, axis=1, keepdims=True)

    return aspect_ratio_difference

def iou_batch(bboxes1, bboxes2):  # det, track
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)  # 1*t*5
    bboxes1 = np.expand_dims(bboxes1, 1)  # d*1*5

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])  # d*t
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)  # d*t
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h  # d*t
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return o  # d*t


def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert (wc > 0).all() and (hc > 0).all()
    area_enclose = wc * hc
    giou = iou - (area_enclose - wh) / area_enclose
    giou = (giou + 1.0) / 2.0  # resize from (-1,1) to (0,1)
    return giou


def diou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0  # resize from (-1,1) to (0,1)


def ciou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.0
    h1 = h1 + 1.0
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi**2)) * (arctan**2)
    S = 1 - iou
    alpha = v / (S + v)
    ciou = iou - inner_diag / outer_diag - alpha * v

    return (ciou + 1) / 2.0  # resize from (-1,1) to (0,1)

# 计算两个检测框之间的距离
def ct_dist(bboxes1, bboxes2):
    """
    Measure the center distance between two sets of bounding boxes,
    this is a coarse implementation, we don't recommend using it only
    for association, which can be unstable and sensitive to frame rate
    and object speed.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    # The linear rescaling is a naive version and needs more study
    ct_dist = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist  # resize to (0,1)

# 计算检测框与预测框的相对速度(y, x)
def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det


def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

## aw：加权成本矩阵
def compute_aw_new_metric(emb_cost, w_association_emb, max_diff=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)
    w_emb_bonus = np.full_like(emb_cost, 0)

    # Needs two columns at least to make sense to boost
    if emb_cost.shape[1] >= 2:
        # Across all rows
        for idx in range(emb_cost.shape[0]):
            inds = np.argsort(-emb_cost[idx])  # 降序
            # Row weight is difference between top / second top
            row_weight = min(emb_cost[idx, inds[0]] - emb_cost[idx, inds[1]], max_diff)  # 最大与第二大的插值
            # Add to row
            w_emb_bonus[idx] += row_weight / 2

    if emb_cost.shape[0] >= 2:
        for idj in range(emb_cost.shape[1]):
            inds = np.argsort(-emb_cost[:, idj])
            col_weight = min(emb_cost[inds[0], idj] - emb_cost[inds[1], idj], max_diff)
            w_emb_bonus[:, idj] += col_weight / 2

    return w_emb + w_emb_bonus

def compute_cm_new_metric(dets, trackers, hits_v=5):
    hits = trackers[:, 5]
    res = 1 - 1 / (hits_v + hits)
    return np.tile(res, (dets.shape[0], 1))

## t-d特征余弦相似度求解
def split_cosine_dist(dets, trks, affinity_thresh=0.55, pair_diff_thresh=0.6, hard_thresh=True):

    cos_dist = np.zeros((len(dets), len(trks)))

    for i in range(len(dets)):
        for j in range(len(trks)):

            cos_d = 1 - sp.distance.cdist(dets[i], trks[j], "cosine")  ## shape = 3x3
            patch_affinity = np.max(cos_d, axis=0)  ## shape = [3,]
            # exp16 - Using Hard threshold
            if hard_thresh:
                if len(np.where(patch_affinity > affinity_thresh)[0]) != len(patch_affinity):
                    cos_dist[i, j] = 0
                else:
                    cos_dist[i, j] = np.max(patch_affinity)
            else:
                cos_dist[i, j] = np.max(patch_affinity)  # can experiment with mean too (max works slightly better)

    return cos_dist

# 数据关联
def associate(
    detections,
    trackers,
    det_embs,
    trk_embs,
    iou_threshold,
    velocities,
    previous_obs,
    vdc_weight,
    w_assoc_emb,
    aw_off,
    aw_param,
    emb_off,
    grid_off,
    result_folder,
    frame_count
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    Y, X = speed_direction_batch(detections, previous_obs)  # 跟踪器速度方向(蓝线)  size: t × d

    # 检测速度 size: t × d
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)  # np.newaxis: (n, ) => (n, 1) 添加一个维度 np.repeat: 重复Y.shape[1](检测框个数)次
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)

    # 两个单位速度向量求内积
    diff_angle_cos = inertia_X * X + inertia_Y * Y  # 向量内积公式：d[i, j] = inx[i, j] * x[i, j] + iny[i, y] * y[i, j]
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)  # 限幅
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])  # (t, )
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0  # 首次不参与计算

    iou_matrix = iou_batch(detections, trackers)  # size: d × t
    cm_matrix = compute_cm_new_metric(detections, trackers, 5)
    ard_matrix = aspect_ratio_difference_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)  # 置信度：d × t
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)  # t * d

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight  # t * d
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores  # d * t

    emb_cost = None
    if not emb_off:
        if grid_off:
            emb_cost = None if (trk_embs.shape[0] == 0 or det_embs.shape[0] == 0) else det_embs @ trk_embs.T  # 向量内积
        else:
            emb_cost = split_cosine_dist(det_embs, trk_embs)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost = 0
            else:
                # emb_cost[iou_matrix <= 0.3] = 0
                pass
            if not aw_off:
                w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)  # w_b权重计算
                emb_cost *= w_matrix
            else:
                emb_cost *= w_assoc_emb

            # final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            final_cost = -(iou_matrix + emb_cost + 0.8*cm_matrix)
            # final_cost = -np.maximum(np.maximum(iou_matrix, angle_diff_cost), emb_cost)

            if frame_count % 100 == 1:
                writer = pd.ExcelWriter(result_folder + f'{frame_count}.xlsx')
                df_iou = pd.DataFrame(iou_matrix)
                df_emb = pd.DataFrame(emb_cost)
                df_adc = pd.DataFrame(angle_diff_cost)
                df_ard = pd.DataFrame(ard_matrix)
                df_cm = pd.DataFrame(cm_matrix)
                df = pd.DataFrame(-final_cost)
                
                df_iou.to_excel(writer, sheet_name='iou', index=False)
                df_emb.to_excel(writer, sheet_name='emb', index=False)
                df_adc.to_excel(writer, sheet_name='angle', index=False)
                df_ard.to_excel(writer, sheet_name='ratio', index=False)
                df_cm.to_excel(writer, sheet_name='cm', index=False)
                df.to_excel(writer, sheet_name='total', index=False)
                writer.save()
            matched_indices = linear_assignment(final_cost)  # C_IOU & C_V  =>  [d, t]
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))  # [det, track]
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate_kitti(detections, trackers, det_cates, iou_threshold, velocities, previous_obs, vdc_weight):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    """
        Cost from the velocity direction consistency
    """
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    """
        Cost from IoU
    """
    iou_matrix = iou_batch(detections, trackers)

    """
        With multiple categories, generate the cost for catgory mismatch
    """
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
        for j in range(num_trk):
            if det_cates[i] != trackers[j, 4]:
                cate_matrix[i][j] = -1e6

    cost_matrix = -iou_matrix - angle_diff_cost - cate_matrix

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
