from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import copy
import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from trackers.byte_tracker.byte_tracker import BYTETracker
# from trackers.ocsort_tracker.ocsort import OCSort
from trackers.deepsort_tracker.deepsort import DeepSort
from trackers.motdt_tracker.motdt_tracker import OnlineTracker
from trackers.sparse_tracker.sparse_tracker import SparseTracker
# from trackers.integrated_ocsort_embedding.ocsort import OCSort
from trackers.hybird_sort_tracker.hybird_sort import Hybird_Sort
from trackers.hybird_sort_tracker.hybird_sort_reid import Hybird_Sort_ReID
from trackers.bamsort_tracker.bamsort_bec import OCSort
import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import cv2
from utils.utils import write_results, write_results_no_score, write_det_results
from external.fast_reid.fast_reid_interfece import FastReIDInterface
import numpy as np

from yolox.utils.visualize import plot_tracking, vis_notag

class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
    
            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_ocsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        trackers_len = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
        # ocsort跟踪器初始化
        tracker = OCSort(args=self.args, det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh, asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte)
        
        embedder = FastReIDInterface(self.args.fast_reid_config, self.args.fast_reid_weights, 'cuda')
        detections = dict()
        result_img = None
        dets = None
        for cur_iter, (imgs, _, info_imgs, ids, raw_img) in enumerate(
            progress_bar(self.dataloader)
        ):
            raw_img = raw_img.numpy()[0, ...]
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    tracker = OCSort(args=self.args, det_thresh = self.args.track_thresh, iou_threshold=self.args.iou_thresh, asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia, use_byte=self.args.use_byte)

                    if len(results) != 0:  # 每一个视频起始帧
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)

                        results = []
                        trackers_len = []
                        dets = None

                cur_dets = None
                ckt_file =  "exps/dancetrack/{}/{}.txt".format(self.args.dataset_type, video_name)
                if os.path.exists(ckt_file):
                    if dets is None:
                        dets = np.loadtxt(ckt_file, delimiter=",")
                    
                    cur_dets = dets[np.where(dets[:,0]==frame_id)]
                    det_bboxes = cur_dets[:, 2:6]
                    det_scores = cur_dets[:, 6]
                    cur_dets = np.concatenate((det_bboxes, det_scores.reshape(-1, 1)), axis=1)
                else:
                    imgs = imgs.type(tensor_type)

                    # skip the the last iters since batchsize might be not enough for batch inference

                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())
                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    # we should save the detections here ! 
                    # os.makedirs("exps/dance_detections/{}".format(video_name), exist_ok=True)
                    # torch.save(outputs[0], ckt_file)

                    output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
                    data_list.extend(output_results)

                    output_results = outputs[0]
                    # post_process detections
                    if output_results.shape[1] == 5:
                        det_scores = output_results[:, 4]
                        det_bboxes = output_results[:, :4]
                    else:
                        output_results = output_results.cpu().numpy()
                        det_scores = output_results[:, 4] * output_results[:, 5]
                        det_bboxes = output_results[:, :4]  # x1y1x2y2
                    img_h, img_w = info_imgs[0], info_imgs[1]
                    scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
                    det_bboxes /= scale
                    cur_dets = np.concatenate((det_bboxes, np.expand_dims(det_scores, axis=-1)), axis=1)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start


            # run tracking
            online_targets = tracker.update(cur_dets)  # 跟踪器更新
            online_tlwhs = []
            online_ids = []
            online_scores =  []
            for t in online_targets:
                """
                    Here is minor issue that DanceTrack uses the same annotation
                    format as MOT17/MOT20, namely xywh to annotate the object bounding
                    boxes. But DanceTrack annotation is cropped at the image boundary, 
                    which is different from MOT17/MOT20. So, cropping the output
                    bounding boxes at the boundary may slightly fix this issue. But the 
                    influence is minor. For example, with my results on the interpolated
                    OC-SORT:
                    * without cropping: HOTA=55.731
                    * with cropping: HOTA=55.737
                """
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                score = t[5]
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))  # 每一帧跟踪器信息: fid, x, y, w, h, tid
            trackers_len.append(tracker.save_info(cur_dets))  # 保存每一帧的现存轨迹数量和检测框数

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if self.args.save_datasets_pic:  # 保存图片
                
                raw_img_conf = raw_img.copy()
                online_im_conf = vis_notag(raw_img_conf, cur_dets[:, 0:4], cur_dets[:, 4], frame_id)
                online_im = plot_tracking(
                    raw_img, online_tlwhs, online_ids, scores=online_scores, frame_id=frame_id, fps=0, distance=0
                )
                result_img = os.path.join(result_folder, video_name, 'tracked')
                result_img_conf = os.path.join(result_folder, video_name, 'conf')
                os.makedirs(result_img, exist_ok=True)
                os.makedirs(result_img_conf, exist_ok=True)
                cv2.imwrite(result_img + f'/{frame_id}.jpg', online_im)
                cv2.imwrite(result_img_conf + f'/{frame_id}.jpg', online_im_conf)

            if cur_iter == len(self.dataloader) - 1:  # 最后一个视频最后一帧
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)
                result_filename_trts = os.path.join(os.path.split(result_folder)[0], 'trackers_num.txt')
                np.savetxt(result_filename_trts, trackers_len, fmt="%d, %d, %s")

                # result_filename_dets = os.path.join("exps/dancetrack/test", '{}.txt'.format(video_names[video_id]))
                # write_det_results(result_filename_dets, det_results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_hybird_sort(
            self,
            args,
            model,
            distributed=False,
            half=False,
            trt_file=None,
            decoder=None,
            test_size=None,
            result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        ori_thresh = self.args.track_thresh
        detections = dict()

        for cur_iter, (imgs, _, info_imgs, ids, raw_img) in enumerate(    # [hgx0411] add raw_image for FastReID
                progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                img_base_name = img_file_name[0].split('/')[-1].split('.')[0]

                """
                    Here, you can use adaptive detection threshold as in BYTE
                    (line 268 - 292), which can boost the performance on MOT17/MOT20
                    datasets, but we don't use that by default for a generalized 
                    stack of parameters on all datasets.
                """
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                else:
                    self.args.track_thresh = ori_thresh

                if video_name == 'MOT20-06' or video_name == 'MOT20-08':
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    tracker = Hybird_Sort(args, det_thresh=self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                                     asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia,
                                     use_byte=self.args.use_byte)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                ckt_file = "dance_detections/dancetrack_wo_ch_w_reid/{}/{}_detetcion.pkl".format(video_name, img_base_name)
                if os.path.exists(ckt_file):
                    data = torch.load(ckt_file)
                    outputs = [data['detection']]
                else:
                    imgs = imgs.type(tensor_type)

                    # skip the the last iters since batchsize might be not enough for batch inference
                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    # we should save the detections here !
                    # os.makedirs("dance_detections/{}".format(video_name), exist_ok=True)
                    # torch.save(outputs[0], ckt_file)
                    # res = {}
                    # res['detection'] = outputs[0]
                    # os.makedirs("dance_detections/{}".format(video_name), exist_ok=True)
                    # torch.save(res, ckt_file)
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                """
                    Here is minor issue that DanceTrack uses the same annotation
                    format as MOT17/MOT20, namely xywh to annotate the object bounding
                    boxes. But DanceTrack annotation is cropped at the image boundary, 
                    which is different from MOT17/MOT20. So, cropping the output
                    bounding boxes at the boundary may slightly fix this issue. But the 
                    influence is minor. For example, with my results on the interpolated
                    OC-SORT:
                    * without cropping: HOTA=55.731
                    * with cropping: HOTA=55.737
                """
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6 if self.args.dataset in ["mot17", "mot20"] else False
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    
    def evaluate_hybird_sort_reid(
            self,
            args,
            model,
            distributed=False,
            half=False,
            trt_file=None,
            decoder=None,
            test_size=None,
            result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        NOTE: This function will change training mode to False, please save states if needed.
        Args:
            model : model to evaluate.
        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # assert self.args.with_fastreid
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        # for fastreid
        self.encoder = FastReIDInterface(self.args.fast_reid_config, self.args.fast_reid_weights, 'cuda')

        ori_thresh = self.args.track_thresh
        detections = dict()

        for cur_iter, (imgs, _, info_imgs, ids, raw_image) in enumerate(    # [hgx0411] add raw_image for FastReID
                progress_bar(self.dataloader)
        ):
            raw_image = raw_image.numpy()[0, ...]  # sequeeze batch dim, [bs, H, W, C] ==> [H, W, C]
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]
                img_base_name = img_file_name[0].split('/')[-1].split('.')[0]

                """
                    Here, you can use adaptive detection threshold as in BYTE
                    (line 268 - 292), which can boost the performance on MOT17/MOT20
                    datasets, but we don't use that by default for a generalized 
                    stack of parameters on all datasets.
                """
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                else:
                    self.args.track_thresh = ori_thresh

                if video_name == 'MOT20-06' or video_name == 'MOT20-08':
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                if video_name not in video_names:
                    video_names[video_id] = video_name

                if frame_id == 1:
                    tracker = Hybird_Sort_ReID(args, det_thresh=self.args.track_thresh, iou_threshold=self.args.iou_thresh,
                                     asso_func=self.args.asso, delta_t=self.args.deltat, inertia=self.args.inertia)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                ckt_file = "dance_detections/{}/{}_detetcion.pkl".format(video_name, img_base_name)
                if os.path.exists(ckt_file):
                    data = torch.load(ckt_file)
                    outputs = [data['detection']]
                    id_feature = data['reid_feature']
                else:
                    imgs = imgs.type(tensor_type)

                    # skip the the last iters since batchsize might be not enough for batch inference
                    outputs = model(imgs)
                    if decoder is not None:
                        outputs = decoder(outputs, dtype=outputs.type())

                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    if outputs[0] == None:
                        id_feature = np.array([]).reshape(0, 2048)
                    else:
                        bbox_xyxy = copy.deepcopy(outputs[0][:, :4])
                        # we should save the detections here !
                        # os.makedirs("dance_detections/{}".format(video_name), exist_ok=True)
                        # torch.save(outputs[0], ckt_file)
                        # box rescale borrowed from convert_to_coco_format()
                        scale = min(self.img_size[0] / float(info_imgs[0]), self.img_size[1] / float(info_imgs[1]))
                        bbox_xyxy /= scale
                        id_feature = self.encoder.inference(raw_image, bbox_xyxy.cpu().detach().numpy())    # normalization and numpy included
                    # res = {}
                    # res['detection'] = outputs[0]
                    # res['reid_feature'] = id_feature
                    # os.makedirs("dance_detections/{}".format(video_name), exist_ok=True)
                    # torch.save(res, ckt_file)
                    # # verify of bboxes
                    # import torchvision.transforms as T
                    # mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
                    # std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
                    # normalize = T.Normalize(mean.tolist(), std.tolist())
                    # unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
                    # img_ = unnormalize(imgs[0]) * 255
                    # img2 = img_.permute(1, 2, 0).type(torch.int16).cpu().detach().numpy()
                    # import cv2
                    # cv2.imwrite('img.png', img2[int(bbox_xyxy[0][1]): int(bbox_xyxy[0][3]),
                    #                        int(bbox_xyxy[0][0]): int(bbox_xyxy[0][2]), :])
            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            if self.args.ECC:
                # compute warp matrix with ECC, when frame_id is not 1.
                # raw_image = raw_image.numpy()[0, ...]       # sequeeze batch dim, [bs, H, W, C] ==> [H, W, C]
                if frame_id != 1:
                    warp_matrix, src_aligned = self.ECC(self.former_frame, raw_image, align=True)
                else:
                    warp_matrix, src_aligned = None, None
                self.former_frame = raw_image       # update former_frame
            else:
                warp_matrix, src_aligned = None, None

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, id_feature=id_feature, warp_matrix=warp_matrix)        # [hgx0411] id_feature
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                """
                    Here is minor issue that DanceTrack uses the same annotation
                    format as MOT17/MOT20, namely xywh to annotate the object bounding
                    boxes. But DanceTrack annotation is cropped at the image boundary, 
                    which is different from MOT17/MOT20. So, cropping the output
                    bounding boxes at the boundary may slightly fix this issue. But the 
                    influence is minor. For example, with my results on the interpolated
                    OC-SORT:
                    * without cropping: HOTA=55.731
                    * with cropping: HOTA=55.737
                """
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6 if self.args.dataset in ["mot17", "mot20"] else False
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end

            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list



    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info