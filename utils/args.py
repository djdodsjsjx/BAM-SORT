import argparse

# yolox_s_MOT20
# def make_parser():
#     parser = argparse.ArgumentParser("OC-SORT parameters")
#     parser.add_argument("--expn", type=str, default="save_det")
#     parser.add_argument("-n", "--name", type=str, default=None, help="model name")

#     # distributed
#     parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
#     parser.add_argument("--output_dir", type=str, default="evaldata/trackers/mot_challenge/improve/")
#     parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
#     parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
#     parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

#     parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
#     parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
#     parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

#     parser.add_argument(  # 跟踪算法评估器
#         "-f", "--exp_file",
#         default="exps/example/mot/yolox_x_mot20_train.py",
#         type=str,
#         help="pls input your expriment description file",
#     )
#     parser.add_argument(
#         "--fp16", dest="fp16",
#         default=True,
#         action="store_true",
#         help="Adopting mix precision evaluating.",
#     )
#     parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn for testing.",)
#     parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
#     parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)  # 自己评估置为False
#     parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
#     parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)

#     # det args
#     parser.add_argument("-c", "--ckpt", default="pretrained/ocsort_x_mot20.pth.tar", type=str, help="ckpt for eval")  # 检测模型
#     parser.add_argument("--conf", default=0.1, type=float, help="test conf")
#     parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
#     parser.add_argument("--tsize", default=None, type=int, help="test img size")
#     parser.add_argument("--seed", default=None, type=int, help="eval seed")

#     # tracking args
#     parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
#     parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
#     parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
#     parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
#     parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
#     parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#     parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
#     parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
#     # parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
#     parser.add_argument("--mot20", dest="mot20", default=True, action="store_true", help="test mot20.")  # 用不到
#     parser.add_argument("--public", action="store_true", help="use public detection")
#     parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis/Height_Modulated_IoU")
#     parser.add_argument("--use_byte", dest="use_byte", default=True, action="store_true", help="use byte in tracking.")
    
#     # for kitti/bdd100k inference with public detections
#     parser.add_argument('--raw_results_path', type=str, default="exps",
#         help="path to the raw tracking results from other tracks")
#     parser.add_argument('--out_path', type=str, default="evaldata/trackers/mot_challenge/improve/MOT20-val-half/baseline+bec+act/", help="path to save output results")  # 推理后输出路径
#     parser.add_argument("--dataset", type=str, default="MOT20", help="MOT17| MOT20 | dancetrack | permatrack_kitti | bdd")
#     parser.add_argument("--dataset_type", type=str, default="val", help="train | val | test")
#     parser.add_argument("--gt_type", type=str, default="val_half", help="| val_half | train_half")
#     parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
#             initializing the tracks (offline).")

#     # for demo video
#     parser.add_argument("--demo_type", default="video", help="demo type, eg. image, video and webcam")
#     parser.add_argument( "--path", default="./videos/input/person.mp4", help="path to images or video")  # 视频路径
#     parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")  # 摄像头id
#     parser.add_argument(
#         "--save_result",
#         default="True",
#         action="store_true",
#         help="whether to save the inference result of image/video",
#     )
#     parser.add_argument(
#         "--aspect_ratio_thresh", type=float, default=1.6,
#         help="threshold for filtering out boxes of which aspect ratio are above the given value."
#     )
#     parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
#     parser.add_argument(
#         "--device",
#         default="gpu",
#         type=str,
#         help="device to run our model, can either be cpu or gpu",
#     )

#     # CMC
#     parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

#     parser.add_argument("--depth_levels", default=3, type=int)  # 距离深度
#     parser.add_argument("--confirm_thresh", default=0.8, type=float)  # 匹配阈值

#     parser.add_argument("--w_emb", type=float, default=0.75, help="Combine weight for emb cost")
#     parser.add_argument(
#         "--alpha_fixed_emb",
#         type=float,
#         default=0.95,
#         help="Alpha fixed for EMA embedding",
#     )
#     parser.add_argument("--emb_off", default=False, type=bool)
#     parser.add_argument("--cmc_off", default=False, type=bool)
#     parser.add_argument("--aw_off", default=True, type=bool)
#     parser.add_argument("--aw_param", type=float, default=0.5)
#     parser.add_argument("--new_kf_off", default=True, type=bool)
#     parser.add_argument("--grid_off", default=True, type=bool)

#     parser.add_argument("--TCM_first_step", default=True, action="store_true", help="use TCM in first step.")
#     parser.add_argument("--TCM_byte_step", default=True, action="store_true", help="use TCM in byte step.")
#     parser.add_argument("--TCM_first_step_weight", type=float, default=1.0, help="TCM first step weight")
#     parser.add_argument("--TCM_byte_step_weight", type=float, default=1.0, help="TCM second step weight")
#     parser.add_argument("--sort_with_reid", default=False, action="store_true", help="use ReID model for Hybird SORT.")

#     # for fast reid
#     parser.add_argument("--EG_weight_high_score", default=0.0, type=float, help="weight of appearance cost matrix when using EG")
#     parser.add_argument("--EG_weight_low_score", default=0.0, type=float, help="weight of appearance cost matrix when using EG")
#     parser.add_argument("--low_thresh", default=0.1, type=float, help="threshold of low score detections for BYTE")
#     parser.add_argument("--high_score_matching_thresh", default=0.8, type=float, help="matching threshold for detections with high score")
#     parser.add_argument("--low_score_matching_thresh", default=0.5, type=float, help="matching threshold for detections with low score")
#     parser.add_argument("--alpha", default=0.8, type=float, help="momentum of embedding update")
#     parser.add_argument("--with_fastreid", dest="with_fastreid", default=False, action="store_true", help="use FastReID flag.")
#     parser.add_argument("--fast_reid_config", dest="fast_reid_config", default=r"external/fast_reid/configs/CUHKSYSU_DanceTrack/sbs_S50.yml", type=str, help="reid config file path")
#     parser.add_argument("--fast_reid_weights", dest="fast_reid_weights", default=r"pretrained/dancetrack_sbs_S50.pth", type=str, help="reid weight path")
#     parser.add_argument("--with_longterm_reid", dest="with_longterm_reid", default=False, action="store_true", help="use long-term reid features for association.")
#     parser.add_argument("--longterm_reid_weight", default=0.0, type=float, help="weight of appearance cost matrix when using long term reid features in 1st stage association")
#     parser.add_argument("--longterm_reid_weight_low", default=0.0, type=float, help="weight of appearance cost matrix when using long term reid features in 2nd stage association")
#     parser.add_argument("--with_longterm_reid_correction", dest="with_longterm_reid_correction", default=False, action="store_true", help="use long-term reid features for association correction.")
#     parser.add_argument("--longterm_reid_correction_thresh", default=1.0, type=float, help="threshold of correction when using long term reid features in 1st stage association")
#     parser.add_argument("--longterm_reid_correction_thresh_low", default=1.0, type=float, help="threshold of correction when using long term reid features in 2nd stage association")
#     parser.add_argument("--longterm_bank_length", type=int, default=30, help="max length of reid feat bank")
#     parser.add_argument("--adapfs", dest="adapfs", default=True, action="store_true", help="Adaptive Feature Smoothing.")
#     # ECC for CMC
#     parser.add_argument("--ECC", dest="ECC", default=False, action="store_true", help="use ECC for CMC.")

#     parser.add_argument("--sort_with_bec", default=True, action="store_true")
#     parser.add_argument("--w_bec", default=0.3, type=float)
#     parser.add_argument("--sort_with_bgtf", default=False, action="store_true")
#     parser.add_argument("--pix_edge_bgtf", default=5, type=int)
#     parser.add_argument("--btfg_start_frame", default=50, type=int)
#     parser.add_argument("--sort_with_eiou", default=False, action="store_true")
#     parser.add_argument("--extendr_eiou", default=0.2, type=float)
#     parser.add_argument("--save_datasets_pic", default=False, action="store_true")  # 是否保存数据集运行后的图片，实验分析用
#     return parser


# baseline(ocsort)
def make_parser():
    parser = argparse.ArgumentParser("OC-SORT parameters")
    parser.add_argument("--expn", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--output_dir", type=str, default="evaldata/trackers/mot_challenge")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")

    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

    parser.add_argument(
        "-f", "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16", dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
    # parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
    parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
    parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--public", action="store_true", help="use public detection")
    parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
    parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")

    # for kitti/bdd100k inference with public detections
    parser.add_argument('--raw_results_path', type=str, default="exps",
        help="path to the raw tracking results from other tracks")
    parser.add_argument('--raw_results_public_path', type=str, default="datasets",
        help="path to the raw tracking results from other tracks")
    # parser.add_argument('--out_path', type=str, default="evaldata/trackers/MOT20/improve/train-half/baseline+now", help="path to save output results")  # 推理后输出路径
    parser.add_argument('--out_path', type=str, default="evaldata/trackers/MOT17/ablation/val-half/4-bec-asso", help="path to save output results")  # 推理后输出路径
    # parser.add_argument('--tn_out_path', type=str, default="evaldata/trackers/MOT20/tracker_num/val-half/baseline", help="path to save output results")  # 推理后输出路径
    parser.add_argument('--det_type', type=str, default="yolox_x")
    parser.add_argument("--dataset", type=str, default="MOT17", help="MOT17| MOT20 | dancetrack | kitti | bdd")
    parser.add_argument("--dataset_type", type=str, default="val", help="train | val | test | all")
    # parser.add_argument("--gt_type", type=str, default="", help="| _val_half | _train_half")
    parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
            initializing the tracks (offline).")

    # for demo video
    parser.add_argument("--demo_type", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument( "--path", default="./videos/demo.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    parser.add_argument("--sort_with_bec", default=True, action="store_true")
    parser.add_argument("--w_bec", default=0.4, type=float)  # dancetrack&MOT20: 0.3  MOT17: 0.4
    parser.add_argument("--bec_num", default=5, type=float, help=" >= 3")  # dancetrack: 4   MOT17: 5  MOT20: 6
    parser.add_argument("--min_hits", type=int, default=2, help="min hits to create track in SORT")   # dancetrack: 5  MOT17&MOT20: 2
    parser.add_argument("--sort_with_std", default=True, action="store_true")
    parser.add_argument("--std_time_since_update", type=int, default=15)  # dancetrack: 5  MOT17: 15   MOT20: 20
    parser.add_argument("--std_switch_cnt", type=int, default=1)  # dancetrack & MOT17&MOT20: 1
    parser.add_argument("--std_max_hits", type=int, default=50)


    parser.add_argument("--sort_with_reid", default=False, action="store_true", help="use ReID model for BAM-SORT.")
    parser.add_argument("--w_emb", type=float, default=0.75, help="Combine weight for emb cost")
    parser.add_argument("--alpha_fixed_emb", type=float, default=0.95)
    parser.add_argument("--fast_reid_config", dest="fast_reid_config", default=r"external/fast_reid/configs/CUHKSYSU_DanceTrack/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast_reid_weights", dest="fast_reid_weights", default=r"pretrained/dancetrack_sbs_S50.pth", type=str, help="reid weight path")
    return parser


# ocsort_x_mot17
# def make_parser():
#     parser = argparse.ArgumentParser("OC-SORT parameters")
#     parser.add_argument("--expn", type=str, default="baseline+bec+act")
#     parser.add_argument("-n", "--name", type=str, default=None, help="model name")

#     # distributed
#     parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
#     parser.add_argument("--output_dir", type=str, default="evaldata/trackers/MOT17/improve/")
#     parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
#     parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
#     parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

#     parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
#     parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
#     parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

#     parser.add_argument(  # 跟踪算法评估器
#         "-f", "--exp_file",
#         default="exps/example/mot/yolox_nano_mot17.py",
#         type=str,
#         help="pls input your expriment description file",
#     )
#     parser.add_argument(
#         "--fp16", dest="fp16",
#         default=True,
#         action="store_true",
#         help="Adopting mix precision evaluating.",
#     )
#     parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn for testing.",)
#     parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
#     parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)  # 自己评估置为False
#     parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
#     parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)

#     # det args
#     parser.add_argument("-c", "--ckpt", default="pretrained/bytetrack_nano_mot17.pth.tar", type=str, help="ckpt for eval")  # 检测模型
#     parser.add_argument("--conf", default=0.1, type=float, help="test conf")
#     parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
#     parser.add_argument("--tsize", default=None, type=int, help="test img size")
#     parser.add_argument("--seed", default=None, type=int, help="eval seed")

#     # tracking args
#     parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
#     parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
#     # parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
#     parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
#     parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
#     parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#     parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
#     parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
#     # parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
#     parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")  # 用不到
#     parser.add_argument("--public", action="store_true", help="use public detection")
#     parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis/Height_Modulated_IoU")
#     parser.add_argument("--use_byte", dest="use_byte", default=True, action="store_true", help="use byte in tracking.")

#     # for kitti/bdd100k inference with public detections
#     parser.add_argument("--sort_with_reid", default=False, action="store_true", help="use ReID model for BAM-SORT.")
#     parser.add_argument("--w_emb", type=float, default=0.75, help="Combine weight for emb cost")
#     parser.add_argument("--fast_reid_config", dest="fast_reid_config", default=r"external/fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
#     parser.add_argument("--fast_reid_weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid weight path")

#     parser.add_argument('--raw_results_path', type=str, default="exps",
#         help="path to the raw tracking results from other tracks")
#     parser.add_argument('--raw_results_public_path', type=str, default="datasets",
#         help="path to the raw tracking results from other tracks")
#     # parser.add_argument('--out_path', type=str, default="evaldata/trackers/MOT20/improve/train-half/baseline+now", help="path to save output results")  # 推理后输出路径
#     parser.add_argument('--out_path', type=str, default="evaldata/trackers/MOT20/improve/val/baseline", help="path to save output results")  # 推理后输出路径
#     parser.add_argument('--tn_out_path', type=str, default="evaldata/trackers/MOT20/tracker_num/val-half/baseline", help="path to save output results")  # 推理后输出路径
#     parser.add_argument('--det_type', type=str, default="yolox_x")
#     parser.add_argument("--dataset", type=str, default="MOT20", help="MOT17| MOT20 | dancetrack | kitti | bdd")
#     parser.add_argument("--dataset_type", type=str, default="all", help="train | val | test | all")
#     parser.add_argument("--gt_type", type=str, default="", help="| _val_half | _train_half")
#     parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
#             initializing the tracks (offline).")

#     # for demo video
#     parser.add_argument("--demo_type", default="video", help="demo type, eg. image, video and webcam")
#     parser.add_argument( "--path", default="./videos/input/person.mp4", help="path to images or video")  # 视频路径
#     parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")  # 摄像头id
#     parser.add_argument(
#         "--save_result",
#         default="True",
#         action="store_true",
#         help="whether to save the inference result of image/video",
#     )
#     parser.add_argument(
#         "--aspect_ratio_thresh", type=float, default=1.6,
#         help="threshold for filtering out boxes of which aspect ratio are above the given value."
#     )
#     parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
#     parser.add_argument(
#         "--device",
#         default="gpu",
#         type=str,
#         help="device to run our model, can either be cpu or gpu",
#     )

#     # ECC for CMC
#     parser.add_argument("--ECC", dest="ECC", default=False, action="store_true", help="use ECC for CMC.")

#     parser.add_argument("--sort_with_bec", default=False, action="store_true")
#     parser.add_argument("--w_bec", default=0.4, type=float)  # dancetrack: 0.3  MOT17: 0.4
#     parser.add_argument("--bec_num", default=5, type=float, help=" >= 3")  # dancetrack: 4   MOT17: 5
#     parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")   # dancetrack: 5  MOT17: 2
#     parser.add_argument("--std_time_since_update", type=int, default=15)  # dancetrack: 5  MOT17: 15
#     parser.add_argument("--std_switch_cnt", type=int, default=1)  # dancetrack & MOT17: 1
#     parser.add_argument("--std_max_hits", type=int, default=50)
#     parser.add_argument("--sort_with_bgtf", default=False, action="store_true")
#     parser.add_argument("--pix_edge_bgtf", default=5, type=int)
#     parser.add_argument("--btfg_start_frame", default=20, type=int)
#     parser.add_argument("--sort_with_eiou", default=False, action="store_true")
#     parser.add_argument("--extendr_eiou", default=0.2, type=float)
#     parser.add_argument("--save_datasets_pic", default=False, action="store_true")  # 是否保存数据集运行后的图片，实验分析用
#     return parser

# ocsort_dance
# def make_parser():
#     parser = argparse.ArgumentParser("OC-SORT parameters")
#     parser.add_argument("--expn", type=str, default="baseline+bec+act")
#     parser.add_argument("-n", "--name", type=str, default=None, help="model name")

#     # distributed
#     parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
#     parser.add_argument("--output_dir", type=str, default="evaldata/trackers/DanceTrack/improve/")
#     parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
#     parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
#     parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

#     parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
#     parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
#     parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

#     parser.add_argument(  # 跟踪算法评估器
#         "-f", "--exp_file",
#         default="exps/example/mot/yolox_x_dancetrack.py",
#         type=str,
#         help="pls input your expriment description file",
#     )
#     parser.add_argument(
#         "--fp16", dest="fp16",
#         default=True,
#         action="store_true",
#         help="Adopting mix precision evaluating.",
#     )
#     parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn for testing.",)
#     parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
#     parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)  # 自己评估置为False
#     parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
#     parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)

#     # det args
#     parser.add_argument("-c", "--ckpt", default="pretrained/ocsort_dance_model.pth.tar", type=str, help="ckpt for eval")  # 检测模型
#     parser.add_argument("--conf", default=0.1, type=float, help="test conf")
#     parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
#     parser.add_argument("--tsize", default=None, type=int, help="test img size")
#     parser.add_argument("--seed", default=None, type=int, help="eval seed")

#     # tracking args
#     parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
#     parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
#     # parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
#     parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
#     parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
#     parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#     parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
#     parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
#     # parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
#     parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")  # 用不到
#     parser.add_argument("--public", action="store_true", help="use public detection")
#     parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis/Height_Modulated_IoU")
#     parser.add_argument("--use_byte", dest="use_byte", default=True, action="store_true", help="use byte in tracking.")

#     # for kitti/bdd100k inference with public detections
#     parser.add_argument("--sort_with_reid", default=False, action="store_true", help="use ReID model for BAM-SORT.")
#     parser.add_argument("--w_emb", type=float, default=0.75, help="Combine weight for emb cost")
#     parser.add_argument("--fast_reid_config", dest="fast_reid_config", default=r"external/fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
#     parser.add_argument("--fast_reid_weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid weight path")

#     parser.add_argument('--raw_results_path', type=str, default="exps",
#         help="path to the raw tracking results from other tracks")
#     parser.add_argument('--raw_results_public_path', type=str, default="datasets",
#         help="path to the raw tracking results from other tracks")
#     # parser.add_argument('--out_path', type=str, default="evaldata/trackers/MOT20/improve/train-half/baseline+now", help="path to save output results")  # 推理后输出路径
#     parser.add_argument('--out_path', type=str, default="evaldata/trackers/dancetrack/ablation/val/0_bec_asso", help="path to save output results")  # 推理后输出路径
#     parser.add_argument('--tn_out_path', type=str, default="evaldata/trackers/dancetrack/tracker_num/val/baseline+bec", help="path to save output results")  # 推理后输出路径
#     parser.add_argument('--det_type', type=str, default="yolox_m")
#     parser.add_argument("--dataset", type=str, default="dancetrack", help="MOT17| MOT20 | dancetrack | kitti | bdd")
#     parser.add_argument("--dataset_type", type=str, default="val", help="train | val | test | all")
#     parser.add_argument("--gt_type", type=str, default="", help="| _val_half | _train_half")
#     parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
#             initializing the tracks (offline).")

#     # for demo video
#     parser.add_argument("--demo_type", default="video", help="demo type, eg. image, video and webcam")
#     parser.add_argument( "--path", default="./videos/input/person.mp4", help="path to images or video")  # 视频路径
#     parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")  # 摄像头id
#     parser.add_argument(
#         "--save_result",
#         default="True",
#         action="store_true",
#         help="whether to save the inference result of image/video",
#     )
#     parser.add_argument(
#         "--aspect_ratio_thresh", type=float, default=1.6,
#         help="threshold for filtering out boxes of which aspect ratio are above the given value."
#     )
#     parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
#     parser.add_argument(
#         "--device",
#         default="gpu",
#         type=str,
#         help="device to run our model, can either be cpu or gpu",
#     )

#     # ECC for CMC
#     parser.add_argument("--ECC", dest="ECC", default=False, action="store_true", help="use ECC for CMC.")

#     parser.add_argument("--sort_with_bec", default=False, action="store_true")
#     parser.add_argument("--w_bec", default=0.3, type=float)  # dancetrack: 0.3  MOT17: 0.4
#     parser.add_argument("--bec_num", default=4, type=float, help=" >= 3")  # dancetrack: 4   MOT17: 5
#     parser.add_argument("--min_hits", type=int, default=5, help="min hits to create track in SORT")   # dancetrack: 5  MOT17: 2
#     parser.add_argument("--std_time_since_update", type=int, default=5)  # dancetrack: 5  MOT17: 15
#     parser.add_argument("--std_switch_cnt", type=int, default=1)  # dancetrack & MOT17: 1
#     parser.add_argument("--std_max_hits", type=int, default=50)
#     parser.add_argument("--sort_with_bgtf", default=False, action="store_true")
#     parser.add_argument("--pix_edge_bgtf", default=5, type=int)
#     parser.add_argument("--btfg_start_frame", default=20, type=int)
#     parser.add_argument("--sort_with_eiou", default=False, action="store_true")
#     parser.add_argument("--extendr_eiou", default=0.2, type=float)
#     parser.add_argument("--save_datasets_pic", default=False, action="store_true")  # 是否保存数据集运行后的图片，实验分析用
#     return parser



# yolox_nano
# def make_parser():
#     parser = argparse.ArgumentParser("OC-SORT parameters")
#     parser.add_argument("--expn", type=str, default=None)
#     parser.add_argument("-n", "--name", type=str, default=None, help="model name")

#     # distributed
#     parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
#     parser.add_argument("--output_dir", type=str, default="evaldata/trackers/myset")
#     parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
#     parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
#     parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

#     parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
#     parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
#     parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

#     parser.add_argument(  # 跟踪算法评估器
#         "-f", "--exp_file",
#         default="exps/example/mot/yolox_x_myset.py",
#         type=str,
#         help="pls input your expriment description file",
#     )
#     parser.add_argument(
#         "--fp16", dest="fp16",
#         default=True,
#         action="store_true",
#         help="Adopting mix precision evaluating.",
#     )
#     parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn for testing.",)
#     parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
#     parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)  # 自己评估置为False
#     parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
#     parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)

#     # det args
#     parser.add_argument("-c", "--ckpt", default="pretrained/ocsort_x_mot20.pth.tar", type=str, help="ckpt for eval")  # 检测模型
#     parser.add_argument("--conf", default=0.1, type=float, help="test conf")
#     parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
#     parser.add_argument("--tsize", default=None, type=int, help="test img size")
#     parser.add_argument("--seed", default=None, type=int, help="eval seed")

#     # tracking args
#     parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
#     parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
#     parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
#     parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
#     parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
#     parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#     parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
#     parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
#     parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
#     parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")  # 用不到
#     parser.add_argument("--public", action="store_true", help="use public detection")
#     parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
#     parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")

#     # for kitti/bdd100k inference with public detections
#     parser.add_argument('--raw_results_path', type=str, default="exps/permatrack_kitti_test/",
#         help="path to the raw tracking results from other tracks")
#     parser.add_argument('--out_path', type=str, default="./video/output/yolox_nano_myset_person.mp4", help="path to save output results")  # 推理后输出路径
#     parser.add_argument("--dataset", type=str, default="mot", help="kitti or bdd")
#     parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
#             initializing the tracks (offline).")

#     # for demo video
#     parser.add_argument("--demo_type", default="video", help="demo type, eg. image, video and webcam")
#     parser.add_argument( "--path", default="./videos/input/dancetrack100.mp4", help="path to images or video")  # 视频路径
#     parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")  # 摄像头id
#     parser.add_argument(
#         "--save_result",
#         default="True",
#         action="store_true",
#         help="whether to save the inference result of image/video",
#     )
#     parser.add_argument(
#         "--aspect_ratio_thresh", type=float, default=1.6,
#         help="threshold for filtering out boxes of which aspect ratio are above the given value."
#     )
#     parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
#     parser.add_argument(
#         "--device",
#         default="gpu",
#         type=str,
#         help="device to run our model, can either be cpu or gpu",
#     )

#     # CMC
#     parser.add_argument("--cmc-method", default="orb", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

#     parser.add_argument("--depth_levels", default=3, type=int)  # 距离深度
#     parser.add_argument("--confirm_thresh", default=0.8, type=float)  # 匹配阈值


#     parser.add_argument("--w_assoc_emb", type=float, default=0.75, help="Combine weight for emb cost")
#     parser.add_argument(
#         "--alpha_fixed_emb",
#         type=float,
#         default=0.95,
#         help="Alpha fixed for EMA embedding",
#     )
#     parser.add_argument("--emb_off", action="store_true")
#     parser.add_argument("--cmc_off", action="store_true")
#     parser.add_argument("--aw_off", action="store_true")
#     parser.add_argument("--aw_param", type=float, default=0.5)
#     parser.add_argument("--new_kf_off", action="store_true")
#     parser.add_argument("--grid_off", action="store_true")
#     return parser


## yolox_tiny_nano
# def make_parser():
#     parser = argparse.ArgumentParser("OC-SORT parameters")
#     parser.add_argument("--expn", type=str, default=None)
#     parser.add_argument("-n", "--name", type=str, default=None, help="model name")

#     # distributed
#     parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
#     parser.add_argument("--output_dir", type=str, default="evaldata/trackers/mot_challenge")
#     parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
#     parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
#     parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

#     parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
#     parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
#     parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

#     parser.add_argument(  # 跟踪算法评估器
#         "-f", "--exp_file",
#         default="exps/example/mot/yolox_tiny_myset.py",
#         type=str,
#         help="pls input your expriment description file",
#     )
#     parser.add_argument(
#         "--fp16", dest="fp16",
#         default=True,
#         action="store_true",
#         help="Adopting mix precision evaluating.",
#     )
#     parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn for testing.",)
#     parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
#     parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)  # 自己评估置为False
#     parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
#     parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)

#     # det args
#     parser.add_argument("-c", "--ckpt", default="pretrained/yolox_tiny_myset.pth.tar", type=str, help="ckpt for eval")  # 检测模型
#     parser.add_argument("--conf", default=0.1, type=float, help="test conf")
#     parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
#     parser.add_argument("--tsize", default=None, type=int, help="test img size")
#     parser.add_argument("--seed", default=None, type=int, help="eval seed")

#     # tracking args
#     parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
#     parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
#     parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
#     parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
#     parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
#     parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
#     parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
#     parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
#     parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
#     parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")  # 用不到
#     parser.add_argument("--public", action="store_true", help="use public detection")
#     parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
#     parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")
    
#     # for kitti/bdd100k inference with public detections
#     parser.add_argument('--raw_results_path', type=str, default="exps/permatrack_kitti_test/",
#         help="path to the raw tracking results from other tracks")
#     parser.add_argument('--out_path', type=str, default="./videos/output/yolox_tiny_myset_person.mp4", help="path to save output results")  # 推理后输出路径
#     parser.add_argument("--dataset", type=str, default="mot", help="kitti or bdd")
#     parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
#             initializing the tracks (offline).")

#     # for demo video
#     parser.add_argument("--demo_type", default="video", help="demo type, eg. image, video and webcam")
#     parser.add_argument( "--path", default="./videos/input/person.mp4", help="path to images or video")  # 视频路径
#     parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")  # 摄像头id
#     parser.add_argument(
#         "--save_result",
#         default="True",
#         action="store_true",
#         help="whether to save the inference result of image/video",
#     )
#     parser.add_argument(
#         "--aspect_ratio_thresh", type=float, default=1.6,
#         help="threshold for filtering out boxes of which aspect ratio are above the given value."
#     )
#     parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
#     parser.add_argument(
#         "--device",
#         default="gpu",
#         type=str,
#         help="device to run our model, can either be cpu or gpu",
#     )
#     return parser
