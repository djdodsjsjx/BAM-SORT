import argparse

# yolox_s_MOT20
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
#         default="exps/example/mot/yolox_s_mot20.py",
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
#     parser.add_argument("-c", "--ckpt", default="pretrained/myocsort_yolox_s_mot20.pth.tar", type=str, help="ckpt for eval")
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
#     parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
#     parser.add_argument("--public", action="store_true", help="use public detection")
#     parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
#     parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")
    
#     # for kitti/bdd100k inference with public detections
#     parser.add_argument('--raw_results_path', type=str, default="exps/permatrack_kitti_test/",
#         help="path to the raw tracking results from other tracks")
#     parser.add_argument('--out_path', type=str, help="path to save output results")
#     parser.add_argument("--dataset", type=str, default="mot", help="kitti or bdd")
#     parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
#             initializing the tracks (offline).")

#     # for demo video
#     parser.add_argument("--demo_type", default="image", help="demo type, eg. image, video and webcam")
#     parser.add_argument( "--path", default="./videos/demo.mp4", help="path to images or video")
#     parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
#     parser.add_argument(
#         "--save_result",
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



# ocsort_x_mot17
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
#         default="exps/example/mot/yolox_x_mot17_train.py",
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
#     parser.add_argument("-c", "--ckpt", default="pretrained/ocsort_x_mot17.pth.tar", type=str, help="ckpt for eval")
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
#     parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
#     parser.add_argument("--public", action="store_true", help="use public detection")
#     parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
#     parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")
    
#     # for kitti/bdd100k inference with public detections
#     parser.add_argument('--raw_results_path', type=str, default="exps/permatrack_kitti_test/",
#         help="path to the raw tracking results from other tracks")
#     parser.add_argument('--out_path', type=str, help="path to save output results")
#     parser.add_argument("--dataset", type=str, default="mot", help="kitti or bdd")
#     parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
#             initializing the tracks (offline).")

#     # for demo video
#     parser.add_argument("--demo_type", default="image", help="demo type, eg. image, video and webcam")
#     parser.add_argument( "--path", default="./videos/demo.mp4", help="path to images or video")
#     parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
#     parser.add_argument(
#         "--save_result",
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

# ocsort_x_mot20
def make_parser():
    parser = argparse.ArgumentParser("OC-SORT parameters")
    parser.add_argument("--expn", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--output_dir", type=str, default="evaldata/trackers/mot_challenge")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--devices", default=1, type=int, help="device for training")

    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

    parser.add_argument(  # 跟踪算法评估器
        "-f", "--exp_file",
        default="exps/example/mot/yolox_x_myset.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16", dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)  # 自己评估置为False
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)

    # det args
    parser.add_argument("-c", "--ckpt", default="pretrained/ocsort_x_mot20.pth.tar", type=str, help="ckpt for eval")  # 检测模型
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
    parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
    parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
    parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")  # 用不到
    parser.add_argument("--public", action="store_true", help="use public detection")
    parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
    parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")
    
    # for kitti/bdd100k inference with public detections
    parser.add_argument('--raw_results_path', type=str, default="exps/permatrack_kitti_test/",
        help="path to the raw tracking results from other tracks")
    parser.add_argument('--out_path', type=str, default="./videos/output/yolox_x_mot20myset_fire_person_test.mp4", help="path to save output results")  # 推理后输出路径
    parser.add_argument("--dataset", type=str, default="mot", help="kitti or bdd")
    parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
            initializing the tracks (offline).")

    # for demo video
    parser.add_argument("--demo_type", default="webcam", help="demo type, eg. image, video and webcam")
    parser.add_argument( "--path", default="./videos/input/fireperson.mp4", help="path to images or video")  # 视频路径
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")  # 摄像头id
    parser.add_argument(
        "--save_result",
        default="True",
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
    return parser


# ocsort_dance
# def make_parser():
#     parser = argparse.ArgumentParser("OC-SORT parameters")
#     parser.add_argument("--expn", type=str, default=None)
#     parser.add_argument("-n", "--name", type=str, default=None, help="model name")

#     # distributed
#     parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
#     parser.add_argument("--output_dir", type=str, default="evaldata/trackers/DanceTrack")
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
#     parser.add_argument("-c", "--ckpt", default="pretrained/ocsort_dance_model.pth.tar", type=str, help="ckpt for eval")  # yolox模型文件
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
#     parser.add_argument('--out_path', type=str, default="./videos/output/yolox_x_dance_walk.mp4", help="path to save output results")  # 推理后输出路径
#     parser.add_argument("--dataset", type=str, default="mot", help="kitti or bdd")
#     parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
#             initializing the tracks (offline).")

#     # for demo video
#     parser.add_argument("--demo_type", default="video", help="demo type, eg. image, video and webcam")
#     parser.add_argument( "--path", default="./videos/input/walk.mp4", help="path to images or video")  # 视频路径
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
#         default="exps/example/mot/yolox_nano_mot.py",
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
#     parser.add_argument("-c", "--ckpt", default="pretrained/yolox_nano_mot20.pth.tar", type=str, help="ckpt for eval")  # 检测模型
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
#     parser.add_argument('--out_path', type=str, default="./video/output/320240.mp4", help="path to save output results")  # 推理后输出路径
#     parser.add_argument("--dataset", type=str, default="mot", help="kitti or bdd")
#     parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
#             initializing the tracks (offline).")

#     # for demo video
#     parser.add_argument("--demo_type", default="webcam", help="demo type, eg. image, video and webcam")
#     parser.add_argument( "--path", default="./videos/input/320240.avi", help="path to images or video")  # 视频路径
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
