from loguru import logger
import sys
sys.path.insert(0, 'D:/Code/python/DeepLearning/track/BAM-SORT/')
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluatorDance as MOTEvaluator

from utils.args import make_parser
from tools.mota import eval

import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
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

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():  # k: 文件名, tsacc: 文件内容
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names  # 输出每一个文件对应的跟踪指标


@logger.catch
def main(exp, args, num_gpu):
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True
    rank = args.local_rank

    

    result_dir = "{}_{}_results".format(args.expn, exp.eval_mode)
    file_name = os.path.join(exp.output_dir, result_dir)  # evaldata/trackers/DanceTracker/yolox_x_dancetrack_val
    file_name = str(increment_path(file_name, exist_ok=False))  # 对已有的文件进行评估，需要注释
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)
    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)  # 评估集加载
    evaluator = MOTEvaluator(  # 评估器初始化
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    results_folder = os.path.join(file_name, "data")
    os.makedirs(results_folder, exist_ok=True)

    # start tracking  评估  跟踪算法加载 + YOLOX模型的评估 + OC-SORT 跟踪轨迹写入到 results_folder 文件夹中
    # 如果已有跟踪轨迹data文件，可注释，直接运行结果
    *_, summary = evaluator.evaluate_ocsort(
            model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
    # *_, summary = evaluator.evaluate_hybird_sort_reid(
    #         args, model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    # )
    # if we evaluate on validation set, 
    logger.info("\n" + summary)

    if args.test:
        # we skip evaluation for inference on test set
        return 
    eval(results_folder, f"datasets/{exp.dataset}/val", "")  # evaluate MOTA


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    exp.output_dir = args.output_dir
    args.dataset_type = exp.dataset_type
    if not args.expn:
        args.expn = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )