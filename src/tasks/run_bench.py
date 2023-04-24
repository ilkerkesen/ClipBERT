import torch
import json
import random
from transformers import BertConfig, BertTokenizerFast
from src.modeling.modeling import ClipBertForPreTraining, ClipBertForVideoTextRetrieval

from src.modeling.e2e_model import ClipBert

from src.datasets.dataset_video_retrieval import (
    ClipBertVideoRetrievalDataset,
    VideoRetrievalCollator,
    ClipBertVideoRetrievalEvalDataset,
)
from src.datasets.dataset_vl_bench import Dataset_v1, process_path
from src.datasets.dataloader import InfiniteIterator, PrefetchLoader
from src.datasets.data_utils import ImageNorm, mk_input_group
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from src.configs.config import shared_configs
from src.utils.misc import set_random_seed, NoOp, zero_none_grad
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter
from src.utils.basic_utils import (
    load_jsonl, load_json, save_json, get_rounded_percentage)
from src.utils.load_save import (ModelSaver,
                                 save_training_meta,
                                 load_state_dict_with_mismatch)
from src.utils.load_save import E2E_TrainingRestorer as TrainingRestorer
from src.optimization.sched import get_lr_sched
from src.optimization.utils import setup_e2e_optimizer

import numpy as np
from tqdm import tqdm
from os.path import join, exists
from easydict import EasyDict as edict
from apex import amp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
# import horovod.torch as hvd
from src.utils.distributed import all_gather_list
from collections import defaultdict

from torchvision.io import read_video
from src.datasets.data_utils import ImageResize, ImagePad, image_to_tensor


def mk_video_ret_datalist(raw_datalist, cfg):
    """
    Args:
        raw_datalist: list(dict)
        cfg:

    Returns:

    """
    LOGGER.info(f"Loaded data size {len(raw_datalist)}")
    if cfg.data_ratio != 1.0:
        random.shuffle(raw_datalist)
        raw_datalist = raw_datalist[:int(len(raw_datalist) * cfg.data_ratio)]
        LOGGER.info(f"Use {100 * cfg.data_ratio}% of the loaded data: {len(raw_datalist)}")

    datalist = []
    qid = 0
    for raw_d in raw_datalist:
        d = dict(
            id=qid,
            txt=raw_d["caption"],
            vid_id=raw_d["clip_name"]
        )
        qid += 1
        datalist.append(d)
    LOGGER.info(f"datalist {len(datalist)}")
    return datalist


def mk_video_ret_dataloader(anno_path, lmdb_dir, cfg, tokenizer, is_train=True):
    """"""
    raw_datalist = load_jsonl(anno_path)
    datalist = mk_video_ret_datalist(raw_datalist, cfg)
    grouped = defaultdict(list)  # examples grouped by image/video id
    for d in datalist:
        grouped[d["vid_id"]].append(d)
    LOGGER.info(f"grouped {len(grouped)}")

    # each group has a single image with multiple questions
    group_datalist = mk_input_group(
        grouped,
        max_n_example_per_group=cfg.max_n_example_per_group if is_train else 1,  # force 1 in eval,
        is_train=is_train
    )
    LOGGER.info(f"group_datalist {len(group_datalist)}")

    frm_sampling_strategy = cfg.frm_sampling_strategy
    if not is_train and frm_sampling_strategy == "rand":
        frm_sampling_strategy = "middle"
    dataset = ClipBertVideoRetrievalDataset(
        datalist=group_datalist,
        tokenizer=tokenizer,
        img_lmdb_dir=lmdb_dir,
        max_img_size=cfg.max_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        frm_sampling_strategy=frm_sampling_strategy,
        itm_neg_size=cfg.itm_neg_size,
        ensemble_n_clips=cfg.train_n_clips,
        random_sample_clips=cfg.random_sample_clips
    )
    LOGGER.info(f"is_train {is_train}, dataset size {len(dataset)} groups, "
                f"each group {cfg.max_n_example_per_group if is_train else 1}")
    if cfg.do_inference:
        batch_size = cfg.inference_batch_size
    else:
        batch_size = cfg.train_batch_size if is_train else cfg.val_batch_size

    vqa_collator = VideoRetrievalCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            # sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=vqa_collator.collate_batch)
    return dataloader


def mk_video_ret_eval_dataloader(anno_path, lmdb_dir, cfg, tokenizer):
    """
    eval_retrieval: bool, will sample one video per batch paired with multiple text.
    Returns:

    """
    raw_datalist = load_jsonl(anno_path)
    datalist = mk_video_ret_datalist(raw_datalist, cfg)
    frm_sampling_strategy = cfg.frm_sampling_strategy
    if frm_sampling_strategy == "rand":
        frm_sampling_strategy = "middle"
    dataset = ClipBertVideoRetrievalEvalDataset(
        datalist=datalist,
        tokenizer=tokenizer,
        img_lmdb_dir=lmdb_dir,
        max_img_size=cfg.max_img_size,
        max_txt_len=cfg.max_txt_len,
        fps=cfg.fps,
        num_frm=cfg.num_frm,
        frm_sampling_strategy=frm_sampling_strategy,
        ensemble_n_clips=cfg.inference_n_clips,
    )

    retrieval_collator = VideoRetrievalCollator(
        tokenizer=tokenizer, max_length=cfg.max_txt_len)
    dataloader = DataLoader(dataset,
                            batch_size=1,  # already batched in dataset
                            shuffle=False,
                            # sampler=sampler,
                            num_workers=cfg.n_workers,
                            pin_memory=cfg.pin_mem,
                            collate_fn=retrieval_collator.collate_batch)
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    dataloader = PrefetchLoader(dataloader, img_norm)
    return dataloader


def setup_dataloaders(cfg, tokenizer):
    LOGGER.info("Init. train_loader and val_loader...")
    train_loader = mk_video_ret_dataloader(
        anno_path=cfg.train_datasets[0].txt,
        lmdb_dir=cfg.train_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer, is_train=True
    )
    val_loader = mk_video_ret_dataloader(
        anno_path=cfg.val_datasets[0].txt,
        lmdb_dir=cfg.val_datasets[0].img,
        cfg=cfg, tokenizer=tokenizer, is_train=False
    )
    img_norm = ImageNorm(mean=cfg.img_pixel_mean, std=cfg.img_pixel_std)
    train_loader = PrefetchLoader(train_loader, img_norm)
    val_loader = PrefetchLoader(val_loader, img_norm)
    return train_loader, val_loader


def setup_model(cfg, device=None):
    LOGGER.info("Setup model...")
    # has to be a BertConfig instance
    model_cfg = load_json(cfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add downstream model config
    add_attr_list = [
        "num_labels", "classifier", "cls_hidden_scale",
        "loss_type", "margin",
    ]
    # for k in add_attr_list:
    #    setattr(model_cfg, k, cfg[k])

    # FIXME: take a look at this
    # transformer_model_cls = ClipBertForVideoTextRetrieval
    transformer_model_cls = ClipBertForPreTraining

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    LOGGER.info("setup e2e model")
    model = ClipBert(
        model_cfg, input_format=cfg.img_input_format,
        detectron2_model_cfg=cfg.detectron2_model_cfg,
        transformer_cls=transformer_model_cls)

    cfg.e2e_weights_path = '/pretrain/clipbert_image_text_pretrained.pt'
    if cfg.e2e_weights_path:
        LOGGER.info(f"Loading e2e weights from {cfg.e2e_weights_path}")
        load_state_dict_with_mismatch(model, cfg.e2e_weights_path)
    else:
        LOGGER.info(f"Loading cnn weights from {cfg.detectron2_weights_path}")
        LOGGER.info(f"Loading bert weights from {cfg.bert_weights_path}")
        model.load_separate_ckpt(
            cnn_weights_path=cfg.detectron2_weights_path,
            bert_weights_path=cfg.bert_weights_path)
    if cfg.freeze_cnn:
        model.freeze_cnn_backbone()
    model.to(device)

    LOGGER.info("Setup model done!")
    return model


def forward_step(model, batch, cfg):
    """shared for training and validation"""
    outputs = model(batch)  # dict
    return outputs


@torch.no_grad()
def start_demo(cfg):
    set_random_seed(cfg.seed)
    n_gpu = 1
    device = torch.device(cfg.device)
    model = setup_model(cfg, device=device)
    model.eval()
    model = amp.initialize(model, enabled=cfg.fp16, opt_level='O2')
    global_step = 0
    tokenizer = BertTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    cfg.data_ratio = 1.

    data = Dataset_v1(
        cfg.json_path,
        quva_dir=cfg.quva_dir,
        something_something_dir=cfg.something_something_dir,
        num_frames=cfg.num_frames,
        tokenizer=tokenizer,
        proficiency=cfg.proficiency,
    )

    num_examples = num_correct = 0
    results = dict()
    for item in tqdm(data):
        batch = dict(
            visual_inputs=item['visual_inputs'],
            text_input_ids=item['text_input_ids'],
            text_input_mask=item['text_input_mask'],
            mlm_labels=None,
            # itm_labels=torch.tensor([0, 1]).to(device),
            n_examples_list=item['n_examples_list'],
        )
        output = forward_step(model, batch, cfg)
        probabilities = output['itm_scores'].softmax(dim=-1)
        item_id = item['item_id']
        results[item_id] = {'scores': probabilities[:, 0].tolist()}

    with open(process_path(cfg['output_file']), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    input_cfg = shared_configs.get_bench_args()
    input_cfg.num_labels = 2
    start_demo(input_cfg)
