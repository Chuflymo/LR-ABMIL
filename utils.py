import os
import math
import torch
import random
import pickle
import numpy as np
import pandas as pd
from git import Repo
import torch.nn as nn
from copy import deepcopy
from random import shuffle
from typing import Optional
from functools import partial
from contextlib import suppress
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset
from torch import distributed as dist
from timm.scheduler import create_scheduler_v2
from torchaug import transforms as ta_transforms
from sklearn.model_selection import StratifiedKFold
from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG


global jpeg
jpeg = TurboJPEG()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

def get_patient_label_surv(args,csv_file):
    if args:
        if args.rank == 0:
            print('[datasets] loading datasets from %s' % (csv_file))
    rows = pd.read_csv(csv_file)
    rows = survival_label(rows)

    label_dist = rows['Label'].value_counts().sort_index()

    if args:
        if args.rank == 0:
            print('[datasets] discrete label distribution: ')
            print(label_dist)
            print('[datasets] datasets from %s, number of cases=%d' % (csv_file, len(rows)))

    return rows

def survival_label(rows):
    n_bins, eps = 4, 1e-6
    uncensored_df = rows[rows['Status'] == 1]
    disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = rows['Event'].max() + eps
    q_bins[0] = rows['Event'].min() - eps
    disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    # missing event data
    disc_labels = disc_labels.values.astype(int)
    disc_labels[disc_labels < 0] = -1
    rows.insert(len(rows.columns), 'Label', disc_labels)

    return rows

def get_patient_label(args,csv_file):
    df = pd.read_csv(csv_file)

    if len(df.columns) == 2:
        df.columns = ['ID', 'Label']
    else:
        df.columns = ['Case', 'ID', 'Label', 'Split']

    patients_list = df['ID']
    labels_list = df['Label']

    label_counts = labels_list.value_counts().to_dict()

    if args:
        if args.rank == 0:
            print(f"patient_len:{len(patients_list)} label_len:{len(labels_list)}")
            print(f"all_counter:{label_counts}")

    return df

def get_kfold(args,k, df, val_ratio=0, label_balance_val=True):
    skf = StratifiedKFold(n_splits=k)

    train_dfs = []
    test_dfs = []
    val_dfs = []

    for train_index, test_index in skf.split(df, df['Label']):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        if val_ratio != 0:
            val_df, train_df = data_split(args.seed,train_df, val_ratio, True, label_balance_val)
            if args.val2test:
                test_df = pd.concat([val_df, test_df], axis=0).reset_index(drop=True)
                args.val_ratio = 0.
        else:
            val_df = pd.DataFrame()
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        val_dfs.append(val_df)

    return train_dfs, test_dfs, val_dfs

def data_split(seed,df, ratio, shuffle=True, label_balance_val=True):
    if label_balance_val:
        val_df = pd.DataFrame()
        train_df = pd.DataFrame()

        for label in df['Label'].unique():
            label_df = df[df['Label'] == label]
            n_total = len(label_df)
            offset = int(n_total * ratio)

            if shuffle:
                label_df = label_df.sample(frac=1, random_state=seed)

            val_df = pd.concat([val_df, label_df.iloc[:offset]])
            train_df = pd.concat([train_df, label_df.iloc[offset:]])
    else:
        n_total = len(df)
        offset = int(n_total * ratio)

        if n_total == 0 or offset < 1:
            return pd.DataFrame(), df

        if shuffle:
            df = df.sample(frac=1, random_state=seed)

        val_df = df.iloc[:offset]
        train_df = df.iloc[offset:]

    return val_df, train_df

def get_data_dfs(args, csv_file):
    if args.rank == 0:
        print(f'[datasets] loading datasets from {csv_file}')

    df = pd.read_csv(csv_file)

    required_columns = ['ID', 'Split', 'Label']

    if args.datasets.lower().startswith('surv') and 'Label' not in df.columns:
        df = survival_label(df)

    if not all(col in df.columns for col in required_columns):
        if len(df.columns) == 2:
            from sklearn.model_selection import train_test_split
            df.columns = ['ID', 'Label']
            if args.rank == 0:
                print(f"[datasets] Split column not found in CSV file, splitting data randomly with val_ratio={args.val_ratio}")
            train_indices, test_indices = train_test_split(
                range(len(df)),
                test_size=args.val_ratio,
                random_state=args.seed
            )
            df['Split'] = 'train'
            df.loc[test_indices, 'Split'] = 'test'
            args.val_ratio = 0.
        elif len(df.columns) == 4:
            df.columns = ['Case', 'ID', 'Label', 'Split']
        else:
            raise ValueError(f"CSV file must contain these columns: {required_columns}")

    if args.rank == 0:
        print(f"Total samples: {len(df)}")
        print(df['Label'].value_counts())
        print(df['Split'].value_counts())

    return df

def get_split_dfs(args,df):
    if 'Split' not in df.columns:
        raise ValueError("CSV file must contain a 'Split' column")

    train_df = df[df['Split'].str.lower() == 'train'].reset_index(drop=True)
    test_df = df[df['Split'].str.lower() == 'test'].reset_index(drop=True)
    val_df = df[df['Split'].str.lower() == 'val'].reset_index(drop=True)

    if args.val2test:
        test_df = pd.concat([val_df, test_df], axis=0).reset_index(drop=True)
        args.val_ratio = 0.

    if len(val_df) == 0:
        val_df = test_df

    return train_df, test_df, val_df

# def collate_fn_img_batch(batch):
#     """处理变长patch序列的自定义collate函数"""
#     inputs = []
#     targets = []
    
#     for item in batch:
#         inputs.append(item['input'])  # 每个是 [n_patches, C, H, W]
#         targets.append(item['target'])  # 每个是 [1]
    
#     # 处理targets
#     targets = torch.cat(targets, dim=0)  # [batch_size]
    
#     return {
#         'input': inputs,  # 保持为列表，每个元素可能不同长度
#         'target': targets
#     }

def collate_fn_img_batch(batch):
    first_item = batch[0]
    batch_size = len(batch)

    _images = torch.empty((first_item['input'].size(0) * batch_size, *first_item['input'].shape[1:]),
                          dtype=first_item['input'].dtype,
                          device=first_item['input'].device,
                          memory_format=torch.channels_last)

    labels = []

    for i, item in enumerate(batch):
        _images[i * first_item['input'].size(0):(i + 1) * first_item['input'].size(0)].copy_(item['input'])
        labels.append(item['target'])

    outputs = {
        'input': _images,
        'target': torch.cat(labels, dim=0)
    }

    optional_keys = ['pos', 'idx', 'event', 'censorship']
    for key in optional_keys:
        if key in first_item:
            outputs[key] = torch.cat([item[key] for item in batch if key in item], dim=0)

    return outputs

def collate_fn_nbs(batch):
    return batch[0]

def _jpegflag(flag: str = 'color', channel_order: str = 'bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return TJPF_BGR
        elif channel_order == 'rgb':
            return TJCS_RGB
    elif flag == 'grayscale':
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')

def imfrombytes(content: bytes,
                flag: str = 'color',
                channel_order: str = 'bgr',
                backend: Optional[str] = None) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`. If backend is
            None, the global imread_backend specified by ``mmcv.use_backend()`
            will be used. Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> img_path = '/path/to/img.jpg'
        >>> with open(img_path, 'rb') as f:
        >>>     img_buff = f.read()
        >>> img = mmcv.imfrombytes(img_buff)
        >>> img = mmcv.imfrombytes(img_buff, flag='color', channel_order='rgb')
        >>> img = mmcv.imfrombytes(img_buff, backend='pillow')
        >>> img = mmcv.imfrombytes(img_buff, backend='cv2')
    """

    img = jpeg.decode(content, _jpegflag(flag, channel_order))
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    return img

def get_same_psize_img(args,_list,same_psize,list_random=None):
    assert same_psize > 0

    if same_psize > 0 and same_psize < 1:
        same_psize = same_psize * len(_list)
        same_psize = max(same_psize,args.min_seq_len)
    same_psize = int(same_psize)
    __list = deepcopy(_list)
    ps = len(__list)

    if ps > same_psize:
        shuffle(__list)
        return __list[:same_psize]

    elif ps < same_psize and args.batch_size > 1:
        random_indices = torch.randint(0, ps, (same_psize - ps,))
        padding = [__list[pad_idx] for pad_idx in random_indices]
        __list += padding

    return __list

def seed_torch(seed=2021):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)

def check_and_commit_changes(args):
    if args.rank == 0:
        # Get git repository of current directory
        repo = Repo(os.path.dirname(os.path.abspath(__file__)))
        # Check for uncommitted changes
        has_changes = repo.is_dirty()
        has_untracked = len(repo.untracked_files) > 0

        if has_changes or has_untracked:
            print("Detected uncommitted changes...")
            # Handle untracked files
            if has_untracked:
                print("Untracked files detected...")
                repo.git.add(repo.untracked_files)
                # raise NotImplementedError

            commit_message = "Run Auto commit"
            # Commit all changes
            repo.git.commit('-a', '-m', commit_message)
            print(f"Changes committed: {commit_message}")
        else:
            print("No changes detected")

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

def distributed_concat(tensor, num_sample):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_sample]

def save_cpk(args, model, random, train_loader, scheduler, optimizer, epoch, early_stopping, _metric_val, _te_metric,
             best_ckc_metric, best_ckc_metric_te, best_ckc_metric_te_tea, wandb):
    random_state = {
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'py': random.getstate(),
        'loader': '',
    }
    ckp = {
        'model': model.state_dict(),
        'lr_sche': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'k': args.fold_curr,
        'early_stop': early_stopping.state_dict(),
        'random': random_state,
        'ckc_metric': _metric_val + _te_metric,
        'val_best_metric': best_ckc_metric,
        'te_best_metric': best_ckc_metric_te + best_ckc_metric_te_tea,
    }
    if args.rank == 0:
        torch.save(ckp, os.path.join(args.output_path, 'ckp.pt'))

def update_best_metric(best_metric, val_metric):
    updated_best_metric = best_metric.copy()

    assert set(best_metric.keys()) == set(val_metric.keys())

    for key in val_metric.keys():
        if key == "epoch":
            continue
        if key == "loss":
            if val_metric[key] < best_metric[key]:
                updated_best_metric[key] = val_metric[key]
        else:
            if val_metric[key] > best_metric[key]:
                updated_best_metric[key] = val_metric[key]

    return updated_best_metric

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1)  # censorship status, 0 or 1
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    uncensored_loss = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(
        torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def build_train(args, model, train_loader):
    # build criterion
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'nll_surv':
        criterion = NLLSurvLoss(alpha=0.0)
    elif args.loss == 'progressive':
        criterion = ProgressiveLoss(num_classes=6)
    else:
        raise NotImplementedError

    # build optimizer
    if args.distributed:
        _model = model.module
    else:
        _model = model

    lr = args.lr
    lr_enc = lr

    general_param_names = ('feature', 'norm', 'norm1', 'classifier')
    general_params = [p for name, p in _model.mil.named_parameters() if
                      p.requires_grad and any(name.startswith(sp) for sp in general_param_names)]
    other_params = [p for name, p in _model.mil.named_parameters() if
                    p.requires_grad and not any(name.startswith(sp) for sp in general_param_names)]

    if args.warmup_epochs > 0:
        params = [
            {'params': filter(lambda p: p.requires_grad, _model.encoder.parameters()), 'lr': lr_enc,
             'weight_decay': args.weight_decay},
            # {'params': filter(lambda p: p.requires_grad, model.mil.parameters()), 'lr': lr, 'weight_decay': args.weight_decay},
            {'params': general_params, 'lr': lr, 'weight_decay': args.weight_decay},
            {'params': other_params, 'lr': lr, 'weight_decay': args.weight_decay},
        ]
        # print(list(params[0]['params']))
    else:
        params = [
            {'params': filter(lambda p: p.requires_grad, _model.encoder.parameters()), 'lr': lr_enc,
             'weight_decay': args.weight_decay},
            # {'params': filter(lambda p: p.requires_grad, model.mil.parameters()), 'lr': lr, 'weight_decay': args.weight_decay}
            {'params': general_params, 'lr': lr, 'weight_decay': args.weight_decay},
            {'params': other_params, 'lr': lr, 'weight_decay': args.weight_decay},
        ]

    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params)
    elif args.opt == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(params)
    else:
        raise NotImplementedError

    # build scheduler
    if args.lr_sche == 'cosine':
        scheduler, _ = create_scheduler_v2(optimizer, sched='cosine', num_epochs=args.num_epoch,
                                           warmup_lr=args.warmup_lr, warmup_epochs=args.warmup_epochs, min_lr=1e-7)
    else:
        raise NotImplementedError

    # build early stopping
    if args.early_stopping:
        # if args.datasets.lower().startswith('surv'):
        #     #patience,stop_epoch = 10,args.max_epoch
        # else:
        # patience,stop_epoch = 20,args.max_epoch
        patience, stop_epoch = args.patient, args.max_epoch
        early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch)
    else:
        early_stopping = None

    return criterion, optimizer, scheduler, early_stopping


class NLLSurvLoss(object):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def __call__(self, Y, c, logits=None, hazards=None, S=None, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if hazards is None:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
        return nll_loss(hazards, S, Y, c, alpha=alpha)
    

class ProgressiveLoss(nn.Module):
    def __init__(self, num_classes=6, reduction='sum'):
        super(ProgressiveLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, labels):
        batch_size = labels.size(0)
        cumulative_labels = torch.zeros(batch_size, self.num_classes, 
                                        device=labels.device, dtype=torch.float32)
        
        for i in range(batch_size):
            label_idx = labels[i].item()
            cumulative_labels[i, :label_idx+1] = 1.0
        
        loss_matrix = self.bce_loss(logits, cumulative_labels)
        loss_per_image = loss_matrix.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss_per_image.mean()
        elif self.reduction == 'sum':
            return loss_per_image.sum()
        return loss_per_image


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False, save_best_model_stage=0.):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        try:
            self.val_loss_min = np.Inf
        except:
            self.val_loss_min = np.inf
        self.save_best_model_stage = save_best_model_stage

    def __call__(self, args, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss if epoch >= self.save_best_model_stage else 0.

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            if args.rank == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def state_dict(self):
        return {
            'patience': self.patience,
            'stop_epoch': self.stop_epoch,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }

    def load_state_dict(self, dict):
        self.patience = dict['patience']
        self.stop_epoch = dict['stop_epoch']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        self.early_stop = dict['early_stop']
        self.val_loss_min = dict['val_loss_min']

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss