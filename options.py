import yaml
import torch
import argparse
from timm.utils import init_distributed_device

# The initial parser exclusively handles the --config argument, which specifies a YAML file
# containing configuration values that will override the default arguments in the main parser
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='E2E MIL Training Script')

##### Dataset
group = parser.add_argument_group('Dataset')
# Paths
parser.add_argument('--dataset_root', default='/data/xxx/TCGA', type=str, help='Dataset root path')
parser.add_argument('--dataset_sub_root', default=None, type=str, help='Dataset root path for multi-scale input')
group.add_argument('--csv_path', default=None, type=str, help='Dataset CSV path for Label and Split')
# Dataset settings
group.add_argument('--datasets', default='panda', type=str, help='Dataset')
group.add_argument('--val_ratio', default=0., type=float, help='Validation set ratio')
group.add_argument('--cv_fold', default=5, type=int, help='Number of cross validation folds [5]')
group.add_argument('--img_size', default=224, type=int, help='Image size [256 or 224]')
group.add_argument('--val2test', action='store_true', help='Use validation set as test set')
group.add_argument('--random_seed', action='store_true', help='Enable random seed for multi-fold validation')
# Multi-scale input settings
group.add_argument('--mul_scale_input', action='store_true', help='Enable multi-scale input')
group.add_argument('--mul_scale_ratio', default=2., type=float, help='Multi-scale ratio [2.0]')
# Patch size settings
group.add_argument('--same_psize', default=0., type=float, help='Keep the same size for all patches [0]')
group.add_argument('--same_psize_pad_type', default='zero', type=str, choices=['zero', 'random', 'none'])
group.add_argument('--same_psize_ratio', default=0., type=float, help='Ratio for same patch size [0]')
# Dataloader settings
group.add_argument('--num_workers', default=6, type=int, help='Number of dataloader workers')
group.add_argument('--pin_memory', action='store_true', help='Enable pinned memory')
group.add_argument('--prefetch_factor', default=2, type=int, help='Prefetch factor [2]')
group.add_argument('--persistence', action='store_true', help='Enable persistent datasets caching')
# Image transformation settings
group.add_argument('--img_transform', default='none', type=str, help='Image transformation type')
group.add_argument('--img_trans_chunk', default=4, type=int, help='Image transformation chunk size')
group.add_argument('--crop_scale', default=0.08, type=float, help='Crop scale for transformation')

##### Training
group = parser.add_argument_group('Training')
group.add_argument('--num_epoch', default=200, type=int, help='Total number of training epochs [200]')
group.add_argument('--early_stopping', action='store_false', help='Enable early stopping')
group.add_argument('--max_epoch', default=130, type=int, help='Maximum training epochs for early stopping [130]')
group.add_argument('--warmup_epochs', default=0, type=int, help='Number of warmup epochs [0]')
group.add_argument('--patient', default=20, type=int, help='Patience epochs for early stopping [20]')
group.add_argument('--input_dim', default=1024, type=int, help='Input feature dimension (r18:256, r50:1024)')
group.add_argument('--n_classes', default=2, type=int, help='Number of classes(subtyping:2, survival:4, grading:6)')
group.add_argument('--batch_size', default=1, type=int, help='Batch size')
group.add_argument('--max_patch_train', default=None, type=int, help='Maximum patches per training batch')
group.add_argument('--p_batch_size', default=512, type=int, help='Patch batch size')
group.add_argument('--p_batch_size_v', default=2048, type=int, help='Patch batch size for validation')
group.add_argument('--loss', default='ce', type=str, choices=['ce', 'bce', 'progressive', 'nll_surv'],help='Loss function type')
group.add_argument('--opt', default='adam', type=str, help='Optimizer type [adam, adamw]')
group.add_argument('--model', default='r18_abmilx', type=str, help='Model name [(r18, r50v2)_(abmil, abmilx, )]')
group.add_argument('--seed', default=2021, type=int, help='Random seed [2021]')
group.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
group.add_argument('--warmup_lr', default=1e-6, type=float, help='Warmup learning rate [1e-6]')
group.add_argument('--lr_sche', default='cosine', type=str, help='Learning rate scheduler [cosine, step, const]')
group.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [1e-5]')
group.add_argument('--best_metric_index', default=-1, type=int, help='Metric index for model selection')
group.add_argument('--no_drop_last', action='store_true', default=False, help='Disable dropping last incomplete batch')

##### Evaluation
group = parser.add_argument_group('Evaluation')
group.add_argument('--num_bootstrap', default=1000, type=int, help='Number of bootstrap iterations')
group.add_argument('--bootstrap_mode', default='test', type=str, choices=['test', 'none', 'val', 'test_val'])

##### Model
group = parser.add_argument_group('Model')
# General
group.add_argument('--dropout', default=0.25, type=float, help='Dropout rate')
group.add_argument('--mil_norm', default=None, choices=['bn', 'ln', 'none'], help='MIL normalization type')
group.add_argument('--no_mil_bias', action='store_true', help='Disable MIL bias')
group.add_argument('--mil_bias', action='store_true', help='Enable MIL bias')
group.add_argument('--inner_dim', default=512, type=int, help='Inner dimension')
# General Transformer
group.add_argument('--pos', default=None, type=str, choices=['ppeg', 'sincos', 'none', ], help='Position encoding type')
group.add_argument('--n_heads', default=8, type=int, help='Number of attention heads')
group.add_argument('--n_layers', default=2, type=int, help='Number of transformer layers')
group.add_argument('--pool', default='cls_token', type=str, help='Pooling method')
group.add_argument('--attn_dropout', default=0., type=float, help='Attention dropout rate')
group.add_argument('--ffn', action='store_true', help='Enable FFN')
group.add_argument('--sdpa_type', default='torch', type=str, choices=['torch', 'flash', 'math', 'memo_effi', 'torch_math'], help='SDPA implementation type')
group.add_argument('--ffn_dp', default=0., type=float, help='FFN dropout rate')
group.add_argument('--ffn_ratio', default=4., type=float, help='FFN expansion ratio')

##### E2E
group = parser.add_argument_group('E2E')
group.add_argument('--load_gpu_later', action='store_true', help='Load model to GPU later')
group.add_argument('--load_gpu_later_train', action='store_true', help='Load model to GPU later during training')
group.add_argument('--no_enc_pt', action='store_true', help='Disable encoder pretraining [Only for ablation]')
# Patch Selection
group.add_argument('--all_patch_train', action='store_true', help='Train on all patches')
# Test
group.add_argument('--test_type', default='main', type=str, choices=['main', 'ema', 'both', 'both_ema'],help='Test mode type')

##### ABMILX
group.add_argument('--abx_D', default=None, type=float, help='ABX D ratio')
group.add_argument('--abx_attn_bias', action='store_true', help='Enable ABX attention bias')
group.add_argument('--abx_attn_plus', action='store_true', help='Enable ABX attention plus')
##### Miscellaneous
group = parser.add_argument_group('Miscellaneous')
group.add_argument('--title', default='default', type=str, help='Experiment title')
group.add_argument('--project', default='mil_new_c16', type=str, help='Project name')
group.add_argument('--log_iter', default=100, type=int, help='Logging frequency')
group.add_argument('--amp', action='store_true', help='Enable automatic mixed precision training')
group.add_argument('--amp_unscale', action='store_true', help='Enable AMP unscaling')
group.add_argument('--output_path', type=str, help='Output path')
group.add_argument('--model_path', default=None, type=str, help='Model initialization path')
group.add_argument("--local-rank", "--local_rank", type=int)
group.add_argument('--mm', default=0.9997, type=float, help='Decay of EMA')
# Wandb
group.add_argument('--wandb', action='store_true', help='Enable Weights & Biases')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

    cfg = {}
    if args_config.config:
        config_files = args_config.config.split(',')

        for config_file in config_files:
            config_file = config_file.strip()
            if config_file:
                try:
                    with open(config_file, 'r') as f:
                        cfg.update(yaml.safe_load(f))
                except Exception as e:
                    print(f"Error loading config file {config_file}: {str(e)}")

        parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    args.config = args_config.config.split(',')

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text


def more_about_config(args):
    # Additional settings
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device = init_distributed_device(args)

    # Training settings
    if not args.mil_bias:
        args.mil_bias = not args.no_mil_bias
    args.drop_last = not args.no_drop_last

    args.seed_ori = args.seed

    if args.best_metric_index == -1:
        args.best_metric_index = 1 if args.n_classes != 2 and not args.datasets.lower().startswith('surv') else 0

    args.max_epoch = min(args.max_epoch, args.num_epoch)

    return args, device