import os
import gc
import time
import torch
import wandb
import random
import traceback
import numpy as np
from collections import OrderedDict
from torch import distributed as dist
from timm.layers import convert_sync_batchnorm
from timm.utils import AverageMeter, distribute_bn
from torch.nn.parallel import DistributedDataParallel as DDP
from einops._torch_specific import allow_ops_in_compiled_graph
import utils
from utils import *
from dataloader import *
from modules import build_model
from engines import build_engine
from options import _parse_args, more_about_config


def main(args, device):
    """
    Main training function that handles cross-validation process
    Args:
        args: Command line arguments
        device: PyTorch device object
    """

    print(args.n_classes)

    utils.seed_torch(args.seed)

    # Load datasets path
    label_path = args.csv_path if args.csv_path else os.path.join(args.dataset_root, 'label.csv')
    if args.cv_fold > 1 and not os.path.isdir(label_path):
        if args.datasets.lower().startswith('surv'):
            df = get_patient_label_surv(args, label_path)
        else:
            df = get_patient_label(args, label_path)

        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        if args.cv_fold > 1:
            train_dfs, test_dfs, val_dfs = get_kfold(args, args.cv_fold, df, val_ratio=args.val_ratio)

    # Initialize metrics storage based on datasets type
    if args.datasets.lower().startswith('surv'):
        cindex, cindex_std, te_cindex = [], [], []
        cindex_ema = cindex_ema_std = []
        ckc_metric = [cindex, cindex_std]
        ckc_metric_ema = [cindex_ema, cindex_ema_std]
        te_ckc_metric = [te_cindex]
    else:
        acs, pre, rec, fs, auc, ck, acs_m, te_auc, te_fs = [], [], [], [], [], [], [], [], []
        acs_std, auc_std, fs_std, ck_std, acs_m_std = [], [], [], [], []
        acs_ema, pre_ema, rec_ema, fs_ema, auc_ema, acs_ema_std, auc_ema_std, fs_ema_std, ck_ema, acs_m_ema, ck_ema_std, acs_m_ema_std = [], [], [], [], [], [], [], [], [], [], [], []
        ckc_metric = [auc, acs, pre, rec, fs, ck, acs_m, acs_std, auc_std, fs_std, ck_std, acs_m_std]
        ckc_metric_ema = [auc_ema, acs_ema, pre_ema, rec_ema, fs_ema, ck_ema, acs_m_ema, acs_ema_std, auc_ema_std,
                          fs_ema_std, ck_ema_std, acs_m_ema_std]
        te_ckc_metric = [te_auc, te_fs]

    # Start k-fold cross validation
    for k in range(0, args.cv_fold):
        # Prepare datasets for current fold
        if args.cv_fold > 1 and not os.path.isdir(label_path):
            dataset = {
                'train': train_dfs[k],
                'test': test_dfs[k],
                'val': val_dfs[k]
            }
        else:
            _label_path = os.path.join(args.csv_path, f'fold_{k}.csv') if os.path.isdir(label_path) else args.csv_path
            df = get_data_dfs(args, _label_path)
            # Split datasets using split field
            train_dfs, test_dfs, val_dfs = get_split_dfs(args, df)

            # Create datasets dictionary
            dataset = {
                'train': train_dfs,
                'test': test_dfs,
                'val': val_dfs
            }

        args.fold_curr = k
        if args.rank == 0:
            print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        ckc_metric, te_ckc_metric, ckc_metric_ema = one_fold(args, device, ckc_metric, te_ckc_metric, ckc_metric_ema,
                                                             dataset)

    # Delete checkpoint file after training is complete
    if args.rank == 0:
        if os.path.isfile(os.path.join(args.output_path, 'ckp.pt')):
            os.remove(os.path.join(args.output_path, 'ckp.pt'))

    # Log validation metrics summary to wandb
    if args.wandb and args.rank == 0:
        if args.datasets.lower().startswith('surv'):
            wandb.log({
                "cross_val/cindex_mean": np.mean(np.array(cindex)),
                "cross_val/cindex_std_mean": np.mean(np.array(cindex_std)),
                "cross_val/cindex_std": np.std(np.array(cindex)),
            })
        else:
            wandb.log({
                "cross_val/acc_mean": np.mean(np.array(acs)),
                "cross_val/auc_mean": np.mean(np.array(auc)),
                "cross_val/f1_mean": np.mean(np.array(fs)),
                "cross_val/acc_std_mean": np.mean(np.array(acs_std)),
                "cross_val/auc_std_mean": np.mean(np.array(auc_std)),
                "cross_val/f1_std_mean": np.mean(np.array(fs_std)),
                "cross_val/pre_mean": np.mean(np.array(pre)),
                "cross_val/recall_mean": np.mean(np.array(rec)),
                "cross_val/ck_mean": np.mean(np.array(ck)),
                "cross_val/acc_micro_mean": np.mean(np.array(acs_m)),
                "cross_val/acc_std": np.std(np.array(acs)),
                "cross_val/auc_std": np.std(np.array(auc)),
                "cross_val/f1_std": np.std(np.array(fs)),
                "cross_val/pre_std": np.std(np.array(pre)),
                "cross_val/recall_std": np.std(np.array(rec)),
                "cross_val/ck_std": np.std(np.array(ck)),
                "cross_val/acc_micro_std": np.std(np.array(acs_m)),
            })

    # Log EMA model metrics to wandb if available
    if args.wandb and args.rank == 0:
        if args.datasets.lower().startswith('surv'):
            wandb.log({
                "cross_val/cindex_ema_mean": np.mean(np.array(cindex_ema)),
                "cross_val/cindex_ema_std_mean": np.mean(np.array(cindex_ema_std)),
                "cross_val/cindex_ema_std": np.std(np.array(cindex_ema)),
            })
        else:
            wandb.log({
                "cross_val/acc_ema_mean": np.mean(np.array(acs_ema)),
                "cross_val/auc_ema_mean": np.mean(np.array(auc_ema)),
                "cross_val/f1_ema_mean": np.mean(np.array(fs_ema)),
                "cross_val/ck_ema_mean": np.mean(np.array(ck_ema)),
                "cross_val/acc_micro_ema_mean": np.mean(np.array(acs_m_ema)),
                "cross_val/acc_ema_std_mean": np.mean(np.array(acs_ema_std)),
                "cross_val/auc_ema_std_mean": np.mean(np.array(auc_ema_std)),
                "cross_val/f1_ema_std_mean": np.mean(np.array(fs_ema_std)),
                "cross_val/ck_ema_std_mean": np.mean(np.array(ck_ema_std)),
                "cross_val/acc_micro_ema_std_mean": np.mean(np.array(acs_m_ema_std)),
                "cross_val/pre_ema_mean": np.mean(np.array(pre_ema)),
                "cross_val/recall_ema_mean": np.mean(np.array(rec_ema)),
                "cross_val/acc_ema_std": np.std(np.array(acs_ema)),
                "cross_val/auc_ema_std": np.std(np.array(auc_ema)),
                "cross_val/f1_ema_std": np.std(np.array(fs_ema)),
                "cross_val/pre_ema_std": np.std(np.array(pre_ema)),
                "cross_val/recall_ema_std": np.std(np.array(rec_ema)),
                "cross_val/ck_ema_std": np.std(np.array(ck_ema)),
                "cross_val/acc_micro_ema_std": np.std(np.array(acs_m_ema)),
            })

    # Print final cross validation results
    if args.rank == 0:
        if args.datasets.lower().startswith('surv'):
            print('Cross validation c-index mean: %.4f, std %.4f ' % (
            np.mean(np.array(cindex)), np.std(np.array(cindex))))

        else:
            print('Cross validation accuracy mean: %.4f, std %.4f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
            print('Cross validation auc mean: %.4f, std %.4f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
            print('Cross validation precision mean: %.4f, std %.4f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
            print('Cross validation recall mean: %.4f, std %.4f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
            print('Cross validation fscore mean: %.4f, std %.4f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))
            print('Cross validation accuracy_micro mean: %.4f, std %.4f ' % (np.mean(np.array(acs_m)), np.std(np.array(acs_m))))

            if args.test_type != 'main':
                print('Cross validation accuracy mean: %.4f, std %.4f ' % (
                np.mean(np.array(acs_ema)), np.std(np.array(acs_ema))))
                print('Cross validation auc mean: %.4f, std %.4f ' % (
                np.mean(np.array(auc_ema)), np.std(np.array(auc_ema))))


def one_fold(args, device, ckc_metric, te_ckc_metric, ckc_metric_ema, dataset):
    """
    Train and evaluate model for a single fold
    Args:
        args: Command line arguments
        device: PyTorch device object
        ckc_metric: List to store metrics for current model
        te_ckc_metric: List to store test metrics
        ckc_metric_ema: List to store metrics for EMA model
        dataset: Dictionary containing train/val/test datasets
    Returns:
        Updated metrics lists after this fold's training
    """
    # --->initiation
    if args.random_seed:
        args.seed = args.seed_ori + args.fold_curr * 100
    utils.seed_torch(args.seed + args.rank)
    torch.cuda.empty_cache()
    gc.collect()

    loss_scaler = torch.amp.GradScaler(device=device, enabled=not args.amp_unscale)

    amp_autocast = torch.autocast

    # --->build data loader
    train_loader, val_loader, test_loader = build_img_dataloader(args, dataset)

    # --->bulid networks
    model,model_others = build_model(args, device, train_loader)
    model = model.to(memory_format=torch.channels_last)

    model_ema = None

    if 'model_ema' in model_others and model_others['model_ema'] is not None:
        model_others['model_ema'] = model_others['model_ema'].to(memory_format=torch.channels_last)

    if args.distributed:
        model = convert_sync_batchnorm(model)

    # --->DDP
    if args.distributed:
        model = DDP(model, device_ids=None if args.load_gpu_later or args.load_gpu_later_train else [device],
                    broadcast_buffers=True) #, find_unused_parameters=True
        if 'model_ema' in model_others:
            state_dict = model.state_dict()

            model_others['model_ema'].load_state_dict(state_dict)

    # --->build criterion,optimizer,scheduler,early-stopping
    criterion, optimizer, scheduler, early_stopping = build_train(args, model, train_loader)

    # --->metric
    best_ckc_metric = [0. for i in range(len(ckc_metric))]
    best_ckc_metric_ema = [0. for i in range(len(ckc_metric))]
    best_ckc_metric_te = [0. for i in range(len(te_ckc_metric))]
    best_ckc_metric_te_ema = [0. for i in range(len(te_ckc_metric))]
    epoch_start, opt_thr, opt_main_ema = 0, 0, 0
    _metric_val_ema = None

    # --->build engine
    train, validate = build_engine(args, device)

    # --->train loop
    train_time_meter = AverageMeter()
    valid_time_meter = AverageMeter()

    try:
        for epoch in range(0, args.num_epoch):

            torch.cuda.empty_cache()
            gc.collect()
            train_loss, start, end = 0, 0, 0
            # if hasattr(dataset_train, 'set_epoch'):
            #     dataset_train.set_epoch(epoch)
            if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)

            train_loss, start, end = train(args, model, model_ema, train_loader, optimizer, device,
                                           amp_autocast, criterion, loss_scaler, scheduler, epoch, model_others)
            train_time_meter.update(end - start)

            _metric_val, stop, test_loss, threshold_optimal, rowd_val = validate(args, model=model,
                                                                                    loader=val_loader, device=device,
                                                                                    criterion=criterion,
                                                                                    amp_autocast=amp_autocast,
                                                                                    early_stopping=early_stopping,
                                                                                    epoch=epoch, others=model_others)
            
            # _metric_val, stop, test_loss, threshold_optimal, rowd_val, total_time, inference_time = validate(args, model=model,
            #                                                                         loader=val_loader, device=device,
            #                                                                         criterion=criterion,
            #                                                                         amp_autocast=amp_autocast,
            #                                                                         early_stopping=early_stopping,
            #                                                                         epoch=epoch, others=model_others)
            

            if args.test_type != 'main':
                _metric_val, _metric_val_ema = _metric_val

            # always run test_set in the training
            _te_metric = [0., 0.]

            # logging and wandb
            if args.rank == 0:
                if args.datasets.lower().startswith('surv'):
                        print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, c-index: %.4f, train_time: %.4f(%.4f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[0], train_time_meter.val,train_time_meter.avg))
                else:
                        print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.4f, auc_value:%.4f, precision: %.4f, recall: %.4f, fscore: %.4f , train_time: %.4f(%.4f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[6], _metric_val[0], _metric_val[2], _metric_val[3], _metric_val[4], train_time_meter.val,train_time_meter.avg))
            # if args.rank == 0:
            #     if args.datasets.lower().startswith('surv'):
            #             print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, c-index: %.4f, train_time: %.4f(%.4f), valid_time: %.4f(%.4f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[0], train_time_meter.val,train_time_meter.avg, total_time,inference_time))
            #     else:
            #             print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.4f, auc_value:%.4f, precision: %.4f, recall: %.4f, fscore: %.4f , train_time: %.4f(%.4f), valid_time: %.4f(%.4f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[6], _metric_val[0], _metric_val[2], _metric_val[3], _metric_val[4], train_time_meter.val,train_time_meter.avg, total_time,inference_time))
            rowd_val['epoch'] = epoch
            if args.wandb and args.rank == 0:
                rowd = OrderedDict([(str(args.fold_curr) + '-fold/val_' + _k, _v) for _k, _v in rowd_val.items()])
                wandb.log(rowd)

            # update best metric
            if epoch == 0:
                best_rowd = rowd_val
            else:
                best_rowd = utils.update_best_metric(best_rowd, rowd_val)

            # save the best model in the val_set
            if _metric_val[args.best_metric_index] > best_ckc_metric[args.best_metric_index]:
                best_ckc_metric = _metric_val + [epoch]
                best_rowd['epoch'] = epoch
                opt_thr = threshold_optimal
                if not os.path.exists(args.output_path):
                    os.mkdir(args.output_path)
                if args.rank == 0:
                    _ema_ckp =  None
                    if 'model_ema' in model_others and model_others['model_ema'] is not None:
                        _ema_ckp = model_others['model_ema'].state_dict()
                    best_pt = {
                        'model': model.state_dict(),
                        'teacher': _ema_ckp,
                        'epoch': epoch
                    }
                    torch.save(best_pt, os.path.join(args.output_path,'fold_{fold}_model_best.pt'.format(fold=args.fold_curr)))

            # save the best model_ema in the val_set
            if _metric_val_ema is not None:
                if _metric_val_ema[args.best_metric_index] > best_ckc_metric_ema[args.best_metric_index]:
                    best_ckc_metric_ema = _metric_val_ema + [epoch]
                    if not os.path.exists(args.output_path):
                        os.mkdir(args.output_path)
                    if args.rank == 0:
                        _ema_ckp =  None
                        if 'model_ema' in model_others and model_others['model_ema'] is not None:
                            _ema_ckp = model_others['model_ema'].state_dict()
                        best_pt = {
                            'teacher': _ema_ckp,
                            'epoch': epoch
                        }
                        torch.save(best_pt, os.path.join(args.output_path, 'fold_{fold}_ema_model_best.pt'.format(fold=args.fold_curr)))

            if args.wandb and args.rank == 0:
                rowd = OrderedDict(
                    [(str(args.fold_curr) + '-fold/val_best_' + _k, _v) for _k, _v in best_rowd.items()])
                wandb.log(rowd)

            # save checkpoint
            utils.save_cpk(args, model, random, train_loader, scheduler, optimizer, epoch, early_stopping,
                           _metric_val, _te_metric, best_ckc_metric, best_ckc_metric_te, best_ckc_metric_te_ema,
                           wandb)

            if stop:
                break
    except KeyboardInterrupt:
        pass

    if args.distributed:
        dist.barrier()

    best_std = None
    best_std_ema = None
    if os.path.exists(os.path.join(args.output_path, f'fold_{args.fold_curr}_model_best.pt')):
        best_std = torch.load(
            os.path.join(args.output_path, 'fold_{fold}_model_best.pt'.format(fold=args.fold_curr)),
            map_location='cpu', weights_only=True)

    if os.path.exists(os.path.join(args.output_path, f'fold_{args.fold_curr}_ema_model_best.pt')):
        best_std_ema = torch.load(
            os.path.join(args.output_path, 'fold_{fold}_ema_model_best.pt'.format(fold=args.fold_curr)),
            map_location='cpu', weights_only=True)

    if best_std is not None:
        info = model.load_state_dict(best_std['model'])
        if args.rank == 0:
            print(f"Epoch {best_std['epoch']} Main Model Loaded: {info}")

    if args.test_type != 'main':
        info = model_others['model_ema'].load_state_dict(best_std_ema['teacher'])
        if args.rank == 0:
            print(f"Epoch {best_std_ema['epoch']} EMA Model Loaded: {info}")

    metric_test, test_loss_log, rowd = validate(args, model=model, loader=test_loader, device=device,
                                                criterion=criterion, amp_autocast=amp_autocast, status='test',
                                                others=model_others)

    if args.wandb and args.rank == 0:
        rowd = OrderedDict([('test_' + _k, _v) for _k, _v in rowd.items()])
        wandb.log(rowd)

    if args.test_type != 'main':
        metric_test, metric_test_ema = metric_test
        [ckc_metric_ema[i].append(metric_test_ema[i]) for i, _ in enumerate(ckc_metric_ema)]

    # update metric for each-fold
    [ckc_metric[i].append(metric_test[i]) for i, _ in enumerate(ckc_metric)]

    return ckc_metric, te_ckc_metric, ckc_metric_ema


if __name__ == '__main__':

    start_time = time.time()
    args, args_text = _parse_args()
    args, device = more_about_config(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if 'WANDB_MODE' in os.environ:
        if os.environ["WANDB_MODE"] == "offline":
            _output_wandb_cache = os.path.join(args.output_path, 'wandb_cache', args.project, args.title)

    if not os.path.exists(os.path.join(args.output_path, args.project)) and args.rank == 0:
        os.mkdir(os.path.join(args.output_path, args.project))
    args.output_path = os.path.join(args.output_path, args.project, args.title)
    if not os.path.exists(args.output_path) and args.rank == 0:
        os.mkdir(args.output_path)

    if args.wandb and args.rank == 0:
        utils.check_and_commit_changes(args)

        _output = args.output_path

        wandb.init(project=args.project, name=args.title, config=args, dir=os.path.join(_output))

        # Get the project root directory
        args.project_root = os.path.dirname(os.path.abspath(__file__))

        # Use os.path.join to create absolute paths
        wandb.save(os.path.join(args.project_root, 'main.py'), base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'options.py'), base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'engines/base_engine.py'), base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'modules/__init__.py'), base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'datasets/data_utils.py'), base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'modules/vit_mil.py'), base_path=args.project_root, policy='now')

        if args.config:
            for _config in args.config:
                wandb.save(os.path.join(args.project_root, f'config/{_config}'), base_path=args.project_root,policy='now')

        wandb.save(os.path.join(args.project_root, 'datasets/dataset.py'), base_path=args.project_root,policy='now')
        wandb.save(os.path.join(args.project_root, 'engines/e2e.py'), base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'modules/e2e.py'), base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'modules/e2e_pooling.py'), base_path=args.project_root, policy='now')

        model_name = args.model.split('_')[-1]
        wandb.save(os.path.join(args.project_root, f'modules/{model_name}.py'), base_path=args.project_root, policy='now')


    if args.rank == 0:
        os.system(f'cat {args.dataset_root} > /dev/null')
    args.env = args.env_train = args.dataset_root
    if int(args.img_size) == 256:
        args.env_train = args.env_train.replace('224', '256')
        if args.dataset_sub_root is not None:
            args.dataset_sub_root = args.dataset_sub_root.replace('224', '256')

    localtime = time.asctime(time.localtime(time.time()))
    if args.rank == 0:
        print(localtime)
    try:
        main(args=args, device=device)
    except:
        traceback.print_exc()
    finally:
        if args.rank == 0:
            end_time = time.time()
            run_time = end_time - start_time

            hours, rem = divmod(run_time, 3600)
            minutes, seconds = divmod(rem, 60)
            readable_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

            print(f"总运行时间: {readable_time}")

            if args.wandb:
                wandb.log({"total_runtime_seconds": run_time})
                wandb.log({"total_runtime_human": readable_time})
                wandb.finish(quiet=True)