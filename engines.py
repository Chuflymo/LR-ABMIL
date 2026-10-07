import time
import torch
import wandb
from utils import *
from functools import partial
from contextlib import suppress
from metrics import get_metric_val
from timm.utils import unwrap_model
from collections import OrderedDict
from timm.models import model_parameters
from torch.profiler import profile, ProfilerActivity
from timm.utils import AverageMeter, dispatch_clip_grad


def build_engine(args,device):

    engine = E2E(args,device)

    trainer = BaseTrainer(engine=engine,args=args)

    if args.datasets.lower().startswith('surv'):
        return trainer.surv_train,trainer.surv_validate
    else:
        return trainer.train,trainer.validate


class BaseTrainer():
    def __init__(self, engine, args, **kwargs):
        self.engine = engine

    def train(self, args, model, model_ema, loader, optimizer, device, amp_autocast, criterion, loss_scaler, scheduler,
              epoch, model_others):
        start = time.time()
        loss_cls_meter = AverageMeter()
        loss_cl_meter = AverageMeter()
        patch_num_meter = AverageMeter()
        keep_num_meter = AverageMeter()
        pad_ratio_meter = AverageMeter()
        mm_meter = AverageMeter()
        update_time_m = AverageMeter()
        data_time_m = AverageMeter()

        _profile = suppress

        has_no_sync = hasattr(model, "no_sync")
        model.train()

        train_loss_log = 0.
        accum_steps = 1
        last_accum_steps = len(loader) % accum_steps
        updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
        num_updates = epoch * updates_per_epoch
        last_batch_idx = len(loader) - 1
        last_batch_idx_to_accum = len(loader) - last_accum_steps

        model_unddp = model.module if args.distributed else model

        data_start_time = update_start_time = time.time()

        self.engine.init_func_train(args, model=model_unddp, others=model_others, epoch=epoch, optimizer=optimizer)
        optimizer.zero_grad()
        update_sample_count = 0

        for batch_idx, batch in enumerate(loader):
            last_batch = batch_idx == last_batch_idx
            need_update = last_batch or (batch_idx + 1) % accum_steps == 0
            update_idx = batch_idx // accum_steps
            if batch_idx >= last_batch_idx_to_accum:
                accum_steps = last_accum_steps

            bag = batch['input']
            label = batch['target']
            batch_size = label.size(0)
            pos = batch.get('pos', None)
            idx = batch.get('idx', None)
            feat = batch.get('feat', None)

            self.engine.after_get_data_func(args, device=device, bag=bag, optimizer=optimizer, loader=loader,
                                            n_iter=epoch * len(loader) + batch_idx, epoch=epoch, model=model_unddp,
                                            others=model_others)

            # multiply by accum steps to get equivalent for full update
            data_time_m.update(accum_steps * (time.time() - data_start_time))

            def _forward():
                with amp_autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    profile_env = _profile if update_idx % args.log_iter == 0 or batch_idx == len(
                        loader) - 1 else suppress
                    with profile_env() as prof:
                        logits, labels, aux_loss, patch_num, keep_num, pad_ratio = self.engine.forward_func(args,
                                                                                                            model=model,
                                                                                                            model_ema=model_ema,
                                                                                                            bag=bag,
                                                                                                            label=label,
                                                                                                            criterion=criterion,
                                                                                                            batch_size=batch_size,
                                                                                                            i=batch_idx,
                                                                                                            epoch=epoch,
                                                                                                            n_iter=epoch * len(
                                                                                                                loader) + batch_idx,
                                                                                                            loader=loader,
                                                                                                            device=device,
                                                                                                            others=model_others,
                                                                                                            pos=pos,
                                                                                                            idx=idx,
                                                                                                            feat=feat)

                    if logits is None:
                        return None, None, None, patch_num, keep_num, pad_ratio
                    # continue
                    bs = logits.size(0)
                    logit_loss = criterion(logits.view(bs, -1), labels.view(bs))
                    loss = logit_loss

                    loss /= accum_steps
                    return loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio

            def _backward(_loss):
                if loss_scaler:

                    loss_scaler.scale(_loss).backward()
                    if need_update:
                        loss_scaler.step(optimizer)
                        loss_scaler.update()
                else:
                    _loss.backward()
                    if need_update:
                        optimizer.step()

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio = _forward()
                    if loss is None:
                        continue
                    _backward(loss)
            else:
                loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio = _forward()
                if loss is None:
                    continue
                _backward(loss)

            self.engine.after_backward_func(args, model=model, others=model_others, num_updates=num_updates)

            if not args.distributed:
                loss_cls_meter.update(logit_loss * accum_steps, batch_size)
                loss_cl_meter.update(0., batch_size)
                patch_num_meter.update(patch_num, 1)
                keep_num_meter.update(keep_num, 1)
                pad_ratio_meter.update(pad_ratio, 1)

            update_sample_count += batch_size
            if not need_update:
                data_start_time = time.time()
                continue

            num_updates += 1
            optimizer.zero_grad()

            mm = 0.
            mm_meter.update(mm, 1)

            time_now = time.time()
            update_time_m.update(time.time() - update_start_time)
            update_start_time = time_now

            if update_idx % args.log_iter == 0 or batch_idx == len(loader) - 1:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if args.distributed:
                    reduced_loss = reduce_tensor(logit_loss.data, args.world_size)
                    loss_cls_meter.update(reduced_loss.item() * accum_steps, batch_size)
                    if isinstance(aux_loss, torch.Tensor):
                        loss_cl_meter.update(
                            reduce_tensor(aux_loss.data, args.world_size).item() * accum_steps * 0.,
                            batch_size)
                    else:
                        loss_cl_meter.update(aux_loss * accum_steps * 0., batch_size)
                    if isinstance(patch_num, torch.Tensor):
                        patch_num_meter.update(reduce_tensor(patch_num, args.world_size).item(), 1)
                    else:
                        patch_num_meter.update(patch_num, 1)
                    if isinstance(keep_num, torch.Tensor):
                        keep_num_meter.update(reduce_tensor(keep_num, args.world_size).item(), 1)
                    else:
                        keep_num_meter.update(keep_num, 1)
                    update_sample_count *= args.world_size

                rowd = OrderedDict([
                    ('main_loss', loss_cls_meter.avg),
                    ('lr', lr),
                    ('aux_loss', loss_cl_meter.avg),
                    ('patch_num', patch_num_meter.avg),
                    ('keep_num', keep_num_meter.avg),
                    ('pad_ratio', pad_ratio_meter.avg),
                    ('mm', mm_meter.avg),
                ])

                if args.rank == 0:

                    if loss_cl_meter.val != 0:
                        loss_cl_meter_avg_format = f'AuxLoss: {loss_cl_meter.val:#.4g} ({loss_cl_meter.avg:#.16g})  '
                    else:
                        loss_cl_meter_avg_format = ''

                    if keep_num_meter.avg == patch_num_meter.avg:
                        keep_num_meter_avg_format = ""
                    else:
                        keep_num_meter_avg_format = f'KN: {keep_num_meter.val:#.1f} ({keep_num_meter.avg:#.1f})  '
                    print(
                        f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                        f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                        f'Loss: {loss_cls_meter.val:#.4g} ({loss_cls_meter.avg:#.16g})  '
                        f'{loss_cl_meter_avg_format}'
                        f'PN: {patch_num_meter.val:#.1f} ({patch_num_meter.avg:#.1f})  '
                        f'{keep_num_meter_avg_format}'
                        f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                        f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                        f'LR: {lr:.3e}  '
                        f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                    )
                    rowd = OrderedDict([(str(args.fold_curr) + '-fold/' + _k, _v) for _k, _v in rowd.items()])
                    if args.wandb and args.rank == 0:
                        wandb.log(rowd)

            update_sample_count = 0
            data_start_time = time.time()
            train_loss_log = train_loss_log + loss.item()

        self.engine.final_train_func(args, model=model)

        end = time.time()
        train_loss_log = train_loss_log / len(loader)

        scheduler.step(epoch + 1)

        return train_loss_log, start, end

    def validate(self, args, model, loader, device, criterion, amp_autocast, early_stopping=None, epoch=None,
                 status="val", others=None):
        start_v = time.time()
        model.eval()
        loss_cls_meter = AverageMeter()
        loader_time_meter = AverageMeter()
        process_time_meter = AverageMeter()
        inference_time_meter = AverageMeter()
        bag_logit, bag_labels = None, None
        bag_logit_sub = None

        model_unddp = model.module if args.distributed else model

        self.engine.init_func_val(args, status=status, amp_autocast=amp_autocast, model=model_unddp, loader=loader,
                                  others=others, epoch=epoch)
        start_all_time = time.time()
        accu_time = 0
        acc_time_loader = 0
        start_all_time = time.time()

        with torch.no_grad():
            for i, batch in enumerate(loader):
                start_time = time.time()

                bag = batch['input']
                label = batch['target']
                pos = batch.get('pos', None)
                idx = batch.get('idx', None)

                batch_size = label.size(0)
                bag_labels = torch.cat((bag_labels, label)) if bag_labels is not None else torch.clone(label)

                self.engine.after_get_data_func(args, device=device, bag=bag, optimizer=None, loader=loader,
                                                n_iter=None, epoch=None, model=model_unddp)
                
                inference_start = time.time()

                with amp_autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    logits, labels = self.engine.validate_func(args, model=model, bag=bag, label=label,
                                                               criterion=criterion, batch_size=batch_size, i=i,
                                                               loader=loader, device=device, others=others, pos=pos,
                                                               idx=idx)
                    
                    inference_end = time.time()

                    if logits is None:
                        continue
                    

                    if type(logits) in (list, tuple):
                        logits, logits_sub = logits
                    else:
                        logits_sub = None

                    bs = logits.size(0)

                    bag_logit = torch.cat((bag_logit, logits)) if bag_logit is not None else logits.clone()
                    if logits_sub is not None:
                        bag_logit_sub = torch.cat(
                            (bag_logit_sub, logits_sub)) if bag_logit_sub is not None else logits_sub.clone()
                    test_loss = criterion(logits.view(bs, -1), labels.view(bs))

                loss_cls_meter.update(test_loss, 1)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()
                iteration_time = end_time - start_time
                accu_time += iteration_time
                all_time_gap = end_time - start_all_time
                loader_time = all_time_gap - acc_time_loader
                acc_time_loader += loader_time
                
                end_v = time.time()
                total_validation_time = end_v - start_v
                batch_inference_time = inference_end - inference_start

                loader_time_meter.update(loader_time, 1)
                process_time_meter.update(iteration_time, 1)

                if args.rank == 0 and (i + 1) == len(loader):
                    print(f"Iter{i + 1}: Loader took {loader_time_meter.avg:.2f} seconds "
                          f"Process took {process_time_meter.avg:.2f} seconds ")

        if args.distributed:
            bag_logit = distributed_concat(bag_logit, len(loader.dataset))
            bag_labels = distributed_concat(bag_labels, len(loader.dataset))
            if bag_logit_sub is not None:
                bag_logit_sub = distributed_concat(bag_logit_sub, len(loader.dataset))

        suffix_main = suffix_sub = None
        if args.test_type in ('both_ema'):
            suffix_main = 'ema'

        output = get_metric_val(args, bag_logit=bag_logit, bag_labels=bag_labels, model=model, status=status,
                                early_stopping=early_stopping, epoch=epoch,
                                loss_cls_meter=loss_cls_meter, suffix=suffix_main)
        output = list(output)
        # output.append(total_validation_time)
        # output.append(batch_inference_time)

        if bag_logit_sub is not None:
            output_sub = get_metric_val(args, bag_logit=bag_logit_sub, bag_labels=bag_labels, model=model,
                                        status=status, early_stopping=None, epoch=epoch,
                                        loss_cls_meter=loss_cls_meter, suffix=suffix_sub)

            output_sub = list(output_sub)
            # output_sub.append(total_validation_time)
            # output_sub.append(batch_inference_time)

            output[-1].update(output_sub[-1])

            if args.test_type in ('both_ema'):
                output[0] = [output_sub[0], output[0]]
            else:
                output[0] = [output[0], output_sub[0]]

        return output

    ############# Survival Prediction ###################
    def surv_train(self, args, model, model_ema, loader, optimizer, device, amp_autocast, criterion, loss_scaler,
                   scheduler, epoch, model_others):
        start = time.time()

        loss_cls_meter = AverageMeter()
        loss_cl_meter = AverageMeter()
        patch_num_meter = AverageMeter()
        keep_num_meter = AverageMeter()
        pad_ratio_meter = AverageMeter()
        mm_meter = AverageMeter()
        update_time_m = AverageMeter()
        data_time_m = AverageMeter()

        model.train()

        model_unddp = model.module if args.distributed else model

        accum_steps = 1
        updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
        num_updates = epoch * updates_per_epoch
        last_batch_idx = len(loader) - 1

        self.engine.init_func_train(args, model=model_unddp, others=model_others, epoch=epoch, optimizer=optimizer)
        optimizer.zero_grad()

        train_loss_log = 0.
        update_sample_count = 0
        data_start_time = update_start_time = time.time()

        for batch_idx, batch in enumerate(loader):

            last_batch = batch_idx == last_batch_idx
            need_update = last_batch or (batch_idx + 1) % accum_steps == 0
            update_idx = batch_idx // accum_steps

            bag = batch['input']

            label = batch['target'].to(dtype=torch.long, device=batch['target'].device)
            data_Censorship = batch['censorship'].to(dtype=torch.float, device=batch['censorship'].device)

            pos = batch.get('pos', None)
            idx = batch.get('idx', None)

            batch_size = label.size(0)

            self.engine.after_get_data_func(args, device=device, bag=bag, optimizer=optimizer, loader=loader,
                                            n_iter=epoch * len(loader) + batch_idx, epoch=epoch, model=model_unddp,
                                            others=model_others)

            data_time_m.update(accum_steps * (time.time() - data_start_time))

            def _forward():
                with amp_autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                    logits, labels_surv, aux_loss, patch_num, keep_num, pad_ratio = self.engine.forward_func(
                        args,
                        model=model,
                        model_ema=model_ema,
                        bag=bag,
                        label=[label, data_Censorship],
                        criterion=criterion,
                        batch_size=batch_size,
                        i=batch_idx,
                        epoch=epoch,
                        n_iter=epoch * len(loader) + batch_idx,
                        loader=loader,
                        device=device,
                        others=model_others,
                        pos=pos,
                        idx=idx
                    )
                    if logits is None:
                        return None, None, None, patch_num, keep_num, pad_ratio

                    with amp_autocast(device_type='cuda', enabled=False):
                        logit_loss = criterion(logits=logits, Y=labels_surv[0], c=labels_surv[1])

                        loss = logit_loss
                        loss /= accum_steps

                    return loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio

            def _backward(_loss):
                if loss_scaler:
                    loss_scaler.scale(_loss).backward()

                    if need_update:
                        loss_scaler.step(optimizer)
                        loss_scaler.update()
                else:
                    _loss.backward()
                    if need_update:
                        optimizer.step()

            has_no_sync = hasattr(model, "no_sync")
            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio = _forward()
                    if loss is None:
                        data_start_time = time.time()
                        continue
                    _backward(loss)
            else:
                loss, aux_loss, logit_loss, patch_num, keep_num, pad_ratio = _forward()
                if loss is None:
                    data_start_time = time.time()
                    continue
                _backward(loss)

            self.engine.after_backward_func(args, model=model, others=model_others, num_updates=num_updates)

            if not args.distributed:
                loss_cls_meter.update(loss.item() * accum_steps, batch_size)
                loss_cl_meter.update(aux_loss, batch_size)
                patch_num_meter.update(patch_num, 1)
                keep_num_meter.update(keep_num, 1)
                pad_ratio_meter.update(pad_ratio, 1)

            update_sample_count += batch_size

            if not need_update:
                data_start_time = time.time()
                continue

            num_updates += 1
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update(model, epoch * len(loader) + batch_idx)
                mm = model_ema.get_decay(epoch * len(loader) + batch_idx)
            else:
                mm = 0.
            mm_meter.update(mm, 1)

            if args.distributed:
                reduced_loss = reduce_tensor(logit_loss.data, args.world_size)
                loss_cls_meter.update(reduced_loss.item() * accum_steps, batch_size)
                if isinstance(aux_loss, torch.Tensor):
                    loss_cl_meter.update(
                        reduce_tensor(aux_loss.data, args.world_size).item() * accum_steps * 0., batch_size)
                else:
                    loss_cl_meter.update(aux_loss * accum_steps * 0., batch_size)
                if isinstance(patch_num, torch.Tensor):
                    patch_num_meter.update(reduce_tensor(patch_num, args.world_size).item(), 1)
                else:
                    patch_num_meter.update(patch_num, 1)
                if isinstance(keep_num, torch.Tensor):
                    keep_num_meter.update(reduce_tensor(keep_num, args.world_size).item(), 1)
                else:
                    keep_num_meter.update(keep_num, 1)
                if isinstance(pad_ratio, torch.Tensor):
                    pad_ratio_meter.update(reduce_tensor(pad_ratio, args.world_size).item(), 1)
                else:
                    pad_ratio_meter.update(pad_ratio, 1)
                update_sample_count *= args.world_size

            time_now = time.time()
            update_time_m.update(time_now - update_start_time)
            update_start_time = time_now

            if update_idx % args.log_iter == 0 or batch_idx == last_batch_idx:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if args.rank == 0:
                    if loss_cl_meter.val != 0:
                        loss_cl_meter_avg_format = f'AuxLoss: {loss_cl_meter.val:#.4g} ({loss_cl_meter.avg:#.16g})  '
                    else:
                        loss_cl_meter_avg_format = ''

                    if keep_num_meter.avg == patch_num_meter.avg:
                        keep_num_meter_avg_format = ""
                    else:
                        keep_num_meter_avg_format = f'KN: {keep_num_meter.val:#.1f} ({keep_num_meter.avg:#.1f})  '

                    print(
                        f'Train(Surv): {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                        f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                        f'Loss: {loss_cls_meter.val:#.4g} ({loss_cls_meter.avg:#.16g})  '
                        f'{loss_cl_meter_avg_format}'
                        f'PN: {patch_num_meter.val:#.1f} ({patch_num_meter.avg:#.1f})  '
                        f'{keep_num_meter_avg_format}'
                        f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                        f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                        f'LR: {lr:.3e}  '
                        f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                    )

                    rowd = OrderedDict([
                        ('main_loss', loss_cls_meter.avg),
                        ('lr', lr),
                        ('aux_loss', loss_cl_meter.avg),
                        ('patch_num', patch_num_meter.avg),
                        ('keep_num', keep_num_meter.avg),
                        ('pad_ratio', pad_ratio_meter.avg),
                        ('mm', mm_meter.avg),
                    ])
                    rowd = OrderedDict([(str(args.fold_curr) + '-fold/' + _k, _v) for _k, _v in rowd.items()])
                    if args.wandb and args.rank == 0:
                        wandb.log(rowd)

            train_loss_log += loss.item()
            update_sample_count = 0
            data_start_time = time.time()

        self.engine.final_train_func(args, model=model)

        end = time.time()
        train_loss_log = train_loss_log / len(loader)

        scheduler.step(epoch + 1)

        return train_loss_log, start, end

    def surv_validate(self, args, model, loader, device, criterion, amp_autocast, early_stopping=None, epoch=None,
                      status="val", others=None):
        model.eval()
        loss_cls_meter = AverageMeter()
        loader_time_meter = AverageMeter()
        process_time_meter = AverageMeter()

        all_risk_scores = None
        all_risk_scores_sub = None
        all_censorships = None
        all_event_times = None

        model_unddp = model.module if args.distributed else model

        self.engine.init_func_val(args, status=status, amp_autocast=amp_autocast, model=model_unddp, loader=loader,
                                  others=others, epoch=epoch)

        start_all_time = time.time()
        acc_time_loader = 0
        accu_time = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                start_time = time.time()

                bag = batch['input']
                label = batch['target'].to(dtype=torch.long, device=batch['target'].device)
                data_Censorship = batch['censorship'].to(dtype=torch.float, device=batch['censorship'].device)
                data_Event = batch['event']
                pos = batch.get('pos', None)
                idx = batch.get('idx', None)
                batch_size = label.size(0)

                self.engine.after_get_data_func(args, device=device, bag=bag, optimizer=None, loader=loader,
                                                n_iter=None, epoch=None, model=model_unddp, others=others)

                with amp_autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):

                    logits, labels = self.engine.validate_func(
                        args,
                        model=model,
                        bag=bag,
                        label=label,
                        criterion=criterion,
                        batch_size=batch_size,
                        i=i,
                        loader=loader,
                        device=device,
                        others=others,
                        pos=pos,
                        idx=idx
                    )

                    if logits is None:
                        end_time = time.time()
                        iteration_time = end_time - start_time
                        accu_time += iteration_time
                        all_time_gap = end_time - start_all_time
                        loader_time = all_time_gap - acc_time_loader
                        acc_time_loader += loader_time
                        loader_time_meter.update(loader_time, 1)
                        process_time_meter.update(iteration_time, 1)
                        if args.rank == 0 and (i + 1) == len(loader):
                            print(f"Iter{i + 1}: Loader took {loader_time_meter.avg:.2f} seconds "
                                  f"Process took {process_time_meter.avg:.2f} seconds ")
                        continue

                    if type(logits) in (list, tuple):
                        logits, logits_sub = logits
                    else:
                        logits_sub = None

                    hazards = torch.sigmoid(logits)
                    S = torch.cumprod(1 - hazards, dim=1)

                    test_loss = criterion(hazards=hazards, S=S, Y=labels, c=data_Censorship)
                    loss_cls_meter.update(test_loss, batch_size)

                    # risk score
                    risk = -torch.sum(S, dim=1)
                    all_risk_scores = torch.cat(
                        (all_risk_scores, risk)) if all_risk_scores is not None else risk.clone()
                    all_censorships = torch.cat(
                        (all_censorships, data_Censorship)) if all_censorships is not None else data_Censorship.clone()
                    all_event_times = torch.cat(
                        (all_event_times, data_Event)) if all_event_times is not None else data_Event.clone()

                    if logits_sub is not None:
                        hazards_sub = torch.sigmoid(logits_sub)
                        S_sub = torch.cumprod(1 - hazards_sub, dim=1)
                        risk_sub = -torch.sum(S_sub, dim=1)
                        all_risk_scores_sub = torch.cat(
                            (all_risk_scores_sub, risk_sub)) if all_risk_scores_sub is not None else risk_sub.clone()

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.time()
                iteration_time = end_time - start_time
                accu_time += iteration_time
                all_time_gap = end_time - start_all_time
                loader_time = all_time_gap - acc_time_loader
                acc_time_loader += loader_time

                loader_time_meter.update(loader_time, 1)
                process_time_meter.update(iteration_time, 1)

                if args.rank == 0 and (i + 1) == len(loader):
                    print(f"Iter{i + 1}: Loader took {loader_time_meter.avg:.2f} seconds "
                          f"Process took {process_time_meter.avg:.2f} seconds ")

        if args.distributed:
            all_risk_scores = distributed_concat(all_risk_scores, len(loader.dataset))
            all_censorships = distributed_concat(all_censorships, len(loader.dataset))
            all_event_times = distributed_concat(all_event_times, len(loader.dataset))
            if all_risk_scores_sub is not None:
                all_risk_scores_sub = distributed_concat(all_risk_scores_sub, len(loader.dataset))

        suffix_main = suffix_sub = None
        if args.test_type in ('both_ema'):
            suffix_main = 'ema'

        output = get_metric_val(args, bag_logit=all_risk_scores,
                                bag_labels={'censorships': all_censorships, 'event_times': all_event_times},
                                model=model, status=status,
                                early_stopping=early_stopping, epoch=epoch,
                                loss_cls_meter=loss_cls_meter, suffix=suffix_main, surv=True)

        output = list(output)

        if all_risk_scores_sub is not None:
            output_sub = get_metric_val(args, bag_logit=all_risk_scores_sub,
                                        bag_labels={'censorships': all_censorships, 'event_times': all_event_times},
                                        model=model, status=status, early_stopping=None, epoch=epoch,
                                        loss_cls_meter=loss_cls_meter, suffix=suffix_sub, surv=True)

            output_sub = list(output_sub)

            output[-1].update(output_sub[-1])

            if args.test_type in ('both_ema'):
                output[0] = [output_sub[0], output[0]]
            else:
                output[0] = [output[0], output_sub[0]]

        return output


class CommonMIL():
    def __init__(self, args) -> None:
        self.training = True

    def init_func_train(self, args, **kwargs):
        self.training = True

    def init_func_val(self, args, **kwargs):
        self.training = False

    def after_get_data_func(self, args, **kwargs):
        pass

    def forward_func(self, args, model, model_ema, bag, label, criterion, batch_size, i, epoch, n_iter, pos, **kwargs):
        pad_ratio = 0.
        logits = model(bag, pos=pos)
        aux_loss, patch_num, keep_num = 0., bag.size(1), bag.size(1)

        return logits, label, aux_loss, patch_num, keep_num, pad_ratio

    def after_backward_func(self, args, **kwargs):
        pass

    def final_train_func(self, args, **kwargs):
        pass

    def validate_func(self, args, model, bag, label, criterion, batch_size, i, pos, **kwargs):
        logits = model(bag, pos=pos)

        return logits, label


class E2E(CommonMIL):
    def __init__(self, args, device) -> None:
        super().__init__(args)
        self.dataset = None
        self.loader = None
        self.device = device
        self.max_psize = args.max_patch_train or args.same_psize  # Maximum patch size for training
        self.norm = Normalize(device, False)  # Normalization module for patches
        self.static_pos = [self.max_psize]  # Static position, seems related to patch selection or positional encoding
        self.inference_mode =  suppress  # Context manager for inference mode

    def init_func_train(self, args, model, others, epoch, optimizer, **kwargs):
        self.training = True  # Set training mode

    def after_get_data_func(self, args, device, bag, optimizer, loader, n_iter, epoch, model, **kwargs):
        pass  # Placeholder for actions after data is fetched

    def get_grad_patch(self, args, bag, device, others, pos, feat):
        """Selects patches for gradient computation based on selection type."""
        patch_num = bag.size(0)  # Total number of patches in the bag
        if patch_num > self.max_psize:
            # If number of patches exceeds max_psize, select a subset
            indices = torch.randperm(patch_num, device=bag.device)
            selected_indices = indices[:self.max_psize]
            remaining_indices = indices[self.max_psize:]

            selected_patches = bag[selected_indices]  # Get selected patches
            remaining_patches = bag[remaining_indices]  # Get remaining patches
            ori_indices = None  # Original indices not tracked here for this branch

        elif patch_num < self.max_psize:
            # If number of patches is less than max_psize, pad if necessary
            # Pad by duplicating existing patches randomly
            ori_indices = torch.arange(patch_num, device=bag.device)
            selected_indices = torch.cat(
                [ori_indices, torch.randint(0, patch_num, (self.max_psize - patch_num,), device=bag.device)])
            selected_patches = bag[selected_indices]
            remaining_patches = None  # No remaining patches in this case

        else:
            # If number of patches is equal to max_psize
            selected_patches = bag  # Use all patches
            remaining_patches = None
            ori_indices = None

        remaining_patches = None  # Ensure remaining_patches is None if not used (or handled above)

        return selected_patches, remaining_patches, ori_indices, pos, feat

    def forward_func(self, args, model, model_ema, bag, label, criterion, batch_size, i, loader, device, others,
                     pos=None, idx=None, feat=None, **kwargs):
        """Main forward pass logic for training."""
        ps = None  # Placeholder for patch selection indices or similar
        # Simple implementation first, consider optimization later
        if type(bag) in (tuple, list):
            # If bag is a tuple/list, it might contain (patches, positions) or (patches, features)
            # Change to pos input, temporarily not considering feat input
            # feat input
            bag, ps = bag  # Unpack bag and patch selection info (ps)

        if len(bag.size()) == 5:  # If bag has 5 dimensions (e.g., [1, N, C, H, W]), squeeze the first dim
            bag = bag.squeeze(0)

        _, C, W, H = bag.size()  # Get patch dimensions

        if args.batch_size > 1:
            # Handle batch processing without EMA selection (all patches are used or randomly selected if get_grad_patch is called)
            patch_num = bag.size(0) / batch_size  # Average number of patches per bag in the batch
            keep_num = patch_num  # All patches are kept (or selected by get_grad_patch if called)
            ori_indices = None
            selected_patches = bag  # The input 'bag' is already the batch of selected patches
            remaining_patches = None

        else:  # Single bag processing (batch_size == 1)
            if ps is None:  # If patch selection info (ps) is not provided directly
                patch_num = bag.size(0)  # Total number of patches in the bag
                # Select patches for gradient computation
                selected_patches, remaining_patches, ori_indices, pos, feat = self.get_grad_patch(args, bag, device,
                                                                                                  others, pos, feat)
                keep_num = selected_patches.size(0)  # Number of patches kept
                if pos is not None:
                    if len(pos.shape) == 2:  # Ensure pos has a batch dimension
                        pos = pos.unsqueeze(0)
            else:
                raise NotImplementedError  # Handling for pre-selected patches (ps) not fully implemented here

        # For other models
        # logits = model((selected_patches,remaining_patches),ori_indices,pos=pos,B=batch_size,feat=feat)
        logits = model(selected_patches, ori_indices, pos=pos, B=batch_size, feat=feat)
        aux_loss = 0.  # No auxiliary loss by default for other models

        return logits, label, aux_loss, patch_num, keep_num, 0.  # Return model outputs and related info

    def final_train_func(self, args, model):
        pass

    def validate_func(self, args, model, bag, label, criterion, batch_size, i, loader, device, others, **kwargs):
        """Validation forward pass logic."""
        if type(bag) in (tuple, list):
            if isinstance(bag[1], torch.Tensor):
                # Input is (patches, positions)
                bag, pos = bag
                ps = None
            # batch input
            else:
                # Input is (patches, patch_selection_info) or similar
                bag, ps = bag
                pos = None
        else:
            ps = None  # No patch selection info
            pos = None  # No positional info

        if len(bag.size()) == 5:  # If bag has 5 dimensions, squeeze the first one
            bag = bag.squeeze(0)

        patch_num = bag.size(0)  # Total number of patches
        all_patches = bag  # Use all patches for validation
        del bag  # Free memory

        keep_num = all_patches.size(0)  # Number of patches kept (all of them)

        # Determine which model to use for validation (main model, EMA model, or both)
        if args.test_type != 'main':
            logits_ema = others['model_ema'](all_patches, pos=pos)  # Get logits from EMA model
            logits_ema = logits_ema
            if args.test_type == 'ema':
                logits = logits_ema  # Use only EMA model logits
            elif args.test_type == 'both':
                logits_main = model(all_patches, pos=pos)  # Get logits from main model
                logits = [logits_main, logits_ema]  # Return logits from both models
            elif args.test_type == 'both_ema':  # Similar to 'both', but EMA logits first
                logits_main = model(all_patches, pos=pos)
                logits = [logits_ema, logits_main]
        else:
            # Use only main model for validation
            logits = model(all_patches, pos=pos)

        return logits, label  # Return logits and labels

    def init_func_val(self, args, amp_autocast, model, loader, status='val', others=None, epoch=0):
        pass


class Normalize(nn.Module):
    def __init__(self, device, channels_last):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(
            [x * 255 for x in IMAGENET_MEAN], device=device).view(1, 3, 1, 1).to(memory_format=torch.channels_last)
        self.std = torch.tensor(
            [x * 255 for x in IMAGENET_STD], device=device).view(1, 3, 1, 1).to(memory_format=torch.channels_last)

        if channels_last:
            self.mean.to(memory_format=torch.channels_last)
            self.std.to(memory_format=torch.channels_last)

    def __call__(self, tensor):
        return tensor.sub_(self.mean).div_(self.std)