# Copyright (c) 2020, princeton-vl
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications Copyright 2026 xfusion authors
# from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim

from xfusion.train.RAFT.core.raft import RAFT
from xfusion.inference.model.edvr_models import EDVRSTFTempRank
from xfusion.train.RAFT.core.calibrator_models import FlowCalibratorModel
import xfusion.train.RAFT.core.datasets_xray as datasets
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from xfusion.train.RAFT.core.utils.dist_utils import init_distributed_mode
from xfusion.utils import yaml_load
from xfusion import config
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def flow_recon_loss(loss, flow_preds, flow_gt, pred, gt, mask, valid, gamma=0.8, max_flow=MAX_FLOW, factor = 1, mask_valid_for_fusion_ok=False):
    flow_loss, metrics, valid_ = sequence_loss(flow_preds, flow_gt, valid * mask[:,0] * mask[:,1], gamma=gamma, max_flow=max_flow)
    diff = (pred - gt) * valid_[:,None] * 255.
    mse = ((diff)**2).sum() / (valid_.sum() + 10**(-10))
    psnr = 10. * torch.log10(255. * 255. / mse)
    if mask_valid_for_fusion_ok:
        recon_loss = loss(pred*valid_[:,None] * valid_.size()[-2] * valid_.size()[-1] / valid_.sum(dim=tuple(range(1, valid_.dim())),keepdim=True)[:,None], gt*valid_[:,None] * valid_.size()[-2] * valid_.size()[-1] / valid_.sum(dim=tuple(range(1, valid_.dim())),keepdim=True)[:,None])
        metrics['recon loss'] = recon_loss.clone().item()
        metrics['flow loss'] = flow_loss.clone().item()
        return (recon_loss * factor + flow_loss), metrics, psnr
    else:
        return (loss(pred,gt) * factor + flow_loss), metrics, psnr

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics, valid

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.cal_train_lr, weight_decay=args.cal_train_wdecay, eps=args.cal_train_epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.cal_train_lr, args.cal_train_num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, rank = 0):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.rank = rank

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            if self.rank == 0:
                self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    save_path = Path(config.get_train_dirs()) / Path(args.cal_train_save_path).name / 'checkpoints'
    save_path.mkdir(exist_ok=True,parents=True)
    save_path = str(save_path)

    dist_mode = init_distributed_mode(args.cal_train_launcher, args.cal_train_dist_url)
    
    device = torch.device(args.cal_train_device)

    opt = yaml_load(args.cal_train_opt_path)
    model_config = opt['network_g']
    del model_config['type']
    model_config['num_frame'] = args.cal_train_num_frame_total
    model_config['num_frame_hi'] = 0
    model_config['center_frame_idx'] = args.cal_train_num_frames//2

    eval_model = EDVRSTFTempRank(**model_config)
    if args.cal_train_full_weights is not None:
        pass
    else:
        eval_model.load_state_dict(torch.load(args.cal_train_evaluator_weights)['params'])
        for p in eval_model.parameters():
            p.requires_grad = False

    flow_model = RAFT(args)
    if args.cal_train_full_weights is not None:
        pass
    else:
        if (args.cal_train_flow_weights is not None) and (Path(args.cal_train_flow_weights).exists()):
            flow_model_weights_ = torch.load(args.cal_train_flow_weights,map_location=torch.device('cpu'))
            flow_model_weights = {}
            for k in list(flow_model_weights_.keys()):
                flow_model_weights[k[7:]] = flow_model_weights_[k]
            flow_model.load_state_dict(flow_model_weights)
        else:
            if dist_mode['rank'] == 0:
                print('training flow model from scratch')
    
    model = FlowCalibratorModel(flow_model, eval_model, pixel_size_ratio=args.cal_train_pixel_size_ratio)
    if (args.cal_train_full_weights is not None) and (Path(args.cal_train_full_weights).exists()):
        full_model_weights_ = torch.load(args.cal_train_full_weights,map_location=torch.device('cpu'))
        full_model_weights = {}
        for k in list(full_model_weights_.keys()):
            full_model_weights[k[7:]] = full_model_weights_[k]
        model.load_state_dict(full_model_weights)
        if (args.cal_train_evaluator_weights is not None) and (Path(args.cal_train_evaluator_weights).exists()):
            eval_model_weights_ = torch.load(args.cal_train_evaluator_weights)['params']
            eval_model_weights = {}
            for k in list(eval_model_weights_.keys()):
                eval_model_weights['model.'+k] = eval_model_weights_[k]
            msg = model.load_state_dict(eval_model_weights,strict=False)
            print(f"missing keys: {msg.missing_keys}")
            print(f"unexpected keys: {msg.unexpected_keys}")
        for p in model.model.parameters():
            p.requires_grad = False

    if dist_mode['distributed']:
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=args.cal_train_find_unused_parameters)
    else:
        if torch.cuda.is_available():
            model = nn.DataParallel(model, device_ids=[torch.cuda.current_device()])
    print("Parameter Count: %d" % count_parameters(model))

    if args.cal_train_restore_ckpt is not None:
        model.load_state_dict(torch.load(args.cal_train_restore_ckpt), strict=False)
    
    if torch.cuda.is_available():
        model.to(device)#.cuda()
    model.train()
    model.module.model.eval()
    # if args.cal_train_stage != 'chairs':
    #     model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.cal_train_mixed_precision)
    logger = Logger(model, scheduler, dist_mode['rank'])

    VAL_FREQ = 5000

    loss = torch.nn.MSELoss()
    should_keep_training = True
    epoch = 0
    while should_keep_training:
        if dist_mode['distributed']:
            train_loader.sampler.set_epoch(epoch)
        for i_batch, data_blob in enumerate(train_loader):
            if dist_mode['rank'] == 0:
                print(f"training step {total_steps}")
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.to(device) if torch.cuda.is_available() else x for x in data_blob]

            

            # flow_predictions = model(image1, image2, iters=args.cal_train_iters)            
            assert args.cal_train_num_frames > 1
            img_lr = image1
            img = torch.cat((image2[:,:args.cal_train_num_frames//2],image2[:,args.cal_train_num_frames//2+1:]),dim=1)
            img_gt = image2[:,args.cal_train_num_frames//2]
            results = model({'hq':img, 'gt':img_gt, 'lq':img_lr, 'flow_gt':flow, 'valid':valid})
            # loss, metrics = sequence_loss(flow_predictions, flow, valid, args.cal_train_gamma)
            l, metrics, psnr = flow_recon_loss(loss, results['flows'], results['flow_gt'], results['pred'], results['gt'], results['mask'], results['valid'], gamma=args.cal_train_gamma, factor=args.cal_train_factor,mask_valid_for_fusion_ok=args.cal_train_mask_valid_for_fusion_ok)
            scaler.scale(l).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.cal_train_clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            if total_steps % SUM_FREQ == (SUM_FREQ - 1):
                if dist_mode['rank'] == 0:
                    print(f"training loss is {l.item()}")
                    print(f"psnr is {psnr.item()} dB")
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = save_path+'/'+('%d_%s.pth' % (total_steps+1, args.cal_train_name))
                torch.save(model.state_dict(), PATH)
            
            total_steps += 1

            if total_steps > args.cal_train_num_steps:
                should_keep_training = False
                break
        epoch = epoch + 1

    logger.close()
    PATH = save_path + '/' + ('%s.pth' % args.cal_train_name)
    torch.save(model.state_dict(), PATH)

    return PATH