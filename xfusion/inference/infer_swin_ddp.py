import os
#import datetime
#from omegaconf import OmegaConf
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#import argparse
import logging
#from main import get_parser
#import yaml
#from collections import OrderedDict
#from lightning import seed_everything
#from lightning.pytorch.trainer import Trainer
#from ldm.util import instantiate_from_config
#from ldm.models.autoencoder import EDVR
from xfusion.train.basicsr.archs.VideoTransformerSTF_arch import PatchTransformerSTF
import numpy as np
import torch
#import sys
import cv2
#from distutils.util import strtobool
#sys.path.insert(0,'/home/beams/FAST/conda/BasicSR_single_channel')
from xfusion.train.basicsr.data import build_dataloader
from xfusion.train.basicsr.utils import tensor2img
from xfusion.train.basicsr.utils.dist_util import get_dist_info
from xfusion.train.basicsr.utils import get_root_logger
from PIL import Image
from pathlib import Path
#from copy import deepcopy
#import matplotlib as mpl
#mpl.use('QtAgg')
import matplotlib.pyplot as plt
#from ldm.data.imagenet import REDSDatasetSTF, VideoTestDatasetSTF
from xfusion.train.basicsr.data.xray_dataset import XrayVideoTestDatasetSTF
#import argparse
from skimage.metrics import structural_similarity as ssim
import torch.multiprocessing as mp
import torch.distributed as dist
from xfusion.utils import yaml_load
from xfusion import config   

def run_inference(ts, args):

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if args.machine == 'tomo':
        rank = int(os.environ['RANK'])
        world_size = torch.cuda.device_count()
    elif args.machine == 'polaris':
        rank = int(os.environ.get('PMI_RANK',-1))
        world_size = int(os.environ.get('NTOTRANKS',-1))
    rank_ = 0
    
    #torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank,world_size=world_size)

    mode = str(args.infer_mode)
    lo_frame_sep = int(args.lo_frame_sep)
    hi_frame_sep = int(args.hi_frame_sep)
    b0 = int(args.b0)
    img_class = str(args.img_class)
    
    opt = yaml_load(args.opt)
    opt['model_type'] = args.model_type
    opt['path']['pretrain_network_g'] = args.model_file
    print(f"{args.opt} and {args.arch_opt}")
    opt['network_g'] = yaml_load(args.arch_opt)['network_g']

    opt['network_g']['embed_dim'] = args.embed_dim
    opt['network_g']['num_feat_ext'] = args.embed_dim
    opt['network_g']['num_frame'] = args.num_frame
    
    opt['network_g']['depths'] = [d * args.scale_depth for d in opt['network_g']['depths']]
    opt['network_g']['window_size'][0] += (args.num_frame - 3)
    opt['network_g']['window_size'][1:] = [args.window_size_spatial] * len(opt['network_g']['window_size'][1:])
    
    opt['network_g']['num_heads'] = [int(np.ceil(h * args.scale_mha)) for h in opt['network_g']['num_heads']]
    opt['network_g']['align_features_ok'] = args.align_features_ok
    opt['network_g']['adapt_deformable_conv'] = args.adapt_deformable_conv
    
    test_set_name = img_class
    #test_set_name = opt['name']
    gt_size = opt['datasets']['val']['gt_size']
    dataroot = config.get_inf_data_dirs(img_class)
    inf_home_dir = Path(dataroot).parent

    out_dir = Path(inf_home_dir / f'{test_set_name}_{mode}_align_{args.align_features_ok}_prepostx_fixed_lr_r_{lo_frame_sep}_hr_d_{hi_frame_sep*2}_b0_{b0}_em_{args.embed_dim}_nf_{args.num_frame}_dep_{args.scale_depth}_nh_{args.scale_mha}_win_{args.window_size_spatial}_model_{args.model_file.parent.stem}_ddp_swinVRHP_rem1_{ts}')
    out_dir.mkdir(exist_ok=True,parents=True)

    log_file = os.path.join(out_dir, f"inference.log")
    logger = get_root_logger(logger_name=__name__, log_level=logging.INFO, log_file=log_file)

    logger.info(f"num frame is {opt['network_g']['num_frame']}")
    logger.info(f"window size is {opt['network_g']['window_size']}")
    logger.info(f'inference under {mode} mode')
    logger.info(f'LR frame separation is {lo_frame_sep}')
    logger.info(f'HR frame separation is {hi_frame_sep}')

    opt['datasets']['val']['dataroot_lq'] = [os.path.join(dataroot,'LR')]
    opt['datasets']['val']['dataroot_gt'] = [os.path.join(dataroot,'HR')]
    opt['datasets']['val']['dataroot_lq'] = [dr+f'_b0_{b0}' if Path(dr).name.lower() != 'lr' else str(Path(dr).parents[0])+f'_b0_{b0}'+f"/{Path(dr).name}" for dr in opt['datasets']['val']['dataroot_lq']]
    opt['datasets']['val']['dataroot_gt'] = [dr+f'_b0_{0}' if Path(dr).name.lower() != 'hr' else str(Path(dr).parents[0])+f'_b0_{b0}'+f"/{Path(dr).name}" for dr in opt['datasets']['val']['dataroot_gt']]


    #opt['datasets']['train']['type'] = 'REDSDatasetSTF'
    opt['datasets']['val']['type'] = 'VideoTestDatasetSTF'

    opt['path']['pretrain_network_g'] = str(args.model_file)
    
    #print(f"data paths are: {opt['datasets']['val']['dataroot_lq']} for LR and {opt['datasets']['val']['dataroot_gt']} for HR")
    logger.info(f"data paths are: {opt['datasets']['val']['dataroot_lq']} for LR and {opt['datasets']['val']['dataroot_gt']} for HR")

    opt['dist'] = True

    if mode not in ['bil','bic']:
        opt['num_gpu'] = 8
        opt['manual_seed'] = 10
        k = 10
        if 'patchsize' in opt['network_g']:
            opt['network_g']['patchsize'] = [(sz,sz) for sz in opt['network_g']['patchsize']]
        first_stage_config = opt['network_g']
        del first_stage_config['type']
        logger.info(first_stage_config)
        if mode == 'stf':
            first_stage_model = PatchTransformerSTF(**first_stage_config)
        elif mode == 'sr':
            raise Exception('ViT for SR is not implemented yet...')
        
        first_stage_model.load_state_dict(torch.load(opt['path']['pretrain_network_g'])['params_ema'])
        first_stage_model.to(rank_)
        first_stage_model.eval()
    else:
        center_frame_idx = 1
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        opt['datasets'][phase]['phase'] = phase
        #if phase == 'train':
        #    continue
        dataset_opt['scale'] = 4
        dataset_opt['gt_size'] = gt_size[:2]
        dataset_opt['num_frame'] = args.num_frame
        #test_set = VideoTestDatasetSTF(dataset_opt)
        #test_set = build_dataset(dataset_opt)
        test_set = XrayVideoTestDatasetSTF(dataset_opt)
        test_set.case_frame_size = {'240':gt_size}
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        test_loaders.append(test_loader)


    logger.info(f"found {len(test_loader.dataset)} images")
    logger.info(f"rank is {rank} and world size is {world_size}")
    for test_loader in test_loaders:
        #test_set_name = test_loader.dataset.opt['name']
        dataset = test_loader.dataset
        #rank, world_size = get_dist_info()
        psnrs, aads, atts, ssims, masks, indices, times = [], [], [], [], [], [], []
        
        for idx in range(rank+1, len(dataset), world_size):
            results = []

            val_data_ = dataset[idx]
            val_data_['lq'].unsqueeze_(0)
            if 'gt' in list(val_data_.keys()):
                val_data_['gt'].unsqueeze_(0)
            else:
                val_data_['image'].unsqueeze_(0)
            
            if 'gt' in list(val_data_.keys()):
                if mode == 'stf':
                    val_data = {'lq':val_data_['lq'].to(rank_), 'gt':val_data_['gt'].to(rank_), 'hq':val_data_['hq'][None,:,:,:].to(rank_)}
                else:
                    val_data = {'lq':val_data_['lq'].to(rank_), 'gt':val_data_['gt'].to(rank_)}
            else:
                if mode == 'stf':
                    val_data = {'lq':val_data_['lq'].to(rank_), 'image':val_data_['image'].to(rank_), 'hq':val_data_['hq'][None,:,:,:].to(rank_)}
                else:
                    val_data = {'lq':val_data_['lq'].to(rank_), 'image':val_data_['image'].to(rank_)}
            
            if mode == 'stf':
                if 'gt' in list(dataset[max(0,int(idx//(hi_frame_sep*2)*hi_frame_sep*2))].keys()):
                    gt_key = 'gt'
                else:
                    gt_key = 'image'
                if len(dataset[max(0,int(idx//(hi_frame_sep*2)*hi_frame_sep*2))][gt_key].size()) == 4:
                    val_data['hq'] = torch.cat((dataset[max(0,int(idx//(hi_frame_sep*2)*hi_frame_sep*2))][gt_key][0,:,:,:].unsqueeze(0), dataset[min(gt_size[2],int((idx//(hi_frame_sep*2)+1)*hi_frame_sep*2))][gt_key][0,:,:,:].unsqueeze(0))).unsqueeze(0).cuda()
                elif len(dataset[max(0,int(idx//(hi_frame_sep*2)*hi_frame_sep*2))][gt_key].size()) == 3:
                    val_data['hq'] = torch.cat((dataset[max(0,int(idx//(hi_frame_sep*2)*hi_frame_sep*2))][gt_key].unsqueeze(0), dataset[min(gt_size[2],int((idx//(hi_frame_sep*2)+1)*hi_frame_sep*2))][gt_key].unsqueeze(0))).unsqueeze(0).cuda()
            
            if (idx == max(0,int(idx//(hi_frame_sep*2)*hi_frame_sep*2))) or (idx == min(len(dataset)-1,int(idx+lo_frame_sep))):
                masks.append(1)
            else:
                masks.append(0)
            
            if mode not in ['bil','bic']:
                with torch.no_grad():
                    
                    time_start = time.time()
                    results = first_stage_model(val_data)
                    torch.cuda.synchronize()
                    time_stop = time.time()
                    if mode.upper() == 'SR':
                        result = results['out']
                    elif mode.upper() == 'STF':
                        result = results['out']
                        if 'corr_score' in list(results.keys()):
                            corr = results['corr_score']
                    
                    result_img = (tensor2img(result).astype(float))
                result = result.detach().cpu()
            else:
                if 'gt' in list(val_data_.keys()):
                    gt_key = 'gt'
                else:
                    gt_key = 'image'
                if mode == 'bil':
                    result = cv2.resize(val_data_['lq'][:,1,:,:,:].squeeze().numpy(), (val_data_[gt_key].squeeze().numpy().shape[1],val_data_[gt_key].squeeze().numpy().shape[0]),\
                                            interpolation=cv2.INTER_LINEAR).astype(float)
                elif mode == 'bic':
                    result = cv2.resize(val_data_['lq'][:,1,:,:,:].squeeze().numpy(), (val_data_[gt_key].squeeze().numpy().shape[1],val_data_[gt_key].squeeze().numpy().shape[0]),\
                                            interpolation=cv2.INTER_CUBIC).astype(float)
                result_img = (tensor2img(torch.tensor(result)[None,None,...]).astype(float))
            result_img[result_img>255] = 255
            
            if 'gt' in list(val_data.keys()):
                gt = val_data['gt'].detach().cpu()
            else:
                gt = val_data['image'].detach().cpu()
            hi_img = tensor2img(gt).astype(float)
            if len(hi_img.shape) == 3:
                hi_img = hi_img[:,:,0]
            elif len(hi_img.shape) == 2:
                pass

            diff = (result_img - hi_img)
            mse = np.mean((diff)**2)
            psnr = 10. * np.log10(255. * 255. / mse)
            aad = abs(diff).mean()
            _ssim = ssim(result_img,hi_img, data_range=255)
            logger.info(f"psnr is {psnr} dB")
            logger.info(f"aad is {aad}")
            logger.info(f"ssim is {_ssim}")
            psnrs.append(psnr)
            aads.append(aad)
            ssims.append(_ssim)
            indices.append(idx)
            times.append(time_stop-time_start)
            if mode not in ['bil','bic']:
                if mode.upper() == 'STF' and 'corr_score' in list(results.keys()):
                    logger.info(f"attention high is {corr.detach().cpu().squeeze().numpy()[-1]}")
                    atts.append(corr.detach().cpu().squeeze().numpy())
            Image.fromarray(result_img.astype(np.uint8)).save(f"{(out_dir / Path(val_data_['lq_path']).stem)}_{psnr}.png")

    if mode not in ['bil','bic']:
        if mode.upper() == 'STF' and 'corr_score' in list(results.keys()):
            att_all = np.vstack(atts)
            
            if att_all.shape[1] == 4:
                result_dict = {'index': indices, 'time': times, 'psnr':psnrs, 'aad': aads, 'ssim': ssims, 't-1 lo':att_all[:,0], 't lo': att_all[:,1], 't+1 lo': att_all[:,2], 't hi': att_all[:,3]}
            elif att_all.shape[1] == 5:
                result_dict = {'index': indices, 'time': times, 'psnr':psnrs, 'aad': aads, 'ssim': ssims, 't-1 lo':att_all[:,0], 't lo': att_all[:,1], 't+1 lo': att_all[:,2], 't-1 hi': att_all[:,3], 't+1 hi': att_all[:,4]}
        else:
            result_dict = {'index': indices, 'time': times, 'psnr':psnrs, 'aad': aads, 'ssim': ssims}
    else:
        result_dict = {'index': indices, 'time': times, 'psnr':psnrs, 'aad': aads, 'ssim': ssims}
    if masks:
        result_dict['mask'] = masks
    import pandas as pd
    pd.DataFrame(result_dict).to_csv(out_dir / f'error_{test_set_name}_{mode}_align_{args.align_features_ok}_lr_r_{lo_frame_sep}_hr_d_{2*hi_frame_sep}_b0_{b0}_swinvx_{rank}.csv')
    logger.info('done')