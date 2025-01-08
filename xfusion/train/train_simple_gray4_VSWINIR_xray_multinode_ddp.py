import datetime
import logging
import math
import time
import torch
import os
from os import path as osp
import numpy as np
from pathlib import Path
from xfusion import config
from xfusion.train.basicsr.data import build_dataloader, build_dataset
from xfusion.train.basicsr.data.data_sampler import EnlargedSampler
from xfusion.train.basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from xfusion.train.basicsr.models import build_model
from xfusion.train.basicsr.utils import (AvgTimer, FullTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from xfusion.train.basicsr.utils.options import copy_opt_file, dict2str
from xfusion.train.basicsr.utils import set_random_seed
from xfusion.train.basicsr.utils.dist_util import get_dist_info, init_dist
from xfusion.train.basicsr.data.xray_dataset import XrayDatasetSTF, XrayVideoTestDatasetSTF
from torch.profiler import profile, record_function, ProfilerActivity
#from distutils.util import strtobool
#from basicsr.data.reds_dataset import REDSDatasetSTF
#from basicsr.data.video_test_dataset import VideoTestDatasetSTF

from mpi4py import MPI
comm = MPI.COMM_WORLD

#TODO: log info to replace print
def parse_options_(root_path, args, opt):
    import random
    from xfusion.utils import _postprocess_yml_value
    
    if args.force_yml == 'none':
        args.force_yml = None
    
    print(args.pretrain_network_g)
    if args.pretrain_network_g != 'none':
        assert os.path.isfile(args.pretrain_network_g)
        opt['path']['pretrain_network_g'] = args.pretrain_network_g
        assert os.path.isdir(str(Path(args.pretrain_network_g).parents[1] / 'training_states'))
        opt['path']['resume_state'] = str(Path(args.pretrain_network_g).parents[1] / 'training_states' / (Path(args.pretrain_network_g).stem.split('_')[-1]+'.state'))
        
        assert os.path.isfile(opt['path']['resume_state'])

        opt['path']['strict_load_g'] = True
        opt['path']['ignore_resume_networks'] = 'network_g'
    
    print(f"debugging---{args.opt}")
    #TODO: change path to yml file
    
    print(opt)
    
    assert opt['train']['scheduler']['type'] == 'CosineAnnealingRestartLR'
    opt['datasets']['train']['hi_frame_sep'] = 1
    opt['train']['scheduler']['periods'] = [p // args.iter_div_factor for p in opt['train']['scheduler']['periods']]
    opt['train']['total_iter'] /= args.iter_div_factor
    opt['train']['single_epoch_ok'] = args.single_epoch_ok
    opt['train']['warmup_iter'] = args.warmup_iter
    opt['datasets']['train']['dataset_enlarge_ratio'] = args.dataset_enlarge_ratio
    opt['datasets']['train']['sub_dataset'] = args.sub_dataset
    opt['datasets']['train']['sub_dataset_frac'] = args.sub_dataset_frac
    opt['datasets']['train']['num_frame'] = args.num_frame
    opt['datasets']['val']['num_frame'] = args.num_frame
    opt['network_g']['num_frame'] = args.num_frame
    opt['network_g']['window_size'][0] += (args.num_frame - 3)
    opt['network_g']['window_size'][1:] = [args.window_size_spatial] * len(opt['network_g']['window_size'][1:])
    opt['network_g']['depths'] = [d * args.scale_depth for d in opt['network_g']['depths']]
    opt['network_g']['num_heads'] = [int(np.ceil(h * args.scale_mha)) for h in opt['network_g']['num_heads']]
    opt['network_g']['embed_dim'] = args.embed_dim
    opt['network_g']['num_feat_ext'] = args.embed_dim
    opt['network_g']['resi_connection'] = args.resi_connection
    opt['network_g']['conv_window_size'] = args.conv_window_size
    
    # distributed settings
    if args.launcher == 'none':
        print(f"debugging...{opt}")
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            #size = int(os.environ['NTOTRANKS'])
            #rank = int(os.environ['PMI_RANK'])
            #print(f"{size}, {rank}")
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    postfix = f"_{args.drop_path_rate}{args.iter_div_factor}{args.num_frame}{args.embed_dim}{args.batch_size}{args.initial_lr}{args.num_workers}{args.warmup_iter}{args.scale_depth}{args.scale_mha}{args.window_size_spatial}{args.dataset_enlarge_ratio}{args.sub_dataset}{args.sub_dataset_frac}{args.resi_connection}{args.conv_window_size}_renorm_base_rem1_world_size_{opt['world_size']}"
    opt['name'] += postfix
    
    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = args.is_train

    # debug setting
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    assert opt['num_gpu'] != 'auto'
    #if opt['num_gpu'] == 'auto':
    #    opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        #if phase == 'train' and args.launcher == 'polaris':
        #    opt['datasets'][phase]['num_worker_per_gpu'] = 0
        #if dataset.get('dataroot_gt') is not None:
        #    dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        #if dataset.get('dataroot_lq') is not None:
        #    dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if args.is_train:
        experiments_root = opt['path'].get('experiments_root')
        if experiments_root is None:
            experiments_root = osp.join(root_path, 'experiments')
        experiments_root = osp.join(experiments_root, opt['name'])

        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = opt['path'].get('results_root')
        if results_root is None:
            results_root = osp.join(root_path, 'results')
        results_root = osp.join(results_root, opt['name'])

        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt, args

def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            #train_set = XrayDatasetSTF(dataset_opt)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            if opt['train']['single_epoch_ok']:
                total_epochs = 1
                total_iters = num_iter_per_epoch * total_epochs
            else:
                total_iters = int(opt['train']['total_iter'])
                total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = XrayVideoTestDatasetSTF(dataset_opt)
            #val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join(opt['root_path'], 'experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        assert device_id == 0
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(args, opt):
    root_path = config.get_train_dirs()
    # parse options, set distributed setting, set random seed
    opt, args = parse_options_(root_path, args, opt)
    print(f"drop path rate is {args.drop_path_rate}")
    opt['network_g']['drop_path_rate'] = args.drop_path_rate
    opt['root_path'] = root_path
    if type(args.dir_hi_train) is not list:
        if str(args.dir_hi_train) != 'none':
            if opt['datasets']['train']['type'] == 'XrayDatasetSTF':
                opt['datasets']['train']['dataroot_gt'] = [str(args.dir_hi_train)]
            else:
                opt['datasets']['train']['dataroot_gt'] = str(args.dir_hi_train)
    else:
        assert opt['datasets']['train']['type'] == 'XrayDatasetSTF'
        if not any([True if str(dh) == 'none' else False for dh in args.dir_hi_train]):
            opt['datasets']['train']['dataroot_gt'] = [str(dh) for dh in args.dir_hi_train]
    
    if type(args.dir_lo_train) is not list:
        if str(args.dir_lo_train) != 'none':
            if opt['datasets']['train']['type'] == 'XrayDatasetSTF':
                opt['datasets']['train']['dataroot_lq'] = [str(args.dir_lo_train)]
            else:
                opt['datasets']['train']['dataroot_lq'] = str(args.dir_lo_train)
    else:
        assert opt['datasets']['train']['type'] == 'XrayDatasetSTF'
        if not any([True if str(dl) == 'none' else False for dl in args.dir_lo_train]):
            opt['datasets']['train']['dataroot_lq'] = [str(dl) for dl in args.dir_lo_train]
    
    if type(args.dir_hi_val) is not list:
        if str(args.dir_hi_val) != 'none':
            if opt['datasets']['val']['type'] == 'XrayVideoTestDatasetSTF':
                opt['datasets']['val']['dataroot_gt'] = [str(args.dir_hi_val)]
            else:
                opt['datasets']['val']['dataroot_gt'] = str(args.dir_hi_val)
    else:
        assert opt['datasets']['val']['type'] == 'XrayVideoTestDatasetSTF'
        if not any([True if str(dh) == 'none' else False for dh in args.dir_hi_val]):
            opt['datasets']['val']['dataroot_gt'] = [str(dh) for dh in args.dir_hi_val]
    
    if type(args.dir_lo_val) is not list:
        if str(args.dir_lo_val) != 'none':
            if opt['datasets']['val']['type'] == 'XrayVideoTestDatasetSTF':
                opt['datasets']['val']['dataroot_lq'] = [str(args.dir_lo_val)]
            else:
                opt['datasets']['val']['dataroot_lq'] = str(args.dir_lo_val)
    else:
        assert opt['datasets']['val']['type'] == 'XrayVideoTestDatasetSTF'
        if not any([True if str(dl) == 'none' else False for dl in args.dir_lo_val]):
            opt['datasets']['val']['dataroot_lq'] = [str(dl) for dl in args.dir_lo_val]
    
    if type(args.path_train_meta_info_file) is not list:
        if str(args.path_train_meta_info_file) != 'none':
            if opt['datasets']['train']['type'] == 'XrayDatasetSTF':
                opt['datasets']['train']['meta_info_file'] = [str(args.path_train_meta_info_file)]
            else:
                opt['datasets']['train']['meta_info_file'] = str(args.path_train_meta_info_file)
    else:
        assert opt['datasets']['train']['type'] == 'XrayDatasetSTF'
        if not any([True if str(pt) == 'none' else False for pt in args.path_train_meta_info_file]):
            opt['datasets']['train']['meta_info_file'] = [str(pt) for pt in args.path_train_meta_info_file]
    
    if type(args.path_val_meta_info_file) is not list:
        if str(args.path_val_meta_info_file) != 'none':
            if opt['datasets']['val']['type'] == 'XrayVideoTestDatasetSTF':
                opt['datasets']['val']['meta_info_file'] = [str(args.path_val_meta_info_file)]
            else:
                opt['datasets']['val']['meta_info_file'] = str(args.path_val_meta_info_file)
    else:
        assert opt['datasets']['val']['type'] == 'XrayVideoTestDatasetSTF'
        if not any([True if str(pv) == 'none' else False for pv in args.path_val_meta_info_file]):
            opt['datasets']['val']['meta_info_file'] = [str(pv) for pv in args.path_val_meta_info_file]
    
    opt['datasets']['train']['num_worker_per_gpu'] = args.num_workers
    opt['datasets']['train']['batch_size_per_gpu'] = args.batch_size
    opt['train']['optim_g']['lr'] = args.initial_lr
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    #copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    #logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    #if opt['name'].split('_')[-1] == 'tfmer':
    opt['datasets']['val']['scale']=4
    opt['datasets']['val']['gt_size'] = [640,1280]
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result
    opt['train']['total_iter'] = total_iters
    opt['train']['scheduler']['periods'] = [total_iters for p in opt['train']['scheduler']['periods']]
    #data = train_loader.dataset.__getitem__(0)
    #data_val = val_loaders[0].dataset.__getitem__(0)
    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    if args.sub_dataset:
        if total_epochs > (start_epoch+1):
            total_epochs -= 1
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer, full_timer = AvgTimer(), AvgTimer(), FullTimer()
    comm.barrier()
    start_time = time.time()

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(skip_first=int(np.ceil(total_iters/100)), wait=5, warmup=1, active=args.profiler_iter_num, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(opt['path']['experiments_root'] + '/newlog'),
        record_shapes=True,
        with_stack=True) as prof:
        for epoch in range(start_epoch, total_epochs):
            train_sampler.set_epoch(epoch)
            prefetcher.reset()
            train_data = prefetcher.next()

            while train_data is not None:
                data_timer.record()

                current_iter += 1
                prof.step()
                if current_iter > total_iters:
                    break
                # update learning rate
                model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                # training
                model.feed_data(train_data)
                model.optimize_parameters(current_iter)
                #iter_timer.record()
                full_timer.record()
                if current_iter == 1:
                    # reset start time in msg_logger for more accurate eta_time
                    # not work in resume mode
                    msg_logger.reset_start_time()
                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': full_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

                # save models and training states
                if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    model.save(epoch, current_iter)

                # validation
                if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                    if len(val_loaders) > 1:
                        logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                    for val_loader in val_loaders:
                        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

                data_timer.start()
                #iter_timer.start()
                full_timer.start()
                train_data = prefetcher.next()
            # end of iter
        

    # end of epoch
    # wait for all processes to resync
    comm.barrier()
    end_time = time.time()

    sections = config.XFUSION_PARAMS
    config.write(os.path.join(opt['path']['experiments_root'],'xfusion.conf'), args=args, sections=sections)
    consumed_time = str(datetime.timedelta(seconds=int(end_time - start_time)))

    times = full_timer.get_full_times()
    for t in times:
        with open(opt['path']['experiments_root'] + '/' + opt['name']+'_world_size_'+str(opt['world_size'])+'_rank'+str(opt['rank'])+'.txt','a') as f:
            f.write(str(t)+'\n')
    
    with open(opt['path']['experiments_root'] + '/' + opt['name']+'_world_size_'+str(opt['world_size'])+'_rank'+str(opt['rank'])+'.txt','a') as f:
        f.write(str(start_time)+'\n')
        f.write(str(end_time)+'\n')
        f.write(consumed_time+'\n')
    
    
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()
