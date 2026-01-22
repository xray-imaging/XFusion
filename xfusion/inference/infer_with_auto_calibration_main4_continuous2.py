import torch
from xfusion.inference.model.calibrator_models2 import FlowCalibratorModel
from xfusion import config  
from xfusion.utils import yaml_load
from xfusion.utils import preprocess_ddm_data3 as preprocess_ddm_data
from xfusion.utils import is_gamma_modified, match_phantom_shimadzu_frame_indices2, is_phantom_frame_continuous
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
from natsort import natsorted
from skimage import transform
from tqdm import tqdm


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def load_images(x, img_type, frame_sel, ph_img_files:List, sh_img_files:List, indices = [1,17], device = 0, use_half_framerate_ok=True):
    
    if img_type in ['actual','actual2']:
        assert frame_sel is not None
        assert indices is not None
        assert ph_img_files is not None
        assert sh_img_files is not None
        img_, img_lr_, imgs_lr_ = [], [], []
        bbox = None
        for i in indices:
            if use_half_framerate_ok:
                idx = frame_sel.index[i]
                idx_pre = frame_sel.index[max(i-1,0)]
                idx_post = frame_sel.index[min(i+1,len(frame_sel.index)-1)]
                img = [ph_img_files[idx_pre],ph_img_files[idx],ph_img_files[idx_post]]

                idx_lr = frame_sel['corresponding_shimadzu_idx'].iloc[i]
                img_lr = sh_img_files[idx_lr]
                imgs_lr = [sh_img_files[max(0,idx_lr-1)],sh_img_files[min(idx_lr+1,len(sh_img_files)-1)]]
            else:
                idx_pre = frame_sel.iloc[i]['phantom_idx1']
                idx_post = frame_sel.iloc[i]['phantom_idx2']
                # with the full framerate, no phantom images are held out as testing data hence only 2 images per inference
                # spatial conditioning is per image, independent of other images
                img = [ph_img_files[idx_pre],ph_img_files[idx_post]]
                img_lr = sh_img_files[i]
                imgs_lr = [sh_img_files[max(0,i-1)],sh_img_files[min(i+1,len(sh_img_files)-1)]]

            imgs_phantom, imgs_shimadzu, bbox, row, col, row_lr, col_lr = preprocess_ddm_data(x, img, img_lr, imgs_lr, base_width_increment_=16, base_height_increment_=16, bbox=bbox,minmax=False)
            if use_half_framerate_ok:
                img = [torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32,device=device) for im in imgs_phantom]
                img_lr = torch.from_numpy(imgs_shimadzu[0]).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32,device=device)
                imgs_lr = [torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32,device=device) for im in imgs_shimadzu[1:]]
            else:
                img = [torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32) for im in imgs_phantom]
                img_lr = torch.from_numpy(imgs_shimadzu[0]).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
                imgs_lr = [torch.from_numpy(im).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32) for im in imgs_shimadzu[1:]]

            img_.append(img)
            img_lr_.append(img_lr)
            imgs_lr_.append(imgs_lr)
        img_new = [torch.cat([img_[j][i] for j in range(len(img_))],dim=0) for i in range(len(img_[0]))]
        img_lr_new = torch.cat(img_lr_,dim=0)
        imgs_lr_new = [torch.cat([imgs_lr_[j][i] for j in range(len(imgs_lr_))],dim=0) for i in range(len(imgs_lr_[0]))]
    
    else:
        raise Exception(f"Image type {img_type} not implemented yet.")
    return img_new, img_lr_new, imgs_lr_new, bbox, row, col, row_lr, col_lr


def run_inference(args):
    input_dir = args.input_dir
    case_list = args.case_list
    tfm_file = Path(args.tfm_file)
    meta_file_scale = Path(args.meta_file_scale)
    img_class = args.img_class
    model_file = args.model_file
    use_half_framerate_ok = args.traverse_mode == 'double'
    exp_root = Path(config.get_inf_data_dirs(input_dir.name))
    device = args.device
    
    opt_path = args.arch_opt
    opt = yaml_load(opt_path)
    opt['manual_seed'] = 10
    torch.manual_seed(opt['manual_seed'])

    if 'patchsize' in opt['network_g']:
        opt['network_g']['patchsize'] = [(sz,sz) for sz in opt['network_g']['patchsize']]
    model_config = opt['network_g']
    del model_config['type']
    
    if args.model_type == 'EDVRModel':
       
        from xfusion.inference.model.edvr_models import EDVRSTFTempRank
        model_config['num_frame'] = 5
        model_config['num_frame_hi'] = 0
        model_config['center_frame_idx'] = 1
        model = EDVRSTFTempRank(**model_config)
    elif args.model_type == 'SwinIRModel':
        
        from xfusion.train.basicsr.archs.VideoTransformerSTF_arch import PatchTransformerSTF
        model_config['num_frame'] = 3
        # model_config['window_size'][0] += (args.num_frame - 3)
        model_config['window_size'][1:] = [4] * len(model_config['window_size'][1:])
        model_config['depths'] = [d * 1 for d in model_config['depths']]
        model_config['num_heads'] = [h * 2 for h in model_config['num_heads']]
        model_config['embed_dim'] = 384
        model_config['num_feat_ext'] = 384
        del model_config['adapt_deformable_conv']
        # model_config['align_features_ok'] = False
        # model_config['adapt_deformable_conv'] = False
        model = PatchTransformerSTF(**model_config)

    
    calibrator_model = torch.nn.DataParallel(FlowCalibratorModel(None,model,model_type=args.model_type))
    calibrator_model = calibrator_model.module
    if args.model_type == 'EDVRModel':
        weights_stf = torch.load(model_file)['params']
    elif args.model_type == 'SwinIRModel':
        weights_stf = torch.load(model_file)['params_ema']
    
    calibrator_model.model.load_state_dict(weights_stf)
    
    calibrator_model.to(device=device)
    calibrator_model.eval()
    try:
        calibrator_model = torch.compile(calibrator_model, mode="max-autotune", fullgraph=False)
    except:
        print('torch.compile not supported by the current virtual environment...')
    
    for case_id in tqdm(case_list):
        
            
        if img_class == 'actual':
            case_id_ = int(case_id.split('.')[0][2:])
        elif img_class == 'actual2':
            case_id_ = int(case_id.split('_')[0])

        root_dir = input_dir / case_id

        # shimadzu metafile check
        sh_metafiles = [o for o in root_dir.glob('*') if (o.is_file() and o.suffix == '.ini')]
        assert len(sh_metafiles) == 1, f"error: number of shimadzu metadata ini files is {len(sh_metafiles)}. Please keep one unique file and rerun the program."
        sh_metafile = sh_metafiles[0]

        sh_dirs = [o for o in root_dir.glob('*') if (o.is_dir() and o.stem == 'TIFF16')]
        sh_dirs_img = list(sh_dirs[0].glob('*'))
        sh_dir_img = sh_dirs_img[0]
        sh_img_files = natsorted(list(sh_dir_img.glob('*.tiff')))

        ph_dirs = [o for o in root_dir.glob('*') if (o.is_dir() and o.stem != 'TIFF16')]
        ph_dir_img = ph_dirs[0]
        ph_img_files = natsorted(list(ph_dir_img.glob('*.tif')))

        ph_temp_metafiles = list(ph_dir_img.glob('*.xml'))
        assert len(ph_temp_metafiles) == 1, f"error: number of phantom xml files is {len(ph_temp_metafiles)}. Please keep one unique file and rerun the program."
        ph_temp_metafile = ph_temp_metafiles[0]

        assert not is_gamma_modified(ph_temp_metafile), f"error: parameter gamma modified during operation. Please repeat the experiment and keep gamma at 0."

        temp_cal, reverse_temp_cal, error_log = match_phantom_shimadzu_frame_indices2(ph_temp_metafile, sh_metafile)
        if not use_half_framerate_ok:
            frame_sel = pd.DataFrame(temp_cal)
        else:
            assert is_phantom_frame_continuous(reverse_temp_cal), f"error: matched phantom frame numbers are not continuous. Please make sure the Shimadzu camera framerate is smaller than Phantom framerate, repeat the experiment and rerun the program."
            frame_sel = pd.DataFrame(reverse_temp_cal)
            frame_sel.set_index('phantom_idx',inplace=True)
        
        scale_factor = np.load(meta_file_scale)
        mat = np.load(tfm_file)
        tfm = transform.EuclideanTransform(mat)
        x_ = np.array([*scale_factor,tfm.rotation,*tfm.translation])
        
        indices_ = list(frame_sel.index)
        indices_ = [i-min(indices_) for i in indices_]
        img_new, img_lr_new, imgs_lr_new, bbox, row, col, row_lr, col_lr = load_images(x_, img_class, frame_sel, ph_img_files, sh_img_files, indices = list(range(len(frame_sel.index))),device=device, use_half_framerate_ok=use_half_framerate_ok)
        bbox = torch.from_numpy(bbox.astype(np.int32))
        full_sizes = torch.tensor([row,col,row_lr,col_lr],dtype=bbox.dtype,device=bbox.device)
        
        if use_half_framerate_ok:
            aads, ssims, psnrs = [], [], []
        if use_half_framerate_ok:
            hi_imgs = []
        result_imgs, lo_imgs = [], []
        if not use_half_framerate_ok:
            result_dir = exp_root / case_id / "continuous"
        else:
            result_dir = exp_root / case_id / "double"
        result_dir.mkdir(exist_ok=True,parents=True)
        tensors_to_save = {
                            'img': img_new,
                            'img_lr': img_lr_new,
                            'imgs_lr': imgs_lr_new,
                            'bbox': bbox,
                            'full_sizes': full_sizes
                            }

        # Save the dictionary of tensors to a file
        torch.save(tensors_to_save, result_dir/ f"{case_id}.pt")
        for sample_idx in range(img_lr_new.size()[0]):
            img = [im[sample_idx:sample_idx+1].to(device) for im in img_new]
            imgs_lr = [im[sample_idx:sample_idx+1].to(device) for im in imgs_lr_new]
            img_lr = img_lr_new[sample_idx:sample_idx+1].to(device)

            imgs_lr_ = torch.cat([imgs_lr[0],img_lr,imgs_lr[1]],dim=0).unsqueeze(0)
            if not use_half_framerate_ok:
                imgs_hr_ = torch.cat([img[0],img[1]],dim=0).unsqueeze(0)
            else:
                imgs_hr_ = torch.cat([img[0],img[2]],dim=0).unsqueeze(0)
            
            results_ = calibrator_model.reconstruct_full(imgs_lr_, imgs_hr_, None if not use_half_framerate_ok else img[1], use_half_framerate_ok=use_half_framerate_ok)
            
            # no external spatial calibration: this will not happen in continuous inference
            if use_half_framerate_ok:
                psnrs.append(results_['psnr'])
                aads.append(results_['aad'])
                ssims.append(results_['ssim'])
            
            result_imgs.append(results_['result_img'][None,...])
            if use_half_framerate_ok:
                hi_imgs.append(results_['hi_img'][None,...])
            lo_imgs.append(results_['lo_img'][None,...])

        # will not happen with continuous inference
        if use_half_framerate_ok:
            result_dict = {'index': indices_, 'psnr':psnrs, 'aad': aads, 'ssim': ssims}
            
        if use_half_framerate_ok:
            out_dir = result_dir
            out_filename = tfm_file.stem
            pd.DataFrame(result_dict).to_csv(out_dir / f'error_{out_filename}_{case_id_}.csv')
        
        result_img = np.concatenate(result_imgs,axis=0)
        if use_half_framerate_ok:
            hi_img = np.concatenate(hi_imgs,axis=0)
        lo_img = np.concatenate(lo_imgs,axis=0)
        if use_half_framerate_ok:
            np.savez(result_dir/case_id,result_img=result_img.astype(np.uint8), hi_img=hi_img.astype(np.uint8), lo_img=lo_img.astype(np.uint8))
        else:
            np.savez(result_dir/case_id,result_img=result_img.astype(np.uint8), lo_img=lo_img.astype(np.uint8))
        print(f"done with case {case_id_}.")

    print('done')