import cv2
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from xfusion.utils import average_selected_frames_phantom, average_frames_shimadzu, normalize_img, remove_calibrator_background, construct_coordinate_systems_quasi2,construct_coordinate_systems_quasi,\
                  pair_square_centers, calculate_isotropic_scale_factor, affine_transformation_shimadzu_img, affine_transformation_shimadzu_img_nonuniform_scaling
from xfusion.constants import FACTOR, BALL_RADIUS, RENORM_THRESH, MEAN_FILTER_KERNEL_SIZE, SIMILARITY_THRESH
from skimage import transform
from skimage.segmentation import clear_border

import time
from xfusion import config

def run_calibration(args):

    start_time = time.time()
    case_id = args.cal_id
    root_dir = args.cal_dir / case_id
    model_path = args.cal_model_file
    tol_pos = args.tol_pos
    tol_area = args.tol_area
    tol_rotation = args.tol_rotation
    device = args.cal_device
    if (model_path.exists()):
        
        from xfusion.train.RAFT.core.raft import RAFT
        flow_model = RAFT(args)
        states_ = torch.load(model_path)
        states = {}
        for k in list(states_.keys()):
            if k.startswith('module.flow_model'):
                states['module'+k[len('module.flow_model'):]] = states_[k]
        flow_model = torch.nn.DataParallel(flow_model)
        msg = flow_model.load_state_dict(states,strict=False)
    else:
        raise Exception(f"Error: {model_path} does not exist.")
    
    flow_model = flow_model.module
    
    flow_model.to(device=device)
    flow_model.eval()
    
    out_dir = Path(config.get_calibration_dirs()) / args.cal_dir.name
    out_dir.mkdir(parents=True,exist_ok=True)
    
    cal_results = {}
    if args.verbose:
        print("loading images and correcting shape and size...", end="")
    
    sh_dirs = [o for o in root_dir.glob('*') if (o.is_dir() and o.stem == 'TIFF16')]
    assert len(sh_dirs) == 1, f"error: number of shimadzu folders is {len(sh_dirs)}. Please keep one unique folder and rerun the program."
    ph_dirs = [o for o in root_dir.glob('*') if (o.is_dir() and o.stem != 'TIFF16')]
    assert len(ph_dirs) == 1, f"error: number of phantom folders is {len(ph_dirs)}. Please keep one unique folder and rerun the program."

    sh_dirs_img = list(sh_dirs[0].glob('*'))
    assert len(sh_dirs_img) == 1, f"error: number of shimadzu image folders is {len(sh_dirs_img)}. Please keep one unique folder and rerun the program."
    sh_dir_img = sh_dirs_img[0]
    ph_dir_img = ph_dirs[0]

    ph_temp_metafiles = list(ph_dir_img.glob('*.xml'))
    assert len(ph_temp_metafiles) <= 1, f"error: number of phantom xml files is {len(ph_temp_metafiles)}. Please keep one unique file and rerun the program."
    if len(ph_temp_metafiles) == 1:
        ph_temp_metafile = ph_temp_metafiles[0]
    else:
        from natsort import natsorted
        ph_temp_imgfiles = natsorted(list(ph_dir_img.glob('*.tif')))
        assert len(ph_temp_imgfiles) > 0, f"error: number of phantom images is {len(ph_temp_imgfiles)}. Please specify a folder with a positive number of images and rerun the program."
        ph_temp_metafile = ph_temp_imgfiles[0]

    img_phantom = average_selected_frames_phantom(ph_temp_metafile)
    img_shimadzu = average_frames_shimadzu(sh_dir_img)

    img_phantom = np.fliplr(img_phantom)
    img_shimadzu = cv2.resize(img_shimadzu,(int(img_shimadzu.shape[1] * FACTOR),int(img_shimadzu.shape[0] * FACTOR)), interpolation=cv2.INTER_LINEAR)
    
    if args.verbose:
        print("done.")
        print("detecting foreground of calibrator...", end="")

    mask_phantom = remove_calibrator_background(img_phantom)
    mask_shimadzu = remove_calibrator_background(img_shimadzu)
    if args.verbose:
        print("done.")
        print("renormalizing image intensities...", end="")

    th_lo_shimadzu = np.percentile(img_shimadzu[mask_shimadzu],RENORM_THRESH*100)
    th_hi_shimadzu = np.percentile(img_shimadzu[mask_shimadzu],100-RENORM_THRESH*100)

    th_lo_phantom = np.percentile(img_phantom[mask_phantom],RENORM_THRESH*100)
    th_hi_phantom = np.percentile(img_phantom[mask_phantom],100-RENORM_THRESH*100)

    phantom_img = normalize_img(img_phantom,th_lo=th_lo_phantom,th_hi=th_hi_phantom)
    shimadzu_img = normalize_img(img_shimadzu,th_lo=th_lo_shimadzu,th_hi=th_hi_shimadzu)
    if args.verbose:
        print("done.")
        print("subtracting uneven illumination...", end="")
    
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BALL_RADIUS + 1, 2 * BALL_RADIUS + 1))

    shimadzu_img = cv2.morphologyEx(shimadzu_img, cv2.MORPH_TOPHAT, disk_kernel)
    phantom_img = cv2.morphologyEx(phantom_img, cv2.MORPH_TOPHAT, disk_kernel)
    if args.verbose:
        print("done.")
        print("Smoothing...", end="")
    mean_filter_kernel_size = (MEAN_FILTER_KERNEL_SIZE,MEAN_FILTER_KERNEL_SIZE)
    shimadzu_img = cv2.blur(shimadzu_img,mean_filter_kernel_size)
    phantom_img = cv2.blur(phantom_img,mean_filter_kernel_size)
    shimadzu_img = (shimadzu_img / shimadzu_img.max() * 255).astype(np.uint8)
    phantom_img = (phantom_img / phantom_img.max() * 255).astype(np.uint8)
    if args.verbose:
        print("done.")
        print("segmenting squares...", end="")

    ret2,mask_phantom = cv2.threshold(phantom_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask_shimadzu = shimadzu_img >= ret2
    mask_phantom = clear_border(mask_phantom)
    mask_shimadzu = clear_border(mask_shimadzu)

    if args.verbose:
        print("done.")
    
    if args.verbose:
        print("pairing squares from phantom and shimadzu...", end="")

    mask_phantom_labeled, mask_shimadzu_labeled, region_props = construct_coordinate_systems_quasi(mask_phantom, mask_shimadzu)
    # compute the angle to be rotated by, from shimadzu to phantom
    rotate_xy_ = (np.arctan2(region_props['b1_shimadzu'][1],region_props['b1_shimadzu'][0])-np.arctan2(region_props['b1_phantom'][1],region_props['b1_phantom'][0]))/np.pi*180
    if not abs(rotate_xy_ / 180 * np.pi) <= tol_rotation:
        row, col = mask_shimadzu.shape
        tf_shift = transform.SimilarityTransform(translation=[-col/2, -row/2])
        tf_shift_inv = transform.SimilarityTransform(translation=[col/2, row/2])
        tf_rotation = transform.EuclideanTransform(rotation=rotate_xy_ / 180 * np.pi)
        mask_shimadzu = transform.warp(mask_shimadzu*255,inverse_map=(tf_shift + (tf_rotation + tf_shift_inv)).inverse) > 0
        shimadzu_img = transform.warp(shimadzu_img,inverse_map=(tf_shift + (tf_rotation + tf_shift_inv)).inverse)

   
    imgs_lr_ = torch.from_numpy(shimadzu_img).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32,device=device)
    row_lr, col_lr = imgs_lr_.size()[-2:]
    
    
    imgs_hr_ = torch.from_numpy(phantom_img).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32,device=device)
    
    row_hr, col_hr = imgs_hr_.size()[-2:]
    padsize_row = row_hr - row_lr
    padsize_col = col_hr - col_lr
    padrad_row = padsize_row // 2
    padrad_col = padsize_col // 2
    imgs_lr_ = F.pad(imgs_lr_,(padrad_col,padsize_col-padrad_col,padrad_row,padsize_row-padrad_row),mode='constant',value=0)
    crop_window_hr = [col_hr//2-args.window_size//2, row_hr//2-args.window_size//2,\
                        col_hr//2+args.window_size//2, row_hr//2+args.window_size//2]
    
    lq_split_ = imgs_lr_[...,crop_window_hr[1]:crop_window_hr[3],crop_window_hr[0]:crop_window_hr[2]]
    hq_split_ = imgs_hr_[...,crop_window_hr[1]:crop_window_hr[3],crop_window_hr[0]:crop_window_hr[2]]
    
    with torch.no_grad():
        flow = flow_model(lq_split_,hq_split_, iters=args.flow_iters, test_mode=True)[1]
        
    disp = np.concatenate([flow[:,1,row_hr//2-crop_window_hr[1],col_hr//2-crop_window_hr[0]].cpu().numpy(), flow[:,0,row_hr//2-crop_window_hr[1],col_hr//2-crop_window_hr[0]].cpu().numpy()],axis=0)
    disp = disp[None,...]
    center_phantom = np.array([[row_hr//2,col_hr//2]]) + disp
    center_shimadzu = np.array([[row_hr//2,col_hr//2]]) - np.array([[padrad_row,padrad_col]])
    
    mask_phantom_labeled, mask_shimadzu_labeled, region_props = construct_coordinate_systems_quasi2(mask_phantom, mask_shimadzu, center_phantom, center_shimadzu, crop_window_hr, padrad_row, padrad_col, flow.cpu().numpy())
    if region_props['changed']:
        print(f"warning: a tie-breaking issue occurred...")
    region_props['tol_pos'] = tol_pos
    region_props['tol_area'] = tol_area
    shimadzu_centroids, list_index_sh_, phantom_centroids, list_index_ph_ = pair_square_centers(**region_props)
    if args.verbose:
        print(f"{shimadzu_centroids.shape[0]} squares paired.")
        print("done.")
    cal_results['num_pairs'] = shimadzu_centroids.shape[0]
    mask_phantom_ = sum([mask_phantom_labeled==i for i in list_index_ph_])
    mask_shimadzu_ = sum([mask_shimadzu_labeled==i for i in list_index_sh_])

    scale_factor, pixel_size_um_phantom, pixel_size_um_shimadzu = calculate_isotropic_scale_factor(mask_phantom_, mask_shimadzu_, SIMILARITY_THRESH, verbose = args.verbose)
    cal_results['scale_factor'] = scale_factor
    cal_results['pixel_size_um_phantom'] = pixel_size_um_phantom
    cal_results['pixel_size_um_shimadzu'] = pixel_size_um_shimadzu
    if args.verbose:
        print("registering shimadzu image to phantom pass 1 affine... ")
    transformation_rigid_matrix, grid_mask_shimadzu_pad_warped, shimadzu_centroids_warped = affine_transformation_shimadzu_img(mask_shimadzu_, mask_phantom_, shimadzu_centroids, phantom_centroids, scale_factor)
    
    delta = phantom_centroids-shimadzu_centroids_warped
    displacement = np.linalg.norm(delta,ord=2,axis=1)
    if args.verbose:
        print(f"Residual error after affine transformation:")
        print(f"Mean is {displacement.mean():.2f} and std is {displacement.std():.2f} pixels")
        print(f"Max is {displacement.max():.2f} and min is {displacement.min():.2f} pixels")
        print("done.")
        print("registering shimadzu image to phantom pass 2 nonuniform scaling...")

    cal_results['mean_residual_post_affine'] = displacement.mean()
    cal_results['std_residual_post_affine'] = displacement.std()
    cal_results['max_residual_post_affine'] = displacement.max()
    cal_results['min_residual_post_affine'] = displacement.min()

    transformation_rigid_matrix, grid_mask_shimadzu_pad_warped, shimadzu_centroids_warped, scale_factor_corrected = affine_transformation_shimadzu_img_nonuniform_scaling(mask_shimadzu_, mask_phantom_, shimadzu_centroids, phantom_centroids, scale_factor, shimadzu_centroids_warped)
    
    if args.verbose:
        print("saving nonuniform scaling factors and affine transformation matrix...", end="")
    
    np.save(out_dir / f'tfm_{case_id}_{tol_pos:.2f}_{tol_area:.2f}_flow_init_rotation_inv_lbl_outliers',transformation_rigid_matrix.params)
    np.save(out_dir / f"scale_factor_{case_id}_{tol_pos:.2f}_{tol_area:.2f}_corrected_flow_init_rotation_inv_lbl_outliers",np.array([scale_factor_corrected[0][1],scale_factor_corrected[0][0]]))
    np.save(out_dir / f'rot_{case_id}_{tol_pos:.2f}_{tol_area:.2f}_flow_init_rotation_inv_lbl_outliers',rotate_xy_)
    if args.verbose:
        print('done.')
        
    delta = phantom_centroids-shimadzu_centroids_warped
    displacement = np.linalg.norm(delta,ord=2,axis=1)

    if args.verbose:
        print(f"Residual error after rescaling:")
        print(f"Mean is {displacement.mean():.2f} and std is {displacement.std():.2f} pixels")
        print(f"Max is {displacement.max():.2f} and min is {displacement.min():.2f} pixels")

    cal_results['mean_residual_post_ns'] = displacement.mean()
    cal_results['std_residual_post_ns'] = displacement.std()
    cal_results['max_residual_post_ns'] = displacement.max()
    cal_results['min_residual_post_ns'] = displacement.min()
    cal_results['changed'] = region_props['changed']
    out_file = out_dir / f"calibration_results_{case_id}_{tol_pos:.2f}_{tol_area:.2f}_flow_init_rotation_inv_lbl_outliers.txt"
    if out_file.exists():
        os.remove(out_file)
    for k,v in cal_results.items():
        with open(out_file,'a') as f:
            f.write(f"{k}:{v}\n")
    end_time = time.time()
    if args.verbose:
        print(f"total calibration time is {(end_time-start_time):.2f}s... done.")
