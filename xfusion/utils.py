#  Copyright 2018-2022 BasicSR Authors

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# Modifications Copyright 2024-2026 xfusion authors

import os
import cv2
import yaml
import numpy as np
import torch
import time
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from collections import OrderedDict
from PIL import Image

from xfusion import log

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])

def compile_dataset(args):
    cases_hi = natsorted(list(args.dir_hi_convert.glob('*')))
    for case_hi in tqdm(cases_hi):
        files_hi = natsorted(list(case_hi.glob('*.png')))
        out_case_hi = args.out_dir_hi / case_hi.stem
        Path(out_case_hi).mkdir(exist_ok=True,parents=True)
        out_case_lo = args.out_dir_lo / case_hi.stem
        Path(out_case_lo).mkdir(exist_ok=True,parents=True)
        for file_hi in files_hi:
            img_hi  = Image.open(file_hi).convert('L')
            file_lo = args.dir_lo_convert / case_hi.stem / file_hi.name
            img_lo  = Image.open(file_lo).convert('L')
            img_hi  = np.array(img_hi)
            img_lo  = np.array(img_lo)
            img_hi  = Image.fromarray(np.concatenate([img_hi[:,:,None],img_hi[:,:,None],img_hi[:,:,None]],axis=2))
            img_lo  = Image.fromarray(np.concatenate([img_lo[:,:,None],img_lo[:,:,None],img_lo[:,:,None]],axis=2))
            log.info("Converting to gray scale low (%s) and high (%s) res images" % (file_lo, file_hi))
            img_hi.save(out_case_hi / file_hi.name)
            img_lo.save(out_case_lo / file_hi.name)

def average_selected_frames_phantom(temp_file):
    if temp_file.suffix == '.xml':
        from lxml import etree
        parser = etree.XMLParser(resolve_entities=False)
        tree = etree.parse((temp_file), parser)
        root = tree.getroot()
        timestamps_all = root.find('TIMEBLOCK').findall('Time')
        frame_idx_e = {idx:int(ts.attrib['Frame']) for idx,ts in enumerate(timestamps_all) if ts.text.split(' ')[-1]=='E'}
    elif temp_file.suffix == '.tif':
        frame_idx_e = {}
    else:
        raise Exception(f"error: phantom file extension is {temp_file.suffix}. Please keep only .tif and .xml files in the directory and rerun the program.")
    frames = []
    if frame_idx_e:
        frame_idx = natsorted(list(frame_idx_e.keys()))
        
    else:
        from xfusion.constants import PHANTOM_FRAME_NUMBER
        assert len(list(temp_file.parent.glob('*.tif'))) - PHANTOM_FRAME_NUMBER >= 0, f"total number of frames is {len(list(temp_file.parent.glob('*.tif')))}.\
             Please make sure there are at least {PHANTOM_FRAME_NUMBER} and rerun the program."
        mid_frame_idx = min(len(list(temp_file.parent.glob('*.tif'))) // 2 + 1, len(list(temp_file.parent.glob('*.tif'))) - PHANTOM_FRAME_NUMBER)
        print(f"phantom has not received any flag from shimadzu, using the {PHANTOM_FRAME_NUMBER} frames starting from {mid_frame_idx} instead...")
        frame_idx = list(range(mid_frame_idx,mid_frame_idx+PHANTOM_FRAME_NUMBER))
    
    for idx in frame_idx:
        img = np.array(Image.open(temp_file.parent / f"img_{str(idx).zfill(6)}.tif"))
        frames.append(img.astype(float))

    img = sum(frames) / len(frames)
    img = img.astype(np.uint16)
    return img

def average_frames_shimadzu(img_dir):
    img_files = natsorted(list(img_dir.glob('*.tiff')))

    frames = []
    for idx,img_file in enumerate(img_files):
        if idx == 0:
            continue
        img = np.array(Image.open(img_file))
        frames.append(img.astype(float))

    img = sum(frames) / len(frames)
    img = img.astype(np.uint16)
    return img

def remove_calibrator_background(img):

    # import cv2
    # from skimage import morphology
    # from skimage.morphology import white_tophat
    # from skimage.filters import rank
    from skimage.segmentation import clear_border
    from scipy import ndimage as ndi

    from xfusion.constants import BALL_RADIUS, MEAN_FILTER_KERNEL_SIZE
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BALL_RADIUS + 1, 2 * BALL_RADIUS + 1))

    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, disk_kernel) #white_tophat(img,footprint=morphology.disk(BALL_RADIUS))
    mean_filter_kernel_size = (MEAN_FILTER_KERNEL_SIZE,MEAN_FILTER_KERNEL_SIZE)
    img = cv2.blur(img,mean_filter_kernel_size) #rank.mean

    ret,_ = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = clear_border(img >= ret)

    disk_kernel_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(BALL_RADIUS,BALL_RADIUS))
    mask = cv2.morphologyEx(mask.astype(img.dtype), cv2.MORPH_CLOSE, disk_kernel_gap)

    mask_label, num_dots = ndi.label(mask)
    list_index = np.arange(1, num_dots + 1)
    
    list_sum = ndi.sum(mask, labels=mask_label, index=list_index)
    mask = mask_label == list_index[np.where(list_sum == list_sum.max())[0][0]]
    # mask = morphology.binary_closing(mask,footprint=morphology.disk(BALL_RADIUS//2))
    return mask

def flatten_distances_unique(list_dist, list_index,similarity_thresh):
    list_index = [[set(pair) for pair in l] for l in list_index ]
    list_dist_linear = []
    list_index_linear = []
    dmin = np.inf
    for l in list_dist:
        list_dist_linear.extend(l)
        dmin = min(dmin,min(l[l!=0]))
    for l in list_index:
        list_index_linear.extend(l)
    list_dist = list_dist_linear
    list_index = list_index_linear
    
    list_dist_keep = []
    list_dist_remove = []
    list_index_keep = []
    for i, d in enumerate(list_dist):
        # for j, d in enumerate(l):
        if list_index[i] not in list_index_keep:
            if abs(d/dmin-1)<=similarity_thresh:
                list_dist_keep.append(d)
                list_index_keep.append(list_index[i])
            else:
                list_dist_remove.append(d)
    return np.array(list_dist_keep), np.array(list_dist_remove)

def detect_area_center(mat):
    import itertools
    from scipy import ndimage as ndi
    from scipy.ndimage.measurements import center_of_mass
    mat_label, num_dots = ndi.label(mat)
    list_index = np.arange(1, num_dots + 1)
    
    list_sum = ndi.sum(mat, labels=mat_label, index=list_index)
    list_cent = np.asarray(
        center_of_mass(mat, labels=mat_label, index=list_index))
    
    list_dists = [np.sort(np.sqrt((dot[0] - list_cent[:, 0]) ** 2
                                 + (dot[1] - list_cent[:, 1]) ** 2))
                 for dot in list_cent]
    list_indices = [list(itertools.product([i],np.argsort(np.sqrt((dot[0] - list_cent[:, 0]) ** 2
                                 + (dot[1] - list_cent[:, 1]) ** 2)))) for i,dot in enumerate(list_cent)]
    return list_sum, list_cent, list_dists, list_indices, mat_label

def label_comp_detect_center_form_basis_quasi(mask, center=None):
    row, col = mask.shape
    num_dots, list_cent, list_dists, list_indices, mask_labeled = detect_area_center(mask>0)
    if center is None:
        center = np.array([[row//2,col//2]])

    dist = np.sqrt(((list_cent - center)**2).sum(1))
    center_idx = np.where(dist==min(dist))[0][0]
    center_ = list_cent[center_idx]
    num_dots_ = num_dots[center_idx]

    # find 4 nearest neighbors phantom image
    dist_ = np.sqrt(((center_.reshape((1,-1))-list_cent)**2).sum(1))
    nn_indices = np.argsort(dist_)[1:5]
    nn_centers = [list_cent[i] for i in nn_indices]
    nn_vec = np.concatenate([(c-center_).reshape((1,-1)) / np.linalg.norm(c-center_) for c in nn_centers],axis=0)
    pitch = np.array([np.linalg.norm(c-center_) for c in nn_centers]).mean()
    nn_num_dots = np.array([num_dots[i] for i in nn_indices]).mean()
    return center_, nn_vec, {'nn_num_dots':nn_num_dots, 'list_cent':list_cent, 'pitch':pitch,'num_dots':num_dots}, mask_labeled

def construct_coordinate_systems_quasi(mask_phantom, mask_shimadzu, center_phantom=None, center_shimadzu=None):
    
    phantom_center_, nn_vec_phantom, region_props_phantom, mask_phantom_labeled = label_comp_detect_center_form_basis_quasi(mask_phantom, center_phantom)
    shimadzu_center_, nn_vec_shimadzu, region_props_shimadzu, mask_shimadzu_labeled = label_comp_detect_center_form_basis_quasi(mask_shimadzu, center_shimadzu)
    region_props = {'nn_num_dots_phantom':region_props_phantom['nn_num_dots'], 'nn_num_dots_shimadzu':region_props_shimadzu['nn_num_dots'],\
                    'list_cent_phantom':region_props_phantom['list_cent'], 'list_cent_shimadzu':region_props_shimadzu['list_cent'],\
                    'pitch_phantom':region_props_phantom['pitch'], 'pitch_shimadzu':region_props_shimadzu['pitch'],\
                    'num_dots_phantom':region_props_phantom['num_dots'], 'num_dots_shimadzu':region_props_shimadzu['num_dots']}

    ps_norm = np.matmul(nn_vec_phantom, nn_vec_shimadzu.T)

    # find the most in phase pair
    b1_indices = np.where(ps_norm==ps_norm.max())
    b1_phantom = nn_vec_phantom[b1_indices[0][0]]
    b1_shimadzu = nn_vec_shimadzu[b1_indices[1][0]]

    # find the out of phase phantom, then pair with shimadzu
    p_norm = np.matmul(nn_vec_phantom,b1_phantom.reshape((-1,1)))
    b2_index_phantom = np.where(abs(p_norm)==abs(p_norm).min())[0][0]
    b2_phantom = nn_vec_phantom[b2_index_phantom]
    b2_index_shimadzu = np.where(ps_norm[b2_index_phantom]==ps_norm[b2_index_phantom].max())[0][0]
    b2_shimadzu = nn_vec_shimadzu[b2_index_shimadzu]

    region_props['phantom_center_'] = phantom_center_
    region_props['b1_phantom'] = b1_phantom
    region_props['b2_phantom'] = b2_phantom
    region_props['shimadzu_center_'] = shimadzu_center_
    region_props['b1_shimadzu'] = b1_shimadzu
    region_props['b2_shimadzu'] = b2_shimadzu
    return mask_phantom_labeled, mask_shimadzu_labeled, region_props

def construct_coordinate_systems_quasi2(mask_phantom, mask_shimadzu, center_phantom=None, center_shimadzu=None, crop_window_hr=None, padrad_row=None, padrad_col=None, flow = None):
    from scipy.ndimage import map_coordinates
    phantom_center_, nn_vec_phantom, region_props_phantom, mask_phantom_labeled = label_comp_detect_center_form_basis_quasi(mask_phantom, center_phantom)
    shimadzu_center_, nn_vec_shimadzu, region_props_shimadzu, mask_shimadzu_labeled = label_comp_detect_center_form_basis_quasi(mask_shimadzu, center_shimadzu)
    if (crop_window_hr is not None) and (padrad_row is not None) and (padrad_col is not None) and (flow is not None):
        center_shimadzu_ = shimadzu_center_ + np.array([[padrad_row,padrad_col]])
        disp_u = map_coordinates(flow[0,0],[center_shimadzu_[:,0]-crop_window_hr[1],center_shimadzu_[:,1]-crop_window_hr[0]],order=3)
        disp_v = map_coordinates(flow[0,1],[center_shimadzu_[:,0]-crop_window_hr[1],center_shimadzu_[:,1]-crop_window_hr[0]],order=3)
        # disp = np.concatenate([flow[:,1,center_shimadzu_[1]-crop_window_hr[1],center_shimadzu_[0]-crop_window_hr[0]].cpu().numpy(), flow[:,0,center_shimadzu_[1]-crop_window_hr[1],center_shimadzu_[0]-crop_window_hr[0]].cpu().numpy()],axis=0)
        disp = np.concatenate([disp_v,disp_u],axis=0)
        disp = disp[None,...]
        center_phantom_ = center_shimadzu_ + disp
        phantom_center2_, nn_vec_phantom2, region_props_phantom2, mask_phantom_labeled2 = label_comp_detect_center_form_basis_quasi(mask_phantom, center_phantom_)

    if not np.array_equal(phantom_center2_, phantom_center_):
        phantom_center_ = phantom_center2_
        nn_vec_phantom = nn_vec_phantom2
        region_props_phantom = region_props_phantom2
        mask_phantom_labeled = mask_phantom_labeled2
        flag_changed = True
    else:
        flag_changed = False

    region_props = {'nn_num_dots_phantom':region_props_phantom['nn_num_dots'], 'nn_num_dots_shimadzu':region_props_shimadzu['nn_num_dots'],\
                    'list_cent_phantom':region_props_phantom['list_cent'], 'list_cent_shimadzu':region_props_shimadzu['list_cent'],\
                    'pitch_phantom':region_props_phantom['pitch'], 'pitch_shimadzu':region_props_shimadzu['pitch'],\
                    'num_dots_phantom':region_props_phantom['num_dots'], 'num_dots_shimadzu':region_props_shimadzu['num_dots']}

    ps_norm = np.matmul(nn_vec_phantom, nn_vec_shimadzu.T)

    # find the most in phase pair
    b1_indices = np.where(ps_norm==ps_norm.max())
    b1_phantom = nn_vec_phantom[b1_indices[0][0]]
    b1_shimadzu = nn_vec_shimadzu[b1_indices[1][0]]

    # find the out of phase phantom, then pair with shimadzu
    p_norm = np.matmul(nn_vec_phantom,b1_phantom.reshape((-1,1)))
    b2_index_phantom = np.where(abs(p_norm)==abs(p_norm).min())[0][0]
    b2_phantom = nn_vec_phantom[b2_index_phantom]
    b2_index_shimadzu = np.where(ps_norm[b2_index_phantom]==ps_norm[b2_index_phantom].max())[0][0]
    b2_shimadzu = nn_vec_shimadzu[b2_index_shimadzu]

    region_props['phantom_center_'] = phantom_center_
    region_props['b1_phantom'] = b1_phantom
    region_props['b2_phantom'] = b2_phantom
    region_props['shimadzu_center_'] = shimadzu_center_
    region_props['b1_shimadzu'] = b1_shimadzu
    region_props['b2_shimadzu'] = b2_shimadzu
    region_props['changed'] = flag_changed
    return mask_phantom_labeled, mask_shimadzu_labeled, region_props

def pair_square_centers(shimadzu_center_, list_cent_shimadzu, pitch_shimadzu, num_dots_shimadzu, nn_num_dots_shimadzu, b1_shimadzu, b2_shimadzu,\
                        phantom_center_, list_cent_phantom, pitch_phantom, num_dots_phantom, nn_num_dots_phantom, b1_phantom, b2_phantom, **kwargs):
    from xfusion.constants import TOL_POS, TOL_AREA
    if 'tol_pos' in list(kwargs.keys()):
        tol_pos = kwargs['tol_pos']
    else:
        tol_pos = TOL_POS
    if 'tol_area' in list(kwargs.keys()):
        tol_area = kwargs['tol_area']
    else:
        tol_area = TOL_AREA
    # map all shimadzu center points
    sh1 = np.matmul((list_cent_shimadzu - shimadzu_center_) / pitch_shimadzu,b1_shimadzu.reshape((-1,1)))
    sh2 = np.matmul((list_cent_shimadzu - shimadzu_center_) / pitch_shimadzu,b2_shimadzu.reshape((-1,1)))
    list_index_sh = [i for i in range(sh1.shape[0]) if (abs(sh1[i]-np.round(sh1[i])) <= tol_pos and abs(sh2[i]-np.round(sh2[i])) <= tol_pos and abs(num_dots_shimadzu[i]-nn_num_dots_shimadzu)/nn_num_dots_shimadzu <= tol_area)]
    coords_sh = [(int(np.round(sh1[i])),int(np.round(sh2[i]))) for i in list_index_sh]
    
    ph1 = np.matmul((list_cent_phantom - phantom_center_) / pitch_phantom,b1_phantom.reshape((-1,1)))
    ph2 = np.matmul((list_cent_phantom - phantom_center_) / pitch_phantom,b2_phantom.reshape((-1,1)))
    list_index_ph = [i for i in range(ph1.shape[0]) if (abs(ph1[i]-np.round(ph1[i])) <= tol_pos and abs(ph2[i]-np.round(ph2[i])) <= tol_pos and abs(num_dots_phantom[i]-nn_num_dots_phantom)/nn_num_dots_phantom <= tol_area)]
    coords_ph = [(int(np.round(ph1[i])),int(np.round(ph2[i]))) for i in list_index_ph]

    # plus 1 to be consistent with ndi labeling convention i.e., 0 for the background
    list_index_coord_sh_ = [(i+1,coord) for i,coord in zip(list_index_sh,coords_sh) if coord in coords_ph]
    list_index_coord_ph_ = [(i+1,coord) for i,coord in zip(list_index_ph,coords_ph) if coord in coords_sh]

    list_index_sh_ = [i for i,coord in list_index_coord_sh_]
    list_index_ph_ = [i for i,coord in list_index_coord_ph_]
    
    coords_sh_ = np.array([coord for i,coord in list_index_coord_sh_],dtype=[('x', int), ('y', int)])
    coords_ph_ = np.array([coord for i,coord in list_index_coord_ph_],dtype=[('x', int), ('y', int)])

    list_cent_phantom_ = [list_cent_phantom[i-1:i,:] for i in list_index_ph_]
    list_cent_shimadzu_ = [list_cent_shimadzu[i-1:i,:] for i in list_index_sh_]

    # sort both shimadzu and phantom coordinates in natural order
    order_sh = np.argsort(coords_sh_)
    order_ph = np.argsort(coords_ph_)
    list_cent_phantom_ = np.vstack(list_cent_phantom_)[order_ph,:]
    list_cent_shimadzu_ = np.vstack(list_cent_shimadzu_)[order_sh,:]
    phantom_centroids = np.hstack((list_cent_phantom_[:,1:2],list_cent_phantom_[:,0:1]))
    shimadzu_centroids = np.hstack((list_cent_shimadzu_[:,1:2],list_cent_shimadzu_[:,0:1]))
    return shimadzu_centroids, list_index_sh_, phantom_centroids, list_index_ph_

def calculate_isotropic_scale_factor(mask_phantom_, mask_shimadzu_,similarity_thresh, verbose = True):
    from xfusion.constants import G400_PITCH_UM, FACTOR
    num_dots_phantom, _, list_dists_phantom, list_indices_phantom, mask_phantom_labeled = detect_area_center(mask_phantom_>0)
    num_dots_shimadzu, _, list_dists_shimadzu, list_indices_shimadzu, mask_shimadzu_labeled = detect_area_center(mask_shimadzu_>0)

    list_dist_phantom_keep, list_dist_phantom_remove = flatten_distances_unique(list_dists_phantom, list_indices_phantom, similarity_thresh)
    if verbose:
        print(f"min and max of the pitches from phantom are: {min(list_dist_phantom_keep):.2f} and {max(list_dist_phantom_keep):.2f}")
        print(f"min and max of the higher harmonics from phantom are: {min(list_dist_phantom_remove[list_dist_phantom_remove!=0]):.2f} and {max(list_dist_phantom_remove[list_dist_phantom_remove!=0]):.2f}")
    pixel_size_um_phantom = G400_PITCH_UM / (list_dist_phantom_keep.mean())
    if verbose:
        print(f"pixel size (um) of phantom is {pixel_size_um_phantom:.2f}")

    list_dist_shimadzu_keep, list_dist_shimadzu_remove = flatten_distances_unique(list_dists_shimadzu, list_indices_shimadzu, similarity_thresh)
    if verbose:
        print(f"min and max of the pitches from shimadzu are: {min(list_dist_shimadzu_keep):.2f} and {max(list_dist_shimadzu_keep):.2f}")
        print(f"min and max of the higher harmonics from shimadzu are: {min(list_dist_shimadzu_remove[list_dist_shimadzu_remove!=0]):.2f} and {max(list_dist_shimadzu_remove[list_dist_shimadzu_remove!=0]):.2f}")
    # list_dist_shimadzu_keep = [d for l in list_dist_shimadzu for d in l if abs(d/l[1]-1)<=0.05]
    pixel_size_um_shimadzu = G400_PITCH_UM / (list_dist_shimadzu_keep.mean()) * FACTOR
    if verbose:
        print(f"pixel size (um) of shimadzu is {pixel_size_um_shimadzu:.2f}")

    scale_factor = (pixel_size_um_shimadzu / pixel_size_um_phantom) / FACTOR
    return scale_factor, pixel_size_um_phantom, pixel_size_um_shimadzu

def affine_transformation_shimadzu_img(mask_shimadzu_, mask_phantom_, shimadzu_centroids, phantom_centroids, scale_factor):
    # import cv2
    from skimage import transform
    grid_mask_sh = cv2.resize(mask_shimadzu_.astype(float),(int(np.floor(mask_shimadzu_.shape[1] * scale_factor)),int(np.floor(mask_shimadzu_.shape[0] * scale_factor))), interpolation=cv2.INTER_LINEAR)

    # pad shimadzu image to the same size of the phantom one
    padsize_row = mask_phantom_.shape[0] - int(np.floor(mask_shimadzu_.shape[0] * scale_factor)) #grid_mask_sh.shape[0]
    padsize_col = mask_phantom_.shape[1] - int(np.floor(mask_shimadzu_.shape[1] * scale_factor)) #grid_mask_sh.shape[1]
    padrad_row = padsize_row // 2
    padrad_col = padsize_col // 2
    grid_mask_shimadzu_pad = np.pad(grid_mask_sh,((padrad_row,padsize_row-padrad_row),(padrad_col,padsize_col-padrad_col)),mode='constant',constant_values=0)

    shimadzu_centroids = shimadzu_centroids*scale_factor+np.array([[padrad_col,padrad_row]])

    transformation_rigid_matrix = transform.estimate_transform('euclidean',shimadzu_centroids, phantom_centroids)

    shimadzu_centroids_warped = transformation_rigid_matrix(shimadzu_centroids)
    grid_mask_shimadzu_pad_warped = transform.warp(grid_mask_shimadzu_pad,inverse_map=transformation_rigid_matrix.inverse)
    return transformation_rigid_matrix, grid_mask_shimadzu_pad_warped, shimadzu_centroids_warped

def affine_transformation_shimadzu_img_nonuniform_scaling(mask_shimadzu_, mask_phantom_, shimadzu_centroids, phantom_centroids, scale_factor, shimadzu_centroids_warped):
    # import cv2
    from skimage import transform
    from scipy.optimize import curve_fit
    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A*x + B

    nrows, ncols = mask_phantom_.shape
    delta = phantom_centroids-shimadzu_centroids_warped

    scale_factor_correction = np.zeros((4))
    popt, pcov = curve_fit(f, (shimadzu_centroids[:,1]-nrows//2),(delta[:,1]))
    scale_factor_correction[0] = popt[0]
    scale_factor_correction[1] = popt[1]

    popt, pcov = curve_fit(f, (shimadzu_centroids[:,0]-ncols//2),(delta[:,0]))
    scale_factor_correction[2] =popt[0]
    scale_factor_correction[3] = popt[1]

    scale_factor_corrected = np.array([[scale_factor*(1+scale_factor_correction[2]),scale_factor*(1+scale_factor_correction[0])]])
    grid_mask_sh = cv2.resize(mask_shimadzu_.astype(float),(int(np.floor(mask_shimadzu_.shape[1] * scale_factor_corrected[0,0])),int(np.floor(mask_shimadzu_.shape[0] * scale_factor_corrected[0,1]))), interpolation=cv2.INTER_LINEAR)
    padsize_row = mask_phantom_.shape[0] - int(np.floor(mask_shimadzu_.shape[0] * scale_factor_corrected[0,1])) #grid_mask_sh.shape[0]
    padsize_col = mask_phantom_.shape[1] - int(np.floor(mask_shimadzu_.shape[1] * scale_factor_corrected[0,0])) #grid_mask_sh.shape[1]
    padrad_row = padsize_row // 2
    padrad_col = padsize_col // 2
    grid_mask_shimadzu_pad = np.pad(grid_mask_sh,((padrad_row,padsize_row-padrad_row),(padrad_col,padsize_col-padrad_col)),mode='constant',constant_values=0)

    shimadzu_centroids = shimadzu_centroids*scale_factor_corrected+np.array([[padrad_col,padrad_row]])
    transformation_rigid_matrix = transform.estimate_transform('euclidean',shimadzu_centroids, phantom_centroids)

    shimadzu_centroids_warped_ = transformation_rigid_matrix(shimadzu_centroids)
    grid_mask_shimadzu_pad_warped = transform.warp(grid_mask_shimadzu_pad,inverse_map=transformation_rigid_matrix.inverse)
    return transformation_rigid_matrix, grid_mask_shimadzu_pad_warped, shimadzu_centroids_warped_, scale_factor_corrected

#compared functions below with any common ones in the calibration (inference) code
def is_gamma_modified(meta_file):
    from lxml import etree
    parser = etree.XMLParser(resolve_entities=False)

    tree = etree.parse((meta_file), parser)
    root = tree.getroot()
    gamma = (root.find('CameraSetup').find('Gamma').text)
    return float(gamma) != 0

def format_date(ds):
    import calendar
    month_abbr_to_num = {month: index for index, month in enumerate(calendar.month_abbr) if month}
    return ds[2]+'-'+str(month_abbr_to_num[ds[0]]).zfill(2)+'-'+ds[1]

def format_time(ts):
    ts_parts = ts[0].split('.')+ts[1].split('.')
    return ts_parts[0],''.join(ts_parts[1:])

def string_to_nanosecond_timestamp(date_string, format_string):
    from datetime import datetime
    """
    Converts a string to a timestamp.

    Args:
        date_string: The date string to convert.
        format_string: The format string that corresponds to the date string.

    Returns:
        An integer representing the timestamp.
    """
    datetime_object = datetime.strptime(date_string, format_string)
    timestamp_s = int(datetime_object.timestamp())
    return timestamp_s

def match_phantom_shimadzu_frame_indices2(temp_metafile_phantom, temp_metafile_shimadzu, verbose = True):
    '''
        shimadzu frames parsed based on "recframe"
    '''
    import re
    import configparser
    from lxml import etree
    format_string = "%Y-%m-%d %H:%M:%S"
    if verbose:
        print(f'loading temporal calibration...',end="")
    
    parser = etree.XMLParser(resolve_entities=False)
    tree = etree.parse((temp_metafile_phantom), parser)
    root = tree.getroot()
    framerate_phantom = float(root.find('CameraSetup').find('FrameRateDouble').text)
    timestamps_all = root.find('TIMEBLOCK').findall('Time')
    dates_all = root.find('TIMEBLOCK').findall('Date')
    #TODO: also obtain phantom's height and width for spatial calibration
    config = configparser.ConfigParser()
    config.read(temp_metafile_shimadzu)
    frametime_shimadzu = config.get('REC_INFO','RecSpeed')
    frame_num_shimadzu = int(config.get('REC_INFO','RecFrame')) #int(config.get('PLAY_INFO','FrameStop'))
    frametime_shimadzu_ = int(re.findall(r'\d+(?:,\d{3})*', frametime_shimadzu)[0].replace(",",""))
    frametime_unit =''.join([i for i in frametime_shimadzu if (not i.isdigit()) and (i!=',')]).lower()
    if frametime_unit == 'ns':
        framerate_shimadzu = 10**9 / frametime_shimadzu_
    else:
        raise Exception('Other frame time units than ns not implemented. Please check the Shimadzu camera acquisition settings, make sure the unit is ns and rerun the program.')

    if verbose:
        print(f"frame rate difference/ratio is: {framerate_shimadzu/framerate_phantom}",end="")

    frame_idx_e = {idx:int(ts.attrib['Frame']) for idx,ts in enumerate(timestamps_all) if ts.text.split(' ')[-1]=='E'}
    if len(frame_idx_e) == 0:
        raise Exception(f"No Shimadzu triggering detected during Phantom's imaging time. Please synchronize the two cameras' imaging time window, enable Phantom to capture Shimadzu's trigger signal, and rerun the program.")
    
    indices_e = list(frame_idx_e.keys())
    frames_e = list(frame_idx_e.values())
    ds_e = [format_date(ds.text.split(' ')[1:]) for ds in dates_all if int(ds.attrib['Frame']) in frames_e]
    ts_e = [format_time(ts.text.split(' ')[:2]) for ts in timestamps_all if int(ts.attrib['Frame']) in frames_e]
    timestamps_e = [string_to_nanosecond_timestamp(ds+' '+ts[0], format_string)*10**9+int(ts[1])*10 for ds,ts in zip(ds_e,ts_e)]
    phantom_ts = [ts-timestamps_e[0] for ts in timestamps_e]
    shimadzu_ts = [int(config.get(f"IMAGE_INFO_{i}",'ImageRelative'))-int(config.get(f"IMAGE_INFO_{1}",'ImageRelative')) for i in range(1,frame_num_shimadzu+1)]
    
    shimadzu_indices = []
    phantom_indices_1 = []
    phantom_indices_2 = []
    ph_times = np.array(phantom_ts)
    sh_times = np.array(shimadzu_ts)
    for idx, sh in enumerate(shimadzu_ts):
        if sh >= ph_times.max():
            break
        ph_idx1 = indices_e[np.where(ph_times<=sh)[0].max()]
        ph_idx2 = indices_e[np.where(ph_times>sh)[0].min()]
        if ph_idx1 <= ph_idx2:
            shimadzu_indices.append(idx)
            phantom_indices_1.append(ph_idx1)
            phantom_indices_2.append(ph_idx2)


    temp_cal = {'shimadzu_idx':shimadzu_indices, 'phantom_idx1': phantom_indices_1, 'phantom_idx2': phantom_indices_2}
    
    # reverse calibration
    phantom_indices = []
    corresponding_shimadzu_indices = []
    error_log = {'phantom_indices':[],'time_difference':[]}
    for idx, ph in enumerate(phantom_ts):
        if ph <= max(sh_times) and ph >= min(sh_times):
            time_diff = (sh_times - ph)
            sh_idx = np.where(abs(time_diff) == abs(time_diff).min())[0][0]
            if abs(time_diff).min() <= 10:
                phantom_indices.append(indices_e[idx])
                corresponding_shimadzu_indices.append(sh_idx)
            error_log['phantom_indices'].append(indices_e[idx])
            error_log['time_difference'].append(time_diff[sh_idx])

    reverse_temp_cal = {'phantom_idx':phantom_indices, 'corresponding_shimadzu_idx':corresponding_shimadzu_indices}
    
    # common_ts = max(list(set(phantom_ts) & set(shimadzu_ts)))
    return temp_cal, reverse_temp_cal, error_log

def is_phantom_frame_continuous(reverse_temp_cal):
    phantom_indices = np.array(natsorted(list(reverse_temp_cal['phantom_idx'])))
    idx_diff = np.diff(phantom_indices)
    return np.all(idx_diff == 1)

def normalize_img(img,th_lo=None,th_hi=None):
    if th_lo is None:
        th_lo = img.min()
    if th_hi is None:
        th_hi = img.max()
    img[img<=th_lo]=th_lo
    img[img>=th_hi]=th_hi
    img = ((img - img.min()) / (img.max()-img.min()) * 255).astype(np.uint8)
    return img

def normalize_img2(img):
    from xfusion.constants import RENORM_THRESH
    nonzero_min = img[img>0].min()
    img[img==0] = nonzero_min
    
    th_lo = np.percentile(img,RENORM_THRESH*100)
    th_hi = np.percentile(img,100-RENORM_THRESH*100)
    img_normed = normalize_img(img,th_lo=th_lo,th_hi=th_hi)
    return img_normed

def preprocess_ddm_data3(x, img, img_lr, imgs_lr, base_width_increment_, base_height_increment_, preprocessed = False, bbox=None, minmax=True):
    from skimage import transform
    import largestinteriorrectangle as lir
    from xfusion.constants import FACTOR
    #beginning the function call of the objective
    img_phantom = np.array(Image.open(img[1]))
    height, width = img_phantom.shape
    # first read shimadzu
    imgs_shimadzu = []
    tfm = transform.EuclideanTransform(rotation=x[2],translation=x[3:5])
    for i, img_file in enumerate([img_lr] + imgs_lr):
        img_shimadzu = np.array(Image.open(img_file))
        if i == 0:
            row_lr, col_lr = img_shimadzu.shape
        #HACK: for the cpu calibration, the scaling is separated from the Euclidean transformation and implemented with resizing
        if not preprocessed:
            img_shimadzu = cv2.resize(img_shimadzu,(int(np.floor(img_shimadzu.shape[1] * (x[1] * FACTOR))),int(np.floor(img_shimadzu.shape[0] * (x[0] * FACTOR)))), interpolation=cv2.INTER_LINEAR)
        else:
            img_shimadzu = cv2.resize(img_shimadzu,(int(np.floor(img_shimadzu.shape[1] * (x[1]))),int(np.floor(img_shimadzu.shape[0] * (x[0])))), interpolation=cv2.INTER_LINEAR)
        
        if not preprocessed:
            if minmax:
                img_shimadzu = normalize_img2(img_shimadzu)
            else:
                img_shimadzu = (img_shimadzu / 257).astype(np.uint8)

        padsize_row = height - img_shimadzu.shape[0]
        padsize_col = width - img_shimadzu.shape[1]
        padrad_row = padsize_row // 2
        padrad_col = padsize_col // 2
        foreground = np.ones_like(img_shimadzu) * 255.
        img_shimadzu = np.pad(img_shimadzu,((padrad_row,padsize_row-padrad_row),(padrad_col,padsize_col-padrad_col)),mode='constant',constant_values=0)
        foreground = np.pad(foreground,((padrad_row,padsize_row-padrad_row),(padrad_col,padsize_col-padrad_col)),mode='constant',constant_values=0)
        img_shimadzu = transform.warp(img_shimadzu,inverse_map=tfm.inverse)
        foreground = transform.warp(foreground,inverse_map=tfm.inverse)
        if (i == 0) and (bbox is None):
            
            bbox = lir.lir(foreground.astype(bool))
            assert (bbox[2]-bbox[0]) >= base_width_increment_
            assert (bbox[3]-bbox[1]) >= base_height_increment_
            bbox[2] -= ((bbox[2]-bbox[0]) % base_width_increment_)
            bbox[3] -= ((bbox[3]-bbox[1]) % base_height_increment_)
        
        img_shimadzu = img_shimadzu[bbox[1]:bbox[3],bbox[0]:bbox[2]] * 255.
        
        imgs_shimadzu.append(img_shimadzu)

    imgs_phantom = []
    for i, img_file in enumerate(img):
        img_phantom = np.array(Image.open(img_file))
        if i == 0:
            row, col = img_phantom.shape
        if not preprocessed:
            img_phantom = np.fliplr(img_phantom)

        if not preprocessed:
            if minmax:
                img_phantom = normalize_img2(img_phantom)
            else:
                img_phantom = (img_phantom / 257).astype(np.uint8)
        img_phantom = img_phantom[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        
        imgs_phantom.append(img_phantom)
    return imgs_phantom, imgs_shimadzu, bbox, row, col, row_lr, col_lr

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    import math
    import torch
    from torchvision.utils import make_grid
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def estimate_nonuniform_similarity_2d_analytical(X, Y):
    """
    Estimate 2D transformation with nonuniform scaling, rotation, and translation analytically.
    Assumes: y ≈ R * D * x + t, where D is diagonal (no shear).
    
    Args:
        X: (N, 2) source points
        Y: (N, 2) target points
        
    Returns:
        A: Full linear transform matrix (R @ D)
        t: Translation vector
        R: Rotation matrix
        D: Diagonal scaling matrix
    """
    # X = np.asarray(X)
    # Y = np.asarray(Y)

    # 1. Center the points
    mu_X = X.mean(dim=0)
    mu_Y = Y.mean(dim=0)
    Xc = X - mu_X
    Yc = Y - mu_Y

    # 2. Estimate diagonal scaling
    #HACK: diagonal scaling is scaled by cos(alpha) and should be down-scaled to correct
    dx = (Yc[:, 0] * Xc[:, 0]).sum() / (Xc[:, 0]**2).sum()
    dy = (Yc[:, 1] * Xc[:, 1]).sum() / (Xc[:, 1]**2).sum()
    D = torch.diag(torch.tensor([dx, dy],device=Xc.device))

    # 3. Apply scaling to centered source
    X_scaled = Xc @ D

    # 4. Estimate rotation via SVD
    H = X_scaled.T @ Yc
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    D[0,0] = 2 * D[0,0] / (R[0,0] + R[1,1])
    D[1,1] = 2 * D[1,1] / (R[0,0] + R[1,1])
    # 5. Compute translation
    A = R @ D
    t = mu_Y - mu_X @ A.T

    return A, t, R, D

def estimate_transformation_parameters(t, R, D):

    alpha = (torch.atan2(R[1, 0], R[0, 0]) + torch.atan2(-R[0, 1], R[1, 1]))/2
    x = torch.tensor([D[1,1], D[0,0], alpha, t[0], t[1]],device=t.device, dtype=t.dtype)
    return x