
from xfusion.utils import tensor2img, estimate_nonuniform_similarity_2d_analytical, estimate_transformation_parameters
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import numpy as np
from torch import linalg as LA
import torch.nn.functional as F
from xfusion.constants import FACTOR

def round_rect_corner_coordinates(bbox, width, height):
    bbox[:,0] = torch.ceil(bbox[:,0])
    bbox[0,1] = torch.floor(bbox[0,1])
    bbox[1,1] = torch.ceil(bbox[1,1])
    bbox[0,2] = torch.ceil(bbox[0,2])
    bbox[1,2] = torch.floor(bbox[1,2])
    bbox[:,3] = torch.floor(bbox[:,3])
    bbox[0] = torch.minimum(torch.maximum(bbox[0],torch.tensor(0)),torch.tensor(width-1))
    bbox[1] = torch.minimum(torch.maximum(bbox[1],torch.tensor(0)),torch.tensor(height-1))
    bbox = bbox.int()
    return bbox

def flip_rect_corner_coordinates_vertical(bbox):
    b1, b2, b3, b4 = torch.clone(bbox[:,0:1]), torch.clone(bbox[:,1:2]), torch.clone(bbox[:,2:3]), torch.clone(bbox[:,3:4])
    bbox = torch.cat((b3,b4,b1,b2),dim=1)
    bbox[1] = -bbox[1]
    return bbox

def pytorch_lir(bbox:torch.Tensor, alpha: torch.Tensor, width, height):
    if alpha == 0:
        bbox = round_rect_corner_coordinates(bbox, width, height)
        return torch.tensor([bbox[0,0],bbox[1,0],bbox[0,1],bbox[1,2]]).int()
    if alpha > 0:
        # Convention within the analytical solution is rotation counter-clockwise. 
        # If rotated clockwise, then to flip the coordinates vertically to be compliant
        bbox_ = flip_rect_corner_coordinates_vertical(bbox)
        alpha_ = torch.clone(alpha).detach()
    else:
        bbox_ = torch.clone(bbox)
        alpha_ = torch.clone(-alpha).detach()

    assert alpha_ > 0
    assert alpha_/torch.tensor(np.pi)*180 <= 90, f"Error: rotation is {alpha_/torch.tensor(np.pi)*180}. Only rotations of 0-90 degrees are implemented."
    bbox_ = compute_lir_from_rect_corners(bbox_, alpha_)
    
    if alpha > 0:
        bbox_ = flip_rect_corner_coordinates_vertical(bbox_)
    bbox_ = round_rect_corner_coordinates(bbox_, width, height)
    
    return torch.tensor([bbox_[0,0],bbox_[1,0],bbox_[0,1],bbox_[1,2]])

def compute_lir_from_rect_corners(bbox_, alpha):
    '''
        largest interior rectangle (lir) from coordinates of corner points of a rectangle rotated by alpha (radian)
        out: coordinates of the 4 corners of the largest interior rectangle
    '''
    edge_len1 = LA.norm(bbox_[:,0]-bbox_[:,1],dim=0)
    edge_len2 = LA.norm(bbox_[:,1]-bbox_[:,3],dim=0)
    edge_len1_ = LA.norm(bbox_[:,2]-bbox_[:,3],dim=0)
    edge_len2_ = LA.norm(bbox_[:,0]-bbox_[:,2],dim=0)
    if (edge_len1+edge_len1_) >= (edge_len2+edge_len2_):
        l = (edge_len1+edge_len1_)/2
        s = (edge_len2+edge_len2_)/2
    else:
        s = (edge_len1+edge_len1_)/2
        l = (edge_len2+edge_len2_)/2
    
    anchor = bbox_[:,2:3]
    
    if l*torch.sin(2*alpha) >= s:
        a = s/(2*torch.sin(alpha))
        b = s/(2*torch.cos(alpha))
        x = s/2
    else:
        a = (l*torch.cos(alpha)-s*torch.sin(alpha))/torch.cos(2*alpha)
        b = (s*torch.cos(alpha)-l*torch.sin(alpha))/torch.cos(2*alpha)
        x = a * torch.sin(alpha)
    b3 = torch.tensor([[anchor[0,0]-x*torch.sin(alpha)],[anchor[1,0]-x*torch.cos(alpha)]])
    b1 = b3 + torch.tensor([[0],[-b]])
    b2 = b1 + torch.tensor([[a],[0]])
    b4 = b3 + torch.tensor([[a],[0]])
    bbox = torch.cat((b1,b2,b3,b4),dim=1)

    return bbox

def compute_transformed_corners(padrad_col, padrad_row, padsize_col, padsize_row, width, height, tfm):
    '''
        out: coordinates of the 4 image corners after resizing, padding, and affine transformation
        order is: top left, top right, bottom left, bottom right
    '''
    bbox_idx = torch.tensor([[padrad_col,width-(padsize_col-padrad_col)-1,padrad_col,width-(padsize_col-padrad_col)-1],[padrad_row,padrad_row,height-(padsize_row-padrad_row)-1,height-(padsize_row-padrad_row)-1],[1,1,1,1]]).to(torch.float32)
    bbox_idx = bbox_idx.int()

    affine_grid_ = F.affine_grid(tfm.unsqueeze(0)[:,:2], size=(1,1,height,width), align_corners=True)
    affine_grid_[...,0] = affine_grid_[...,0] * (width-1)/2 + (width-1)/2
    affine_grid_[...,1] = affine_grid_[...,1] * (height-1)/2 + (height-1)/2

    bbox = torch.tensor([[affine_grid_[0,bbox_idx[1,0],bbox_idx[0,0],0], affine_grid_[0,bbox_idx[1,1],bbox_idx[0,1],0], affine_grid_[0,bbox_idx[1,2],bbox_idx[0,2],0], affine_grid_[0,bbox_idx[1,3],bbox_idx[0,3],0]],[affine_grid_[0,bbox_idx[1,0],bbox_idx[0,0],1], affine_grid_[0,bbox_idx[1,1],bbox_idx[0,1],1], affine_grid_[0,bbox_idx[1,2],bbox_idx[0,2],1], affine_grid_[0,bbox_idx[1,3],bbox_idx[0,3],1]]])
    
    return bbox

def get_trs_transformations(x, two_way_ok = True, device=0):
    # forward transformations are independent and do not participate in the grad descent
    rotation_mat = torch.tensor([[torch.cos(x[2]), -torch.sin(x[2]), 0],[torch.sin(x[2]), torch.cos(x[2]), 0],[0, 0, 1]]).to(device)
    translation_mat = torch.tensor([[1, 0, x[3]],[0, 1, x[4]],[0, 0, 1]]).to(device)
    scaling_mat = torch.tensor([[x[1], 0, 0],[0, x[0], 0],[0, 0, 1]]).to(device)

    if two_way_ok:
        rotation_mat_inverse = torch.stack([torch.stack([torch.cos(x[2]), torch.sin(x[2]), torch.tensor(0,dtype=x.dtype, device=device)]),torch.stack([-torch.sin(x[2]), torch.cos(x[2]), torch.tensor(0,dtype=x.dtype, device=device)]),torch.tensor([0, 0, 1],dtype=x.dtype, device=device)],dim=0)
        translation_mat_inverse = torch.stack([torch.stack([torch.tensor(1,dtype=x.dtype, device=device), torch.tensor(0,dtype=x.dtype, device=device), -x[3]]),torch.stack([torch.tensor(0,dtype=x.dtype, device=device), torch.tensor(1,dtype=x.dtype, device=device), -x[4]]),torch.tensor([0, 0, 1],dtype=x.dtype, device=device)],dim=0)
        scaling_mat_inverse = torch.stack([torch.stack([1/x[1], torch.tensor(0,dtype=x.dtype, device=device), torch.tensor(0,dtype=x.dtype, device=device)]),torch.stack([torch.tensor(0,dtype=x.dtype, device=device), 1/x[0], torch.tensor(0,dtype=x.dtype, device=device)]),torch.tensor([0, 0, 1],dtype=x.dtype, device=device)],dim=0)
        return rotation_mat, translation_mat, scaling_mat, rotation_mat_inverse, translation_mat_inverse, scaling_mat_inverse
    return rotation_mat, translation_mat, scaling_mat

class FlowCalibratorModel(nn.Module):
    def __init__(self, flow_model, model, pixel_size_ratio = 4, base_height_increment=32, base_width_increment=32,  base_height_increment_=16, base_width_increment_=16, model_type=None):
        super().__init__()
        # flow_model is trainable and model is the evaluator model with frozen weights to compute loss
        self.flow_model = flow_model
        self.model = model
        self.pixel_size_ratio = pixel_size_ratio
        self.base_height_increment = base_height_increment
        self.base_width_increment = base_width_increment
        self.base_height_increment_ = base_height_increment_
        self.base_width_increment_ = base_width_increment_
        self.model_type = model_type
    
    def fit_calibration_parameters(self, flow_, mask = None, device=0):
        flow = torch.clone(flow_)
        affine_grid_original = F.affine_grid(torch.tensor([[1., 0., 0.],[0., 1., 0.]]).unsqueeze(0), size=(flow.size()[0],1,*flow.size()[2:]), align_corners=True)
        affine_grid_original = affine_grid_original.permute(0,3,1,2).to(device=device)
        if mask is None:
            mask = torch.ones_like(flow[:,0:1],dtype=bool)
        else:
            assert mask.size() == flow[:,0:1].size()
        flow[:,0] = flow[:,0] * 2 / (flow.size()[-1]-1)
        flow[:,1] = flow[:,1] * 2 / (flow.size()[-2]-1)
        affine_grid = affine_grid_original - flow
        X = torch.cat((affine_grid[0,0][mask[0,0]].unsqueeze(1), affine_grid[0,1][mask[0,0]].unsqueeze(1)),dim=1)
        Y = torch.cat((affine_grid_original[0,0][mask[0,0]].unsqueeze(1), affine_grid_original[0,1][mask[0,0]].unsqueeze(1)),dim=1)
        A, t, R, D = estimate_nonuniform_similarity_2d_analytical(X,Y)
        x = estimate_transformation_parameters(t, R, D)
        return x

    def fit_calibration_parameters_skimage_compatible(self, flow,row,col,bbox, row_lr, col_lr, mask=None, device=0):
        affine_grid_original = F.affine_grid(torch.tensor([[1., 0., 0.],[0., 1., 0.]]).unsqueeze(0), size=(flow.size()[0],1,row,col), align_corners=True)
        affine_grid_original = affine_grid_original.permute(0,3,1,2).to(device=device)
        affine_grid_original[0,0] = affine_grid_original[0,0] * ((col-1)/2) + ((col-1)/2)
        affine_grid_original[0,1] = affine_grid_original[0,1] * ((row-1)/2) + ((row-1)/2)
        affine_grid_original = affine_grid_original[...,bbox[1]:bbox[3],bbox[0]:bbox[2]]
        if mask is None:
            mask = torch.ones_like(flow[:,0:1],dtype=bool)
        else:
            assert mask.size() == flow[:,0:1].size()

        affine_grid = affine_grid_original - flow
        X = torch.cat((affine_grid[0,0][mask[0,0]].unsqueeze(1), affine_grid[0,1][mask[0,0]].unsqueeze(1)),dim=1)
        Y = torch.cat((affine_grid_original[0,0][mask[0,0]].unsqueeze(1), affine_grid_original[0,1][mask[0,0]].unsqueeze(1)),dim=1)
        A, t, R, D = estimate_nonuniform_similarity_2d_analytical(X,Y)
        #TODO: check exact offset in the translation
        del_d0 = col_lr*(1-D[0,0])*FACTOR/2
        del_d1 = row_lr*(1-D[1,1])*FACTOR/2
        del_d = torch.tensor([del_d0,del_d1],dtype=t.dtype,device=t.device)
        #HACK: to match skimage and pytorch inplementations ideal to use scale estimated from calibrator
        # use D from the images to make estimates independent of inplementations
        dt = R@del_d + R@(torch.tensor([(col-FACTOR*col_lr)*(1-D[0,0])/2, (row-FACTOR*row_lr)*(1-D[1,1])/2],dtype=R.dtype,device=R.device))
        t = t-dt
        x = estimate_transformation_parameters(t, R, D)
        return x
    '''
        row_lr and col_lr reparameterized as the initially calibrated lr image size
    '''
    def fit_calibration_parameters_skimage_compatible2(self, flow,row,col,bbox, row_lr, col_lr, mask=None, device=0):
        affine_grid_original = F.affine_grid(torch.tensor([[1., 0., 0.],[0., 1., 0.]]).unsqueeze(0), size=(flow.size()[0],1,row,col), align_corners=True)
        affine_grid_original = affine_grid_original.permute(0,3,1,2).to(device=device)
        affine_grid_original[0,0] = affine_grid_original[0,0] * ((col-1)/2) + ((col-1)/2)
        affine_grid_original[0,1] = affine_grid_original[0,1] * ((row-1)/2) + ((row-1)/2)
        affine_grid_original = affine_grid_original[...,bbox[1]:bbox[3],bbox[0]:bbox[2]]
        if mask is None:
            mask = torch.ones_like(flow[:,0:1],dtype=bool)
        else:
            assert mask.size() == flow[:,0:1].size()

        affine_grid = affine_grid_original - flow
        X = torch.cat((affine_grid[0,0][mask[0,0]].unsqueeze(1), affine_grid[0,1][mask[0,0]].unsqueeze(1)),dim=1)
        Y = torch.cat((affine_grid_original[0,0][mask[0,0]].unsqueeze(1), affine_grid_original[0,1][mask[0,0]].unsqueeze(1)),dim=1)
        A, t, R, D = estimate_nonuniform_similarity_2d_analytical(X,Y)
        #TODO: check exact offset in the translation
        del_d0 = col_lr*(1-D[0,0])/2
        del_d1 = row_lr*(1-D[1,1])/2
        del_d = torch.tensor([del_d0,del_d1],dtype=t.dtype,device=t.device)
        #HACK: to match skimage and pytorch inplementations ideal to use scale estimated from calibrator
        # use D from the images to make estimates independent of inplementations
        dt = R@del_d + R@(torch.tensor([(col-col_lr)*(1-D[0,0])/2, (row-row_lr)*(1-D[1,1])/2],dtype=R.dtype,device=R.device))
        t = t-dt
        x = estimate_transformation_parameters(t, R, D)
        return x

    def reconstruct_mosaic(self, imgs_lr, imgs_hr, img_gt, img_size, iters=12, test_mode=True):
        imgs = torch.cat((imgs_hr,img_gt.unsqueeze(1)),dim=1)

        lq_splits_h = list(torch.split(imgs_lr,img_size,dim=-2))
        hq_splits_h = list(torch.split(imgs,img_size,dim=-2))
        recon_splits_h = {}
        for i, (lq_split_h, hq_split_h) in enumerate(zip(lq_splits_h,hq_splits_h)):
            lq_splits_w = list(torch.split(lq_split_h,img_size,dim=-1))
            hq_splits_w = list(torch.split(hq_split_h,img_size,dim=-1))
            recon_splits_w = {}
            for j, (lq_split,hq_split) in enumerate(zip(lq_splits_w,hq_splits_w)):
                if (lq_split.size()[-2] != img_size) and (lq_split.size()[-1] != img_size):
                    # do reverse padding and then crop accordingly
                    img_size_h, img_size_w = lq_split.size()[-2:]
                    lq_split_ = imgs_lr[...,-img_size:,-img_size:]
                    hq_split_ = imgs[...,-img_size:,-img_size:]
                    assert torch.equal(lq_split,lq_split_[...,-img_size_h:,-img_size_w:])
                    assert torch.equal(hq_split,hq_split_[...,-img_size_h:,-img_size_w:])
                    with torch.no_grad():
                        sample = {'lq':lq_split_,'hq':hq_split_[:,:-1],'gt':hq_split_[:,-1]}
                        out = self(sample, iters=iters, test_mode=test_mode)
                        
                    for k in list(out.keys()):
                        if k != 'psnr':
                            if k == 'flows':
                                out[k] = [o[...,-img_size_h:,-img_size_w:] for o in out[k]]
                            elif k == 'lq':
                                out[k] = out[k][...,-int(img_size_h/self.pixel_size_ratio):,-int(img_size_w/self.pixel_size_ratio):]
                            else:
                                out[k] = out[k][...,-img_size_h:,-img_size_w:]
                    if out['flows'][0] is None:
                            print('a')
                
                elif (lq_split.size()[-2] != img_size):
                    img_size_h = lq_split.size()[-2]
                    lq_split_ = imgs_lr[...,-img_size:,j*img_size:(j+1)*img_size]
                    hq_split_ = imgs[...,-img_size:,j*img_size:(j+1)*img_size]
                    assert torch.equal(lq_split,lq_split_[...,-img_size_h:,:])
                    assert torch.equal(hq_split,hq_split_[...,-img_size_h:,:])
                    with torch.no_grad():
                        sample = {'lq':lq_split_,'hq':hq_split_[:,:-1],'gt':hq_split_[:,-1]}
                        out = self(sample, iters=iters, test_mode=test_mode)
                        
                    for k in list(out.keys()):
                        if k != 'psnr':
                            if k == 'flows':
                                out[k] = [o[...,-img_size_h:,:] for o in out[k]]
                            elif k == 'lq':
                                out[k] = out[k][...,-int(img_size_h/self.pixel_size_ratio):,:]
                            else:
                                out[k] = out[k][...,-img_size_h:,:]
                    if out['flows'][0] is None:
                        print('a')
                
                elif (lq_split.size()[-1] != img_size):
                    img_size_w = lq_split.size()[-1]
                    lq_split_ = imgs_lr[...,i*img_size:(i+1)*img_size,-img_size:]
                    hq_split_ = imgs[...,i*img_size:(i+1)*img_size,-img_size:]
                    assert torch.equal(lq_split,lq_split_[...,-img_size_w:])
                    assert torch.equal(hq_split,hq_split_[...,-img_size_w:])
                    with torch.no_grad():
                        sample = {'lq':lq_split_,'hq':hq_split_[:,:-1],'gt':hq_split_[:,-1]}
                        out = self(sample, iters=iters, test_mode=test_mode)
                        
                    for k in list(out.keys()):
                        if k != 'psnr':
                            if k == 'flows':
                                out[k] = [o[...,-img_size_w:] for o in out[k]]
                            elif k == 'lq':
                                out[k] = out[k][...,-int(img_size_w/self.pixel_size_ratio):]
                            else:
                                out[k] = out[k][...,-img_size_w:]
                    if out['flows'][0] is None:
                        print('a') 
                    
                else:
                    with torch.no_grad():
                        sample = {'lq':lq_split,'hq':hq_split[:,:-1],'gt':hq_split[:,-1]}
                        out = self(sample, iters=iters, test_mode=test_mode)
                        
                for k in list(out.keys()):
                    if k not in list(recon_splits_w.keys()):
                        if k == 'flows':
                            recon_splits_w[k] = [[o] for o in out[k]]
                        else:
                            recon_splits_w[k] = [out[k]]
                    else:
                        if k == 'flows':
                            for t,r in enumerate(recon_splits_w[k]):
                                r.append(out[k][t])
                        else:
                            recon_splits_w[k].append(out[k])
            for k in list(recon_splits_w.keys()):
                if k not in list(recon_splits_h.keys()):
                    if k == 'flows':
                        recon_splits_h[k] = [[torch.cat(r,dim=-1)] for r in recon_splits_w[k]]
                    elif k == 'psnr':
                        recon_splits_h[k] = recon_splits_w[k]
                    else:
                        recon_splits_h[k] = [torch.cat(recon_splits_w[k],dim=-1)]
                else:
                    if k == 'flows':
                        for t,r in enumerate(recon_splits_h[k]):
                            r.append(torch.cat(recon_splits_w[k][t],dim=-1))
                    elif k == 'psnr':
                        recon_splits_h[k].extend(recon_splits_w[k])
                    else:
                        recon_splits_h[k].append(torch.cat(recon_splits_w[k],dim=-1))
        recon = {}
        for k in list(recon_splits_h.keys()):
            if k == 'flows':
                recon[k] = [torch.cat(r,dim=-2) for r in recon_splits_h[k]]
            elif k == 'psnr':
                recon[k] = recon_splits_h[k]
            else:
                recon[k] = torch.cat(recon_splits_h[k],dim=-2)
        return recon
    
    def reconstruct_full(self, img_lr, img, img_gt, use_half_framerate_ok=True):
        
        b, t, c, h, w = img_lr.size()
        b_, t_, c_, h_, w_ = img.size()
        assert h == h_
        assert w == w_

        
        flow = None
        bbox = [0,0,w,h]

        img = (img / 255.).float()
        if use_half_framerate_ok:
            img_gt = (img_gt / 255.).float()

        img_lr = (F.interpolate(img_lr.view(b*t,c,bbox[3]-bbox[1],bbox[2]-bbox[0]), size=[(bbox[3]-bbox[1])//self.pixel_size_ratio, (bbox[2]-bbox[0])//self.pixel_size_ratio], mode='bilinear', align_corners=False) / 255.).view(b,t,c,(bbox[3]-bbox[1])//self.pixel_size_ratio, (bbox[2]-bbox[0])//self.pixel_size_ratio)
        row,col = img_lr.size()[-2:]
        ph_lr = (row // self.base_height_increment) * self.base_height_increment
        pw_lr = (col // self.base_width_increment) * self.base_width_increment
        img_lr = img_lr[...,:ph_lr,:pw_lr]

        ph = ph_lr * self.pixel_size_ratio
        pw = pw_lr * self.pixel_size_ratio
        img = img[...,:ph,:pw]
        if use_half_framerate_ok:
            img_gt = img_gt[...,:ph,:pw]
        
        
        with torch.no_grad():
            results = self.model({'lq':img_lr,'hq':img})
        mid = results['out']
        result_img = tensor2img(mid).astype(float)
        if use_half_framerate_ok:
            hi_img = tensor2img(img_gt).astype(float)
        lo_img = tensor2img(img_lr[:,1]).astype(float)
        
        max_val = result_img.max()
        if use_half_framerate_ok:
            diff = (result_img - hi_img)
            mse = np.mean((diff)**2)
            # if result_img.max() < 255:
            psnr = 10. * np.log10(max_val**2 / mse)
            
            # else:
            #     psnr = 10. * np.log10(255. * 255. / mse)
            #TODO: how to normalize aad wrt the actual intensity range
            aad = abs(diff).mean()
            _ssim = ssim(result_img,hi_img, data_range=255)

            result = {'pred': mid, 'gt': img_gt, 'lq': img_lr, 'hq': img, 'flow': flow, 'psnr': psnr, 'aad': aad, 'ssim': _ssim, 'result_img':result_img, 'hi_img':hi_img, 'lo_img':lo_img}
        else:
            result = {'pred': mid, 'gt': img_gt, 'lq': img_lr, 'hq': img, 'flow': flow, 'result_img':result_img, 'lo_img':lo_img}
        result['max_val'] = max_val

        return result


    def calibrate_image_by_transformation(self, img_lr, x):
        batch,channel,height,width = img_lr.size()
        assert len(x.size()) == 1, f"error: transformation parameters are found to be {len(x.size())}-dimensional. Currently only 1-D vectors implemented."
        rotation_mat, translation_mat, scaling_mat, rotation_mat_inverse, translation_mat_inverse, scaling_mat_inverse = get_trs_transformations(x, two_way_ok=True, device=img_lr.device)
        theta_forward = torch.matmul(torch.matmul(translation_mat,rotation_mat),scaling_mat)
        theta = torch.matmul(torch.matmul(scaling_mat_inverse, rotation_mat_inverse), translation_mat_inverse)[:2].unsqueeze(0)
        theta = theta.repeat((batch,1,1))
        affine_grid = F.affine_grid(theta, size=(batch,channel,height,width), align_corners=True)

        img_lr = F.grid_sample(img_lr, affine_grid, align_corners=True)
        
        bbox_ = compute_transformed_corners(0, 0, 0, 0, width, height, theta_forward)
        bbox = pytorch_lir(bbox_, x[2], width, height)
        assert (bbox[2]-bbox[0]) >= self.base_width_increment_
        assert (bbox[3]-bbox[1]) >= self.base_height_increment_
        bbox[2] -= ((bbox[2]-bbox[0]) % self.base_width_increment_)
        bbox[3] -= ((bbox[3]-bbox[1]) % self.base_height_increment_)
        
        return img_lr, bbox

    def estimate_flow_mosaic(self, imgs_lr, imgs, img_size, iters=12, test_mode=True):
        lq_splits_h = list(torch.split(imgs_lr,img_size,dim=-2))
        hq_splits_h = list(torch.split(imgs,img_size,dim=-2))
        flow_splits_h = []
        for i, (lq_split_h, hq_split_h) in enumerate(zip(lq_splits_h,hq_splits_h)):
            lq_splits_w = list(torch.split(lq_split_h,img_size,dim=-1))
            hq_splits_w = list(torch.split(hq_split_h,img_size,dim=-1))
            flow_splits_w = []
            for j, (lq_split,hq_split) in enumerate(zip(lq_splits_w,hq_splits_w)):
                if (lq_split.size()[-2] != img_size) and (lq_split.size()[-1] != img_size):
                    # do reverse padding and then crop accordingly
                    img_size_h, img_size_w = lq_split.size()[-2:]
                    lq_split_ = imgs_lr[...,-img_size:,-img_size:]
                    hq_split_ = imgs[...,-img_size:,-img_size:]
                    assert torch.equal(lq_split,lq_split_[...,-img_size_h:,-img_size_w:])
                    assert torch.equal(hq_split,hq_split_[...,-img_size_h:,-img_size_w:])
                    with torch.no_grad():
                        flow = self.flow_model(lq_split_,hq_split_, iters=iters, test_mode=test_mode)[1]
                    flow = flow[...,-img_size_h:,-img_size_w:]
                elif (lq_split.size()[-2] != img_size):
                    img_size_h = lq_split.size()[-2]
                    lq_split_ = imgs_lr[...,-img_size:,j*img_size:(j+1)*img_size]
                    hq_split_ = imgs[...,-img_size:,j*img_size:(j+1)*img_size]
                    assert torch.equal(lq_split,lq_split_[...,-img_size_h:,:])
                    assert torch.equal(hq_split,hq_split_[...,-img_size_h:,:])
                    with torch.no_grad():
                        flow = self.flow_model(lq_split_,hq_split_, iters=iters, test_mode=test_mode)[1]
                    flow = flow[...,-img_size_h:,:]
                elif (lq_split.size()[-1] != img_size):
                    img_size_w = lq_split.size()[-1]
                    lq_split_ = imgs_lr[...,i*img_size:(i+1)*img_size,-img_size:]
                    hq_split_ = imgs[...,i*img_size:(i+1)*img_size,-img_size:]
                    assert torch.equal(lq_split,lq_split_[...,-img_size_w:])
                    assert torch.equal(hq_split,hq_split_[...,-img_size_w:])
                    with torch.no_grad():
                        flow = self.flow_model(lq_split_,hq_split_, iters=iters, test_mode=test_mode)[1]
                    flow = flow[...,-img_size_w:]
                else:
                    with torch.no_grad():
                        flow = self.flow_model(lq_split,hq_split, iters=iters, test_mode=test_mode)[1]
                flow_splits_w.append(flow)
            flow_splits_h.append(torch.cat(flow_splits_w,dim=-1))
        flow = torch.cat(flow_splits_h,dim=-2)
        return flow

    def calibrate_image(self, img_lr, flow, batch, frame_num, channel, height, width):
        # warp each time slice of img_lr with flow
        # then crop img_lr and img consistently to remove black margins
        img_lr_ = []
        affine_grid_original = F.affine_grid(torch.tensor([[1., 0., 0.],[0., 1., 0.]],device=flow.device,dtype=flow.dtype).unsqueeze(0).repeat((batch,1,1)), size=(batch,channel,height,width), align_corners=True)
        affine_grid = affine_grid_original - flow
        for t in range(frame_num):
            img_lr_.append(F.grid_sample(img_lr[:,t], affine_grid, align_corners=True).unsqueeze(1))
        img_lr = torch.cat(img_lr_,dim=1)
        # compute corners and largest interior rectangle implicitly
        # by creating a mask to record values in affine_grid exceeding the range [-1,1]
        lir_mask = ~((affine_grid>1) + (affine_grid<(-1)))
        lir_mask = lir_mask.permute(0,3,1,2)
        return img_lr, lir_mask

    def forward(self, sample, iters=12, test_mode=True):
        if self.training:
            # HACK tensors in sample are unnormalized 8-bit int cast to torch.float32
            img_lr, img, img_gt, flow_gt, valid = torch.clone(sample['lq']), torch.clone(sample['hq']), torch.clone(sample['gt']), torch.clone(sample['flow_gt']), torch.clone(sample['valid'])
        else:
            img_lr, img, img_gt = torch.clone(sample['lq']), torch.clone(sample['hq']), torch.clone(sample['gt'])
        b, t, c, h, w = img_lr.size()
        b_, t_, c_, h_, w_ = img.size()
        assert h == h_
        assert w == w_
        flows = self.flow_model(img_lr[:,self.model.center_frame_idx], img_gt, iters=iters, test_mode=test_mode)

        flow = torch.clone(flows[-1])
        flow[:,0] = flow[:,0] / ((w-1)/2)
        flow[:,1] = flow[:,1] / ((h-1)/2)
        flow = flow.permute(0,2,3,1)
        img_lr, lir_mask = self.calibrate_image(img_lr, flow, b, t, c, h, w)

        img = (img / 255.).float()
        img_gt = (img_gt / 255).float()

        img_lr = (F.interpolate(img_lr.view(b*t,c,h,w), size=[h//self.pixel_size_ratio, w//self.pixel_size_ratio], mode='bilinear', align_corners=False) / 255.).view(b,t,c,h//self.pixel_size_ratio, w//self.pixel_size_ratio)
        row,col = img_lr.size()[-2:]
        ph_lr = (row // self.base_height_increment) * self.base_height_increment
        pw_lr = (col // self.base_width_increment) * self.base_width_increment
        img_lr = img_lr[...,:ph_lr,:pw_lr]

        ph = ph_lr * self.pixel_size_ratio
        pw = pw_lr * self.pixel_size_ratio
        img = img[...,:ph,:pw]
        img_gt = img_gt[...,:ph,:pw]
        if self.training:
            flow_gt = flow_gt[...,:ph,:pw]
        lir_mask = lir_mask[...,:ph,:pw]
        if self.training:
            valid = valid[...,:ph,:pw]
        flows = [fl[...,:ph,:pw] for fl in flows]
        if self.model_type == 'EDVRModel':
            results = self.model({'lq':img_lr, 'hq':img, 'gt':img_gt})

            mid = results['out']
        elif self.model_type == 'SwinIRModel':
            mid = self.model(img_lr, img)

        result_img = tensor2img(mid).astype(float)
        hi_img = tensor2img(img_gt).astype(float)
        lo_img = tensor2img(img_lr).astype(float)
        diff = (result_img - hi_img)
        mse = np.mean((diff)**2)
        psnr = 10. * np.log10(255. * 255. / mse)
        
        return {'pred': mid, 'gt': img_gt, 'lq': img_lr, 'mask': lir_mask, 'flows': flows, 'psnr': psnr}