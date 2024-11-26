import torch
from torch.nn import functional as F

from xfusion.train.basicsr.utils.registry import MODEL_REGISTRY
#from .sr_model import SRModel
from xfusion.train.basicsr.models.video_base_model import VideoBaseModel

@MODEL_REGISTRY.register()
class SwinIRModel(VideoBaseModel):
    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']

        if len(window_size) == 2 and isinstance(window_size,list):
            for ws in window_size:
                assert len(ws) == 3
            window_size = [tuple(ws) for ws in window_size]
            #self.split_input_ok = True
        elif isinstance(window_size,tuple):
            assert len(window_size) == 3
            window_size = [window_size]
            #self.split_input_ok = False
        elif len(window_size) == 3 and isinstance(window_size,list):
            for ws in window_size:
                assert isinstance(ws,int)
            window_size = [tuple(window_size)]
            #self.split_input_ok = False
        else:
            raise Exception(f'window size {window_size} not expected')

        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = [0 for _ in range(len(window_size))], [0 for _ in range(len(window_size))]
        _, _, _, h, w = self.lq.size()
        for i in range(len(window_size)):
            if h % window_size[i][1] != 0:
                mod_pad_h[i] = window_size[i][1] - h % window_size[i][1]
            if w % window_size[i][2] != 0:
                mod_pad_w[i] = window_size[i][2] - w % window_size[i][2]

        sample = {}
        sample['lq'] = F.pad(self.lq.view((-1,1,h,w)), (0, mod_pad_w[0], 0, mod_pad_h[0]), 'reflect').view(-1,self.opt['datasets']['train']['num_frame'],1,(h+mod_pad_h[0]),(w+mod_pad_w[0]))
        sample['hq'] = F.pad(self.hq.view((-1,1,h * scale,w * scale)), (0, int(mod_pad_w[1 % len(window_size)] * scale), 0, int(mod_pad_h[1 % len(window_size)] * scale)), 'reflect').view((-1,self.opt['datasets']['train']['num_frame_hi'],1,scale*(h+mod_pad_h[1 % len(window_size)]),scale*(w+mod_pad_w[1 % len(window_size)])))
        #print(sample['lq'].size())
        #print(sample['hq'].size())
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(sample)['out']
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(sample)['out']
            self.net_g.train()

        if self.output.dim() == 4:
            _, _, h, w = self.output.size()
            out = self.output[:, :, 0:h - mod_pad_h[1 % len(window_size)] * scale, 0:w - mod_pad_w[1 % len(window_size)] * scale]
        elif self.output.dim() == 5:
            _, _, _, h, w = self.output.size()
            out = self.output[:, :, :, 0:h - mod_pad_h[1 % len(window_size)] * scale, 0:w - mod_pad_w[1 % len(window_size)] * scale]
        else:
            d = self.output.dim()
            raise Exception(f'GT image tensor has {d} dimensions: can not handle...')

        self.output = {'out': out}

'''
    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
'''