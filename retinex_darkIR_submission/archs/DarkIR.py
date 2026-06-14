import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from arch_model import EBlock, DBlock
    from arch_util import CustomSequential
    from illumination import IlluminationBranch
except ImportError:
    from archs.arch_model import EBlock, DBlock
    from .arch_util import CustomSequential
    from .illumination import IlluminationBranch

def _rgb_to_value(x: torch.Tensor) -> torch.Tensor:
    return x.max(dim=1, keepdim=True)[0]


class DarkIR(nn.Module):
    
    def __init__(self, img_channel=3, 
                 width=32, 
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2, 
                 enc_blk_nums=[1, 2, 3], 
                 dec_blk_nums=[3, 1, 1],  
                 dilations = [1, 4, 9], 
                 extra_depth_wise = True,
                 use_retinex: bool = False,
                 illum_base: int = 32,
                 input_mode: str = "rgb"):
        super(DarkIR, self).__init__()
        self.img_channel = img_channel
        self.use_retinex = use_retinex
        self.input_mode = input_mode.lower()

        if use_retinex:
            intro_in = img_channel * 2
        elif self.input_mode in ("illum4", "hsv4"):
            intro_in = img_channel + 1
        else:
            intro_in = img_channel

        if use_retinex or self.input_mode == "illum4":
            self.illum = IlluminationBranch(img_channel, illum_base)
        
        self.intro = nn.Conv2d(in_channels=intro_in, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_enc = \
            CustomSequential(
                *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = \
            CustomSequential(
                *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                CustomSequential(
                    *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** len(self.encoders)        
        
        # this layer is needed for the computing of the middle loss. It isn't necessary for anything else
        self.side_out = nn.Conv2d(in_channels = width * 2**len(self.encoders), out_channels = img_channel, 
                                kernel_size = 3, stride=1, padding=1)

    def _encode_input(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_retinex:
            illum = self.illum(input)
            ref = (input / illum.clamp(min=1e-2)).clamp(0.0, 1.0)
            return torch.cat([ref, input], dim=1)
        if self.input_mode == "illum4":
            return torch.cat([input, self.illum(input)], dim=1)
        if self.input_mode == "hsv4":
            return torch.cat([input, _rgb_to_value(input)], dim=1)
        if self.input_mode == "log":
            return torch.log1p(input.clamp(min=0.0))
        return input
        
    def forward(self, input, side_loss = False, use_adapter = None):

        _, _, H, W = input.shape

        input = self.check_image_size(input)
        x = self.intro(self._encode_input(input))
        
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # we apply the encoder transforms
        x_light = self.middle_blks_enc(x)
        
        if side_loss:
            out_side = self.side_out(x_light)
        # apply the decoder transforms
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input
        out = x[:, :, :H, :W] # we recover the original size of the image
        if side_loss:
            return out_side, out
        else:        
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x      

if __name__ == '__main__':
    
    img_channel = 3
    width = 32
    
    enc_blks = [1, 2, 3]
    middle_blk_num_enc = 2
    middle_blk_num_dec = 2
    dec_blks = [3, 1, 1]
    residual_layers = None
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    net = DarkIR(img_channel=img_channel, 
                  width=width, 
                  middle_blk_num_enc=middle_blk_num_enc,
                  middle_blk_num_dec= middle_blk_num_dec,
                  enc_blk_nums=enc_blks, 
                  dec_blk_nums=dec_blks,
                  dilations = dilations,
                  extra_depth_wise = extra_depth_wise)
    
    new_state_dict = net.state_dict()

    inp_shape = (3, 256, 256)

    net.load_state_dict(new_state_dict)

    try:
        from ptflops import get_model_complexity_info  # optional: pip install ptflops

        macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
        print(macs, params)
    except ImportError:
        n = sum(p.numel() for p in net.parameters())
        print(f"params: {n / 1e6:.2f} M  (pip install ptflops for MAC count)")
    
    weights = net.state_dict()
    adapter_weights = {k: v for k, v in weights.items() if 'adapter' not in k}

    
    
