import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level=1, wavelet='haar'):
        """
        Wavelet Convolution Layer (2D).
        Performs a Discrete Wavelet Transform, applies a linear transform 
        to the coefficients, and performs an Inverse Discrete Wavelet Transform.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = wavelet
        
        # For Haar wavelet, we have 4 components per level: LL, LH, HL, HH
        # Each level of decomposition produces 3 detail subbands (LH, HL, HH)
        # and one approximation subband (LL) which is further decomposed.
        
        # For simplicity, we implement a 1-level decomposition for now, 
        # or a multi-level where each subband has its own learnable weight.
        
        n_subbands = 3 * level + 1
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_channels, in_channels, 1, 1) * (2 / (in_channels + out_channels))**0.5)
            for _ in range(n_subbands)
        ])
        
        # Haar filters
        self.register_buffer('filter_LL', torch.tensor([[0.5, 0.5], [0.5, 0.5]]).view(1, 1, 2, 2))
        self.register_buffer('filter_LH', torch.tensor([[0.5, -0.5], [0.5, -0.5]]).view(1, 1, 2, 2))
        self.register_buffer('filter_HL', torch.tensor([[0.5, 0.5], [-0.5, -0.5]]).view(1, 1, 2, 2))
        self.register_buffer('filter_HH', torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).view(1, 1, 2, 2))

    def dwt(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        
        LL = F.conv2d(x, self.filter_LL, stride=2)
        LH = F.conv2d(x, self.filter_LH, stride=2)
        HL = F.conv2d(x, self.filter_HL, stride=2)
        HH = F.conv2d(x, self.filter_HH, stride=2)
        
        LL = LL.view(B, C, H//2, W//2)
        LH = LH.view(B, C, H//2, W//2)
        HL = HL.view(B, C, H//2, W//2)
        HH = HH.view(B, C, H//2, W//2)
        
        return LL, LH, HL, HH

    def idwt(self, LL, LH, HL, HH):
        B, C, H2, W2 = LL.shape
        H, W = H2 * 2, W2 * 2
        
        # We can use conv_transpose2d with the same filters but transposed
        LL = LL.reshape(B * C, 1, H2, W2)
        LH = LH.reshape(B * C, 1, H2, W2)
        HL = HL.reshape(B * C, 1, H2, W2)
        HH = HH.reshape(B * C, 1, H2, W2)
        
        out = (F.conv_transpose2d(LL, self.filter_LL, stride=2) +
               F.conv_transpose2d(LH, self.filter_LH, stride=2) +
               F.conv_transpose2d(HL, self.filter_HL, stride=2) +
               F.conv_transpose2d(HH, self.filter_HH, stride=2))
        
        return out.view(B, C, H, W)

    def forward(self, x):
        # Multi-level decomposition
        subbands = []
        curr_ll = x
        for l in range(self.level):
            LL, LH, HL, HH = self.dwt(curr_ll)
            subbands.append(LH)
            subbands.append(HL)
            subbands.append(HH)
            curr_ll = LL
        subbands.append(curr_ll)
        
        # Apply learnable weights in wavelet domain
        # Reverse order to apply weights from finest to coarsest
        processed_subbands = []
        for i, sb in enumerate(subbands):
            # weight: (out_channels, in_channels, 1, 1)
            # Use F.conv2d for channel-wise linear transform
            processed_subbands.append(F.conv2d(sb, self.weights[i]))
            
        # Reconstruct
        curr_ll = processed_subbands[-1]
        for l in range(self.level):
            idx = (self.level - 1 - l) * 3
            LH = processed_subbands[idx]
            HL = processed_subbands[idx + 1]
            HH = processed_subbands[idx + 2]
            curr_ll = self.idwt(curr_ll, LH, HL, HH)
            
        return curr_ll

class WaveletConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, level=2, wavelet='haar', device=None):
        """
        Generic Wavelet Convolution.
        Currently implements 2D.
        """
        super().__init__()
        self._n_modes = n_modes
        self.order = len(n_modes)
        if self.order == 2:
            self.conv = WaveletConv2d(in_channels, out_channels, level=level, wavelet=wavelet)
        else:
            raise NotImplementedError("Only 2D Wavelet Convolution is implemented currently.")

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])
        if output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            # Resample x to output_shape
            return F.interpolate(x, size=out_shape, mode='bilinear', align_corners=True)

    @property
    def n_modes(self):
        return self._n_modes
    
    @n_modes.setter
    def n_modes(self, n_modes):
        self._n_modes = n_modes
        self.order = len(n_modes)

    def forward(self, x, output_shape=None):
        if output_shape is not None:
            x = self.transform(x, output_shape=output_shape)
        return self.conv(x)
