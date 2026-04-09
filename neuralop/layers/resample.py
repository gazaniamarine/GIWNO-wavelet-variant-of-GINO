import torch
import torch.nn.functional as F

def resample(x, res_scale, axis, output_shape=None):
    """
    Resample a tensor x along given axes.
    x: (batch, channels, d1, ..., dN)
    res_scale: factor to scale (or 1.0)
    axis: dimensions to resample
    output_shape: specific target shape
    """
    if output_shape is None:
        # Compute output shape based on res_scale
        in_shape = list(x.shape[2:])
        output_shape = [int(s * res_scale) for s in in_shape]
    
    # Use interpolate for resampling
    # mode depends on dimensionality
    dim = len(x.shape) - 2
    if dim == 1:
        mode = 'linear'
    elif dim == 2:
        mode = 'bilinear'
    elif dim == 3:
        mode = 'trilinear'
    else:
        # Fallback for higher dims or irregular resampling
        # For Neural Operators, we usually stick to 1D, 2D, 3D
        mode = 'area' 

    return F.interpolate(x, size=output_shape, mode=mode, align_corners=False if mode != 'area' else None)