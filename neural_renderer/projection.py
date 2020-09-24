from __future__ import division

import torch
import numpy as np

def projection(vertices, K, R, t, dist_coeffs, orig_size, eps=1e-9):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    # instead of P*x we compute x'*P'
    vertices = torch.matmul(vertices, R.transpose(2,1)) + t
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:, None, 0]
    k2 = dist_coeffs[:, None, 1]
    p1 = dist_coeffs[:, None, 2]
    p2 = dist_coeffs[:, None, 3]
    k3 = dist_coeffs[:, None, 4]

    # we use x_ for x' and x__ for x'' etc.
    r = torch.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
    y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
    vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
    vertices = torch.matmul(vertices, K.transpose(1,2))
    u, v = vertices[:, :, 0], vertices[:, :, 1]
    v = orig_size - v
    # map u,v from [0, img_size] to [-1, 1] to use by the renderer
    u = 2 * (u - orig_size / 2.) / orig_size
    v = 2 * (v - orig_size / 2.) / orig_size
    vertices = torch.stack([u, v, z], dim=-1)
    return vertices

def projection_fov(vertices, orig_size, fy, R, t, bias=None, eps=1e-5):
    # camera transform
    # vertices = torch.matmul(vertices, R.transpose(2,1)) + t
    if isinstance(t, np.ndarray):
        t = torch.cuda.FloatTensor(t)
    if isinstance(bias, np.ndarray):
        t = torch.cuda.FloatTensor(t)
    if isinstance(R, np.ndarray):
        R = torch.cuda.FloatTensor(R)
    t = t.view(-1,1,3)

    # compute fov
    fov = torch.atan( orig_size/(2*fy) )

    # ==== old
    vertices = vertices - t
    vertices = torch.matmul(vertices, R.transpose(1,2))
    # ==== new
    # vertices = torch.matmul(vertices, R.transpose(1,2))
    # vertices = vertices - t
    # ==== end

    if bias is not None:
        # radius = torch.norm(t,dim=2).view((-1,1,1))
        # bias_range = torch.tan(fov/2) * radius
        # bias = bias.view(-1,1,3) * bias_range
        bias = bias.view(-1,1,3)
        vertices = vertices - bias 
    # compute perspective distortion
    width = torch.tan(fov/2)
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = torch.stack((x,y,z), dim=2)
    return vertices






