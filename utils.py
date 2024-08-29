"""
__author__ = "Ben Batten"
__email__ = "b.batten@imperial.ac.uk"
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib as mpl
# mpl.use('Qt5Agg')

# rotation_matrix_generator = lambda angle: torch.tensor([[torch.cos(angle), torch.sin(angle)],
                                          # [-torch.sin(angle), torch.cos(angle)]], dtype=torch.float64)

def rotation_matrix_generator(angle_tensor):
    # to_return = torch.empty((2, 2, len(angle_tensor)))#Takes 1D tensor of size (batch,).
    return torch.stack((torch.stack((torch.cos(angle_tensor), torch.sin(angle_tensor)), dim=1), torch.stack((-torch.sin(angle_tensor), torch.cos(angle_tensor)), dim=1)), dim=2)
    # return torch.stack((torch.cos(angle_tensor), torch.sin(angle_tensor), -torch.sin(angle_tensor), torch.cos(angle_tensor)), dim=0).view(len(angle_tensor), 2, 2)


def scale_matrix_generator(scale_tensor):
    return torch.stack((torch.stack((1/scale_tensor, 0*scale_tensor), dim=1),
                 torch.stack((0 * scale_tensor, 1/scale_tensor), dim=1)), dim=2)
    # return torch.stack((1/scale_tensor, 0*scale_tensor, 0*scale_tensor, 1/scale_tensor), dim=0).view(len(scale_tensor), 2, 2)


def shear_matrix_generator(shear_tensor):
    return torch.stack((torch.stack(((0*shear_tensor)+1, -shear_tensor), dim=1), torch.stack((0*shear_tensor, (0*shear_tensor)+1), dim=1)), dim=2)
    # return torch.stack((0*shear_tensor, -shear_tensor, 0*shear_tensor, (0*shear_tensor)+1), dim=0).view(len(shear_tensor), 2, 2)

# scale_matrix_generator = lambda scale: torch.tensor([[1 / scale, 0],
#                                        [0, 1 / scale]], dtype=torch.float64)
# shear_matrix_generator = lambda shear: torch.tensor([[1, -shear],
#                                        [0, 1]], dtype=torch.float64)


matrix_generators = {
    'rotate' : rotation_matrix_generator,
    'scale' : scale_matrix_generator,
    'shear' : shear_matrix_generator
    # 'rotate_scale' : rotation_scale_generator,
    # 'scale_shear' : scale_shear_generator,
    # 'rotation_scale_shear' : rotation_scale_shear_generator
}

invert_rot_param = lambda angle: -angle
invert_scale_param = lambda scale: 1/scale
invert_shear_param = lambda shear: -shear

param_inverters = {
    'rotate': invert_rot_param,
    'scale': invert_scale_param,
    'shear': invert_shear_param
}

def max_rotate_grad_parallel(active_intervals, xy_grid):
    '''
    For the x and y coordinate there are four candidates for the maximum gradient. two of the candidates are the turning points (contained, for each pixel, in _static_points_. The other two candidates are the bounds on the range.
    I suggest computing the value at all four candidates across the image in parallel (and batched), multiplying by a boolean tensor to remove static points NOT in the range. Then returning the entire batchximageshapex2 tensor.
    The maximum total transformation gradient can then be computed in congress with the interpolation gradients in _main_. 
    '''
    active_intervals = active_intervals[..., 0]
    x, y = xy_grid
    # static_points = torch.stack((
    #     torch.arctan(xy_grid[0, ...] / xy_grid[1, ...]), torch.arctan(-xy_grid[1, ...] / xy_grid[0, ...])))
    # static_bool = (active_intervals[:, None, None, None, 0] < static_points[None, ...]) & (static_points[None, ...] < active_intervals[:, None, None, None, 1])
    # candidate_grads = torch.stack((
    #     -xy_grid[0, ...] * torch.sin(candidate_params) - xy_grid[1, ...] * torch.cos(candidate_params), xy_grid[0, ...] * torch.cos(candidate_params) - xy_grid[1, ...] * torch.sin(candidate_params)), dim=1)# Should be the same shape as candidate params with an extra dim size 2
    static_points = torch.stack((
        torch.arctan(x / y), torch.arctan(-y / x))).to(active_intervals.device)
    static_bool = (active_intervals[:, :1] < static_points[None, ...]) & (static_points[None, ...] < active_intervals[:, 1:])
    candidate_params = torch.cat((static_points[None, ...].expand((active_intervals.shape[0], -1)), active_intervals), dim=1) # Contains all the candidate parameters for a maximum gradient. For a pixels, for the entrire batch.


    candidate_grads = torch.stack((
        -x * torch.sin(candidate_params) - y * torch.cos(candidate_params), x * torch.cos(candidate_params) - y * torch.sin(candidate_params)), dim=1)# Should be the same shape as candidate params with an extra dim size 2
    candidate_grads[:, :, :2] = candidate_grads[:, :, :2] * static_bool.unsqueeze(1)
    return candidate_grads.unsqueeze(2)


def max_rotate_grad(theta_low, theta_high, x_hat, y_hat):
    cands = [[], []]
    theta_cand = rotation_static_point(x_hat, y_hat).flatten()
    for i in range(2):
        if theta_low[0] <= theta_cand[i] <= theta_high[0]:
            cands[i].append(rotation_grad(theta_cand[i], x_hat, y_hat)[i])
        elif theta_low[0] <= theta_cand[i] + (np.pi) <= theta_high[0]:
            cands[i].append(rotation_grad(theta_cand[i], x_hat, y_hat)[i])
        elif theta_low[0] <= theta_cand[i] - (np.pi) <= theta_high[0]:
            cands[i].append(rotation_grad(theta_cand[i], x_hat, y_hat)[i])
        else:
            cands[i].append(rotation_grad(theta_low[0], x_hat, y_hat)[i])
            cands[i].append(rotation_grad(theta_high[0], x_hat, y_hat)[i])
    max_x_grad = 0
    max_y_grad = 0
    for i in range(len(cands[0])):
        max_x_grad = max(max_x_grad, abs(cands[0][i]))
    for i in range(len(cands[1])):
        max_y_grad = max(max_y_grad, abs(cands[1][i]))
    return torch.tensor([max_x_grad, max_y_grad]).unsqueeze(1)


def max_scale_grad(theta_low, theta_high, x_hat, y_hat):
    return torch.tensor([[abs(x_hat)], [abs(y_hat)]])


def max_scale_grad_parallel(active_intervals, xy_grid):
    x, y = xy_grid
    return torch.tensor([[abs(x)], [abs(y)]], device=active_intervals.device).unsqueeze(0).repeat(active_intervals.shape[0], 1, 1).unsqueeze(-1)


def max_shear_grad(theta_low, theta_high, x_hat, y_hat):
    return torch.tensor([[abs(y_hat)], [0]])

def max_translate_grad(*args):
    # return torch.tensor([[1], [1]], dtype=torch.float64)
    return torch.cat((torch.tensor([[1], [0]], dtype=torch.float64), torch.tensor([[0], [1]], dtype=torch.float64)), dim=1)


def max_rotate_scale_grad(theta_low, theta_high, x_hat, y_hat):
    '''Now for dv/dscale (depend only on theta)'''
    cands = [[], []]
    theta_opt = torch.arctan(-y_hat/x_hat)
    grad = lambda theta: x_hat*torch.cos(theta) - y_hat*torch.sin(theta)
    '''X first'''
    if theta_low[0] <= theta_opt <= theta_high[0]:
        cands[0].append(grad(theta_opt))
    elif theta_low[0] <= theta_opt + (np.pi) <= theta_high[0]:
        cands[0].append(grad(theta_opt))
    elif theta_low[0] <= theta_opt - (np.pi) <= theta_high[0]:
        cands[0].append(grad(theta_opt))
    else:
        cands[0].append(grad(theta_low[0]))
        cands[0].append(grad(theta_high[0]))

    theta_opt = torch.arctan(x_hat/y_hat)
    grad = lambda theta: x_hat*torch.sin(theta) + y_hat*torch.cos(theta)
    '''Now Y'''
    if theta_low[0] <= theta_opt <= theta_high[0]:
        cands[1].append(grad(theta_opt))
    elif theta_low[0] <= theta_opt + (np.pi) <= theta_high[0]:
        cands[1].append(grad(theta_opt))
    elif theta_low[0] <= theta_opt - (np.pi) <= theta_high[0]:
        cands[1].append(grad(theta_opt))
    else:
        cands[1].append(grad(theta_low[0]))
        cands[1].append(grad(theta_high[0]))

    max_x_grad = 0
    max_y_grad = 0
    for i in range(len(cands[0])):
        max_x_grad = max(max_x_grad, abs(cands[0][i]))
    for i in range(len(cands[1])):
        max_y_grad = max(max_y_grad, abs(cands[1][i]))
    scale_grads = torch.tensor([[max_x_grad], [max_y_grad]])

    max_rot_only_grad = max_rotate_grad([theta_low[0]], [theta_high[0]], x_hat, y_hat)
    max_x_grad = max(abs(theta_low[1] * max_rot_only_grad[0]), abs(theta_high[1] * max_rot_only_grad[0]))
    max_y_grad = max(abs(theta_low[1] * max_rot_only_grad[1]), abs(theta_high[1] * max_rot_only_grad[1]))
    rot_grads = torch.tensor([max_x_grad, max_y_grad]).unsqueeze(dim=1)
    # return torch.tensor(np.hstack((rot_grads, scale_grads)), dtype=torch.float64)
    return torch.cat((rot_grads, scale_grads), dim=1)


def max_rotate_shear_grad_parallel(active_intervals, xy_grid):
    x, y = xy_grid
    x = x.to(active_intervals.device)
    y = y.to(active_intervals.device)
    '''dx/drotate'''
    theta_max = lambda m: torch.arctan(
        (x + m * y) / y)  # Function of theta that maximises grad for a given 'm' (shear)
    grad = lambda m, theta: -x * torch.sin(theta) - y * m * torch.sin(theta) - y * torch.cos(theta)

    theta_candidates = torch.stack((theta_max(active_intervals[:, 0, 1]), active_intervals[:, 0, 1]))
    theta_candidates = torch.stack((theta_candidates, torch.stack((theta_max(active_intervals[:, 0, 1]*0), torch.zeros(active_intervals.shape[0], device=active_intervals.device)))))
    theta_candidates = torch.cat((theta_candidates, torch.stack((theta_max(active_intervals[:, 1, 1]), active_intervals[:, 1, 1]))[None, ...]), dim=0)

    candidates = grad(theta_candidates[:, 1, :], theta_candidates[:, 0, :])

    bools = (active_intervals[:, 0, 0] <= theta_candidates[:, 0, :]) & (theta_candidates[:, 0, :] <= active_intervals[:, 1, 0]) & (active_intervals[:, 0, 1] <= theta_candidates[:, 1, :]) & (theta_candidates[:, 1, :] <= active_intervals[:, 1, 1])

    candidates = candidates * bools

    candidates = torch.cat((candidates, grad(active_intervals[:, 0, 1], active_intervals[:, 0, 1])[None, :], grad(active_intervals[:, 0, 1], active_intervals[:, 1, 1])[None, :],
               grad(active_intervals[:, 1, 1], active_intervals[:, 0, 1])[None, :], grad(active_intervals[:, 1, 1], active_intervals[:, 1, 1])[None, :],
               grad(torch.zeros_like(active_intervals[:, 0, 0], device=active_intervals.device), active_intervals[:, 0, 1])[None, :], grad(torch.zeros_like(active_intervals[:, 0, 0], device=active_intervals.device), active_intervals[:, 1, 1])[None, :]))

    # grad(active_intervals[:, 0, 1], active_intervals[:, 0, 1])[None, :]
    # grad(active_intervals[:, 0, 1], active_intervals[:, 1, 1])[None, :]
    #
    # grad(active_intervals[:, 1, 1], active_intervals[:, 0, 1])[None, :]
    # grad(active_intervals[:, 1, 1], active_intervals[:, 1, 1])[None, :]
    #
    # grad(torch.zeros_like(active_intervals[:, 0, 0]), active_intervals[:, 0, 1])[None, :]
    # grad(torch.zeros_like(active_intervals[:, 0, 0]), active_intervals[:, 1, 1])[None, :]

    drotx_dtheta_max = torch.max(abs(candidates), dim=0).values

    '''dy/drotate - Same again but change the two functions for theta_max and grad'''

    theta_max = lambda m: torch.arctan(
        -y / (x+y*m))  # Function of theta that maximises grad for a given 'm' (shear)
    grad = lambda m, theta: x * torch.cos(theta) + y * m * torch.cos(theta) - y * torch.sin(theta)

    theta_candidates = torch.stack((theta_max(active_intervals[:, 0, 1]), active_intervals[:, 0, 1]))
    theta_candidates = torch.stack((theta_candidates, torch.stack((theta_max(active_intervals[:, 0, 1]*0), torch.zeros(active_intervals.shape[0], device=active_intervals.device)))))
    theta_candidates = torch.cat((theta_candidates, torch.stack((theta_max(active_intervals[:, 1, 1]), active_intervals[:, 1, 1]))[None, ...]), dim=0)

    candidates = grad(theta_candidates[:, 1, :], theta_candidates[:, 0, :])

    bools = (active_intervals[:, 0, 0] <= theta_candidates[:, 0, :]) & (theta_candidates[:, 0, :] <= active_intervals[:, 1, 0]) & (active_intervals[:, 0, 1] <= theta_candidates[:, 1, :]) & (theta_candidates[:, 1, :] <= active_intervals[:, 1, 1])

    candidates = candidates * bools

    candidates = torch.cat((candidates, grad(active_intervals[:, 0, 1], active_intervals[:, 0, 1])[None, :], grad(active_intervals[:, 0, 1], active_intervals[:, 1, 1])[None, :],
               grad(active_intervals[:, 1, 1], active_intervals[:, 0, 1])[None, :], grad(active_intervals[:, 1, 1], active_intervals[:, 1, 1])[None, :],
               grad(torch.zeros_like(active_intervals[:, 0, 0], device=active_intervals.device), active_intervals[:, 0, 1])[None, :], grad(torch.zeros_like(active_intervals[:, 0, 0], device=active_intervals.device), active_intervals[:, 1, 1])[None, :]))


    droty_dtheta_max = torch.max(abs(candidates), dim=0).values


    '''dx/dshear'''

    grad = lambda theta: y * torch.cos(theta)

    theta_candidates = torch.zeros_like(active_intervals[:, 0, 0], device=active_intervals.device)
    candidates = grad(theta_candidates)

    bools = (active_intervals[:, 0, 0] <= theta_candidates) & (
                theta_candidates <= active_intervals[:, 1, 0])

    candidates = candidates * bools

    candidates = torch.stack((candidates, grad(active_intervals[:, 0, 0]), grad(active_intervals[:, 1, 0])))

    dshearx_dm_max = torch.max(abs(candidates), dim=0).values

    '''dy/dshear'''
    grad = lambda theta: y * torch.sin(theta)

    theta_candidates = torch.ones_like(active_intervals[:, 0, 0])*torch.pi / 2
    candidates = grad(theta_candidates)

    bools = (active_intervals[:, 0, 0] <= theta_candidates) & (
                theta_candidates <= active_intervals[:, 1, 0])

    candidates = candidates * bools

    candidates = torch.stack((candidates, grad(active_intervals[:, 0, 0]), grad(active_intervals[:, 1, 0])))

    dsheary_dm_max = torch.max(abs(candidates), dim=0).values
    return torch.stack((torch.stack((drotx_dtheta_max, droty_dtheta_max),dim=1), torch.stack((dshearx_dm_max, dsheary_dm_max), dim=1)), dim=2).unsqueeze(dim=-1)

def max_rotate_shear_grad(theta_low, theta_high, x_hat, y_hat):
    '''Grad wrt rotate first.'''
    '''We only do dx/drotate first as it is much harder than for dy/drotate'''
    theta_max = lambda m: torch.arctan((x_hat + m*y_hat) / (y_hat - m*x_hat))   # Function of theta that maximises grad for a given 'm' (shear)
    grad = lambda m, theta: -x_hat*torch.sin(theta) + m*x_hat*torch.cos(theta) - y_hat*torch.cos(theta) - m*y_hat*torch.sin(theta)
    cands = []
    theta_cands = [(theta_max(0), 0), (theta_max(theta_low[1]), theta_low[1]), (theta_max(theta_high[1]), theta_high[1])]
    for theta_cand in theta_cands:
        if theta_low[0] <= theta_cand[0] <= theta_high[0]:
            cands.append(grad(theta_cand[1], theta_cand[0]))
        elif theta_low[0] <= theta_cand[0] + (torch.pi) <= theta_high[0]:
            cands.append(grad(theta_cand[1], theta_cand[0]))
        elif theta_low[0] <= theta_cand[0] - (torch.pi) <= theta_high[0]:
            cands.append(grad(theta_cand[1], theta_cand[0]))
        else:
            cands.append(grad(theta_cand[1], theta_low[0]))
            cands.append(grad(theta_cand[1], theta_high[0]))
    max_x_grad = 0
    for cand in cands:
        max_x_grad = max(max_x_grad, abs(cand))

    '''Now for dy/drotate'''
    grad = lambda theta: x_hat*torch.cos(theta) - y_hat*torch.sin(theta)
    theta_opt = torch.arctan(-y_hat/x_hat)
    cands = []
    if theta_low[0] <= theta_opt <= theta_high[0]:
        cands.append(grad(theta_opt))
    elif theta_low[0] <= theta_opt + (torch.pi) <= theta_high[0]:
        cands.append(grad(theta_opt))
    elif theta_low[0] <= theta_opt - (torch.pi) <= theta_high[0]:
        cands.append(grad(theta_opt))
    else:
        cands.append(grad(theta_low[0]))
        cands.append(grad(theta_high[0]))
    max_y_grad = 0
    for cand in cands:
        max_y_grad = max(max_y_grad, abs(cand))

    rot_grads = torch.tensor([[max_x_grad], [max_y_grad]])

    '''Now for dv/dshear (depend only on theta)'''
    cands = [[], []]
    theta_opt = torch.tensor(0)
    '''X first'''
    if theta_low[0] <= theta_opt <= theta_high[0]:
        cands[0].append(y_hat*torch.cos(theta_opt))
    elif theta_low[0] <= theta_opt + (torch.pi) <= theta_high[0]:
        cands[0].append(y_hat*torch.cos(theta_opt))
    elif theta_low[0] <= theta_opt - (torch.pi) <= theta_high[0]:
        cands[0].append(y_hat*torch.cos(theta_opt))
    else:
        cands[0].append(y_hat*torch.cos(theta_low[0]))
        cands[0].append(y_hat*torch.cos(theta_high[0]))

    theta_opt = torch.pi/2
    if theta_low[0] <= theta_opt <= theta_high[0]:
        cands[1].append(y_hat*torch.sin(theta_opt))
    elif theta_low[0] <= theta_opt + (torch.pi) <= theta_high[0]:
        cands[1].append(y_hat*torch.sin(theta_opt))
    elif theta_low[0] <= theta_opt - (torch.pi) <= theta_high[0]:
        cands[1].append(y_hat*torch.sin(theta_opt))
    else:
        cands[1].append(y_hat * torch.sin(theta_low[0]))
        cands[1].append(y_hat * torch.sin(theta_high[0]))

    max_x_grad = 0
    max_y_grad = 0
    for i in range(len(cands[0])):
        max_x_grad = max(max_x_grad, abs(cands[0][i]))
    for i in range(len(cands[1])):
        max_y_grad = max(max_y_grad, abs(cands[1][i]))
    shear_grads = torch.tensor([[max_x_grad], [max_y_grad]])
    return torch.cat((rot_grads, shear_grads), dim=1)


def max_scale_shear_grad(theta_low, theta_high, x_hat, y_hat):
    '''dv/dscale first'''
    max_y_grad = abs(y_hat)
    max_x_grad = max(abs(x_hat + theta_low[0]*y_hat), abs(x_hat + theta_high[0]*y_hat))
    scale_grads = torch.tensor([[max_x_grad], [max_y_grad]])

    '''now dv/dshear. This is the same as the gradient for 1D shear'''
    shear_grads = max_shear_grad([theta_low[1]], [theta_high[1]], x_hat, y_hat)
    return np.hstack((scale_grads, shear_grads))


gradient_function_names = {
    # 'rotate': max_rotate_grad,
    'rotate': max_rotate_grad_parallel,
    'scale': max_scale_grad_parallel,
    'shear': max_shear_grad,
    'translate_x_translate_y' : max_translate_grad,
    'rotate_scale' : max_rotate_scale_grad,
    # 'rotate_shear' : max_rotate_shear_grad,
    'rotate_shear' : max_rotate_shear_grad_parallel,
    'scale_shear' : max_scale_shear_grad,
}


# Returns an array of the rotation gradient in x and y (dx/d_theta and dy/d_theta)
def rotation_grad(angle, x_hat, y_hat):
    return torch.tensor([[-x_hat*torch.sin(angle)-y_hat*torch.cos(angle)],
                     [x_hat*torch.cos(angle)-y_hat*torch.sin(angle)]])




# Returns an array containing the values of theta that minimize the rotation gradient in x and y.
def rotation_static_point(x_hat, y_hat):
    return torch.tensor([[torch.arctan(x_hat/y_hat)],
                     [torch.arctan(-y_hat/x_hat)]])


def degrees_to_radians(degrees):
    return np.pi * degrees / 180


'''This is now obsolete as it only maximises the partial gradient (of interpolation).'''
def get_interpolation_gradient_grid(image):
    """Returns a grid with shape: (self.image.shape, 2) where the last dimension contains the maximum abs(gradients) wtf x (first entry) and y (second entry)"""

    # First we compute the bracket common to each partial (P_{i+1,j+1} + P_{i, j} - P_{i+1, j} - P_{i, j+1})
    interior_bracket = image.clone()[:1, :1, :] # P_{i,j}
    interior_bracket = interior_bracket + image[1:, 1:, :] - image[:-1, 1:, :] - image[1:, :-1, :]

    didx = image.clone()[:-1, :-1, :] * -1 # P_{i,j}
    didx += image[:-1, 1:, :]

    didy = image.clone()[:-1, :-1, :] * -1  # P_{i,j}
    didy += image[1:, :-1, :]

    # For each pixel we want to either add the constant or not (delta x/y = 0/1), with the goal of finding the maximum absolute gradient.
    didx = torch.maximum(abs(didx), abs(didx+interior_bracket))
    didy = torch.maximum(abs(didy), abs(didy+interior_bracket))
    return torch.stack([didx, didy], dim=-1)


def sample_from_edges(parameter_bounds, num_samples=10):
    """
    Generate samples from the edges of a hyperrectangle.

    :param lower_bounds: List of lower bounds for each dimension.
    :param upper_bounds: List of upper bounds for each dimension.
    :param num_samples: Number of samples to generate on each edge.
    :return: List of samples from the edges of the hyperrectangle.
    """
    lower_bounds = parameter_bounds[0]
    upper_bounds = parameter_bounds[1]
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("Lower and upper bounds must have the same length")

    if len(lower_bounds) == 1:  # 1D case
        # The edges are simply the two endpoints
        # parameter_samples = torch.linspace(0, 1, sample_number)
        # parameter_samples = lb + (ub - lb) * parameter_samples[:, None]
        samps = torch.linspace(lower_bounds[0], upper_bounds[0], int(num_samples/2))
        return torch.tensor(np.concatenate((samps, torch.flip(samps, dims=(0,)))), dtype=torch.float64).unsqueeze(1)

    elif len(lower_bounds) == 2:  # 2D case
        x_lower, x_upper = lower_bounds[0], upper_bounds[0]
        y_lower, y_upper = lower_bounds[1], upper_bounds[1]

        # Sample from each edge
        y_low = torch.flip(torch.stack((torch.linspace(x_lower, x_upper, int(num_samples / 4)), y_lower.expand(int(num_samples / 4))), dim=1), dims=(0,))
        y_high = torch.stack((torch.linspace(x_lower, x_upper, int(num_samples / 4)), y_upper.expand(int(num_samples / 4))), dim=1)

        x_low = torch.stack((x_lower.expand(int(num_samples / 4)), torch.linspace(y_lower, y_upper, int(num_samples/4))), dim=1)
        x_high = torch.flip(torch.stack((x_upper.expand(int(num_samples / 4)), torch.linspace(y_lower, y_upper, int(num_samples/4))), dim=1), dims=(0,))
        # samples = y_high + x_high + y_low + x_low
        return torch.cat((y_high, x_high, y_low, x_low), dim=0)

    else:
        raise ValueError("Function only supports 1 or 2 dimensions")


def dist_to_bound(bounds_params, param_samples, points):
    if len(param_samples.shape) == 1:
        param_samples = param_samples.unsqueeze(0)
    bound_values = torch.einsum('ijkl,ln->ijkn', bounds_params[:, :, :, :-1], param_samples) + bounds_params[:, :, :, -1:]
    distances = abs(bound_values - points)
    return distances.sum(dim=3) / bound_values.shape[3]


def single_bound_integral(bound, param_lb, param_ub):
    sum = 0
    for i, param in enumerate(bound[:-1]):
        sum += 0.5 * param * (param_ub[i]**2 - param_lb[i]**2)
    return sum


def pw_bound_integral(bound_1, bound_2, param_lb, param_ub, bound_type):
    parameter_samples = np.linspace(param_lb, param_ub, 10)
    if len(param_lb) == 2:
        parameter_samples_1, parameter_samples_2 = np.meshgrid(parameter_samples[..., 0], parameter_samples[..., 1])
        parameter_samples = np.vstack((parameter_samples_1.flatten(), parameter_samples_2.flatten())).T
    parameter_samples = np.hstack((parameter_samples, np.ones((parameter_samples.shape[0], 1))))
    bound_1_vals = np.dot(parameter_samples, bound_1)
    bound_2_vals = np.dot(parameter_samples, bound_2)
    if bound_type == 'lower':
        pw_bound_values = np.maximum(bound_1_vals, bound_2_vals)
    else:
        pw_bound_values = np.minimum(bound_1_vals, bound_2_vals)
    return np.sum(pw_bound_values)


def pw_bound_size_comparison(lbs, ubs, param_lb, param_ub):
    '''Checks the envelope integral of lb1_lb2/ub and lb/ub1_ub2 and returns an indication of whether pwl ub or pwl lb is tighter.'''

    '''First we check the lb as pw.'''
    ub_int = single_bound_integral(ubs[0], param_lb, param_ub)
    lb_int = pw_bound_integral(lbs[0], lbs[1], param_lb, param_ub, 'lower')
    lbpw_vol = ub_int - lb_int

    ub_int = pw_bound_integral(ubs[0], ubs[1], param_lb, param_ub, 'upper')
    lb_int = single_bound_integral(lbs[0], param_lb, param_ub)
    ubpw_vol = ub_int - lb_int
    return (ubpw_vol <= lbpw_vol)


import gurobipy as gp
from gurobipy import GRB

def gurobi_lin(c, A, b):
    # Create a Gurobi model
    model = gp.Model("linear_programming")
    model.setParam('OutputFlag', 0)

    # Define decision variables
    n = len(c)
    x = model.addVars(n, name="x", lb=-GRB.INFINITY)
    # intercept = model.addVar(name="intercept", lb=-GRB.INFINITY)

    # Set objective function
    model.setObjective(gp.quicksum(c[i] * x[i] for i in range(n)), sense=GRB.MINIMIZE)

    # Add inequality constraints
    for i in range(len(A)):
        model.addConstr(gp.quicksum(A[i][j] * x[j] for j in range(n)) <= b[i], f"constraint_{i}")

    # Optimize the model
    model.optimize()

    # Print the results
    if model.status == GRB.OPTIMAL:
        if len(c) == 2:
            return model.getVars()[0].X, model.getVars()[1].X
        elif len(c) == 3:
            return (model.getVars()[0].X, model.getVars()[1].X), model.getVars()[2].X
    else:
        print("No solution found.")