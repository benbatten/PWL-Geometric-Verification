"""
__author__ = "Ben Batten"
__email__ = "b.batten@imperial.ac.uk"
"""

import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
import scipy
import logging
import time
import pickle

from utils import gurobi_lin
import utils
import visualize
# import test_suite
from timeit import Timer as timer

logging.basicConfig(level=logging.DEBUG)

'''Args'''
parser = argparse.ArgumentParser()
parser.add_argument("--transformation", default=None, required=True, nargs='*', type=str, help="List of the transformation(s).")
parser.add_argument("--LB", default=None, required=True, nargs='*', type=float, help="List of the transformation lower bound(s).")
parser.add_argument("--UB", default=None, required=True, nargs='*', type=float, help="List of the transformation upper bound(s).")
parser.add_argument("--image_number", default=100, required=True, type=int, help="Number of images to obtain bounds for on MNIST")
parser.add_argument("--save_bounds", required=True, help="Indicates whether or not to save the bounds.")
parser.add_argument("--bound_type", required=True, help="Indicates whether we are using 'linear' or 'pw_linear' bounds.")
parser.add_argument("--dset", required=True, help="The dataset used, MNIST or CIFAR.")

parser.add_argument("--init_tasks", default=125, required=False, type=int, help="Initial subproblem splits for lipschitz bnb.")
parser.add_argument("--split_num", default=3, required=False, type=int, help="Splits at each iter of lipschitz bnb.")


parser.add_argument("--use_gpu", action='store_true', help="Sets the default device for Tensor computation.")
parser.add_argument("--sample_num", default=100, required=False, type=int, help="Number of samples used in the empirical approximation of the pixel value curve.")
parser.add_argument("--lipschitz_error", default=0.05, required=False, type=float, help="Final error allowed in lipschitz BaB.")

args = parser.parse_args()

'''Testing params'''
# args.transformation = ['translate_x', 'translate_y']
# args.LB = [-2, -2]
# args.UB = [-1, -1]
# args.image_number = 100
# args.lipschitz_error = 5
# args.save_bounds = True
# args.bound_type = 'pw_linear'
# args.samp_num = 10
# args.dset = 'MNIST'

# args.use_gpu = True

with_vis=False
'''End Testing params'''

if args.use_gpu:
    print("USING GPU")
    # args.use_gpu = 'cuda'
    device = torch.device('cuda')
    # torch.set_default_device(torch.device('cuda'))
else:
    args.use_gpu = 'cpu'
    device = torch.device('cpu')

print(args.UB)
print(args.transformation)

# if 'rotate' in args.transformation:
#     ind = args.transformation.index('rotate')
#     args.LB[ind] = utils.degrees_to_radians(args.LB[ind])
#     args.UB[ind] = utils.degrees_to_radians(args.UB[ind])

'''End Args'''

# args.transformation = ['scale', 'shear']
# args.LB = [1, 0]
# args.UB = [1.5, 1.5]

# args.transformation = ['rotate']
# args.LB = [0]
# args.UB = [0.5]
# args.sample_num = 10
# args.use_gpu = 'cuda'
# args.image_number = 5

all_args = zip(args.transformation, args.LB, args.UB)
all_args = sorted(all_args)
args.transformation, args.LB, args.UB = list(zip(*all_args))
args.transformation = list(args.transformation)
args.LB = torch.tensor(np.array(args.LB, dtype='float64')).unsqueeze(0)
args.UB = torch.tensor(np.array(args.UB, dtype='float64')).unsqueeze(0)

original_lb = args.LB.clone()
original_ub = args.UB.clone()

if 'rotate' in args.transformation:
    args.LB[0, 0] = args.LB[0, 0] * torch.pi / 180
    args.UB[0, 0] = args.UB[0, 0] * torch.pi / 180
    print("Converted upper bound to {} radians".format(args.UB[0]))

# for i, transformation in enumerate(args.transformation):
#     args.LB[i] = utils.param_inverters[transformation](args.LB[i])
#     args.UB[i] = utils.param_inverters[transformation](args.UB[i])


class Image:
    def __init__(self, image, transformations, lb, ub):
        self.image = image.type(torch.float64)#.to(device)
        self.j_size = image.shape[1]
        self.i_size = image.shape[0]
        if len(image.shape) == 2:
            self.n_chans = 1
            self.image = self.image[:, :, None]
        else:
            self.n_chans = image.shape[2]
        self.transformations = transformations
        self.lb = lb
        self.ub = ub
        self.origin = ((self.j_size+1)/2, (self.i_size+1)/2)
        self.interpolation_gradient_grid = None

        # self.pad_size = 10
        # self.padded_image = torch.empty(self.i_size+(2*self.pad_size), self.j_size+(2*self.pad_size), self.n_chans)
        # for chan in range(self.padded_image.shape[-1]):
        #     self.padded_image[:, :, chan] = torch.nn.functional.pad(self.image[:, :, chan], pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
        #                                            mode='constant', value=0)

    def _get_cart_coord(self, ind):
        i, j = ind
        if torch.is_tensor(i):
            return torch.stack((j - self.origin[0], self.origin[1] - i))
        else:
            return torch.tensor([j - self.origin[0], self.origin[1] - i], dtype=torch.float64)

    def _get_index_coord(self, cart):
        if torch.is_tensor(cart):
            x, y = cart[:, :1, ...], cart[:, 1:, ...]
            return torch.cat((self.origin[1] - y, x + self.origin[0]), dim=1)
        else:
            x, y = cart
            return torch.cat((self.origin[1] - y, x + self.origin[0]), dim=1)#.type(torch.int32)


    def linear_interpolation(self, transformed_indices, clean_im, chan=0):
        # if args.use_gpu == 'cuda':
        #     transformed_indices = transformed_indices.cuda()
        #     clean_im = clean_im.cuda()
        #     # torch.set_default_device(torch.device('cuda'))
        # with torch.device('cuda') if args.use_gpu == 'cuda' else torch.device('cpu'):
        if len(clean_im.shape) == 3:
            clean_im = clean_im[0]
        corner_locations = torch.floor(
            transformed_indices)#[..., 0]  # Contains the corners for each pixel in the 3rd dimension. In order ((min_i, min_j), (max_i, min_j), (min_i, max_j), (max_i, max_j))
        corner_locations = torch.concat((corner_locations, (corner_locations[:, :, :1, :] + torch.tensor([1, 0], device=corner_locations.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))), dim=2)
        corner_locations = torch.concat((corner_locations, (corner_locations[:, :, :1, :] + torch.tensor([0, 1], device=corner_locations.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))), dim=2)
        corner_locations = torch.concat((corner_locations, (corner_locations[:, :, :1, :] + torch.tensor([1, 1], device=corner_locations.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))), dim=2)
        corner_locations = corner_locations.to(torch.int)
        pad_size = abs(torch.max(corner_locations) - min(self.i_size, self.j_size)) + 1
        padded_image = torch.nn.functional.pad(clean_im[:, :], pad=(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        corner_locations += pad_size

        # padded_image = self.padded_image[:, :, chan]
        # corner_locations += self.pad_size

        corner_values = torch.empty((corner_locations.shape))[:, 0, :, :]
        corner_values[:, 0, :] = padded_image[corner_locations[:, 0, 0, :], corner_locations[:, 1, 0, :]]
        corner_values[:, 1, :] = padded_image[corner_locations[:, 0, 1, :], corner_locations[:, 1, 1, :]]
        corner_values[:, 2, :] = padded_image[corner_locations[:, 0, 2, :], corner_locations[:, 1, 2, :]]
        corner_values[:, 3, :] = padded_image[corner_locations[:, 0, 3, :], corner_locations[:, 1, 3, :]]

        interpolation_value_matrix = torch.empty((corner_values.shape))
        interpolation_value_matrix[:, 0, :] = corner_values[:, 3, :] - corner_values[:, 1, :] - corner_values[:,
                                                                                       2, :] + corner_values[:, 0, :]
        interpolation_value_matrix[:, 1, :] = corner_values[:, 1, :] - corner_values[:, 0, :]
        interpolation_value_matrix[:, 2, :] = corner_values[:, 2, :] - corner_values[:, 0, :]
        interpolation_value_matrix[:, 3, :] = corner_values[:, 0, :]

        delta_matrix = transformed_indices % 1
        interpolation_delta_matrix = torch.empty((interpolation_value_matrix.shape))
        interpolation_delta_matrix[:, 0, :] = delta_matrix[:, 0, 0, :] * delta_matrix[:, 1, 0, :]    # dx*dy
        interpolation_delta_matrix[:, 1, :] = delta_matrix[:, 0, 0, :]     # dx
        interpolation_delta_matrix[:, 2, :] = delta_matrix[:, 1, 0, :]     #dy
        interpolation_delta_matrix[:, 3, :] = torch.ones((corner_locations.shape[0], corner_locations.shape[-1]))
        # return torch.unflatten(torch.diagonal(torch.mm(interpolation_delta_matrix, interpolation_value_matrix.T)), 0, (self.i_size, self.j_size)) # a tensor with the same dimensions as indices that contains the new pixel values
        return (interpolation_delta_matrix*interpolation_value_matrix).sum(dim=1) # a tensor with the same dimensions as indices that contains the new pixel values


    '''Creates dictionary of image samples within the parameter space'''
    def sample(self, sample_number, lb, ub): #lb should be shape (batch_size, transformation_dims)
        # with torch.device('cuda') if args.use_gpu == 'cuda' else torch.device('cpu'):
        # parameter_samples = np.random.uniform(lb, ub, (sample_number, len(self.lb)))
        assert len(lb.shape) == 2, "Incorrect bound shape."
        if lb.shape[1] == 2:
            sample_number = int(sample_number**0.5)
        parameter_samples = torch.linspace(0, 1, sample_number).unsqueeze(0).unsqueeze(0).expand(lb.shape[0], lb.shape[1], -1).to(lb.device)  #Shape: (batch, transformation, sample_num)
        parameter_samples = lb[:, :, None] + ((ub - lb)[:, :, None] * parameter_samples)
        if lb.shape[1] == 2:
            parameter_samples_1, parameter_samples_2 = torch.meshgrid(parameter_samples[0, 0, :], parameter_samples[0, 1, :])
            parameter_samples = torch.vstack((parameter_samples_1.flatten(), parameter_samples_2.flatten())).unsqueeze(0)
        sample_images = torch.empty((lb.shape[0], self.i_size, self.i_size, self.n_chans, parameter_samples.shape[-1]), dtype=torch.float64).to(lb.device)
        index_pairs = [[i, j] for i in range(self.i_size) for j in range(self.j_size)]
        cart_coords = torch.stack(list(map(self._get_cart_coord, index_pairs)))[:, :, None].to(lb.device)
        for s in range(parameter_samples.shape[-1]):  # over the samples
            # clean_im = self.image#.cuda()
            for chan in range(self.n_chans):
                clean_im = self.image[:, :, chan].to(lb.device)
                for t, transform in enumerate(self.transformations):
                    if transform in ['translate_x', 'translate_y']:
                        new_cart_coords = cart_coords + torch.tensor([s, s]).to(lb.device).unsqueeze(-1)
                    else:
                        transformation_matrix = utils.matrix_generators[transform](parameter_samples[:, t, s]).to(lb.device)
                        new_cart_coords = torch.matmul(transformation_matrix.unsqueeze(1), cart_coords.unsqueeze(0))
                    new_index_pairs = self._get_index_coord(torch.permute(new_cart_coords, (0, 2, 3, 1))).to(lb.device)#[:, :, None]
                    # intermediate = self.linear_interpolation(new_index_pairs, clean_im, chan=chan)
                    clean_im = torch.unflatten(self.linear_interpolation(new_index_pairs, clean_im, chan=chan), 1, (self.i_size, self.j_size)).to(lb.device)
                sample_images[:, :, :, chan, s] = clean_im
        self.image_samples = sample_images
        self.parameter_samples = parameter_samples


    '''Get's the unsafe linear bounds'''
    def get_unsafe_lin_bounds(self, bound_type, pwl_constraints=None):
        """
           Finds a linear upper bound on image values based on sample parameters using PyTorch with GPU support.

           :param images: A numpy array of shape (i, j, c, n) representing n samples of an image with shape (i, j, c).
           :param parameters: A numpy array of shape (n, p) where n is the number of samples and p is the number of parameters.
           :return: A dictionary containing the coefficients and intercepts for each pixel/channel.
           """
        images = self.image_samples
        parameters = self.parameter_samples[0].T

        images_reshaped = images.reshape(-1, images.shape[-1])  # Shape: (i*j*c, n)

        # Add a column of ones to the parameters for the intercept term
        ones = np.ones((images.shape[-1], 1))
        augmented_parameters = np.hstack((parameters, ones))

        # Initialize dictionary to store the coefficients and intercepts
        linear_bounds = torch.zeros((images_reshaped.shape[0], len(self.transformations)+1), dtype=torch.float64)

        # For each pixel/channel, solve the linear programming problem
        for idx in range(images_reshaped.shape[0]):
            # The constraints are that the linear function should be greater than each image value
            if bound_type == 'upper':
                if pwl_constraints is not None:
                    c = np.hstack((pwl_constraints[idx], np.ones((pwl_constraints.shape[1], 1)))).sum(0)
                else:
                    c = augmented_parameters.sum(0)
                A_ub = -augmented_parameters
                b_ub = -images_reshaped[idx]
            else:
                if pwl_constraints is not None:
                    c = -np.hstack((pwl_constraints[idx], np.ones((pwl_constraints.shape[1], 1)))).sum(0)
                else:
                    c = -augmented_parameters.sum(0)
                A_ub = augmented_parameters
                b_ub = images_reshaped[idx]

            coefficients, intercept = gurobi_lin(c, A_ub, b_ub)
            linear_bounds[idx, :-1] = torch.tensor(coefficients)
            linear_bounds[idx, -1] = intercept
            # # Solve the linear programming problem
            # result = scipy.optimize.linprog(c=c, A_ub=A_ub, b_ub=b_ub, method='highs', bounds=[(-1e6, 1e6) for i in range(c.shape[0])])
            #
            # if result.success:
            #     # Extract the coefficients for the linear function
            #     linear_function_coeffs = result.x
            #     coefficients, intercept = linear_function_coeffs[:-1], linear_function_coeffs[-1]
            #     linear_bounds[idx, :-1] = coefficients
            #     linear_bounds[idx, -1] = intercept
            # else:
            #     raise ValueError(f"Linear programming failed to converge for pixel/channel index {idx}")

        return linear_bounds.reshape((self.i_size, self.j_size, self.n_chans, len(self.transformations)+1))

    '''GPU TESTING'''

    def get_violation_maxima(self, linear_bounds, bound_type, epsilon, pw_indicator=None): # Original
        violation_bounds = torch.zeros((self.i_size, self.j_size, self.n_chans), dtype=torch.float64)
        for chan in range(self.n_chans):
            for i in range(self.i_size):
                # print(i)
                for j in range(self.j_size):
                    if pw_indicator is not None:
                        if pw_indicator['lower'][i, j, chan]:
                            bound_type = 'lower'
                        else:
                            bound_type = 'upper'
                    regions = [(self.lb, self.ub)]
                    maximas = []
                    maximum_observed_value = torch.zeros((1), dtype=torch.float64).to(device)
                    x, y = self._get_cart_coord([i, j])

                    batch_size = 10000
                    init_tasks = args.init_tasks
                    if self.lb.shape[1] == 2:
                        init_tasks = int(init_tasks**0.5)
                        parameter_samples = torch.linspace(0, 1, steps=init_tasks).unsqueeze(0).expand(
                            self.lb.shape[0], self.lb.shape[1], -1)
                        parameter_samples = (self.lb[:, :, None] + ((self.ub - self.lb))[:, :, None] * parameter_samples)
                        parameter_samples_1, parameter_samples_2 = torch.meshgrid(parameter_samples[0, 0, :],
                                                                                  parameter_samples[0, 1, :])
                        tasks_1 = torch.stack((parameter_samples_1[:-1, :], parameter_samples_1[1:, :]), dim=2).reshape((-1, 2))
                        tasks_2 = torch.stack((parameter_samples_2[:, :-1], parameter_samples_2[:, 1:]), dim=2).reshape((-1, 2))
                        tasks = torch.stack((tasks_1, tasks_2), dim=2)
                    else:
                        tasks = torch.linspace(self.lb.item(), self.ub.item(), steps=init_tasks, dtype=torch.float64).unsqueeze(-1)
                        tasks = torch.stack((tasks[:-1, 0:], tasks[1:, 0:]), dim=1)

                    max_interpolation_grad = self.interpolation_gradient_grid.to(device)
                    max_interpolation_grad = torch.stack((torch.max(
                        max_interpolation_grad[..., chan, 0].reshape(-1), dim=0).values, torch.max(
                        max_interpolation_grad[..., chan, 1].reshape(-1), dim=0).values))

                    while tasks.shape[0] > 0:
                        # print(len(tasks))
                        active_tasks = tasks[:batch_size].to(device)  #Shape: (batch, 2(lb/ub), transformation dim)
                        pixel_linear_bounds = linear_bounds[i, j, chan].to(device)
                        tasks = tasks[batch_size:]

                        self.sample(10, active_tasks[:, 0, :], active_tasks[:, 1, :])
                        # del self.image_samples
                        # del self.parameter_samples

                        '''Removing this an taking interpolation grad as [1,1]'''
                        # Compute smooth regions covered given input transformation parameter ranges: (this needs to be done per pixel otherwise it's just a mess - can parallelize over the smooth regions)
                        # regions_covered = self.get_regions_covered([i, j],  [lb, ub]) + 0.5
                        # regions_covered = self.get_regions_covered_parallel(active_tasks) + 0.5
                        # # regions_covered = torch.stack(list(map(self._get_index_coord, regions_covered)), dtype=torch.int32)
                        # regions_covered = self._get_index_coord(torch.permute(regions_covered, (1, 0))[:, :, None]).type(torch.int32)
                        # regions_covered = regions_covered[regions_covered[:, 0] >= 0]
                        # regions_covered = regions_covered[regions_covered[:, 1] >= 0]
                        # regions_covered = regions_covered[regions_covered[:, 0] <= self.i_size - 2]
                        # regions_covered = regions_covered[regions_covered[:, 1] <= self.j_size - 2]

                        '''Moved to outside while loop as this is constant in the current format'''
                        # try:
                        #     max_interpolation_grad = self.interpolation_gradient_grid.to(device)
                        #     # max_interpolation_grad = self.interpolation_gradient_grid[regions_covered[:, 0], regions_covered[:, 1], chan]
                        # except IndexError:
                        #     max_interpolation_grad = torch.tensor([[0, 0]], dtype=torch.float64).to(device)

                        # max_interpolation_grad = torch.tensor([1., 1.], dtype=torch.float64).unsqueeze(0)
                        '''End experiment of setting constant interpolation grad'''

                        '''Let's call the appropriate maximize function which returns the max(abs(nabla(transformation))) for the COMPOSITION of the
                        transformations.'''
                        # max_transformation_grad = utils.gradient_function_names['_'.join(self.transformations)](lb, ub, x, y)      # Returns a 2xn array. the first row is dx/d(transformation param n) and the second for y etc.
                        # xy_grid = self._get_cart_coord(torch.stack((torch.meshgrid(torch.arange(self.i_size), torch.arange(self.j_size), indexing='ij')), dim=0)).to(device)
                        max_transformation_grad = utils.gradient_function_names['_'.join(self.transformations)](active_tasks, [x, y])      # This is dimension (batch, 2(x/y), transformation, candidates)
                        '''
                        Now I am going to cheat and take the maximum interpolation grad for x and y independently. Maybe come back and change this if you want to make it tighter. Remember we are
                        already being very loose by considering all interpolation regions for every pixel and every transformation/transformation range. This is definitely an area to improve..
                        '''
                        # max_interpolation_grad = torch.stack((torch.max(max_interpolation_grad[..., 0].reshape(-1, self.n_chans), dim=0).values, torch.max(max_interpolation_grad[..., 1].reshape(-1, self.n_chans), dim=0).values))

                        total_diffs = torch.sum(max_transformation_grad * max_interpolation_grad[None, :, None, None], dim=1)
                        total_diffs = torch.sum(total_diffs * (active_tasks[:, 1] - active_tasks[:, 0])[:, None], dim=1)
                        total_diffs = torch.max(abs(total_diffs), dim=1).values

                        if bound_type == 'upper':  # viol_func = pixel_value_func - UB_func   (we want maximum of this)
                            viol_samples = self.image_samples[:, i, j, chan, :] - (torch.sum(self.parameter_samples[:, :, :] * pixel_linear_bounds[None, :-1, None], dim=1) + pixel_linear_bounds[None, -1, None])
                            max_violation_sample = torch.max(viol_samples, dim=1).values # The maximum real sample of the violation function
                            # min_viol_params = self.parameter_samples[torch.argmin(viol_samples), :]
                            # min_viol_sample = torch.min(viol_samples)
                            low_range_viol_value = viol_samples[..., 0]           # The value of the violation function at the low parameter value
                            high_range_viol_value = viol_samples[..., -1]          # The value of the violation function at the high parameter value
                        elif bound_type == 'lower':  # viol_func = LB_func - pixel_value_func   (we want maximum of this)
                            viol_samples = (torch.sum(self.parameter_samples[:, :, :] * pixel_linear_bounds[None, :-1, None], dim=1) + pixel_linear_bounds[None, -1, None]) - self.image_samples[:, i, j, chan, :]
                            max_violation_sample = torch.max(viol_samples, dim=1).values
                            low_range_viol_value = viol_samples[..., 0]           # The value of the violation function at the low parameter value
                            high_range_viol_value = viol_samples[..., -1]          # The value of the violation function at the high parameter value
                        else:
                            raise ValueError(f"Invalid bound type")

                        # Compute upper bound in range
                        upper_bound_viol_function = (low_range_viol_value + high_range_viol_value + total_diffs)/2#torch.mm(max_grad, (ub - lb).unsqueeze(1)))/2

                        maximum_observed_value = torch.max(torch.max(max_violation_sample), maximum_observed_value)

                        halt_tensor = (upper_bound_viol_function - max_violation_sample < epsilon)#.reshape(max_violation_sample.shape[0], -1)  # Halt because we have found a valid upper bound.
                        halt_tensor = halt_tensor | (maximum_observed_value > upper_bound_viol_function)#.reshape(max_violation_sample.shape[0], -1)   # Halt because the region cannot be the maximum for this pixel
                        halt_tensor = halt_tensor | (upper_bound_viol_function <= 0)  # Halt because the region cannot be the maximum for this pixel
                        # halt_tensor = torch.min(halt_tensor)

                        active_tasks = active_tasks[~halt_tensor, :].to('cpu')

                        split_num = args.split_num
                        divider = torch.linspace(0, 1, split_num, dtype=torch.float64)
                        if tasks.shape[-1] == 1:
                            if not active_tasks.shape[0] == 0:#here we set the new ranges to be split into n-1 pieces
                                for iter in range(split_num-1):
                                    tasks = torch.cat((tasks, torch.stack((active_tasks[:, 0] + (divider[iter] * (active_tasks[:, 1] - active_tasks[:, 0])), active_tasks[:, 0] + (divider[iter+1] * (active_tasks[:, 1] - active_tasks[:, 0]))), dim=1)), dim=0)
                        else:
                            if not active_tasks.shape[0] == 0:
                                for iter_1 in range(split_num-1):
                                    for iter_2 in range(split_num-1):
                                        to_stack_1 = torch.stack((active_tasks[:, 0, 0] + (divider[iter_1] * (active_tasks[:, 1, 0] - active_tasks[:, 0, 0])), active_tasks[:, 0, 0] + (divider[iter_1+1] * (active_tasks[:, 1, 0] - active_tasks[:, 0, 0]))), dim=1)
                                        to_stack_2 = torch.stack((active_tasks[:, 0, 1] + (divider[iter_2] * (active_tasks[:, 1, 1] - active_tasks[:, 0, 1])), active_tasks[:, 0, 1] + (divider[iter_2+1] * (active_tasks[:, 1, 1] - active_tasks[:, 0, 1]))), dim=1)
                                        tasks = torch.cat((tasks, torch.stack((to_stack_1, to_stack_2), dim=2)), dim=0)

                    violation_bounds[i, j, chan] = maximum_observed_value + epsilon
        return violation_bounds


    '''END GPU TESTING'''


    def get_regions_covered_parallel(self, active_ranges):

        return

    def get_regions_covered(self, coord, parameter_ranges):
        i, j = coord

        # Now we have a continuous boundary we must fill it with ALL smooth regions from interior.

        '''new way just uses both of the adjacent smooth regions when there is a jump.'''
        # Create a sampling list of the coords in parameter space (from edges)
        edge_samples = utils.sample_from_edges(parameter_ranges, num_samples=100)

        # Now we need to sample the transformation using these parameters
        cart_samples = None
        # cart_samples = []
        for sample_num in range(edge_samples.shape[0]):
            init_coord = self._get_cart_coord(coord).unsqueeze((1))
            for t, transformation in enumerate(self.transformations):
                if transformation in ['translate_x', 'translate_y']:
                    init_coord = init_coord + torch.tensor([edge_samples[sample_num, t], edge_samples[sample_num, t]]).unsqueeze(-1)
                else:
                    transformation_matrix = utils.matrix_generators[transformation]
                    init_coord = torch.mm(transformation_matrix(edge_samples[sample_num, t]), init_coord)
            # cart_samples.append(init_coord)
            if cart_samples is None:
                cart_samples = init_coord
            else:
                cart_samples = torch.cat((cart_samples, init_coord), dim=1)
        cart_samples = torch.floor(cart_samples.t())
        cart_diffs = torch.sum(abs(cart_samples[1:] - cart_samples[:-1]), dim=1)

        cart_diffs = torch.cat((torch.tensor(1,).unsqueeze(0), cart_diffs))
        clean_path = torch.unique(cart_samples[cart_diffs == 1, :], dim=0)

        for ind in torch.where(cart_diffs > 1)[0]:   # for each discontinuous point
            clean_path = torch.cat((torch.tensor([cart_samples[ind, [0]], cart_samples[ind-1, [1]]]).unsqueeze(dim=0), clean_path), dim=0)
            clean_path = torch.cat((torch.tensor([cart_samples[ind-1, [0]], cart_samples[ind, [1]]]).unsqueeze(dim=0), clean_path), dim=0)

        all_x = torch.unique(clean_path[:, 0], dim=0)
        interior_regions = None
        for x in all_x:
            valids_ys = clean_path[clean_path[:, 0] == x][:, 1]
            min_y = torch.min(valids_ys)
            max_y = torch.max(valids_ys)
            for y_val in range(int(min_y), int(max_y)+1):
                if interior_regions is None:
                    interior_regions = torch.tensor([x, y_val], dtype=torch.float64).unsqueeze(0)
                else:
                    interior_regions = torch.cat((interior_regions, torch.tensor([x, y_val], dtype=torch.float64).unsqueeze(0)), dim=0)
        if len(interior_regions.shape) == 2:
            full_region = torch.unique(torch.cat((clean_path, interior_regions)), dim=0)
        else:
            full_region = torch.unique(clean_path, dim=0)
        return full_region


    '''Get's the safe linear bounds by shifting the unsafe candidates.'''
    def shift_safe_bounds(self, linear_bounds, bound_type, epsilon):
        ''' To do this we need, for every pixel, to map the interpolation zones the transformation area intersects. We
         can solve the set of linear equations described by the transformation matrix, pixel coord, and interpolation grid coords to get the
         transformation parameter values at each intersection.
         '''

        # We can precompute the maximum interpolation gradient in every smooth region - this does not change per pixel
        if self.interpolation_gradient_grid is None:
            self.interpolation_gradient_grid = utils.get_interpolation_gradient_grid(self.image)
        c_time = time.time()
        violation_bounds = self.get_violation_maxima(linear_bounds, bound_type, epsilon).to('cpu')
        print("Time Taken:", time.time() - c_time)
        if bound_type == 'upper':
            linear_bounds[:, :, :, -1] += violation_bounds
        else:
            linear_bounds[:, :, :, -1] -= violation_bounds

        return linear_bounds



    '''Takes the upper and lower linear bounds and generates a second either upper or lower linear bound to combine with the bounds to make one PWL and a non-PWL bound.'''
    def get_unsafe_pw_bound(self, lin_lb, lin_ub):
        self.sample(args.sample_num, self.lb, self.ub)

        bound_midpoint = ((self.ub - self.lb)/2) + self.lb
        bounds = {'lower' : lin_lb, 'upper' : lin_ub}
        """Get the average distance from points to bounds for the four quadrants."""
        if bound_midpoint.shape[1] == 2:
            filter_1 = ((self.parameter_samples[0, 0, :] > bound_midpoint[:, 0]) & (
                    self.parameter_samples[0, 1, :] > bound_midpoint[:, 1]))
            filter_2 = ((self.parameter_samples[0, 0, :] > bound_midpoint[:, 0]) & (
                    self.parameter_samples[0, 1, :] < bound_midpoint[:, 1]))
            filter_3 = ((self.parameter_samples[0, 0, :] < bound_midpoint[:, 0]) & (
                    self.parameter_samples[0, 1, :] > bound_midpoint[:, 1]))
            filter_4 = ((self.parameter_samples[0, 0, :] < bound_midpoint[:, 0]) & (
                    self.parameter_samples[0, 1, :] < bound_midpoint[:, 1]))
        else:
            filter_1 = (self.parameter_samples[0, 0, :] > bound_midpoint[0])
            filter_2 = (self.parameter_samples[0, 0, :] > bound_midpoint[0])
            filter_3 = (self.parameter_samples[0, 0, :] < bound_midpoint[0])
            filter_4 = (self.parameter_samples[0, 0, :] < bound_midpoint[0])
        pw_bound_params = {}
        for key, bound in bounds.items():
            dist_1 = utils.dist_to_bound(bound, self.parameter_samples[0, :, filter_1], self.image_samples[0, :, :, :, filter_1])
            dist_2 = utils.dist_to_bound(bound, self.parameter_samples[0, :, filter_2], self.image_samples[0, :, :, :, filter_2])
            dist_3 = utils.dist_to_bound(bound, self.parameter_samples[0, :, filter_3], self.image_samples[0, :, :, :, filter_3])
            dist_4 = utils.dist_to_bound(bound, self.parameter_samples[0, :, filter_4], self.image_samples[0, :, :, :, filter_4])

            quadrants = torch.argmax(torch.stack([dist_1, dist_2, dist_3, dist_4]), dim=0) # 0 = >mid1 >mid2,   1 = >mid1 <mid2,    2= <mid1 >mid2,    3  =  <mid1  <mid2

            '''Now we need to obtain stack of image samples where each pixel is sampled from only the quadrant in question (just send the filtered one for each pixel?)'''
            quadrant_lookup = {
                0: self.parameter_samples[0, :, filter_1],
                1: self.parameter_samples[0, :, filter_2],
                2: self.parameter_samples[0, :, filter_3],
                3: self.parameter_samples[0, :, filter_4]
            }

            pw_params = torch.zeros(self.i_size, self.j_size, self.n_chans, torch.sum(filter_1), len(self.transformations))

            for n in range(self.n_chans):
                for j in range(self.j_size):
                    for i in range(self.i_size):
                        pw_params[i, j, n, ...] = quadrant_lookup[quadrants[i, j, n].item()].T
            pw_bound_params[key] = self.get_unsafe_lin_bounds(key, np.reshape(pw_params, (-1, pw_params.shape[3], pw_params.shape[4])))


        '''Now we have the bound parameters for a second, pw upper and lower bound. We need to decide which one to use. There are two choices: ub and lb1/2 or ub1/2 and lb.
        Let's check integral of the two options and pick for each one.'''
        pw_bound_key = torch.ones(self.i_size, self.j_size, self.n_chans).bool()
        for n in range(self.n_chans):
            for i in range(self.i_size):
                for j in range(self.j_size):
                    pw_bound_key[i, j, n] = utils.pw_bound_size_comparison([lin_lb[i, j, n], pw_bound_params['lower'][i, j, n]], [lin_ub[i, j, n], pw_bound_params['upper'][i, j, n]], self.lb[0], self.ub[0])
        pw_indicator = {}
        pw_indicator['upper'] = pw_bound_key
        pw_indicator['lower'] = ~pw_bound_key

            # pw_pixel_locs = np.zeros((self.i_size, self.j_size, self.n_chans, sum(filter_1), 2, 1))
            # for n in range(self.n_chans):
            #     for j in range(self.j_size):
            #         for i in range(self.i_size):
            #             # pw_params[i, j, n, :] = quadrant_lookup[quadrants[i, j, n].item()]
            #             pixel_params = quadrant_lookup[quadrants[i, j, n].item()]
            #             x, y = self._get_cart_coord([i, j])
            #             pixel_locs = np.array([[x, y] for i in range(pixel_params.shape[0])])[:, :, np.newaxis]
            #             for t, transformation in enumerate(self.transformations):
            #                 transformation_matrix = utils.matrix_generators[transformation]
            #                 pixel_locs = np.matmul(np.array(list(map(transformation_matrix, pixel_params[:, t]))), pixel_locs)
            #             pw_pixel_locs[i, j, n, :] = pixel_locs
            # pw_pixel_locs = torch.tensor(np.reshape(pw_pixel_locs, (-1, pw_pixel_locs.shape[3], 2, 1)))
            # pw_pixel_values = torch.empty((self.i_size, self.j_size, self.n_chans, pw_pixel_locs.shape[1]))
            # for n in range(self.n_chans):
            #     for slice in range(pw_pixel_locs.shape[1]):
            #         pw_pixel_values[:, :, n, slice] = torch.unflatten(self.linear_interpolation(pw_pixel_locs[:, slice, :, :], self.image[:, :, n]), 0, (self.i_size, self.j_size))


        # '''Now we need the values on the bounds which correspond to the sampling parameters'''
        # lower_bound_vals = (self.image_samples - torch.bmm
        # (lin_lb[:, :, :, :-1], self.parameter_samples[:, :]) + lin_lb[:, :, :, -1:])
        # upper_bound_vals = lin_ub[:, :, :, :-1] * self.parameter_samples[:, 0] + lin_ub[:, :, :, -1:] - self.image_samples
        #
        # lower_bound_vals, lower_indices = torch.max(lower_bound_vals, dim=3)
        # upper_bound_vals, upper_indices = torch.max(upper_bound_vals, dim=3)
        #
        # lower_split_points = self.parameter_samples[lower_indices[:, :, 0]]
        # upper_split_points = self.parameter_samples[upper_indices[:, :, 0]]
        #
        # pw_indicator = upper_bound_vals > lower_bound_vals  # 0 means lower bound is pwl
        # mid_points = (pw_indicator * upper_split_points) + (~pw_indicator*lower_split_points)    # Gives the midpoint in parameter space of the split.

        '''We used to create linear bounds for all pixels together. Now we have different bounds for each one, how do I do this?'''
        '''We need to get samples for each pixel as they all have different bounds in the parameter space. We also need to decide what the orientation of the boundary will be
        in the case of two transformations in composition.'''

        return pw_bound_params, pw_indicator


    def shift_pw_bound(self, pw_bounds, pw_indicators, epsilon):
        if self.interpolation_gradient_grid is None:
            self.interpolation_gradient_grid = utils.get_interpolation_gradient_grid(self.image)

        mixed_bounds = torch.zeros(self.i_size, self.j_size, self.n_chans, len(self.transformations)+1, dtype=torch.float64)
        mixed_bounds += pw_indicators['upper'].unsqueeze(-1) * pw_bounds['upper']
        mixed_bounds += pw_indicators['lower'].unsqueeze(-1) * pw_bounds['lower']
        violation_bounds = self.get_violation_maxima(mixed_bounds, None, epsilon, pw_indicator=pw_indicators).to('cpu')
        pw_bounds['lower'][:, :, :, -1] -= pw_indicators['lower'] * violation_bounds
        pw_bounds['upper'][:, :, :, -1] += pw_indicators['upper'] * violation_bounds
        return pw_bounds


'''
Takes a single image and returns a verification result.
'''
def main(im_num, with_vis=False):
    assert set(args.transformation).issubset({'rotate', 'scale', 'shear', 'translate_x', 'translate_y'}), "Transformation list includes unsupposed transformations."
    assert args.LB.shape[1] == args.UB.shape[1] == len(args.transformation), "Bound lists and transformation lists are not of same size."
    if args.dset == 'MNIST':
        mnist_deset = datasets.MNIST(root='./data', train=False, download=True, transform=None)   # the data is under 'data' and the labels under 'targets'
        im = Image(mnist_deset.data[im_num, :, :]/255., args.transformation, args.LB, args.UB)
    elif args.dset == 'CIFAR':
        cifar_deset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)   # the data is under 'data' and the labels under 'targets'
        im = Image(torch.tensor(cifar_deset.data[im_num, :, :]/255.), args.transformation, args.LB, args.UB)
    elif args.dset == 'fashion':
        fashion_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=None)   # the data is under 'data' and the labels under 'targets'
        im = Image(torch.tensor(fashion_dataset.data[im_num, :, :]/255., dtype=torch.float64), args.transformation, args.LB, args.UB)
    im.sample(args.sample_num, args.LB, args.UB)

    import test_suite
    # start = time.time()
    # unsafe_lin_lb = test_suite.custom_lin_solver(im.image_samples, im.parameter_samples, 'lower', num_iterations=1000, lr=0.01)
    # unsafe_lin_ub = test_suite.custom_lin_solver(im.image_samples, im.parameter_samples, 'upper', num_iterations=10000, lr=0.5)
    # unsafe_lin_lb = test_suite.ols_soln(im.image_samples, im.parameter_samples)
    # unsafe_lin_ub = unsafe_lin_lb.clone().detach()
    # print(time.time() - start)

    # start = time.time()
    unsafe_lin_lb = im.get_unsafe_lin_bounds('lower')
    # print(time.time() - start)
    unsafe_lin_ub = im.get_unsafe_lin_bounds('upper')

    # import importlib
    # importlib.reload(visualize)
    # visualize.visualize_bounds(15, 17, im.parameter_samples[0], im.image_samples[0], unsafe_lin_lb, unsafe_lin_ub)

    safe_lin_lb = im.shift_safe_bounds(unsafe_lin_lb, 'lower', args.lipschitz_error)
    safe_lin_ub = im.shift_safe_bounds(unsafe_lin_ub, 'upper', args.lipschitz_error)
    if args.bound_type == 'linear':
        return (safe_lin_lb, safe_lin_ub, args.transformation, args.LB, args.UB)

    elif args.bound_type == 'pw_linear':
        unsafe_pw_bounds, pw_indicator = im.get_unsafe_pw_bound(torch.tensor(safe_lin_lb, dtype=torch.float64), torch.tensor(safe_lin_ub, dtype=torch.float64))
        '''This is calling gurobi and contains many numpy functions. Fix it.'''

        '''Adjust the PW bounds to be safe. Let's write a custom function that takes both bounds and then only process the one indicated in pw_indicator. We can use much
        of the same code as for the single linear bounds.'''
        safe_pw_bounds = im.shift_pw_bound(unsafe_pw_bounds, pw_indicator, args.lipschitz_error)
        if with_vis:
            im.sample(args.sample_num, args.LB, args.UB)
            visualize.visualize_pw_bounds(10, 17, im.parameter_samples, im.image_samples, safe_lin_lb, safe_lin_ub, safe_pw_bounds, pw_indicator)
            # visualize.visualize_bounds(15, 17, im.parameter_samples, im.image_samples, unsafe_lin_lb, unsafe_lin_ub)
        return (safe_lin_lb, safe_lin_ub, safe_pw_bounds, pw_indicator, args.transformation, args.LB, args.UB)


def load_and_vis_pw(i, j, num, transformation='rotate', bound_str='0.0_10.0', dset='MNIST'):
    with open("bounds/{3}/{0}/{1}/pw_linear_bounds_im_{2}.pkl".format(transformation, bound_str, num, dset), 'rb') as f:
        bound_tup = pickle.load(f)
    safe_lin_lb, safe_lin_ub, safe_pw_bounds, pw_indicator, _, LB, UB = bound_tup
    trans = [transformation]
    # LB = np.array([0])
    # UB = np.array([0.5])
    if dset == 'MNIST':
        mnist_deset = datasets.MNIST(root='./data', train=False, download=True,
                                     transform=None)  # the data is under 'data' and the labels under 'targets'
        im = Image(mnist_deset.data[num, :, :] / 255., trans, LB, UB)
    elif dset == 'CIFAR':
        cifar_deset = datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=None)  # the data is under 'data' and the labels under 'targets'
        im = Image(torch.tensor(cifar_deset.data[num, :, :] / 255.), trans, LB, UB)
    elif dset == 'fashion':
        fashion_dataset = datasets.FashionMNIST(root='./data', train=False, download=True,
                                                transform=None)  # the data is under 'data' and the labels under 'targets'
        im = Image(torch.tensor(fashion_dataset.data[num, :, :] / 255., dtype=torch.float64), trans,
                   LB, UB)
    im.sample(100, LB, UB)

    visualize.visualize_pw_bounds(i, j, im.parameter_samples[0, 0, ...], im.image_samples[0, ...], safe_lin_lb, safe_lin_ub, safe_pw_bounds, pw_indicator)

# load_and_vis_pw(8, 12, 0, bound_str='[1.0]_[1.02]', transformation='scale', dset='fashion')

if __name__ == '__main__':
    # Run script
    from tqdm import tqdm
    for im in tqdm(range(args.image_number)):
        # print("Starting image {}".format(im))
        if args.bound_type == 'linear':
            with torch.no_grad():
                # make sure automatic torch gradients are not unnecessarily computed
                bounds = main(im, with_vis=with_vis)
        elif args.bound_type == 'pw_linear':
            with torch.no_grad():
                # make sure automatic torch gradients are not unnecessarily computed
                bounds = main(im, with_vis=with_vis)
                # pass
        if args.save_bounds:
            lbs = '-'.join(list(map(str,original_lb.tolist())))
            ubs = '-'.join(list(map(str,original_ub.tolist())))
            trans = '-'.join(args.transformation)
            import os
            save_dir = "./bounds/{3}/{2}/{0}_{1}/".format(lbs, ubs, trans, args.dset)
            os.makedirs(save_dir, exist_ok=True)
            with open(save_dir + "{0}_bounds_im_{1}.pkl".format(args.bound_type, im), 'wb') as f:
                pickle.dump(bounds, f)
    print("done")
