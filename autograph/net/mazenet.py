import functools, sys
from torchsummary import summary
import torch
import re
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F

from autograph.lib.envs.mazeenv import n_hot_grid
from autograph.lib.util import const_plane


class Mazenet(nn.Module):
    def __init__(self, maze_shape, actions_n, in_channels, initstride=1, initpadding=0, separate_networks=False):
        super(Mazenet, self).__init__()
        self.separate_networks = separate_networks
        self.maze_shape = maze_shape
        self.in_channels = in_channels
        self.maze_size = 1
        self.rounds_trained = 0
        for dim in maze_shape:
            self.maze_size *= dim

        self.net_common_before_flatten = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(2, 2), stride=initstride,
                      padding=0).float(),
            # n*trans(x)*trans(y) -> 32*x*y
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), padding=0, stride=initstride).float(),
            nn.ReLU(),
        )



        output_before_flatten: tensor = self.net_common_before_flatten(torch.zeros(1, in_channels, *maze_shape))

        x = output_before_flatten.flatten(1)

        lin_layer_size = output_before_flatten.flatten(1).shape[1]

        self.net_common_after_flatten = nn.Sequential(
            nn.Linear(lin_layer_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # if we uncomment this, change LP function for larger net
        '''
        self.net_common_after_flatten_v = nn.Sequential(
            nn.Linear(lin_layer_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        '''
        self.net_policy = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_n),
        )

        self.net_value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        common: tensor = self.net_common_before_flatten(x)
        common_flat = common.flatten(1)
        common_after_linear = self.net_common_after_flatten(common_flat)

        #if self.separate_networks:
        #    common_after_linear_v = self.net_common_after_flatten_v(common_flat)
        #   val = self.net_value(common_after_linear_v)
        #else:
        val = self.net_value(common_after_linear)


        pol = self.net_policy(common_after_linear)

        return pol, val

    def batch_states(self, states):
        processed = torch.stack(tuple(self.rewrite_obs(s) for s in states))
        return self(processed)

    def _process_single_tensor(self, tens):
        return tens.float()

    def forward_obs(self, x, device):
        info = maze_obs_rewrite(self.maze_size, x)
        pols, vals = self(info.to(device).unsqueeze(0))
        return pols[0].cpu(), vals[0].cpu()

    # given a mazenet, get the output shapes at each layer; ordered with value head at end
    def get_combined_output_shapes(self, policy=True, value=False):
        if policy == value:
            return ValueError("policy or value not both")
        og_stdout = sys.stdout

        with open('temp_print.txt', 'w') as f:
            sys.stdout = f
            summary(self, (self.in_channels, self.maze_shape[0], self.maze_shape[1]))
            sys.stdout = og_stdout

        # init shape with ones
        shape = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                 1, 1, 1, 1, 1, 1, 1, 1]
        shape[0] = [self.maze_shape[0], self.maze_shape[1], self.in_channels]
        jdx = 0
        with open('temp_print.txt', 'r') as f:
            for idx, line in enumerate(f):
                if idx < 3 or idx > 15:
                    pass
                else:
                    dims = re.compile(r'\[(.*?)\]')

                    found = dims.findall(line)

                    if len(found) > 0:
                        dim_split = found[0].split(',')
                        if len(dim_split) == 4:
                            shape[idx - 2] = [int(dim_split[2]), int(dim_split[3]), int(dim_split[1])]
                        else:
                            shape[idx - 2] = int(dim_split[1])

        # print('pre', shape)
        temp = shape[13]
        temps = shape[11:13]
        shape[11] = temp
        shape[12:14] = temps
        if policy:
            shape = shape[0:-1]
            shape[-1] = 6
            shape[5] = shape[4][0] * shape[4][1] * shape[4][2]

        return shape


    # given a mazenet, get the output shapes at each layer; ordered with value head at end
    def get_output_shapes(self, policy=True, value=False, num_actions=6):

        og_stdout = sys.stdout

        with open('temp_print.txt', 'w') as f:
            sys.stdout = f
            summary(self, (self.in_channels, self.maze_shape[0], self.maze_shape[1]))
            sys.stdout = og_stdout

        # init shape with ones
        shape = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                 1, 1, 1, 1, 1, 1, 1, 1]
        shape[0] = [self.maze_shape[0], self.maze_shape[1], self.in_channels]
        jdx = 0
        with open('temp_print.txt', 'r') as f:
            for idx, line in enumerate(f):
                if idx < 3 or idx > 15:
                    pass
                else:
                    dims = re.compile(r'\[(.*?)\]')

                    found = dims.findall(line)

                    if len(found) > 0:
                        dim_split = found[0].split(',')
                        if len(dim_split) == 4:
                            shape[idx - 2] = [int(dim_split[2]), int(dim_split[3]), int(dim_split[1])]
                        else:
                            shape[idx - 2] = int(dim_split[1])

        # print('pre', shape)
        temp = shape[13]
        temps = shape[11:13]
        shape[11] = temp
        shape[12:14] = temps
        if policy and value:
            temp1_1 = shape[-1]

            shape = shape[0:-1]

            temp1_2 = shape[-1]
            shape[-1] = num_actions
            shape[5] = shape[4][0] * shape[4][1] * shape[4][2]

            shape.append(temp)
            shape.append(temp)
            shape.append(temp1_1)
            shape.append(temp1_2)


        elif policy:
            shape = shape[0:-1]

            shape[-1] = num_actions
            shape[5] = shape[4][0] * shape[4][1] * shape[4][2]

        return shape

    # TRYING WITH TRUNKS
    def get_output_shapes_two_trunk(self, policy=True, value=True):

        og_stdout = sys.stdout

        with open('temp_print.txt', 'w') as f:
            sys.stdout = f
            summary(self, (self.in_channels, self.maze_shape[0], self.maze_shape[1]))
            sys.stdout = og_stdout

        # init shape with ones
        shape = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        shape[0] = [self.maze_shape[0], self.maze_shape[1], self.in_channels]
        jdx = 0
        with open('temp_print.txt', 'r') as f:
            for idx, line in enumerate(f):
                if idx < 3 or idx > 18:
                    pass
                else:
                    dims = re.compile(r'\[(.*?)\]')

                    found = dims.findall(line)

                    if len(found) > 0:
                        dim_split = found[0].split(',')
                        if len(dim_split) == 4:
                            shape[idx - 2] = [int(dim_split[2]), int(dim_split[3]), int(dim_split[1])]
                        else:
                            shape[idx - 2] = int(dim_split[1])

        # print('pre', shape)
        temp = shape[13]
        temps = shape[11:13]
        shape[11] = temp
        shape[12:14] = temps
        if policy and value:
            temp1_1 = shape[-1]

            shape = shape[0:-1]

            temp1_2 = shape[-1]
            shape[-1] = 6
            # policy
            shape[5] = shape[4][0] * shape[4][1] * shape[4][2]

            #value
            shape[9] = shape[5]


            shape.append(temp)
            shape.append(temp)
            shape.append(temp1_1)
            shape.append(temp1_2)
            shape.pop(15)

            temp = shape[9:13]
            shape[9:16] = shape[13:20]
            shape[16:20] = temp
            shape = shape
        elif policy:
            shape = shape[0:-1]

            shape[-1] = 6
            shape[5] = shape[4][0] * shape[4][1] * shape[4][2]

        return shape

@functools.lru_cache(16384)
def maze_obs_rewrite(shape, obs):
    fuel_level = const_plane(shape, obs[0])
    others = tuple(torch.from_numpy(n_hot_grid(shape, layer)).float() for layer in obs[1:])
    return torch.stack((fuel_level, *others), dim=0).float()
