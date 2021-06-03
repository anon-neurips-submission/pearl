import functools

from autograph.lib.envs.mineworldenv import MineWorldEnv
from autograph.net.curiosity.rnd_models import RND
from autograph.net.mazenet import Mazenet
from autograph.net.minenet import Minenet, minecraft_obs_rewrite, minecraft_obs_rewrite_go, minecraft_obs_rewrite_chess, minecraft_obs_rewrite_checkers


def num_channels(env: MineWorldEnv):
    """How many output channels the environment will have"""
    return len(env.config.inventory) + len(env.config.placements) + 1


def minenet_v1(env: MineWorldEnv, num_blocks, **kwargs):
    return Minenet(env.shape, num_channels(env), num_blocks, env.action_space.n, **kwargs)


def mine_mazenet_v1(env: MineWorldEnv, separate_networks=False):
    return Mazenet(env.shape, actions_n=env.action_space.n, in_channels=num_channels(env), initstride=1, initpadding=0, separate_networks=separate_networks)

def mine_mazenet_v1_go(env, separate_networks=False):
    return Mazenet(env.env.env.env.env.shape, actions_n=env.action_spaces['black_0'].n, in_channels=3, initstride=1, initpadding=0, separate_networks=separate_networks)

def mine_mazenet_v1_chess(env, separate_networks=False):
    return Mazenet(env.env.env.env.env.shape, actions_n=env.action_spaces['player_0'].n, in_channels=20, initstride=1, initpadding=0, separate_networks=separate_networks)

def mine_mazenet_v1_checkers(env, separate_networks=False):
    return Mazenet(env.env.env.env.env.shape, actions_n=env.action_spaces['player_0'].n, in_channels=4, initstride=1, initpadding=0, separate_networks=separate_networks)


def minernd_v1(env: MineWorldEnv, feature_space: int):
    return RND(num_channels(env), env.shape, feature_space, init_stride=1)


def mine_obs_rewriter_creator(env: MineWorldEnv):
    return functools.partial(minecraft_obs_rewrite, env.shape)

def mine_obs_go_rewriter_creator(env: MineWorldEnv):
    return functools.partial(minecraft_obs_rewrite_go, env.env.env.env.env.shape)

def mine_obs_chess_rewriter_creator(env: MineWorldEnv):
    return functools.partial(minecraft_obs_rewrite_chess, env.env.env.env.env.shape)

def mine_obs_checkers_rewriter_creator(env: MineWorldEnv):
    return functools.partial(minecraft_obs_rewrite_checkers, env.env.env.env.env.shape)
