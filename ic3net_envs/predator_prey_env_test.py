#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a predator prey environment.
Each agent can just observe itself (it's own identity) i.e. s_j = j and vision sqaure around it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
         -1 out of bound,
         indexing for predator agent (from 2?)
         ??? for prey agent (1 for fixed case, for now)
    - Action Space & Observation Space are according to an agent
    - Rewards -0.05 at each time step till the time
    - Episode never ends
    - Obs. State: Vocab of 1-hot < predator, preys & units >
"""

# core modules
from ctypes import sizeof
from enum import EnumMeta
from pickle import FALSE
import random
import math
import curses
import csv
from xml.dom.minidom import TypeInfo

# 3rd party modules
import gym
import numpy as np
from gym import spaces


class PredatorPreyEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.PREDATOR_CLASS_2 = 4
        self.PREDATOR_CLASS_3 = 5
        self.PREDATOR_CLASS_4 = 6
        self.DOOR_CLASS = 7
        self.SWITCH_CLASS = 9
        self.PLATE_CLASS = 11
        self.DISTRACTOR_CLASS = 13
        self.SYNC_CLASS = 15
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        self.episode_over = False

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)
        curses.init_pair(5, curses.COLOR_WHITE, -1)
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)
        curses.init_pair(7, curses.COLOR_BLUE, -1)


    def init_args(self, parser):
        env = parser.add_argument_group('Prey Predator task')
        env.add_argument('--nenemies', type=int, default=1,
                         help="Total number of preys in play")
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box")
        env.add_argument('--dim_col', type=int, default=5,
                         help="Dimension of box")
        env.add_argument('--vision', type=int, default=2,
                         help="Vision of predator")
        env.add_argument('--moving_prey', action="store_true", default=False,
                         help="Whether prey is fixed or moving")
        env.add_argument('--no_stay', action="store_true", default=False,
                         help="Whether predators have an action to stay in place")
        parser.add_argument('--mode', default='mixed', type=str,
                        help='cooperative|competitive|mixed (default: mixed)')
        env.add_argument('--enemy_comm', action="store_true", default=False,
                         help="Whether prey can communicate.")
        env.add_argument('--load_grid', type=str, default="5x5_PP.csv",
                         help="Load a pre-made grid csv file")
        env.add_argument('--task_type', type=str, default="default",
                         help="Change functionality and reward functions for specific tasks")

    def multi_agent_init(self, args):

        # General variables defining the environment : CONFIG
        params = ['dim', 'dim_col', 'vision', 'moving_prey', 'mode', 'enemy_comm','load_grid']
        for key in params:
            setattr(self, key, getattr(args, key))

        self.nprey = args.nenemies
        self.npredator = args.nfriendly
        self.dims = dims = (self.dim, self.dim_col)
        self.stay = not args.no_stay
        self.task_type = args.task_type
        self.task_count = 0

        if args.moving_prey:
            self.moving_prey = True
            self.prey_actions = 5
        else:
            self.prey_actions = 0

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do -
        if self.stay:
            self.naction = 5
        else:
            self.naction = 4

        self.action_space = spaces.MultiDiscrete([self.naction])

        self.BASE = (dims[0] * dims[1])
        self.file_dims = []
        self._init_grid_2()
        self.OUTSIDE_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE
        self.PREDATOR_CLASS_2 += self.BASE
        self.PREDATOR_CLASS_3 += self.BASE
        self.PREDATOR_CLASS_4 += self.BASE
        self.DOOR_CLASS += self.BASE
        self.SWITCH_CLASS += self.BASE
        self.PLATE_CLASS += self.BASE
        self.SYNC_CLASS += self.BASE
        self.DISTRACTOR_CLASS += self.BASE

        # Setting max vocab size for 1-hot encoding
        self.vocab_size = self.BASE + 17
        #          predator + prey + grid + outside

        # Observation for each agent will be vision * vision ndarray
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1), dtype=int)
        # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        return

    def step(self, action):
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :

            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")
        action = np.array(action).squeeze()
        action = np.atleast_1d(action)

        # Insert update environment here?
        if self.moving_prey:
            stopped = False
            for r in self.reached_prey:
                if r == 1:
                    stopped = True
            if not stopped:
                act = np.random.choice(4,1)[0]
                self.prey_loc[0] = self.movement(act,self.prey_loc[0])[:]

        for i, a in enumerate(action):
            self._take_action(i, a)


        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."


        self.episode_over = False
        self.obs = self._get_obs()

        debug = {'predator_locs':self.predator_loc,'prey_locs':self.prey_loc}
        return self.obs, self._get_reward(), self.episode_over, debug

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.reached_prey = np.zeros(self.npredator)

        # Locations
        locs = self._get_cordinates()
        #print(locs)
        #self.predator_loc, self.prey_loc = locs[:self.npredator], locs[self.npredator:]
        # print("Step")
        # print(self.predator_loc)

        self._set_grid()
        self._set_grid_2()

        # stat - like success ratio
        self.stat = dict()

        # Observation will be npredator * vision * vision ndarray
        self.obs = self._get_obs()
        return self.obs

    def seed(self):
        return

    def _get_cordinates(self):
        idx = np.random.choice(np.prod(self.dims),(self.npredator + self.nprey), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)

        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def _set_grid_2(self):
        self.task_count = 0
        self.grid2 = np.array(self.ref_grid, copy=True)
        self.wall_loc = self.wall_loc_ref[:]
        self.door_loc = self.door_loc_ref[:]
        self.switch_loc = self.switch_loc_ref[:]
        self.plate_loc = self.plate_loc_ref[:]
        self.sync_loc = self.sync_loc_ref[:]
        self.distract_loc = self.distract_loc_ref[:]
        self.predator_loc = []
        self.prey_loc = []
        idx = np.random.choice(len(self.agent_spawns),self.npredator)
        for i in idx:
            self.predator_loc.append(self.agent_spawns[i])
        idx = np.random.choice(len(self.prey_spawns),self.nprey)
        for i in idx:
            self.prey_loc.append(self.prey_spawns[i])
        for p in self.door_loc:
            p[2] = 0
        for p in self.switch_loc:
            p[2] = 0
        for p in self.plate_loc:
            p[2] = 0
        for p in self.sync_loc:
            p[2] = 0
            p[4] = 0
        self.predator_loc = np.vstack(self.predator_loc)
        self.prey_loc = np.vstack(self.prey_loc)

        # print(self.switch_loc_ref)
        # print(self.switch_loc)
        # print(self.door_loc)

    def _init_grid_2(self):
        print("Loading Grid From File: " + self.load_grid)
        if self.load_grid[1] == "x":
            self.file_dims = [int(self.load_grid[0]), int(self.load_grid[2])]
        else:
            self.file_dims = [int(self.load_grid[0:2]), int(self.load_grid[3:5])]
        self.dims = dims = (self.file_dims[0], self.file_dims[1])
        self.BASE = (dims[0] * dims[1])
        ref_grid_str = np.array(np.zeros(self.dims),dtype='string')
        self.wall_loc_ref = []
        self.door_loc_ref = []
        self.switch_loc_ref = []
        self.plate_loc_ref = []
        self.sync_loc_ref = []
        self.distract_loc_ref = []
        self.agent_spawns = []
        self.prey_spawns = []
        filepath = "Grids/Info_" + self.load_grid + "/Grid_"  + self.load_grid + ".csv"
        print(filepath)
        with open(filepath,'r') as f:
            grid = csv.reader(f, delimiter='\t')
            for i,line in enumerate(grid):
                print(line)
                for j,val in enumerate(line):
                    ref_grid_str[i][j] = val
                    if val[0] == "D":
                        self.door_loc_ref.append([i,j,0,int(val[1])])
                        ref_grid_str[i][j] = str(self.DOOR_CLASS)
                    if val[0] == "S":
                        self.switch_loc_ref.append([i,j,0,int(val[1])])
                        ref_grid_str[i][j] = str(self.SWITCH_CLASS)
                    if val[0] == "L":
                        self.plate_loc_ref.append([i,j,0,int(val[1])])
                        ref_grid_str[i][j] = str(self.PLATE_CLASS)
                    if val[0] == "Y":
                        self.sync_loc_ref.append([i,j,0,int(val[1]),0])
                        ref_grid_str[i][j] = str(self.SYNC_CLASS)
                    if val == "W":
                        self.wall_loc_ref.append([i,j])
                        ref_grid_str[i][j] = str(self.OUTSIDE_CLASS)
                    if val == "R":
                        self.distract_loc_ref.append([i,j])
                        ref_grid_str[i][j] = str(self.DISTRACTOR_CLASS)
                    if val == "P":
                        self.prey_spawns.append([i,j])
                        ref_grid_str[i][j] = "0"
                    if val == "A":
                        self.agent_spawns.append([i,j])
                        ref_grid_str[i][j] = "0"
                    if val == "AP":
                        self.prey_spawns.append([i,j])
                        self.agent_spawns.append([i,j])
                        ref_grid_str[i][j] = "0"

        self.ref_grid = ref_grid_str.astype(np.int)
        print(self.ref_grid)
        self.ref_grid += self.BASE

    def _get_obs(self):
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        for i, p in enumerate(self.predator_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1
        
        for i, p in enumerate(self.prey_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1

        for i, p in enumerate(self.wall_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.OUTSIDE_CLASS] += 1

        for i, p in enumerate(self.door_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.DOOR_CLASS+p[2]] += 1

        for i, p in enumerate(self.switch_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.SWITCH_CLASS+p[2]] += 1

        for i, p in enumerate(self.plate_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PLATE_CLASS+p[2]] += 1

        for i, p in enumerate(self.sync_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.SYNC_CLASS+p[2]] += 1

        for i, p in enumerate(self.distract_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.DISTRACT_CLASS] += 1

        obs = []
        for p in self.predator_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])

        if self.enemy_comm:
            for p in self.prey_loc:
                slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
                slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
                obs.append(self.bool_base_grid[slice_y, slice_x])

        obs = np.stack(obs)
        return obs

    def _take_action(self, idx, act):
        # prey action
        # if idx >= self.npredator:
        #     # This is only called if enemy comm and moving_prey is enabled
        #     if not self.moving_prey:
        #         return
        #     else:
        #         for r in self.reached_prey:
        #             if r == 1:
        #                 return
        #         self.prey_loc[0] = self.movement(act,self.prey_loc[0])[:]
        #         return

        if self.reached_prey[idx] == 1:
            return

        self.predator_loc[idx] = self.movement(act,self.predator_loc[idx])[:]

        # This is the plate and door toggle
        for p in self.plate_loc:
            prev_state = p[2]
            p[2] = 0
            for pr in self.predator_loc:
                if pr[0] == p[0] and pr[1] == p[1]:
                    p[2] = 1
            self.grid2[p[0]][p[1]] = self.PLATE_CLASS+p[2]
            if prev_state != p[2]:
                for d in self.door_loc:
                    if d[3] == p[3]:
                        d[2] = p[2]
                        self.grid2[d[0]][d[1]] = self.DOOR_CLASS+d[2]

        # This is the synchronized plate and door toggle
        if len(self.sync_loc) > 0:
            for p in self.sync_loc:
                p[4] = p[2]
                p[2] = 0
                for pr in self.predator_loc:
                    if pr[0] == p[0] and pr[1] == p[1]:
                        p[2] = 1
                self.grid2[p[0]][p[1]] = self.SYNC_CLASS+p[2]
            door_open = True
            for p in self.sync_loc:
                if p[2] == 0 or p[4] == 1:
                    door_open = False
            if door_open:
                for d in self.door_loc:
                    if d[3] == self.sync_loc[0][3]:
                        d[2] = 1
                        self.grid2[d[0]][d[1]] = self.DOOR_CLASS+d[2]
                        # print("Door Opening")
                        # print(p)

    def movement(self,act,agent_loc):
        [y,x] = agent_loc[:]
        
        # STAY action
        if act==5:
            return [y, x]

        # UP
        if act==0 and self.valid_action([max(0, y - 1), x]):
            y = max(0, y-1)

        # RIGHT
        elif act==1 and self.valid_action([y, min(self.dims[1]-1, x + 1)]):
            x = min(self.dims[1]-1, x+1)

        # DOWN
        elif act==2 and self.valid_action([min(self.dims[0]-1, y + 1), x]):
            y = min(self.dims[0]-1, y+1)

        # LEFT
        elif act==3 and self.valid_action([y, max(0, x - 1)]):
            x = max(0, x-1)
        
        agent_loc = [y, x]
        return agent_loc
    
    def valid_action(self,target_loc):
        # This will check if an action is valid
        #print(target_loc)
        [y,x] = target_loc[:]
        grid_val = self.grid2[y][x]
        if  grid_val == self.OUTSIDE_CLASS or grid_val == self.DOOR_CLASS:
            return False

        # This is the switch and door toggle
        for p in self.switch_loc:
            if y == p[0] and x == p[1]:
                p[2] = (p[2]+1)%2
                self.grid2[p[0]][p[1]] = self.SWITCH_CLASS+p[2]
                for d in self.door_loc:
                    if d[3] == p[3]:
                        d[2] = (d[2]+1)%2
                        self.grid2[d[0]][d[1]] = self.DOOR_CLASS+p[2]

        return True
    
    def _get_reward(self):
        # self.TIMESTEP_PENALTY = -0.05
        # self.PREY_REWARD = 0
        # self.POS_PREY_REWARD = 0.05
        # Default mode = mixed
        #print("testing reward")
        n = self.npredator if not self.enemy_comm else self.npredator + self.nprey
        reward = np.full(n, self.TIMESTEP_PENALTY)
        
        # max_dist = np.linalg.norm(self.dims)
        # for i,pred in enumerate(self.predator_loc):
        #     for prey in self.prey_loc:
        #         diffy = pred[0]-prey[0]
        #         diffx = pred[1]-prey[1]
        #         dist = np.linalg.norm([diffx,diffy])
        #         dist_reward = 1 - dist/max_dist
        #         reward[i] += dist_reward * 0.03
                

        on_prey = np.where(np.all(self.predator_loc == self.prey_loc,axis=1))[0]
        nb_predator_on_prey = on_prey.size
        
        if self.mode == 'cooperative':
            reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey
        elif self.mode == 'competitive':
            if nb_predator_on_prey:
                reward[on_prey] = self.POS_PREY_REWARD / nb_predator_on_prey
        elif self.mode == 'mixed':
            if self.task_type == "default":
                reward[on_prey] = self.PREY_REWARD
            elif self.task_count == 1:
                reward[on_prey] = self.PREY_REWARD
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

        self.reached_prey[on_prey] = 1

        # if np.all(self.reached_prey == 1) and self.mode == 'mixed':
        #     self.episode_over = True
            
        # Prey reward
        if nb_predator_on_prey == 0:
            reward[self.npredator:] = -1 * self.TIMESTEP_PENALTY
        else:
            # TODO: discuss & finalise
            reward[self.npredator:] = 0

        # Success ratio
        if self.task_type == "sequence":
            if nb_predator_on_prey >= (self.npredator-len(self.plate_loc)):
                if self.task_count == 0:
                    self.prey_loc = []
                    idx = np.random.choice(len(self.agent_spawns),self.nprey)
                    for i in idx:
                        self.prey_loc.append(self.agent_spawns[i])
                    self.task_count = 1
                    self.stat['success'] = 0.0
                    #print("Task 1 Complete")
                else:
                    self.episode_over = True
                    self.stat['success'] = 1
                    #print("Task 2 Complete")
            else:
                self.stat['success'] = self.task_count * 0.0
            return reward
        else:
            if self.mode != 'competitive':
                if nb_predator_on_prey >= (self.npredator-len(self.plate_loc)):
                    self.episode_over = True
                    self.stat['success'] = 1
                else:
                    self.stat['success'] = 0
            return reward

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())


    def _onehot_initialization(self, a):
        ncols = self.vocab_size
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def render(self, mode='human', close=False):
        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.stdscr.clear()

        for p in self.predator_loc:
            if grid[p[0]][p[1]] != 0 and (grid[p[0]][p[1]][0] == 'X' or grid[p[0]][p[1]][1] == 'X'):
                if grid[p[0]][p[1]][0] == 'X':
                    grid[p[0]][p[1]] = '2X'
                else:
                    n = int(grid[p[0]][p[1]][0]) + 1
                    grid[p[0]][p[1]] = str(n)+'X'
            elif grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'X'
            else:
                grid[p[0]][p[1]] = 'X'

        for p in self.prey_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        for p in self.wall_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'W'
            else:
                grid[p[0]][p[1]] = 'W'

        for p in self.door_loc:
            if p[2] == 0:
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'DC'
                else:
                    grid[p[0]][p[1]] = 'DC'
            else:
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'DO'
                else:
                    grid[p[0]][p[1]] = 'DO'

        for p in self.switch_loc:
            if p[2] == 0:
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'SC'
                else:
                    grid[p[0]][p[1]] = 'SC'
            else:
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'SO'
                else:
                    grid[p[0]][p[1]] = 'SO'

        for p in self.plate_loc:
            if p[2] == 0:
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'LC'
                else:
                    grid[p[0]][p[1]] = 'LC'
            else:
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'LO'
                else:
                    grid[p[0]][p[1]] = 'LO'

        for p in self.sync_loc:
            if p[2] == 0:
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'YC'
                else:
                    grid[p[0]][p[1]] = 'YC'
            else:
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'YO'
                else:
                    grid[p[0]][p[1]] = 'YO'

        for p in self.distract_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'R'
            else:
                grid[p[0]][p[1]] = 'R'

        #grid[2][2] = chr(self.OUTSIDE_CLASS)
        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if 'X' in item and 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif 'X' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    elif 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(2))
                    elif 'W' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(5))
                    elif 'DC' in item:
                        self.stdscr.addstr(row_num, idx * 4, 'D'.center(3), curses.color_pair(6))
                    elif 'DO' in item:
                        self.stdscr.addstr(row_num, idx * 4, 'D'.center(3), curses.color_pair(2))
                    elif 'SC' in item:
                        self.stdscr.addstr(row_num, idx * 4, 'S'.center(3), curses.color_pair(6))
                    elif 'SO' in item:
                        self.stdscr.addstr(row_num, idx * 4, 'S'.center(3), curses.color_pair(2))
                    elif 'LC' in item:
                        self.stdscr.addstr(row_num, idx * 4, 'L'.center(3), curses.color_pair(6))
                    elif 'LO' in item:
                        self.stdscr.addstr(row_num, idx * 4, 'L'.center(3), curses.color_pair(2))
                    elif 'YC' in item:
                        self.stdscr.addstr(row_num, idx * 4, 'Y'.center(3), curses.color_pair(6))
                    elif 'YO' in item:
                        self.stdscr.addstr(row_num, idx * 4, 'Y'.center(3), curses.color_pair(2))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(7))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()
