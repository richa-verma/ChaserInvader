import math
import gym
import random
from gym import spaces
from gym.utils import seeding
import numpy as np
from rllab.spaces import Box
from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
from rllab.misc.overrides import overrides
from rllab.misc import logger
#continous mountain car
#author: richa
class ChaserInvaderEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.viewer = None
        self.reward = 0.0
        self.L  = 2.0
        self.prev_reward = 0.0
        self.phi_max = math.pi/4
        self.rho_min = self.L/math.tan(self.phi_max)
        self.min_position = 0.0
        self.max_position = 30.0
        self.vel_g = 1.2
        self.vel_b = 1.0
        self.vel_i = 1.0
        self.min_action_vel = -1.0
        self.max_action_vel = 1.2
        self.thresh_distance = 1.0 #difference between distances at which invader is caught
        self.thresh_distance_gi = 0.25 #distance between invader and guard when invader is considered caught
        self.thresh_distance_ib = 0.75 #distance between invader and base station when the invader has reached the base station
        #state: guard stats, invader stats, base station stats 
        self.low_state  = np.array([self.min_position,self.min_position, -self.vel_g, -self.phi_max,self.min_position,self.min_position, -self.vel_i, -self.phi_max,self.min_position,self.min_position, -self.vel_b, -self.phi_max])
        self.high_state = np.array([self.max_position,self.max_position,self.vel_g, self.phi_max,self.max_position,self.max_position,self.vel_i, self.phi_max,self.max_position,self.max_position,self.vel_b, -self.phi_max])
        self.observation_space = Box(low=self.low_state, high=self.high_state)
        self.action_space      = Box(np.array([self.min_action_vel,-self.phi_max]), np.array([self.vel_g,+self.phi_max]))  # speed, angle
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        position_g = [self.state[0],self.state[1]]  #guard
        velocity_g = self.state[2]
        phi_g      = self.state[3]
        position_i = [self.state[4],self.state[5]]  #invader
        velocity_i = self.state[6]
        phi_i      = self.state[7]
        position_b = [self.state[8],self.state[9]]  #base station
        velocity_b = self.state[10]
        phi_b      = self.state[11]
        action_vel = action[0]      #velocity from action
        action_phi = action[1]      #phi from action

        d3 = math.sqrt((position_g[1]-position_i[1])**2+(position_g[0]-position_i[0])**2) #d3 = position_g - position_i
        d1 = math.sqrt((position_b[1]-position_i[1])**2+(position_b[0]-position_i[0])**2) #d1 = position_b - position_i

        done   = 0
        reward = 0.0
        info   = []
        if((d3-d1)>0 and (d3-d1)<self.thresh_distance):
            reward = 10.0
            done   = True
            info   = ["invader caught"]
        elif((d1-d3)>0 and (d1-d3)<self.thresh_distance):
            reward = -10.0
            done   = True
            info   = ["invader won"]
        elif(d3<self.thresh_distance_gi):
            reward = 10.0
            done   = True
            info   = ["invader caught"]
        elif(d1<self.thresh_distance_ib):
            reward = -10.0
            done   = True
            info   = ["invader won"]
        else:
            if action_vel  > self.max_action_vel:
                action_vel = self.max_action_vel
            if action_vel  < self.min_action_vel:
                action_vel = self.min_action_vel
            #compute position of guard
            theta = (action_vel/self.L)*math.tan(action_phi)
            x = position_g[0] + action_vel*math.cos(theta)
            y = position_g[1] + action_vel*math.sin(theta)
            if x < self.min_position:
                x = 0.0
            if x > self.max_position:
                x = 30.0
            if y < self.min_position:
                y = 0.0
            if y > self.max_position:
                y = 30.0
            #computing invader's position
            self.state[6] = random.uniform(0,1)
            self.state[7] = random.uniform(-math.pi/2,math.pi/2)
            x_i = self.state[4] + self.state[6]*math.cos(self.state[7])
            y_i = self.state[5] + self.state[6]*math.sin(self.state[7])
            if x_i < self.min_position:
                x_i = 0.0
            if x_i > self.max_position:
                x_i = 30.0
            if y_i < self.min_position:
                y_i = 0.0
            if y_i > self.max_position:
                y_i = 30.0
            #base station is stationary
            info = ["still working"]
            self.state = np.array([x,y, action_vel, action_phi, x_i,y_i, self.state[6], self.state[7], 10,0,0,1.0]) #keep velocity of invader and base station constant
        return Step(observation=self.state, reward=reward, done=done,info=info)

    def reset(self):
        self.state = np.array([20,0, 1.2, 0, 20,10, 1.0, -float((math.pi)/4), 10,0,0.0,0.0]) #acc to constraint d3 <= d1
        #self.state = np.array([self.min_position, self.min_vel_g, -self.phi_max, self.min_position, self.min_vel_i, -self.phi_max, self.min_position, self.min_vel_b, -self.phi_max])
        return np.array(self.state)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        cartwidth=40
        cartheight=20
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            guard = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.guardtrans = rendering.Transform()
            guard.add_attr(self.guardtrans)
            self.viewer.add_geom(guard)
            invader = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.invadertrans = rendering.Transform()
            invader.add_attr(self.invadertrans)
            self.viewer.add_geom(invader)
            base = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.basetrans = rendering.Transform()
            base.add_attr(self.basetrans)
            self.viewer.add_geom(base)
            if self.state is None: return None
        x = self.state
        guardx = x[0]*scale+screen_width/2.0
        guardy = x[1]*scale+screen_height/2.0
        invaderx = x[4]*scale+screen_width/2.0
        invadery = x[5]*scale+screen_height/2.0
        basex = x[8]*scale+screen_width/2.0
        basey = x[9]*scale+screen_width/2.0
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()