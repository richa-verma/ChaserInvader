import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
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
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)
        self.action_space      = spaces.Box(np.array([self.min_action_vel,-self.phi_max]), np.array([self.vel_g,+self.phi_max]))  # speed, angle
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
        print (action)
        action_vel = action      #velocity from action
        action_phi = action      #phi from action

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
            x_i = self.state[3] - self.state[5]*self.state[6]
            y_i = self.state[4] - self.state[5]*self.state[6]
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
            self.state = np.array([x,y, action_phi, action_vel, x_i,y_i, self.state[4], self.state[5], 10,0,0,1.0]) #keep velocity of invader and base station constant
        return self.state, reward, done, info

    def reset(self):
        self.state = np.array([20,0, 1.2, 0, 20,10, 1.0, float((math.pi)/4), 10,0,0.0,1.0]) #acc to constraint d3 <= d1
        #self.state = np.array([self.min_position, self.min_vel_g, -self.phi_max, self.min_position, self.min_vel_i, -self.phi_max, self.min_position, self.min_vel_b, -self.phi_max])
        return np.array(self.state)
    '''def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()'''

