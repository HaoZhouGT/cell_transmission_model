# Create a gym env that simulates 3-lane highway traffic
import os
import gym
from gym import spaces
import numpy as np
import random
import math
from traffic_functions import *
import matplotlib.pyplot as plt


'''
we give an initial condition, which is k_old
every step, we update k_old
at the last step, after updating, we should also output k_old as the result


for the reimman problem, the boundary condition is density

'''


class AV_MB_Env(gym.Env):

    def __init__(self, Up, Dw, AV_init_pos, Upstream_den, Downstream_den, \
        Initial_cond, Init_steps, Drop_b, Drop_e, Drop_cell, Drop_lane):
        '''
        Arrival flow, boundary condition, initial condition 
        '''
        self.Drop_b = Drop_b
        self.Drop_e = Drop_e
        self.Drop_cell = Drop_cell # lane drop cell position
        self.Drop_lane = Drop_lane # index of the drop lane

        self.Up_N = int(Up) # Upstream observable cells
        self.Dw_N = int(Dw) # Downstream observable cells

        self.up_den = Upstream_den # should be a (3,T) array
        self.down_den = Downstream_den # it is unnecessary if we have open downstream boundary
        
        self.L = 2+ np.shape(Initial_cond)[1] # Initial_cond is a array of (3,L-2)
        self.T = np.shape(Upstream_den)[1] # (3,T) array       
        self.end = L * Dx # unit is ft
        self.k_old = kj * np.ones((3,self.L)) # this needs to be used for computing, and updated every time
        self.v = u * np.ones((3,self.L))
        Q = kj/(1/u+1/w)*u
        self.sending_cap = Q * np.ones((3,self.L))
        self.receiving_cap = Q * np.ones((3,self.L+1)) # this means downstream boundary (L+1) is always open
        # for the sending_cap and receiving_cap, we can also use them to simulate the bottleneck

        self.AV_speed = 0 # wait for reset() to overwrite
        self.AV_pos = AV_init_pos # it should be the location of cell, in unit of ft
        assert (0< self.AV_pos and self.AV_pos < self.end), " AV location {} invalid".format(self.AV_pos)
    
        
        # set action/observation spaces are continuous, use Box to define 
#         self.action_space = spaces.Box(low=-4.0, high=2.0, dtype=np.float32) # will be used for clipping later
#         self.state = spaces.Box(low=0.0, high=kj, shape=(self.Up_N+self.Dw_N, 3), dtype=np.float32)
        self.state = None
        self.action_space = None
        
        self.time_step = 0 # this is the time index used to find the corresponding boundary condition 

        self.score = 0
        self.Init_steps = int(Init_steps)
        self.reset()
        

    def reset(self):
        '''
        reset function should return the state for the policy to produce action
        '''
      # initialize the simualtion by letting all lanes run for a few seconds, 
      # for initialization, we assume all lanes have same density, so there is no lane changes at all
        self.score = 0
        for t in range(self.Init_steps):
            self.time_step +=1
            # run the three lane simulation without LCs or AV control
            k_now = kj * np.ones((3,self.L)) # initialize k_now, (3*L-1), note that downstream boundary do not count
            k_now[:,0] = self.up_den[:,self.time_step] # get the upstream boundary condition at current timestep 
            ############## for the left lane ##################################################################
            for j in range(1,L):
                s = self.k_old[0][j] + 1/u * (Flux(u, w, kj, self.k_old[0][j-1], self.k_old[0][j], self.sending_cap[0][j-1], self.receiving_cap[0][j]) -
                                         Flux(u, w, kj, self.k_old[0][j], self.k_old[0][j+1], self.sending_cap[0][j], self.receiving_cap[0][j+1]))
                k_now[0][j] = max(0.0,s)

            ############## for the middle lane ##################################################################
            for j in range(1,L):
                s = self.k_old[1][j] + 1/u * (Flux(u, w, kj, self.k_old[1][j-1], self.k_old[1][j], self.sending_cap[1][j-1], self.receiving_cap[1][j]) -
                                         Flux(u, w, kj, self.k_old[1][j], self.k_old[1][j+1], self.sending_cap[1][j], self.receiving_cap[1][j+1]))
                k_now[1][j] = max(0.0,s)

            ############## for the right lane ##################################################################
            for j in range(1,L):
                s = self.k_old[2][j] + 1/u * (Flux(u, w, kj, self.k_old[2][j-1], self.k_old[2][j], self.sending_cap[2][j-1], self.receiving_cap[2][j]) -
                                         Flux(u, w, kj, self.k_old[2][j], self.k_old[2][j+1], self.sending_cap[2][j], self.receiving_cap[2][j+1]))
                k_now[2][j] = max(0.0,s)
            ############## update self.k_old using k_now ############################
            self.k_old = k_now

        AV_cell = math.floor(self.AV_pos/Dx) # the road cell index the AV is located
        # self.AV_pos += Dt * den_to_v(k_old[1][AV_cell], u, w, kj) * 1.46667 # (mi/h to ft/s)
        self.AV_speed = den_to_v(k_now[1][AV_cell], u, w, kj)

        self.state = self.k_old[:][AV_cell-self.Up_N:AV_cell+self.Dw_N] # using the surrouding density segment 
        return self.state
        
    def step(self, action):
        '''
        In step function, we update: i) the density of all cells on three lanes; ii) speed and position of all MBs 
        including the AV_MB; and iii) apply the acceleration action on the AV_MB using the policy
        
        As an external input, we also need to simulate an active bottleneck, such as a lane drop  

        And we, collect the reward, update the cumulative score
        Use the density solution of all road cells, extract the observational road segment around the AV_MB, 
        then output the current state.
        Check the termination condition of the episode
        '''
        k_now = kj * np.ones((3,self.L)) # every t we create such a new array to receive density updates
        k_now[:][0] = self.up_den[:][self.time_step] # get the upstream boundary condition at current timestep 
        # k_now[:][L-1] = Downstream_den[:][self.time_step]


        ############### forcing the lane-drop constraint before using CT model ##############################
        if self.Drop_b < self.time_step and self.time_step < self.Drop_e:
            receiving_cap[self.Drop_lane][self.Drop_cell] == 0
            sending_cap[self.Drop_lane][self.Drop_cell] == 0

        ############## for the left lane ##################################################################
        for j in range(1,L):
            s = self.k_old[0][j] + 1/u * (Flux(u, w, kj, self.k_old[0][j-1], self.k_old[0][j], self.sending_cap[0][j-1], self.receiving_cap[0][j]) -
                                     Flux(u, w, kj, self.k_old[0][j], self.k_old[0][j+1], self.sending_cap[0][j], self.receiving_cap[0][j+1]))
            k_now[0][j] = max(0.0,s)

        ############## for the middle lane ##################################################################
        for j in range(1,L):
            s = self.k_old[1][j] + 1/u * (Flux(u, w, kj, self.k_old[1][j-1], self.k_old[1][j], self.sending_cap[1][j-1], self.receiving_cap[1][j]) -
                                     Flux(u, w, kj, self.k_old[1][j], self.k_old[1][j+1], self.sending_cap[1][j], self.receiving_cap[1][j+1]))
            k_now[1][j] = max(0.0,s)

        ############## for the right lane ##################################################################
        for j in range(1,L):
            s = self.k_old[2][j] + 1/u * (Flux(u, w, kj, self.k_old[2][j-1], self.k_old[2][j], self.sending_cap[2][j-1], self.receiving_cap[2][j]) -
                                     Flux(u, w, kj, self.k_old[2][j], self.k_old[2][j+1], self.sending_cap[2][j], self.receiving_cap[2][j+1]))
            k_now[2][j] = max(0.0,s)

        
        self.k_old = k_now # note that the k_old has been updated by k_now after each time step

        ############ update the speed and position of all exisiting MBs, including the AV_MB using the policy action #####
        assert self.action_space.contains(action), "acceleration out of bounds!"
        # action = min(action, a_bound)
        action = 0
        AV_speed += action * Dt

        # TO DO
        ############## generate new LCs using the poisson process ####################################
        '''
        create a list for tracking all LCs
        [[lane index, cell index, speed], ...]
        a regular MV is no longer active when it catches up with the downstream, 
        '''

        ############## update the capacity constraint for next time step, using all MBs and the bottleneck ###########
        '''
        note that all updated positions of existing MBs, and new LCs, will have impact on next time-step capacity
        '''
        AV_cell = math.floor(self.AV_pos/Dx)

        ############################################### output the state, reward, done, {info} ##################
        self.state = self.k_old[:][AV_cell-self.Up_N:AV_cell+self.Dw_N] # consider add AV speed into the state


        ########################## compute reward: the throughput at the downstream of bottleneck ###########
        # we are measuring the 5 cells downstream from the bottleneck to calculate the throughput
        reward = den_to_flow(self.k_old[self.Drop_lane][self.Drop_cell+5])*Dt # immediate reward
        self.score += reward
          
        done = (self.AV_position >= self.end)
        
        self.time_step = self.time_step + 1 # move time step foward by 1

        return self.state, reward, done, {'current_den':self.k_old}

    # def render(self):
    #     '''
    #     the rendering function can be used for visualization after each episode
    #     show how the AV_MB passes the bottleneck, and help dampen the osscilation
    #     '''
    #     screen_width = 600
    #     screen_height = 400

    #     world_width = self.x_threshold * 2
    #     scale = screen_width/world_width
    #     carty = 100  # TOP OF CART
    #     polewidth = 10.0
    #     polelen = scale * (2 * self.length)
    #     cartwidth = 50.0
    #     cartheight = 30.0

    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    #         axleoffset = cartheight / 4.0
    #         cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         self.carttrans = rendering.Transform()
    #         cart.add_attr(self.carttrans)
    #         self.viewer.add_geom(cart)
    #         l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
    #         pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         pole.set_color(.8, .6, .4)
    #         self.poletrans = rendering.Transform(translation=(0, axleoffset))
    #         pole.add_attr(self.poletrans)
    #         pole.add_attr(self.carttrans)
    #         self.viewer.add_geom(pole)
    #         self.axle = rendering.make_circle(polewidth/2)
    #         self.axle.add_attr(self.poletrans)
    #         self.axle.add_attr(self.carttrans)
    #         self.axle.set_color(.5, .5, .8)
    #         self.viewer.add_geom(self.axle)
    #         self.track = rendering.Line((0, carty), (screen_width, carty))
    #         self.track.set_color(0, 0, 0)
    #         self.viewer.add_geom(self.track)

    #         self._pole_geom = pole

    #     if self.state is None:
    #         return None

    #     # Edit the pole polygon vertex
    #     pole = self._pole_geom
    #     l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
    #     pole.v = [(l, b), (l, t), (r, t), (r, b)]

    #     x = self.state
    #     cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    #     self.carttrans.set_translation(cartx, carty)
    #     self.poletrans.set_rotation(-x[2])

    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None


if __name__ == '__main__':
    # some parameters for trial
    L = 200
    T = 500
    kc = kj/(1/u+1/w)
    Upstream_den = kc*np.ones((3,T))
    Downstream_den = np.zeros((3,T))
    Initial_cond = kc*np.ones((3,L-2))
    Init_steps = 20
    Drop_b= 100
    Drop_e=200
    Drop_cell= 100
    Drop_lane=2

    env_test = AV_MB_Env(Up=10,Dw=10,AV_init_pos=20, Upstream_den= Upstream_den, \
                     Downstream_den= Downstream_den, Initial_cond=Initial_cond, \
                     Init_steps=Init_steps, Drop_b=Drop_b, Drop_e=Drop_e,\
                     Drop_cell=Drop_cell, Drop_lane=Drop_lane)
    observation = env_test.reset()
    print(observation)
    plt.imshow(observation)
    plt.show()


