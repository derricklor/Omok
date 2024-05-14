#code for RL agent using approximation function
#function approximation code for mountaincar in hw6 to be reused in hw8


import numpy as np
import matplotlib.pyplot as plt
#import gymnasium as gym

dlen = 256 # (n+1)^k, number of features
num_actions = 2
dprime = num_actions*dlen

#begin hw6
def X1d(s,n):
    #s is the state space somewhere in the range [0,1] with 1 dimensions
    #n is the number of features from 0 to n

    ret = np.zeros(n)
    for i in range(0,n):
        ret[i]= np.cos(i*np.pi*s)
    return ret


def calcV(s, w):
    #s is the state
    #w is the weight vector 1D (16) with length =nfeatures, transposed to be column vertical
    xs = X2d(s)# feature vector, row horizontal
    V = np.dot(xs,w) #vector vector multiplication from numpy, outputs a single scalar
    return V


def normalize(arg, min, max):
    #takes arg and normalizes it from min to max. output is in range [0,1] inclusive
    return (arg-min)/(max-min)

def generateC(n,k):
    #dlen = 256 in cartpole
    c = np.zeros((dlen,k))#c is the coefficient vector

    #build c
    ithfeature = 0
    for i in range(n+1):
        for j in range(n+1):
            for l in range(n+1):
                for e in range(n+1):
                    c[ithfeature][0] = i
                    c[ithfeature][1] = j
                    c[ithfeature][2] = l
                    c[ithfeature][3] = e
                    ithfeature +=1
    return c

c = generateC(3,4) # lenth of c is 256


def X2d(s):
    #s is the state space vector with 4 states normalized in the range [0,1]
    #n is number of features = (n+1)^k = d =256 in cartpole
    
    ret = np.zeros(dlen)
    for i in range(0, dlen):
        ret[i]= np.cos(np.pi*s.dot(c[i]))
                                    
    return ret



def calcQ(s,a,w):
    #s is the state
    #a is the action
    #w is the weight vector with length =nfeatures, transposed to be column vertical
    #Q(s,a,w) = Q=wX(sa) = Q=w(a)X(s) = np.dot(w(a),X(s))
    xs = X2d(s) # feature vector, row horizontal 
    Q = np.dot(xs, w[a]) #vector vector multiplication from numpy, outputs a single scalar
    return Q


def greedy_action(s,w):
    #s is state
    #w is weight vector
    #compute Q(s,*,w) for each action and argmax over it to get best action
    temp = np.zeros(3) # assumes action space is 3
    for i in range(0,3):
        temp[i] = calcQ(s,i,w)

    return np.argmax(temp)
    
#end hw6
#######################################begin hw8


def h(s, a, theta):
    #theta = parameterized 1D np array (512)
    #outputs a single scalar value
    x = xsa(s,a) # follow equation h() = theta.dot( x(s,a) )
    h = theta.dot(x)
    return h

def softmax(s, theta):
    #theta = parameterized 1D np array (32)
    #assumes action space is 2
    #outputs same size np array as theta, with probability distribution equivalent to each action's value / sum of all actions
    pdistribution = np.zeros(2)
    e1, e2 = np.exp(h(s,0,theta)), np.exp(h(s,1,theta)) #assignment
    esum = e1 + e2
    pdistribution[0] = e1 / esum
    pdistribution[1] = e2 / esum
    return pdistribution

def xsa(s,a):
    # x(s,0) = [256 features, 0], x(s,1) = [0, 256 features]. 
    # where 0 indicates 256 features of zeros
    # a is action in range 0, 1
    ret = np.zeros(dprime) # init all actions*dlen to zero
    for i in range(0, dlen): #index the bits we need changing (256*a)+i
        #fourier basis 
        ret[(256*a)+i] = np.cos(np.pi*s.dot(c[i]))# c is global variable defined in my py file
    return ret

def eligibilityvector(a, s, theta):
    #\nabla lnpi(a|s, theta) = x(s,a) - sum_b pi(b|s)*x(s,b)
    #a is action in range 0, 1
    #returns a feature vector size dprime = 32
    xa = xsa(s,a)
    sm = softmax(s,theta)
    summation = xsa(s,0)*sm[0] + xsa(s,1)*sm[1]
    return xa - summation



######################################end
#hw6 below
'''
def ESSarsa(alpha, epsilon, gamma, numEpisodes):
    #init
    
    nfeatures = 16 #using fourier basis (n+1)^k, where k=dimensions of state and n = 3
    
    w = np.zeros((3, nfeatures)) # actions x nfeatures
    rewards_per_episode = np.zeros(numEpisodes) # collect data to plot
    steps_per_episode = np.zeros(numEpisodes)

    env = gym.make('MountainCar-v0')
    #env._max_episode_steps = 200
    #epsilondelta = epsilon-0.01/numEpisodes #linear epsilon decay
    for episode in range(numEpisodes):
        
        epsilon = epsilon*0.95
        state, _ = env.reset()
        #normalize
        pos = normalize(state[0], -1.2, 0.6) #position obs space min and max
        vel = normalize(state[1], -0.07, 0.07) #velocity obs space min and max
        state = np.zeros(2)
        state[0] = pos
        state[1] = vel
        if np.random.rand() >= epsilon:# epsilon greedy
            #greedy, compute Q(s,*,w) for each action and argmax over it
            action = greedy_action(state,w)
        else:
            action = np.random.choice(3) #explore
            
        terminated = False
        step_counter = 0
        while not terminated:
            #take action, observe R, S'
            next_state, reward, terminated, _, _ = env.step(action)
            new_pos = normalize(next_state[0], -1.2, 0.6)
            new_vel = normalize(next_state[1], -0.07, 0.07)
            next_state = np.zeros(2)
            next_state[0] = new_pos
            next_state[1] = new_vel

            rewards_per_episode[episode] += reward
            
            if terminated:
                #Q(s,a,w) = Q=wX(sa) = Q=w(a)X(s) = np.dot(w(a),X(s))
                #w[a] and X(s) must have same size in order to dot product
                w[action] = w[action] + alpha*(reward - calcQ(state,action,w) )*X2d(state) #gradient is just feature vector
            else:
                if np.random.rand() >= epsilon:# epsilon greedy
                    next_action = greedy_action(state,w)
                else:
                    next_action =  np.random.choice(3) #explore
                #Q(s,a,w) = Q=wX(sa) = Q=w(a)X(s) = np.dot(w(a),X(s))
                w[action] = w[action] + alpha*(reward + gamma*calcQ(next_state,next_action,w)-\
                                                               calcQ(state,action,w) )*X2d(state)


            state = next_state #last to be updated before next iteration
            action = next_action
            step_counter +=1
        #collect data
        steps_per_episode[episode] = step_counter

    env.close()
    return w, rewards_per_episode, steps_per_episode
    '''
#end

