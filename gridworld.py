
# state = (i,j)
# action in {0=up, 1=right, 2=down, 3=left}

import numpy as np

############################################################################################################
#
# Problem setting
#

def valid_move( world, i, j ):
    if i<0 or j<0 or i>=world.shape[0] or j>=world.shape[1]:
        return False
    return world[i,j]

def move( world, i, j, d ):
    pstay = 0
    ret   = []

    # main direction
    di_main = [ 0, 1, 0, -1 ]
    dj_main = [ -1, 0, 1, 0 ]
    (i0, j0) = (i+di_main[d], j+dj_main[d])
    if valid_move(world, i0, j0):
        ret.append( (i0,j0,0.8) )
    else:
        pstay += 0.8

    # perpendicular 1 ~left
    di_p1 = [ -1, 0, 1, 0 ]
    dj_p1 = [  0,-1, 0, 1 ]
    (i1, j1) = (i+di_p1[d], j+dj_p1[d])
    if valid_move(world, i1, j1):
        ret.append( (i1,j1,0.1) )
    else:
        pstay += 0.1

    # perpendicular 2 ~right
    di_p2 = [ 1, 0,-1, 0 ]
    dj_p2 = [ 0, 1, 0,-1 ]
    (i2, j2) = (i+di_p2[d], j+dj_p2[d])
    if valid_move(world, i2, j2):
        ret.append( (i2,j2,0.1) )
    else:
        pstay += 0.1

    if pstay>0:
        ret.append( (i,j,pstay) )
    return ret

############################################################################################################
def value_iteration( world, reward, end_state, v_old, gamma ):
    (w,h) = (world.shape[0], world.shape[1])
    v_new = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            if end_state[i,j]:
                v_new[i,j] = v_old[i,j]
            else:
                sum = np.zeros(4)
                for a in range (4):
                    l = move(world,i,j,a)
                    for(r,s,p) in l:
                        sum[a] = sum[a] + p*v_old[r,s]
                v_new[i,j] = reward[i,j] + gamma*np.max(sum)
    return v_new


#     ...
#
 def derive_policy( world, end_state, v ):
    (w,h) = (world.shape[0], world.shape[1])
    v_new = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            if end_state[i,j]:
                pi[i,j] = 0
            else:
                sum = np.zeros(4)
                for a in range (4):
                    l = move(world,i,j,a)
                    for(r,s,p) in l:
                        sum[a] = sum[a] + p*v_old[r,s]
                pi[i,j] = reward[i,j] + gamma*np.argmax(sum)
    return pi
#     ...
#
# def value_determination( world, reward, end_state, v_init, gamma, pi ):
#     ...
#
#
# # select action at random
# def qaction( world, i, j, q, T ):
#     ...
#
# # return new state from (i,j) with action a
# def qmove( world, i, j, a ):
#     ...
#
# def qlearning( world, reward, end_state, gamma, n, T ):
#     (w,h) = (world.shape[0], world.shape[1])
#     q = np.zeros((w,h,4))
#     for i in range(w):
#         for j in range(h):
#             for a in range(4):
#                 if end_state[i,j]:
#                     q[i,j,a] = reward[i,j] # init
#                 else:
#                     q[i,j,a] = 0.25
#
#     ...
#
#     return q
#
# def qpolicy( world, reward, end_state, q, gamma ):
#     ...

def qvalue( world, reward, end_state, q, v_init, gamma ):
    (w,h) = (world.shape[0], world.shape[1])
    v = v_init
    for i in range(w):
        for j in range(h):
            if not end_state[i,j]:
                v[i,j] = np.max(q[i,j,:])
    return v

############################################################################################################
#
# Main
#
if __name__ == "__main__":

    world = (0 == np.array( [ [0, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 1, 0, 1] ] ))
    reward = np.array( [ [0, 0, 0, 1],
                         [0, 0, 0,-1],
                         [0, 0, 0, 0] ] )
    end_state = (1 == np.array( [ [0, 0, 0, 1],
                                  [0, 1, 0, 1],
                                  [0, 0, 0, 0] ] ))
    v_init = np.zeros((3,4))
    v_init[0,3] = 1
    v_init[1,3] = -1

    gamma = 0.99


    v = v_init
    for i in range(100):
        v_new = value_iteration( world, reward, end_state, v, gamma )
        e = np.max( np.abs(v - v_new) )
        v = v_new
        if e<0.001:
            break
    print ('VI end')

    pi = derive_policy( world, end_state, v )

    print (v)
    print (pi)

    qi = derive_policy( world, end_state, v_init )
    for i in range(100):
        u = value_determination( world, reward, end_state, v_init, gamma, qi )
        new_qi = derive_policy( world, end_state, u )
        e  = np.max(np.abs(qi - new_qi))
        qi = new_qi
        if e<0.1:
            break
    print ('PI end')

    print (u)
    print (qi)

    q = qlearning( world, reward, end_state, gamma, 10000, 100 )
    pi_q = qpolicy( world, reward, end_state, q, gamma )
    qv   = qvalue( world, reward, end_state, q, v_init, gamma )

    print ('--------------')
    print (qv)
    print (pi_q)
