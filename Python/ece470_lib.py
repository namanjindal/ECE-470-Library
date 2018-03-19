import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm, logm

def bracket(v):
    v = np.asarray(v)
    rtn = []
    if(v.shape == (6,1)):
        rtn = bracket(v[:3])
        rtn = np.concatenate((rtn, v[3:]), axis=1)
        rtn = np.concatenate((rtn, np.zeros((1,4))))
    elif(v.shape == (3,1)):
        rtn = np.zeros((3,3))
        rtn[0][1] = - v[2]
        rtn[0][2] =   v[1]
        rtn[1][2] = - v[0]
        rtn = rtn - rtn.transpose()
    return rtn

def inv_bracket(m):
    rtn = []
    if(m.shape == (4,4)):
        rtn = inv_bracket(m[:3,:3])
        rtn = np.concatenate((rtn, np.array([m[:3,3]]).transpose()))
    elif(m.shape == (3,3)):
        rtn = np.zeros((3,1))
        rtn[2] = -m[0][1]
        rtn[1] = m[0][2]
        rtn[0] = -m[1][2]
    return rtn

def adj_T(T):
    r = T[:3,:3]
    p = T[:3,3:4]
    temp1 = np.concatenate((r, np.zeros((3,3))), axis=1)
    temp2 = np.concatenate((bracket(p).dot(r), r), axis=1)
    return np.concatenate((temp1, temp2), axis=0)

def toPose(rot, pos):
	temp = np.concatenate((rot, pos), axis=1)
	return np.concatenate((temp, [[0,0,0,1]]), axis=0)

def fromPose(T):
	return (T[:3,:3], T[:3, 3:4])

def toScrew(a, q=None):
    a = np.asarray([a]).transpose()
    if(q==None):
        return np.concatenate((np.zeros((3,1)), a))
    q = np.asarray([q]).transpose()
    return np.concatenate((a, bracket(q).dot(a)))

def toTs(S, theta):
    return [expm(bracket(s) * t) for s, t in zip(S, theta)]

def evalT(S, theta, M):
    ret = np.identity(4)
    for t in toTs(S, theta):
        ret = ret.dot(t)

    return ret.dot(M)

def evalJ(S, theta):
    T = toTs(S, theta)
    J = [S[0]]
    for i in range(1, len(S)):
        col = T[0]
        for j in range(1, i):
            col = col.dot(T[j])
        J.append(adj_T(col).dot(S[i]))
    return np.hstack(J)

def findIK(endT, S, M, theta=None, max_iter=100, max_err = 0.001, mu=0.05):
    if  theta==None:
        theta = np.zeros((len(S),1))
    V = np.ones((6,1))
    while np.linalg.norm(V) > max_err and max_iter > 0:
        curr_pose = evalT(S, theta, M)
        V = inv_bracket(logm(endT.dot(inv(curr_pose))))
        J = evalJ(S, theta)
        pinv = inv(J.transpose().dot(J) + mu*np.identity(len(S))).dot(J.transpose())
        thetadot = pinv.dot(V)
        theta = theta + thetadot
    return (theta, np.linalg.norm(V))