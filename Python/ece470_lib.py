import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm, logm

"""
ECE 470 Lib

------------
Created by Naman Jindal

"""

def combine_matrix(m):
    """
    Combines a list of matricies into a single numpy matrix
    Essentially stacks every row of the input list into a numpy matrix,
    then stacks all the rows vertically.
    :param m: The list of lists to convert
    """
    return np.vstack([np.hstack(row) for row in m])


def bracket(v):
    """
    Returns the 'bracket' operator of a 3x1 vector or 6x1 twist
    :param v: the 3x1 vector or 6x1 twist, can be of type list or numpy.ndarray - Must be convertible to a numpy array!
    :returns: a 3x3 or 4x4 numpy array based on the input matrix or an empty list otherwise
    """
    v = np.asarray(v)
    rtn = []
    if(v.shape == (6,1)):
        rtn =   [[ bracket(v[:3]),  v[3:]   ],
                 [ np.zeros((1,4))          ]]
        rtn = combine_matrix(rtn)
    elif(v.shape == (3,1)):
        rtn = np.zeros((3,3))
        rtn[0][1] = - v[2]
        rtn[0][2] =   v[1]
        rtn[1][2] = - v[0]
        rtn = rtn - rtn.transpose()
    return rtn

def inv_bracket(m):
    """
    Performs the inverse 'bracket' operation on a 3x3 or 4x4 matrix
    :param m: the 3x3 skew-symmetric matrix or 4x4 bracket of a twist - Must be convertible to a numpy array!
    :returns: the vector or twist representation of the input matrix or an empty list otherwise
    """
    rtn = []
    m = np.asarray(m)
    if(m.shape == (4,4)):
        rtn =   [[ inv_bracket(m[:3,:3])],
                 [ m[:3,3:]             ]]
        rtn = combine_matrix(rtn)
    elif(m.shape == (3,3)):
        m = m - m.transpose()
        rtn = np.zeros((3,1))
        rtn[2] = - m[0][1]/2
        rtn[1] =   m[0][2]/2
        rtn[0] = - m[1][2]/2
    return rtn

def adj_T(T):
    """
    Returns the adjoint transformation matrix of T
    :param T: the pose whose 6x6 adjoint matrix to return
    """
    rot, pos = fromPose(T)
    return combine_matrix([[ rot,                   np.zeros((3,3)) ],
                           [ bracket(pos).dot(rot), rot             ]])

def toPose(rot, pos):
    return combine_matrix([[ rot, pos  ],
                           [ [0,0,0,1] ]])

def fromPose(T):
    T = np.asarray(T)
    return (T[:3,:3], T[:3, 3:4])

def toScrew(a, q=None):
    """
    Returns the screw of some prismatic or revolute joint as a 6x1 numpy array
    """
    if type(a) is list:
        a = np.transpose([a])
    if q is not None and type(q) is list:
        q = np.transpose([q])
        return combine_matrix([[ a                 ],
                               [ bracket(q).dot(a) ]])

    return combine_matrix([[ np.zeros((3,1)) ],
                           [ a               ]])

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
    if  theta is None:
        theta = np.zeros((len(S),1))
    V = np.ones((6,1))
    while np.linalg.norm(V) > max_err and max_iter > 0:
        curr_pose = evalT(S, theta, M)
        V = inv_bracket(logm(endT.dot(inv(curr_pose))))
        J = evalJ(S, theta)
        pinv = inv(J.transpose().dot(J) + mu*np.identity(len(S))).dot(J.transpose())
        thetadot = pinv.dot(V)
        theta = theta + thetadot
        max_iter -= 1;
    return (theta, np.linalg.norm(V))