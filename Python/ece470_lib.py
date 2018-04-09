import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import expm, logm

"""
----------------------------------------------------------------------------------------------------
ece470_lib.py

Created for ECE 470 CBTF Exams - Python Version

Portions of this library will be included with the course exams.

Also feel free to use it for your Course Project

Hosted at https://github.com/namanjindal/ECE-470-Library
If you would like to contribute/have any issues, reply to @289 on the course Piazza
----------------------------------------------------------------------------------------------------
TO USE:

Import the library into Jupyter Notebook by placing this file in the same directory as your notebook.
Then run:

import ece470_lib as ece470

and then you can use the functions as

ece470.bracket(np.ones((6,1)))

----------------------------------------------------------------------------------------------------
FUNCTION OVERVIEW:

For a more detailed description, see the doc-string associated with each function/class

* skew4             - hardcoded 'bracket' operator on a 6x1 twist
* bracket           - evaluates the 'bracket' operator on a 3x1 or 6x1 input
* inv_bracket       - performs the inverse of 'bracket' on a 3x3 or 4x4 input
* adj_T             - Returns the 6x6 adjoint matrix of a 4x4 transformation matrix
* toPose            - Combines a 3x3 rotation matrix and 3x1 vector into a 4x4 HCT matrix
* fromPose          - Splits a 4x4 HCT matrix into a 3x3 rotation matrix and 3x1 position vector
* toScrew           - Calculates the 6x1 screw matrix for a revolute or prismatic joint
* toTs              - Returns a list of HCT matricies from a screw-axis matrix and joint variables
* sequential_Ts     - toTs, but each element is also multiplied by the previous elements
* evalT             - Evals the space-pose based on S, theta, and optionally, M
* evalJ             - Evals the space-jacobian based on S and theta
* findIK            - Finds a set of thetas to achieve a given pose based on S and M
* matrix_linspace   - Creates a list of linearly spaced matricies between two endpoints
* Tree Class        - Simple, generic tree data structure - supports insert, parents, and iteration

Possibly not included with Exam 5:

* multi_transform   - ?
* Dist3D            - why not just use np.linalg.norm(p1-p2)
* final_pos         - ?
* checkselfcollision- ?

----------------------------------------------------------------------------------------------------
CONTRIBUTORS:

Naman Jindal    (namanj2)
Kyle Jensen     (kejense2)

If you have any issues, questions or suggestions, email us or reply to @289 on Piazza! 

April 2018 Edition
Pre-Exam 5.1
"""


def skew4(V_b):
    return np.array([[0,-1*V_b[2],V_b[1],V_b[3]],
                     [V_b[2],0,-1*V_b[0],V_b[4]],
                     [-1*V_b[1],V_b[0],0,V_b[5]],
                     [0,0,0,0]])


def bracket(v):
    """
    Returns the 'bracket' operator of a 3x1 vector or 6x1 twist
    :param v: the 3x1 vector or 6x1 twist, can be of type list or numpy.ndarray - Must be convertible to a numpy array!
    :returns: a 3x3 or 4x4 numpy array based on the input matrix or an empty list otherwise
    """
    v = np.asarray(v)
    rtn = []
    if(v.shape == (6,1)):
        rtn = np.block([[ bracket(v[:3]),  v[3:]   ],
                        [ np.zeros((1,4))          ]])
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
        rtn = np.block([[ inv_bracket(m[:3,:3])],
                        [ m[:3,3:]             ]])
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
    return np.block([[ rot,                   np.zeros((3,3)) ],
                     [ bracket(pos).dot(rot), rot             ]])

def toPose(rot, pos):
    """
    Returns a 4x4 HCT matrix given by the 3x3 rotation matrix and 3x1 postion vector
    :param rot: A 3x3 Rotation Matrix
    :param pos: A 3x1 Position Vector
    :returns: A 4x4 HTC matrix as a numpy array
    """
    return np.block([[ rot, pos  ],
                     [ [0,0,0,1] ]])

def fromPose(T):
    """
    Returns a rotation matrix and position vector from a 4x4 HCT matrix
    :param T: The 4x4 HCT matrix as either python lists or numpy array
    :returns: a tuple with the first element being a 3x3 numpy array representing the rotation matrix
              and the second element being a 3x1 numpy array position vector
    """
    T = np.asarray(T)
    return (T[:3,:3], T[:3, 3:4])

def toScrew(a, q=None):
    """
    Returns the space screw of some prismatic or revolute joint as a 6x1 numpy array.
    If a q is supplied, the returned screw will be revolute; if no q, screw will be prismatic.
    Can use either python list, list of lists, or numpy array as inputs in XYZ order
    :param a: The axis of motion for a prismatic screw or axis of revolution. Should have norm 1 (not checked)
    :param q: A point passing through the axis if a revolute joint
    :returns: A 6x1 numpy matrix representing the screw axis
    """
    a = np.atleast_2d(a).reshape((3,1))
    # Revolute Screw
    if q is not None:
        q = np.atleast_2d(q).reshape((3,1))
        return np.block([[ a                 ],
                         [ bracket(q).dot(a) ]])
    # Prismatic Screw
    return np.block([[ np.zeros((3,1)) ],
                     [ a               ]])

def toTs(S, theta):
    """
    Generates a list of HCT matricies from a list of screw axes and joint variables. Not that useful for general work,
    but used by other functions. 
    :param S: A python list of 6x1 screw axes
    :param theta: A list/numpy array of joint vars. Should have the same number of elements as S
    :returns: A python list of 4x4 HCT matricies representing a transformation by each of the screw axes
    """
    if isinstance(S, np.ndarray):
        S = np.hsplit(S, S.shape[1])
    return [expm(bracket(s) * t) for s, t in zip(S, theta)]

def sequential_Ts(S, theta):
    """
    TODO: Validation/Documentation
    """
    T = toTs(S, theta)
    ret = [T[0]]
    for t in T[1:]:
        ret.append(ret[-1].dot(t))
    return ret

def evalT(S, theta, M=np.identity(4)):
    """
    Basically Forward Kinematics 
    Finds the end position of a robot based on space screw axes, joint vars and the space 'zero HCT'
    Note that numpy arrays of screw axes are not supported, only python lists of screw axes.
    Use np.hsplit(S, N) to generate a list of screw axes given a numpy array S where N is the number of joints (cols in the matrix) 
    :param S: A python list of 6x1 screw axes from the base to the manipulator
    :param theta: A python list/numpy array of joint vars in the same order as S.
    :param M: A 4x4 HCT transformation matrix representing the pose of the end effector when theta = 0 for all joint vars
    :returns: A numpy 4x4 HCT transformation matrix representing the pose of the end effector at the given joint vars
    """
    ret = np.identity(4)
    for t in toTs(S, theta):
        ret = ret.dot(t)
    return ret.dot(M)

def evalJ(S, theta):
    """
    Finds the space jacobian of a robot with given screw axes at a given joint positions:
    TODO: Improve efficeny by removing the need to recompute the transformation for each screw
    :param S: a python list of 6x1 screw axes
    :param theta: a python list/numpy array of joint vars. Should be same number of elements as S
    :returns: A 6xN matrix representing the space Jacobian of the robot with the given screw axes at the given joint vars
    """
    if isinstance(S, np.ndarray):
        S = np.hsplit(S, S.shape[1])
    T = sequential_Ts(S, theta)
    J = [S[0]]
    for t, s in zip(T, S[1:]):
        J.append(adj_T(t).dot(s))
    return np.hstack(J)

def findIK(endT, S, M, theta=None, max_iter=100, max_err = 0.001, mu=0.05):
    """
    Basically Inverse Kinematics
    Uses Newton's method to find joint vars to reach a given pose for a given robot. Returns joint positions and 
    the error. endT, S, and M should be provided in the space frame. Stop condiditons are when the final pose is less than a given
    twist norm from the desired end pose or a maximum number of iterations are reached. 
    TODO: Improve internal type flexibilty of input types
    :param endT: the desired end pose of the end effector
    :param S: a python list of 6x1 screw axes in the space frame
    :param M: the pose of the end effector when the robot is at the zero position
    :param theta: Optional - An initial guess of theta. If not provided, zeros are used. Should be a Nx1 numpy matrix
    :param max_iter: Optional - The maximum number of iterations of newtons method for error to fall below max_err. Default is 10
    :param max_err: Optional - The maximum error to determine the end of iterations before max_iter is reached. Default is 0.001 and should be good for PL/quizes
    :param mu: The normalizing coefficient (?) when computing the pseudo-inverse of the jacobian. Default is 0.05
    :returns: A tuple where the first element is an Nx1 numpy array of joint variables where the algorithm ended. Second 
              element is the norm of the twist required to take the found pose to the desired pose. Essentially the error that PL checks against.
    """
    if isinstance(S, list):
        S = np.hstack(S)
    
    if  theta is None:
        theta = np.zeros((S.shape[1],1))
    V = np.ones((6,1))
    while np.linalg.norm(V) > max_err and max_iter > 0:
        curr_pose = evalT(S, theta, M)
        V = inv_bracket(logm(endT.dot(inv(curr_pose))))
        J = evalJ(S, theta)
        pinv = inv(J.transpose().dot(J) + mu*np.identity(S.shape[1])).dot(J.transpose())
        thetadot = pinv.dot(V)
        theta = theta + thetadot
        max_iter -= 1;
    return (theta, np.linalg.norm(V))

def matrix_linspace(m_start, m_end, num, to_end=False):
    """
    np.linspace equivilant that also works for numpy arrays as well as numbers
    Can be set to include the endpoint in the given list or not. Can use either a number, python list, or numpy
    array for input, but m_start and m_end must be the same type and shape.
    :param m_start: The start point of the linspace. Will be the first matrix in the returned list
    :param m_end:   The end point of the linspace. Will NOT be included unless to_end is True and will then be the last element
    :param num:     A positive number that indicates the number of divisions
    :param to_end:  Default False - If True, will return num+1 elements and the last will be m_end

    """
    if type(m_start) is list:
        m_start = np.asarray(m_start)
        m_end = np.asarray(m_end)
    diff = (m_end - m_start)/num
    if to_end:
        num += 1
    ret = []
    for i in range(num):
        ret.append(m_start + i * diff)
    return ret 

class Tree:
    """
    Simple generic tree data structure - Uses a dictionary as underlying implementation
    Can take any value for internal elements that supports string cast.
    
    Note - string casting is used as the hash function so identical strings will mean identical elements.

    Not performance optimized, but should work fo the course. 
    Note: does not support removal/modification of tree structure other than insert

    Class is also iterable and supports python loops such as
    
    t1 = Tree("el1")
    ...
    for e in t1:
        print(e)
    ==>
    "el1"
    ...

    TODO: Complete Documentation
    """
    def __init__(self, root):
        self.__tree = {}
        self.__tree[self._hash(root)] = None
        self.__elem = [root]

    def size(self):
        return len(self.__elem)

    def insert(self, data, parent):
        if not self._hash(parent) in self.__tree.keys():
            raise KeyError("Parent element not in tree")
        if self._hash(data) in self.__tree.keys():
            raise IndexError("Data element already in tree")
        self.__elem.append(data)
        self.__tree[self._hash(data)] = parent

    def getElements(self):
        return list(self.__elem)

    def parent(self, data):
        if not self._hash(data) in self.__tree.keys():
            raise KeyError("Data element not in tree")
        return self.__tree[self._hash(data)]

    def __iter__(self):
        return list.__iter__(self.getElements())
   
    @staticmethod
    def _hash(el):
        return hash(str(el))

## -------------------------------------------------------------------------------------------------
## The following code will (possbibly) not be included with Exam 5
## -------------------------------------------------------------------------------------------------

def multi_transform(pts, S, theta):
    """
    Transforms a list of points by a given screw axis and theta combination. The number of points should be greater than or
    equal to the number of screw axes. Points will be transformed as follows:
    TODO DOC
    """ 
    Ns = 0
    Np = 0
    if isinstance(S, np.ndarray):
        Ns = S.shape[1]
        S = np.hsplit(S, Ns)
    else:
        Ns = len(S)
    theta = np.asarray(theta).flatten()
    if isinstance(pts, np.ndarray):
        Np = pts.shape[1]
        pts = np.hsplit(pts, Np)
    else:
        Np = len(pts)
    pts = [np.vstack([p, [[1]] ]) for p in pts]
    T = [evalT(S[:n], theta[:n]) for n in range(1,Ns+1)]
    while Ns < Np:
        T.insert(0, np.identity(4))
        Ns += 1
    return np.hstack([t.dot(p) for t, p in zip(T, pts)])[:3]

def Dist3D(p1, p2):
    """Euclidean distance function for three dimensions. Assumes that p1 and p2 are
    column matrices in the order of px, py, pz respectively.
    """
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def finalpos(S, theta, start):
    """ This function finds the final position of all joints. Solution to 5.1.3.
    Parameters:
    S: the 6x6 matrix of the spatial screw axes for all joints.
    theta: a 6x1 matrix representing a certain configuration of the robot. 
    start: a 3x8 where the i'th column represents the initial position of the ith joint in terms of frame 0. """
    
    position = start[:,:2]
    
    for i in range(2,8):
        M = np.identity(4)
        M[0,3] = start[0, i]
        M[1,3] = start[1, i]
        M[2,3] = start[2, i]
        T = evalT(S[:, 0:i-1], theta[0:i-1], M)
        position = np.concatenate((position, T[:3, 3:4]),axis=1)

    return position

def checkcollision(p, r, q, s):
    """checkcollision is the solution to 5.1.2.
    Parameters:
    p: a 3xn matrix which represents the positions of all spheres.
    r: a 1xn matrix representing the radii of every sphere. 
    q: a 3x1 matrix representing the position of the final sphere.
    s: the radius of the final sphere"""
    
    c = np.zeros(r.shape[1])
    for i in range(p.shape[1]):
        dis = Dist3D(q, p[:,[i]])
        if(dis < (r[:,i]+s)):
            c[i]+=1
    return c
        

def checkselfcollision(S, theta, start, r):
    """checkselfcollisions checks whether a certain series of configurations causes a self collision. Solution to 5.1.5
    S: a 6x6 matrix of the spatial screw axes for all joints.
    theta: a 6xn matrix of configurations. Notice that the ith column of this matrix is a 6x1 theta matrix representing a certain configuration of the robot
    start: a 3x8 matrix of joint starting positions in frame 0. 
    r: a given radius for surrounding spheres"""
    
    c = np.zeros(theta.shape[1])
    for i in range(theta.shape[1]):
        t = theta[:,[i]]
        jointpos = finalpos(S,t,start)
        for j in range(jointpos.shape[1]):
            joint2check = jointpos[:,[j]]
            for k in range(jointpos.shape[1]):
                if(k != j):
                    if(Dist3D(joint2check, jointpos[:,[k]])< (r+r)):
                        c[i] = 1
                    
    return c



