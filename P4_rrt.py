import numpy as np
import matplotlib.pyplot as plt
from utils import plot_line_segments, plot_robot_outline

class RRT(object):
    """ Represents a motion planning problem to be solved using the RRT algorithm"""
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacleCloud, b_x, b_y, backG = None):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5, -np.pi/2])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5, np.pi/2])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacle_cloud = obstacleCloud             # obstacle set (point cloud)
        self.bx = b_x                                   # robot width
        self.by = b_y                                   # robot height
        self.background = backG                         # image illustrating the background
        self.path = None        # the final path as a list of states

    def is_free_state(self, obstacle_cloud, x, bx, by):
        """
        Returns whether a rectangular robot of size bx x by
        would collide with any point in the point cloud describing
        the set of obstacles in the environment

        Inputs:
            obstacle_cloud: 2D np.array of points describing occupied space
            x: state of robot
            bx: robot width
            by: robot height
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_state must be overriden by a subclass of RRT")

    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V to x
        """
        raise NotImplementedError("find_nearest must be overriden by a subclass of RRT")

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRT")

    def solve(self, eps, max_iters=1000, goal_bias=0.05):
        """
        Constructs an RRT rooted at self.x_init with the aim of producing a
        dynamically-feasible and obstacle-free trajectory from self.x_init
        to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
            goal_bias: probability during each iteration of setting
                x_rand = self.x_goal (instead of uniformly randly sampling
                from the state space)
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """

        state_dim = len(self.x_init)

        # V stores the states that have been added to the RRT (pre-allocated at its maximum size
        # since numpy doesn't play that well with appending/extending)
        V = np.zeros((max_iters + 1, state_dim))
        V[0,:] = self.x_init    # RRT is rooted at self.x_init
        n = 1                   # the current size of the RRT (states accessible as V[range(n),:])

        # P stores the parent of each state in the RRT. P[0] = -1 since the root has no parent,
        # P[1] = 0 since the parent of the first additional state added to the RRT must have been
        # extended from the root, in general 0 <= P[i] < i for all i < n
        P = -np.ones(max_iters + 1, dtype=int)


        success = False

        ## Intermediate Outputs
        #    - V, P, n: the represention of the planning tree
        #    - success: whether or not you've found a solution within max_iters RRT iterations
        #    - self.path: if success is True, then must contain list of states (tree nodes)
        #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
        #          trajectories connecting the states in order is obstacle-free.
        for i in range(max_iters):
            # Sample a random state
            if np.random.random() < goal_bias:
                # With goal_bias probability, sample the goal
                x_rand = self.x_goal.copy()
            else:
                # Otherwise, sample uniformly from state space
                x_rand = np.random.uniform(self.statespace_lo, self.statespace_hi)
            
            # Find nearest node in current tree
            nearest_idx = self.find_nearest(V[:n], x_rand)
            x_nearest = V[nearest_idx]
            
            # Steer towards x_rand from x_nearest
            x_new = self.steer_towards(x_nearest, x_rand, eps)
            
            # Check if motion from x_nearest to x_new is collision-free
            if self.is_free_state(self.obstacle_cloud, x_new, self.bx, self.by):
                # Add x_new to the tree
                V[n] = x_new
                P[n] = nearest_idx
                n += 1
                
                # Check if we've reached the goal
                if np.linalg.norm(x_new - self.x_goal) < eps:
                    # We've found a path to the goal
                    success = True
                    
                    # Reconstruct the path
                    path = []
                    current_idx = n - 1  # Index of the node closest to goal
                    while current_idx != -1:
                        path.append(V[current_idx].copy())
                        current_idx = P[current_idx]
                    
                    # Reverse to get path from start to goal
                    self.path = list(reversed(path))
                    break

        plt.figure()
        self.plot_problem()
        self.plot_tree(V, P, color="blue", linewidth=.5, label="RRT tree", alpha=0.5)
        if success:
            self.plot_path(color="green", linewidth=2, label="Solution path")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
            plt.scatter(V[:n,0], V[:n,1])
        else:
            print("Solution not found!")

        return success

    def plot_problem(self):
        if self.background is not None:
            plt.imshow(self.background, cmap='gray_r')
        plt.plot(self.obstacle_cloud[:,0],self.obstacle_cloud[:,1], "r.", markersize=1, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        plt.axis('scaled')

class MidtermRRT(RRT):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric) and collisions are detected with a point cloud
    """

    def find_nearest(self, V, x):
        # Consult function specification in parent (RRT) class.
        distances = np.linalg.norm(V - x, axis=1)
        return np.argmin(distances)

    def steer_towards(self, x1, x2, eps):
        # Consult function specification in parent (RRT) class.
        direction = x2 - x1
        distance = np.linalg.norm(direction)
        if distance <= eps:
            return x2
        else:
            return x1 + eps * (direction / distance)

    def is_free_state(self, obstacle_cloud, x, bx, by):
        # Returns True if the robot in pose "x" is not in collision with the obstacle point cloud, otherwise returns False

        # State information
        tau = np.reshape(x[:2],(2,1)) #displacement
        theta = x[2] #angle

        ########## Code starts here ##########
        # Hint: If one of the points in the point cloud is contained in the drone body, return False immediately; you don't need to check the rest.
        
        # Rotation matrix
        R = None

        # Matrix A defining the planes of the rectangle boundaries
        A = None

        # b defining distance to the planes
        b = None
        
        ########## Code ends here ##########

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_path(self,show_robot_outlines = True,**kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)

        if show_robot_outlines:

            outline_kwargs = {
                'color': kwargs.get('color','blue'),
                'linewidth': 1,
                'alpha': 0.2
            }

            stride = max(1,len(path) // 20) # show 20 outlines along the path

            for i in range(0,np.shape(path)[0], stride):
                plot_robot_outline(path[i],self.bx,self.by,**outline_kwargs)

            # Final pose is always plotted
            plot_robot_outline(self.x_goal, self.bx,self.by, **outline_kwargs)
