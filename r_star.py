import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.spatial import KDTree

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def steer(from_node, to_node, max_step_size):
    """Move from `from_node` toward `to_node` by a maximum step size."""
    dist = euclidean_distance(from_node[:2], to_node[:2])
    if dist <= max_step_size:
        return to_node
    direction = np.array(to_node[:2]) - np.array(from_node[:2])
    direction = direction / dist  # Normalize
    new_pos = np.array(from_node[:2]) + direction * max_step_size
    return (new_pos[0], new_pos[1])


class RStarPlanner:
    def __init__(self, start, goal, x_range, y_range, max_step_size=1.0, search_radius=2.0, max_iterations=1000):
        self.start = start  # (x, y)
        self.goal = goal  # (x, y)
        # self.obstacles = obstacles  # [(ox, oy, radius), ...]
        self.x_range = x_range  # (min_x, max_x)
        self.y_range = y_range  # (min_y, max_y)
        self.max_step_size = max_step_size
        self.search_radius = search_radius
        self.max_iterations = max_iterations

        self.nodes = [start]
        self.parents = {}
        self.kd_tree = KDTree([start[:2]])

    def sample_point(self):
        """Randomly sample a point in the space."""
        x = random.uniform(self.x_range[0], self.x_range[1])
        y = random.uniform(self.y_range[0], self.y_range[1])
        return (x, y)

    def find_nearest(self, point):
        """Find the nearest node in the tree."""
        _, idx = self.kd_tree.query(point)
        return self.nodes[idx]

    def find_nearby(self, node, radius):
        """Find nearby nodes within the given radius."""
        indices = self.kd_tree.query_ball_point(node[:2], radius)
        return [self.nodes[i] for i in indices]

    def rewire(self, new_node):
        """Rewire the tree to ensure optimal connections."""
        nearby_nodes = self.find_nearby(new_node, self.search_radius)
        for node in nearby_nodes:
            # if is_collision_free(node, new_node, self.obstacles):
            new_cost = self.cost(new_node) + euclidean_distance(new_node[:2], node[:2])
            if new_cost < self.cost(node):
                self.parents[node] = new_node

    def cost(self, node):
        """Calculate the cost to reach the node."""
        cost = 0
        while node in self.parents:
            parent = self.parents[node]
            cost += euclidean_distance(node[:2], parent[:2])
            node = parent
        return cost

    def build_path(self):
        """Build the path from start to goal."""
        path = []
        node = self.goal
        while node in self.parents:
            path.append(node)
            node = self.parents[node]
        path.append(self.start)
        path.reverse()
        return path

    def plan(self):
        """Plan the path using the R* algorithm."""
        for _ in range(self.max_iterations):
            random_point = self.sample_point()
            nearest_node = self.find_nearest(random_point)
            new_node = steer(nearest_node, random_point, self.max_step_size)

            # if is_collision_free(nearest_node, new_node, self.obstacles):
            self.nodes.append(new_node)
            self.parents[new_node] = nearest_node
            self.kd_tree = KDTree([node[:2] for node in self.nodes])  # Rebuild tree
            self.rewire(new_node)

            # Check if the goal is reachable
            if euclidean_distance(new_node[:2], self.goal[:2]) <= self.max_step_size:
                # if is_collision_free(new_node, self.goal, self.obstacles):
                self.parents[self.goal] = new_node
                return self.build_path()

        return None  # No path found


def compute_theta(path):
    """Compute orientation theta for each point on the path."""
    thetas = []
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        theta = math.atan2(dy, dx)
        thetas.append(theta)
    thetas.append(thetas[-1])  # Keep the last theta constant
    return thetas


if __name__=='__main__':
    # Define the environment
    start = (3, 9)
    goal = (0, 0)
    # obstacles = [(3, 3, 1.0), (7, 7, 1.5), (5, 5, 1.0)]
    x_range = (0, 12)
    y_range = (0, 12)

    # Plan using R* algorithm
    planner = RStarPlanner(start, goal, x_range, y_range)
    path = planner.plan()

    # Compute orientation
    if path:
        thetas = compute_theta(path)

        # Plot the result
        plt.figure(figsize=(8, 8))
        # for ox, oy, radius in obstacles:
        #     circle = plt.Circle((ox, oy), radius, color='r', alpha=0.5)
        #     plt.gca().add_patch(circle)
        plt.plot([p[0] for p in path], [p[1] for p in path], 'b-', label="Path")
        plt.scatter([p[0] for p in path], [p[1] for p in path], c='b')
        plt.quiver(
            [p[0] for p in path], [p[1] for p in path],
            np.cos(thetas), np.sin(thetas), angles='xy', scale_units='xy', scale=1, color='g', label='Theta'
        )
        plt.scatter([start[0], goal[0]], [start[1], goal[1]], c='g', label='Start/Goal')
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.legend()
        plt.grid()
        plt.title("R* Path with Orientation")
        plt.show()
    else:
        print("No path found.")
