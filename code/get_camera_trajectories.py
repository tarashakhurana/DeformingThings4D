# python code for sampling points on a unit sphere randomly

import math
import random
import numpy as np
from math import comb

def sample_neighboring_point_on_sphere(point, delta_alpha):
    R, theta, phi = point

    # Generate random angle
    alpha = random.uniform(0, delta_alpha)

    # Calculate new coordinates
    theta_prime = theta + alpha # * math.cos(phi)
    phi_prime = phi + alpha * math.sin(phi)

    # Convert to Cartesian coordinates
    x = R * math.sin(theta_prime) * math.cos(phi_prime)
    y = R * math.sin(theta_prime) * math.sin(phi_prime)
    z = R * math.cos(theta_prime)

    return x, y, z


def random_polynomial_trajectory(p1, p2, n, num_points):
    # Calculate the vector difference between the points
    v = p2 - p1

    # Generate a random polynomial of degree n
    coeffs = np.random.rand(n+1)
    poly = np.polynomial.Polynomial(coeffs)

    # Evaluate the polynomial at n+1 evenly spaced values of t
    t = np.linspace(0, 1, num_points+1)
    traj_x = poly(t)
    traj_y = np.sqrt(1 - traj_x**2) * np.sign(v[1])
    traj_z = np.sqrt(1 - traj_x**2 - traj_y**2) * np.sign(v[2])
    traj = np.array([traj_x, traj_y, traj_z]).T

    print(traj)
    return traj


def random_bezier_trajectory(p1, p2, n, num_points):
    # Generate random control points
    control_points = p1 + np.random.rand(n-1, 3) * (p2 - p1) * 3
    control_points = np.insert(control_points, 0, p1, axis=0)
    control_points = np.append(control_points, [p2], axis=0)

    # Evaluate the Bezier curve at n+1 evenly spaced values of t
    t = np.linspace(0, 1, num_points+1)
    traj = np.array([bezier_curve(t_i, control_points) for t_i in t])

    return traj


def bezier_curve(t, control_points):
    n = len(control_points) - 1
    return sum([bernstein_polynomial(i, n, t) * control_points[i] for i in range(n+1)])


def bernstein_polynomial(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


if __name__ == '__main__':

    # number of points to sample
    num_points = 1000
    # sample points on a unit sphere
    points = []
    for i in range(num_points):
        z = 2 * random.random() - 1
        t = 2 * math.pi * random.random()
        r = math.sqrt(1 - z*z)
        x = r * math.cos(t)
        y = r * math.sin(t)
        points.append([x, y, z])
    # save the points to a file
    points = np.array(points)

    # sample 50 different trajectories and color the trajectory with a different color every time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.scatter(points[:,0], points[:,1], points[:,2], color='b', alpha=0.2, s=10)

    trajectories = []

    for i in range(1000):
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # sample a single point again
        z = 2 * random.random() - 1
        t = 2 * math.pi * random.random()
        r = math.sqrt(1 - z*z)
        x = r * math.cos(t)
        y = r * math.sin(t)

        # convert thjs point to spherical coordinates
        r = math.sqrt(x*x + y*y + z*z)
        theta = math.acos(z / r)
        phi = math.atan2(y, x)

        # sample a neighboring point
        delta_alpha = 60 * math.pi / 180
        point = (r, theta, phi)
        xn, yn, zn = sample_neighboring_point_on_sphere(point, delta_alpha)

        # choose a color for this trajectory
        color = np.random.rand(3)

        # show the original point in this color but bigger
        ax.scatter(x, y, z, color=color, s=200)

        # show the neighboring point in this color but bigger
        ax.scatter(xn, yn, zn, color=color, s=200)

        # create a camera trajectory from the start point to the end point
        start_point = np.array((x, y, z))
        end_point = np.array((xn, yn, zn))
        num_points = 50

        # randomly sample the degree of polynomial to fit the trajectory
        degree = random.randint(2, 4)
        trajectory = random_bezier_trajectory(start_point, end_point, degree, num_points)
        trajectories.append(trajectory)

        # plot the camera trajectory
        ax.scatter(trajectory[:,0], trajectory[:,1], trajectory[:,2], color=color, s=10)
        ax.set_box_aspect([1,1,1])

    # save all trajectories to a npy file
    trajectories = np.array(trajectories)
    print(trajectories.shape, "Total number of trajectories")
    np.save('camera_trajectories.npy', trajectories)

    ax.set_box_aspect([1,1,1])
    plt.show()
