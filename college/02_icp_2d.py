import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors
def euclidean_distance(point1, point2):
    """
    Euclidean distance between two points.
    :param point1: the first point as a tuple (a_1, a_2, ..., a_n)
    :param point2: the second point as a tuple (b_1, b_2, ..., b_n)
    :return: the Euclidean distance
    """
    a = np.array(point1)
    b = np.array(point2)

    return np.linalg.norm(a - b, ord=2)


def point_based_matching(point_pairs):
    """
    This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
    by F. Lu and E. Milios.
    :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
    :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
    """

    x_mean = 0
    y_mean = 0
    xp_mean = 0
    yp_mean = 0
    n = len(point_pairs)

    if n == 0:
        return None, None, None

    for pair in point_pairs:

        (x, y), (xp, yp) = pair

        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = 0
    s_y_yp = 0
    s_x_yp = 0
    s_y_xp = 0
    for pair in point_pairs:

        (x, y), (xp, yp) = pair

        s_x_xp += (x - x_mean)*(xp - xp_mean)
        s_y_yp += (y - y_mean)*(yp - yp_mean)
        s_x_yp += (x - x_mean)*(yp - yp_mean)
        s_y_xp += (y - y_mean)*(xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
    translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

    return rot_angle, translation_x, translation_y


def icp(reference_points, points, max_iterations=100, distance_threshold=0.3, convergence_translation_threshold=1e-3,
        convergence_rotation_threshold=1e-4, point_pairs_threshold=10, verbose=False):
    """
    An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
    of N 2D (reference) points.
    :param reference_points: the reference point set as a numpy array (N x 2)
    :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2)
    :param max_iterations: the maximum number of iteration to be executed
    :param distance_threshold: the distance threshold between two points in order to be considered as a pair
    :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                              transformation to be considered converged
    :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                               to be considered converged
    :param point_pairs_threshold: the minimum number of point pairs the should exist
    :param verbose: whether to print informative messages about the process (default: False)
    :return: the transformation history as a list of numpy arrays containing the rotation (R) and translation (T)
             transformation in each iteration in the format [R | T] and the aligned points as a numpy array M x 2
    """

    transformation_history = []

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)

    for iter_num in range(max_iterations):
        if verbose:
            print('------ iteration', iter_num, '------')

        closest_point_pairs = []  # list of point correspondences for closest point rule

        distances, indices = nbrs.kneighbors(points)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

        # if only few point pairs, stop process
        if verbose:
            print('number of pairs found:', len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # compute translation and rotation using point correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
        if closest_rot_angle is not None:
            if verbose:
                print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                print('Translation:', closest_translation_x, closest_translation_y)
        if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
            if verbose:
                print('No better solution can be found!')
            break

        # transform 'points' (using the calculated rotation and translation)
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s],
                        [s, c]])
        aligned_points = np.dot(points, rot.T)
        aligned_points[:, 0] += closest_translation_x
        aligned_points[:, 1] += closest_translation_y

        # update 'points' for the next iteration
        points = aligned_points

        # update transformation history
        transformation_history.append(np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))

        # check convergence
        if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                and (abs(closest_translation_x) < convergence_translation_threshold) \
                and (abs(closest_translation_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    return transformation_history, points
if __name__ == '__main__':
    ## Read pickle file and get data
    data=pickle.load(open('./data/college/2dicp.pickle','rb'))
    fig = plt.figure()
    ## Using matplotlib to plot the data
    T_all=np.eye(3)
    show_animation = True
    import time
    a=time.time()
    for i in range(len(data)-1):
        c=time.time()
        reference_points=data[i][:,0:2]
        points_to_be_aligned=data[i+1][:,0:2]
        test_points=np.hstack([points_to_be_aligned,np.ones((len(points_to_be_aligned),1))])
        
        transformation_history, aligned_points = icp(reference_points, \
            points_to_be_aligned, distance_threshold=30,verbose=False)
        T_sum=np.zeros((3,3))
        T_sum[0:2,0:3]=transformation_history[0][0:2,0:3]
        T_sum[2,2]=1
        for i in range(1,len(transformation_history)):
            T=np.zeros((3,3))
            T[0:2,0:3]=transformation_history[i][0:2,0:3]
            T[2,2]=1
            T_sum=np.dot(T_sum,T)
        T_all=np.dot(T_all,T_sum)
        aligned_points=np.dot(test_points,T_sum)
        d=time.time()
        print(f'cost time:{d-c} s')
        plt.scatter(T_all[0,2],T_all[1,2],c="r", marker=".") 
    b=time.time()
    print(f'cost time:{b-a} s')
    print(T_all)
    # plt.plot(reference_points[:, 0], reference_points[:, 1], 'rx', label='reference points')
    # plt.plot(points_to_be_aligned[:, 0], points_to_be_aligned[:, 1], 'b1', label='points to be aligned')
    # plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
    # plt.legend()
    plt.show()
    
# if __name__ == '__main__':
#     # set seed for reproducible results
#     np.random.seed(12345)

#     # create a set of points to be the reference for ICP
#     xs = np.random.random_sample((50, 1))
#     ys = np.random.random_sample((50, 1))
#     reference_points = np.hstack((xs, ys))

#     # transform the set of reference points to create a new set of
#     # points for testing the ICP implementation

#     # 1. remove some points
#     points_to_be_aligned = reference_points[1:47]
#     np.random.shuffle(points_to_be_aligned)
#     # 2. apply rotation to the new point set
#     theta = math.radians(12)
#     c, s = math.cos(theta), math.sin(theta)
#     rot = np.array([[c, -s],
#                     [s, c]])
#     points_to_be_aligned = np.dot(points_to_be_aligned, rot)

#     # 3. apply translation to the new point set
#     points_to_be_aligned += np.array([np.random.random_sample(), np.random.random_sample()])

#     # run icp
#     transformation_history, aligned_points = icp(reference_points, points_to_be_aligned, verbose=True)

#     # show results
#     plt.plot(reference_points[:, 0], reference_points[:, 1], 'rx', label='reference points')
#     plt.plot(points_to_be_aligned[:, 0], points_to_be_aligned[:, 1], 'b1', label='points to be aligned')
#     plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
#     plt.legend()
#     plt.show()