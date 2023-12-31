#
#
#      0===========================0
#      |    MAREVA 3D Modelling    |
#      0===========================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script of the practical session. Plane detection by RANSAC
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 19/09/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    # Initialize the chosen point and the normal vector
    point = np.zeros((3,1))
    normal = np.zeros((3,1))
    
    # Choose one point from the three
    point = points[0]
    
    # Calculate the vectors of the other two points in relation to the chosen point
    vet1 = points[1] - point
    vet2 = points[2] - point
    
    # Calculate the normal of the plane using the cross product of vectors
    normal = np.cross(vet1, vet2)
    
    # Normalize the normal vector
    normal /= np.linalg.norm(normal)
    
    # Reshape point and normal
    point = point.reshape(3, 1)
    normal = normal.reshape(3, 1)
    
    return point, normal


def in_plane(points, ref_pt, normal, threshold_in=0.1):
    
    indices = np.zeros(len(points), dtype=bool)
    
    # Calculate the vector from the plane point to the input point
    vector_to_points = points - ref_pt.T
    
    # Calculate the distance between the points and the plane
    distances = np.abs(np.dot(vector_to_points, normal))
    
    # Boolean mask of points whose distance to the plane are smaller than the threshold
    indices = distances < threshold_in
        
    return indices


def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    
    best_ref_pt = np.zeros((3,1))
    best_normal = np.zeros((3,1))
    
    N = len(points)
    
    # Initialize the maximum number of votes
    most_votes = 0
    
    # Implement RANSAC
    for i in range(NB_RANDOM_DRAWS):
        # Get randomly three points from the cloud
        pts = points[np.random.randint(0, N, size=3)]
        
        # Compute the plane they define
        ref_pt, normal = compute_plane(pts)
        
        # Count how many points from the cloud are in range of this plane as votes
        indices = in_plane(points, ref_pt, normal, threshold_in)
        count_votes = sum(indices)
        
        # Update output
        if count_votes > most_votes:
            best_ref_pt = ref_pt
            best_normal = normal
            most_votes = count_votes
                
    return best_ref_pt, best_normal


def multi_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    
    all_planes_inds = np.zeros((len(points), 1), np.int32)
    plane_inds = np.zeros((0,), np.int32)
    remaining_inds = np.ones((len(points), 1), np.int32)
    plane_labels = np.zeros((0,), np.int32)
    
    for i in range(NB_PLANES):
        # Implement RANSAC
        ref_pt, normal = RANSAC(points[remaining_inds.nonzero()[0]], NB_RANDOM_DRAWS, threshold_in)
        
        # Find points in the plane and others
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        
        # Update the remaining points without all the planes' points
        all_planes_inds = np.logical_or(all_planes_inds, points_in_plane)
        remaining_inds = (1-all_planes_inds)
        
        # Update the plane inds
        plane_inds = np.concatenate((plane_inds,points_in_plane.nonzero()[0]))
        
        # Update the plane labels
        plane_labels = np.concatenate((plane_labels,((1+i)*np.ones((np.sum(points_in_plane))))))
    
    # Get the remaining points index
    remaining_inds = remaining_inds.nonzero()[0]

    return plane_inds, remaining_inds, plane_labels





# ----------------------------------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    N = len(points)

    # Computes the plane passing through 3 randomly chosen points
    # ***********************************************************
    #

    if False:

        # Define parameter
        threshold_in = 0.1

        # Take randomly three points
        pts = points[np.random.randint(0, N, size=3)]

        # Computes the plane passing through the 3 points
        t0 = time.time()
        ref_pt, normal = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]

        # Save the 3 points and their corresponding plane for verification
        pts_clr = np.zeros_like(pts)
        pts_clr[:, 0] = 1.0
        write_ply('../triplet.ply',
                  [pts, pts_clr],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../triplet_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #

    if False:

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 100
        threshold_in = 0.05

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]

        # Save the best extracted plane and remaining points
        print(f"The point cloud contains {N} points.")
        print(f"The prominent plane contains {len(plane_inds)} points.")
        print(f"The prominent plane has {100*(len(plane_inds)/N)}% of the points in the point cloud.")

        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Find multiple planes in the cloud
    # *********************************
    #

    if True:

        # Define parameters of multi_RANSAC
        NB_RANDOM_DRAWS = 200
        threshold_in = 0.05
        NB_PLANES = 5

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
        t1 = time.time()
        print('\nmulti RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels.astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
