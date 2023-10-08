# TP MAREVA Nuages de Points et Mod�lisation 3D - Python - FG 24/09/2020
# coding=utf8

# Import Numpy
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D

# Import functions from scikit-learn : KDTree
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply
# utils.ply est le chemin relatif utils/ply.py

import time


def read_data_ply(path):
# Lecture de nuage de points sous format ply
    '''
    Lecture de nuage de points sous format ply
    Inputs :
        path = chemin d'acc�s au fichier
    Output :
        data = matrice (3 x n)
    '''
    data_ply = read_ply(path)
    data = np.vstack((data_ply['x'], data_ply['y'], data_ply['z']))
    return(data)

def write_data_ply(data,path):
    '''
    Ecriture de nuage de points sous format ply
    Inputs :
        data = matrice (3 x n)
        path = chemin d'acc�s au fichier
    '''
    write_ply(path, data.T, ['x', 'y', 'z'])
    
def show3D(data):
    '''
    Visualisation de nuages de points avec MatplotLib'
    Input :
        data = matrice (3 x n)
    '''
    #plt.cla()
    # Aide en ligne : help(plt)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[0], data[1], data[2], '.')
    #ax.plot(data_aligned[0], data_aligned[1], data_aligned[2], '.')
    #plt.axis('equal')
    plt.show()


def decimate(data,k_ech):
    '''
    Decimation
    # ----------
    Inputs :
        data = matrice (3 x n)
        k_ech : facteur de decimation
    Output :
        decimated = matrice (3 x (n/k_ech))
    '''

    if False:
        # 1ere methode : boucle for
        n = data.shape[1]
        n_ech=int(n/k_ech)
        
        decimated = np.vstack(data[:, 0])
        # Selectionnez n_ech colonnes à un pas de k_ech
        for i in range(1, n_ech):
            Xi = np.vstack(data[:, k_ech*i])
            decimated = np.hstack((decimated, Xi))  

    else:
        # 2e methode : fonction de Numpy array
        n = data.shape[1]
        n_ech=int(n/k_ech)
        
        # Selectionner des colonnes avec un facteur d'echelle k_ech
        col_index = np.arange(0, n, k_ech)[:n_ech]

        # Sous-echantillonager le nuage de points
        decimated = data[:, col_index]
    
    return(decimated)




def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
        ref = (d x N) matrix where "N" is the number of point and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # Barycenters
    # definir les barycentres ref_center et data_center
    ref_center = np.mean(ref, axis=1)
    ref_center = ref_center.reshape(3, 1)
    
    data_center = np.mean(data, axis=1)
    data_center = data_center.reshape(3, 1)
    
    # Centered clouds
    # calculer les nuages de points centres ref_c et data_c
    ref_c = ref - ref_center
    data_c = data - data_center
    
    # H matrix
    # calculer la matrice H
    H = np.dot(data_c, ref_c.T)

    # SVD on H
    # calculer U, S, et Vt en utilisant np.linalg.svd
    U, S, Vt = np.linalg.svd(H)

    # Checking R determinant
    # si le determinant de U est -1, prendre son oppose
    if np.linalg.det(U) == -1:
        U = -U

    # Getting R and T
    # calculer R et T
    R = np.dot(Vt.T, U.T)
    T = ref_center - np.dot(R, data_center)

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iteratice closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
        ref = (d x N) matrix where "N" is the number of point and "d" the dimension
        max_iter = stop condition on the number of iteration
        RMS_threshold = stop condition on the distance
    Returns :
        R = (d x d) rotation matrix aligning data on ref
        T = (d x 1) translation vector aligning data on ref
        data_aligned = data aligned on ref
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Create a neighbor structure on ref
    search_tree = KDTree(ref.T)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    for i in range(max_iter):

        # Find the nearest neighbors
        distances, indices = search_tree.query(data_aligned.T, return_distance=True)

        # Compute average distance
        RMS = np.sqrt(np.mean(np.power(distances, 2)))

        # Distance criteria
        if RMS < RMS_threshold:
            break

        # Find best transform
        R, T = best_rigid_transform(data, ref[:, indices.ravel()])

        # Update lists
        R_list.append(R)
        T_list.append(T)
        neighbors_list.append(indices.ravel())
        RMS_list.append(RMS)

        # Aligned data
        data_aligned = R.dot(data) + T


    return data_aligned, R_list, T_list, neighbors_list, RMS_list





#
#           Main
#       \**********/
#

if __name__ == '__main__':


    # Fichiers de nuages de points
    bunny_o_path = 'data/bunny_original.ply'
    bunny_p_path = 'data/bunny_perturbed.ply'
    bunny_r_path = 'data/bunny_returned.ply'
    NDC_o_path = 'data/Notre_Dame_Des_Champs_1.ply'
    NDC_o2_path = 'data/Notre_Dame_Des_Champs_2.ply'
    NDC_r_path = 'data/Notre_Dame_Des_Champs_returned.ply'

    # Lecture des fichiers
    bunny_o=read_data_ply(bunny_o_path)                    
    bunny_p=read_data_ply(bunny_p_path)
    NDC_o=read_data_ply(NDC_o_path)
    NDC_o2=read_data_ply(NDC_o2_path)

    # Visualisation du fichier d'origine
    if True:
        show3D(bunny_o)

    # Transformations : d�cimation, rotation, translation, �chelle
    # ------------------------------------------------------------
    if True:
        # Decimation
        k_ech=10
        
        start_time = time.time()
        decimated = decimate(bunny_o,k_ech)
        end_time = time.time()
        print("Execution time =", end_time-start_time, "seconds")
        
        # Visualisation sous Python et par ecriture de fichier
        show3D(decimated)
        
        # Visualisation sous CloudCompare apres ecriture de fichier
        write_data_ply(decimated,bunny_r_path)
        # Puis ouvrir le fichier sous CloudCompare pour le visualiser

    if True:
        show3D(NDC_o)
        decimated = decimate(NDC_o,1000)
        show3D(decimated)
        write_data_ply(decimated,NDC_r_path)

    if True:
        # Translation
        translation = np.array([[0, -0.1, 0.1]]).T
        points = bunny_o + translation
        show3D(points)
        
        # Find the centroid of the cloud and center it
        centroid = np.mean(points, axis=1)
        centroid = centroid.reshape(3, 1)
        points = points - centroid
        show3D(points)
        
        # Echelle
        # points = points divises par 2
        points /= 2
        show3D(points)
        
        # Define the rotation matrix (rotation of angle around z-axis)
        theta = np.pi / 3
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        
        # Apply the rotation
        points=bunny_o
        # centrer le nuage de points
        centroid = np.mean(points, axis=1)
        centroid = centroid.reshape(3, 1)
        points = points - centroid
        # appliquer la rotation
        points = np.dot(R, points)
        # appliquer la translation opposee
        points -= translation
        
        show3D(points)


    # Meilleure transformation rigide (R,Tr) entre nuages de points
    # -------------------------------------------------------------
    if True:

        show3D(bunny_p)
        
        # Find the best transformation
        R, Tr = best_rigid_transform(bunny_p, bunny_o)
        
        # Apply the tranformation
        opt = R.dot(bunny_p) + Tr
        bunny_r_opt = opt
        
        # Show and save cloud
        show3D(bunny_r_opt)
        write_data_ply(bunny_r_opt,bunny_r_path)
        
        # Get average distances
        distances2_before = np.sum(np.power(bunny_p - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))
        
        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
    
   
    # Test ICP and visualize
    # **********************
    if True:
        # Nuage bunny
        start_time = time.time()
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        end_time = time.time()
        print("Execution time =", end_time-start_time, "seconds")
        
        # Nuages Notre Dame des Champs 1 (reference) et 2 (modifie)
        start_time = time.time()
        NDC_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(NDC_o2, NDC_o, 25, 1e-4)
        end_time = time.time()
        print("Execution time =", end_time-start_time, "seconds")
        
        # Nuages NDC echantillonnes
        decimated_1 = decimate(NDC_o,1000)
        decimated_2 = decimate(NDC_o2,1000)
        start_time = time.time()
        NDC_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(decimated_2, decimated_1, 25, 1e-4)
        end_time = time.time()
        print("Execution time =", end_time-start_time, "seconds")
        
        # Tracer l'evolution de f
        plt.xlabel("Number of iterations")
        plt.ylabel("f (R,t)")
        plt.plot(RMS_list)
        plt.show()
        
