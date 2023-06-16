import pywavefront
import matplotlib.pyplot as plt
import os
import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import HeatKernel, Scaler, PersistenceImage
from gtda.plotting import plot_diagram
import plotly.offline as pyo
import random
from tqdm import tqdm
import sys
import open3d as o3d


def plot3D(file_path):
    """
    Plots 3d point cloud for .obj file with given path
    """
    point_cloud = getPointCloud(file_path)
    point_cloud = normalizePointCloud(point_cloud)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if file_path.find("aneurysm") != -1:
        name = "with aneurysm"
    else:
        name = "healthy"
    ax.set_title(name)
    plt.show()


def normalizePointCloud(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud) ** 2, axis=-1)))
    point_cloud /= furthest_distance

    return point_cloud


def getPointCloud(file_path):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"{file_path} is not valid file path")
        sys.exit(1)

    tm = o3d.io.read_triangle_mesh(file_path)
    point_cloud = tm.sample_points_uniformly(400)
    point_cloud = np.array(point_cloud.points)
    point_cloud = normalizePointCloud(point_cloud)

    return point_cloud


def getPointClouds(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        print(f"{path} is not valid directory path")
        sys.exit(1)

    point_clouds = [(getPointCloud(os.path.join(path, file))) for file in
                    tqdm(os.listdir(path), desc="EXTRACTING POINT CLOUDS")]
    return point_clouds


def makeDiagram(path, homology_dimensions=[0, 1, 2], max_edge_length=4):
    """
    Creates persistence diagram from .obj file
    """
    # Reshape point cloud so it fits VietorisRipsPersistence api
    point_cloud = getPointCloud(path)

    point_cloud = np.reshape(point_cloud, (1,) + point_cloud.shape)
    vr = VietorisRipsPersistence(homology_dimensions=homology_dimensions, max_edge_length=max_edge_length)
    diagram = vr.fit_transform(point_cloud)

    return diagram


def makeDiagrams(folder_path, homology_dimensions=[0, 1, 2], max_edge_length=4):
    """
    Creates persistence diagrams from .obj files that are located in given folder.
    """
    point_clouds = getPointClouds(folder_path)
    vr = VietorisRipsPersistence(homology_dimensions=homology_dimensions, max_edge_length=max_edge_length)
    diagrams = [vr.fit_transform(point_cloud.reshape((1,) + point_cloud.shape)) for point_cloud in
                tqdm(point_clouds, desc="CREATING DIAGRAMS")]

    return diagrams


def makePersistenceImage(diagram, sigma=1, n_bins=100):
    persistence_image = PersistenceImage(sigma=sigma, n_bins=n_bins)
    persistence_image = persistence_image.fit_transform(diagram)[0]

    return persistence_image


def makePersistenceImages(folder_path, homology_dimensions=[0, 1, 2], max_edge_length=4, sigma=1, n_bins=100):
    diagrams = makeDiagrams(folder_path=folder_path, homology_dimensions=homology_dimensions,
                            max_edge_length=max_edge_length)
    persistenceImages = [makePersistenceImage(diagram, sigma=sigma, n_bins=n_bins) for diagram in diagrams]
    return persistenceImages


def makeHeatDiagram(diagram, sigma=0.15, n_bins=60):
    scaler = Scaler()
    diagram = scaler.fit_transform(diagram)

    hk = HeatKernel(sigma=sigma, n_bins=n_bins)
    heat_diagram = np.array(hk.fit_transform(diagram))

    return heat_diagram


def makeHeatDiagrams(folder_path, homology_dimensions=[0, 1, 2], max_edge_length=4, sigma=0.15, n_bins=60):
    diagrams = makeDiagrams(folder_path=folder_path, homology_dimensions=homology_dimensions,
                            max_edge_length=max_edge_length)
    heat_diagrams = [makeHeatDiagram(diagram, sigma, n_bins) for diagram in diagrams]

    return heat_diagrams


def plotDiagram(diagram):
    """
    Plots given persistence diagram.
    """
    fig = plot_diagram(diagram)
    pyo.plot(fig)


def saveDiagram(diagram, folder_path, file_name):
    """
    Saves given persistence diagram in given folder, with given name.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig = plot_diagram(diagram)
    path = os.path.join(folder_path, file_name + ".png")
    fig.write_image(path)


def saveDiagrams(diagrams, folder_path):
    """
    Saves diagrams to folder with give path, If folder doesn't exist, it creates that folder.
    """
    for i, diagram in enumerate(diagrams):
        saveDiagram(diagram, folder_path, f"diagram{i + 1}")


def visualizeRandomExample(aneurysm=True, plot=True, homology_dimensions=[0, 1, 2], max_edge_length=4):
    """
    Visualizes random example with/without aneurysm, saves diagram in examples folder and plots it.
    Set aneurysm=False to get example of healthy blood vessel, and plot=False to avoid plotting.
    """
    # Select random file
    if aneurysm:
        input_folder = "aneurysm"
        output_folder = "aneurysm"
    else:
        input_folder = "vessel"
        output_folder = "healthy"

    dir_path = os.path.join("IntrA", "generated", input_folder, "obj")
    files = os.listdir(dir_path)
    selected_file = random.choice(files)

    # Visualize point cloud
    path = os.path.join(dir_path, selected_file)
    if plot:
        plot3D(path)

    # Plot and save diagram
    diagram = makeDiagram(path, homology_dimensions=homology_dimensions, max_edge_length=max_edge_length)
    if plot:
        plotDiagram(diagram)

    saveDiagram(diagram, os.path.join("examples", "diagrams", output_folder), selected_file)
