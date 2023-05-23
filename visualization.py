import pywavefront
import matplotlib.pyplot as plt
import os
import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import HeatKernel, Scaler
from gtda.plotting import plot_diagram
import plotly.offline as pyo
import random
from tqdm import tqdm


def plot3D(path):
    """
    Plots 3d point cloud for .obj file with given path
    """

    # Load the .obj file
    scene = pywavefront.Wavefront(path)

    # Extract point cloud data
    point_cloud = scene.vertices

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z coordinates from the points
    x = [pt[0] for pt in point_cloud]
    y = [pt[1] for pt in point_cloud]
    z = [pt[2] for pt in point_cloud]

    # Create a 3D scatter plot
    ax.scatter(x, y, z)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if path.find("aneurysm") != -1:
        name = "with aneurysm"
    else:
        name = "healthy"
    ax.set_title(name)

    plt.show()


def makeDiagram(path, homology_dimensions=[0, 1, 2], max_edge_length=4):
    """
    Creates persistence diagram from .obj file with given path
    """
    # Load the .obj file
    scene = pywavefront.Wavefront(path)

    # Extract point cloud data
    point_cloud = scene.vertices

    # Create diagram from point cloud
    point_cloud = np.array(list(map(list, point_cloud)))
    point_cloud = point_cloud.reshape(1, point_cloud.shape[0], point_cloud.shape[1])
    vr = VietorisRipsPersistence(homology_dimensions=homology_dimensions, n_jobs=-1, max_edge_length=max_edge_length)
    diagram = vr.fit_transform(point_cloud)[0]

    return diagram


def makeDiagrams(folder_path, homology_dimensions=[0, 1, 2], max_edge_length=4):
    """
    Creates persistence diagrams from .obj files that are located in given folder.
    """
    if not os.path.exists(folder_path):
        print(f"Folder with path: {folder_path} doesn't exist")

    diagrams = []
    for file_name in tqdm(os.listdir(folder_path)):
        path = os.path.join(folder_path, file_name)
        diagram = makeDiagram(path=path, homology_dimensions=homology_dimensions, max_edge_length=max_edge_length)
        diagrams.append(diagram)

    return diagrams


def makeHeatDiagram(diagram, sigma=0.15, n_bins=60):
    diagram = np.reshape(diagram, (1,) + diagram.shape)

    scaler = Scaler()
    diagram = scaler.fit_transform(diagram)

    hk = HeatKernel(sigma=sigma, n_bins=n_bins)
    heat_diagram = np.array(hk.fit_transform(diagram))

    return heat_diagram


def makeHeatDiagrams(folder_path, homology_dimensions=[0, 1, 2], max_edge_length=4, sigma=0.15, n_bins=60):
    diagrams = makeDiagrams(folder_path=folder_path, homology_dimensions=homology_dimensions, max_edge_length=max_edge_length)
    heat_diagrams = []
    for diagram in diagrams:
        heat_diagram = makeHeatDiagram(diagram, sigma=sigma, n_bins=n_bins)
        heat_diagrams.append(heat_diagram)

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
        saveDiagram(diagram, folder_path, f"diagram{i+1}")


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
