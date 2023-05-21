import pywavefront
import matplotlib.pyplot as plt
import os
import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
import plotly.offline as pyo
import random


def plot3D(path) -> None:
    """
    Plots 3d point cloud (works only for .obj files)
    :param path: path to file where 3d model is stored
    :return: None
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

    # Simple check for anomaly for plot name
    if path.find("aneurysm") != -1:
        name = "with aneurysm"
    else:
        name = "healthy"
    ax.set_title(name)

    # Show the plot
    plt.show()


def makeDiagram(path, homology_dimensions=[0, 1, 2], max_edge_length=4):
    """
    Creates persistence diagram
    :param path: path to .obj file where 3d model is stored
    :param homology_dimensions:
    :param max_edge_length:
    :return: persistence diagram
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


def plotDiagram(diagram):
    """
    Plots persistence diagram
    :param diagram: persistence diagram
    :return: None
    """
    fig = plot_diagram(diagram)
    pyo.plot(fig)


def saveDiagram(diagram, folder_path, file_name):
    """
    :param diagram: persistence diagram
    :param folder_path: folder location to save diagram
    :param file_name: name of a file where diagram is stored
    :return:
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig = plot_diagram(diagram)
    path = os.path.join(folder_path, file_name + ".png")
    fig.write_image(path)


def visualizeRandomExample(aneurysm=True, plot=True, homology_dimensions=[0, 1, 2], max_edge_length=4):
    """
    Visualizes random example with/without aneurysm, saves diagram and plots it
     :param aneurysm: True by default - set False to visualize healthy blood vessel
     :param plot: True by default - set False to omit plotting
     :param max_edge_length: parameter that controls filtration
     :param homology_dimensions: selected homologies
     :return: None
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
