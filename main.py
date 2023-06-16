import numpy

import data
import visualization
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data.makeHeatData(0.7)
    X, y = data.loadData()
    # heatDiagrams = visualization.makeHeatDiagrams("small_data/aneurysm", max_edge_length=0.5, sigma=0.15, n_bins=100)
    # for diagram in heatDiagrams:
    #     plt.imshow(diagram[0][2], cmap='jet')
    #     plt.show()
    #
    # heatDiagrams = visualization.makeHeatDiagrams("small_data/healthy", max_edge_length=0.5, sigma=0.15, n_bins=100)
    # for diagram in heatDiagrams:
    #     plt.imshow(diagram[0][2], cmap='jet')
    #     plt.show()

    # path = "IntrA/annotated/obj/AN128_full.obj"
    # diagram = visualization.makeDiagram(path, max_edge_length=0.5)
    # heat = visualization.makeHeatDiagram(diagram, sigma=0.15, n_bins=60)
    # visualization.plot3D(path)
    # plt.imshow(heat[0][2], cmap="jet")
    # plt.show()
    #
    # path = "IntrA/generated/aneurysm/obj/ArteryObjAN167-1.obj"
    # diagram = visualization.makeDiagram(path, max_edge_length=0.5)
    # heat = visualization.makeHeatDiagram(diagram, sigma=0.15, n_bins=60)
    # visualization.plot3D(path)
    # plt.imshow(heat[0][2], cmap="jet")
    # plt.show()
    #
    # path = "IntrA/generated/vessel/obj/ArteryObjAN3-15.obj"
    # diagram = visualization.makeDiagram(path, max_edge_length=0.5)
    # heat = visualization.makeHeatDiagram(diagram, sigma=0.15, n_bins=60)
    # visualization.plot3D(path)
    # plt.imshow(heat[0][2], cmap="jet")
    # plt.show()
    # path = "IntrA/generated/aneurysm/obj/ArteryObjAN168-3.obj"
    # diagram = visualization.makeDiagram(path, max_edge_length=0.5)
    # image = visualization.makeHeatDiagram(diagram, n_bins=100, sigma=1)
    #
    # visualization.plot3D(path)
    #
    # plt.imshow(image[0][1], cmap="jet")
    # plt.show()
    #
    # plt.imshow(image[0][2], cmap="jet")
    # plt.show()

