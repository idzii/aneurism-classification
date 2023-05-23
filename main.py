import data as d
import visualization as v
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # d.makeData(1)
    x, y = d.loadData()
    print(x.shape)
    print(y)
    # diagram = v.makeDiagram("IntrA/generated/vessel/obj/ArteryObjAN2-7.obj", max_edge_length=4)
    # heat_diagram = v.makeHeatDiagram(diagram)
    #
    # plt.imshow(heat_diagram[0][0], cmap="jet")
    # plt.show()
    #
    # plt.imshow(heat_diagram[0][1], cmap="jet")
    # plt.show()
    #
    # plt.imshow(heat_diagram[0][2], cmap="jet")
    # plt.show()
