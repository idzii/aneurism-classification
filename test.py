import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import visualization as v
from keras.models import load_model
import visualization


def isAneurysm(path):
    model = load_model('./models/model.h5')
    diagram = visualization.makeDiagram(path, max_edge_length=0.5)
    heat_kernel = visualization.makeHeatDiagram(diagram)[0][2]
    heat_kernel = np.expand_dims(heat_kernel, axis=0)
    prediction = model.predict(heat_kernel)
    print(prediction)
    prediction = (prediction > 0.5).astype(int)
    print(prediction)
    if prediction == 1:
        return True
    else:
        return False


if __name__ == "__main__":
    path = "IntrA/generated/aneurysm/obj/ArteryObjAN170-9.obj"
    if isAneurysm(path):
        print("Aneurysm")
    else:
        print("Healthy")

    # path = "IntrA/generated/aneurysm/obj/ArteryObjAN168-29.obj"
    # tm = o3d.io.read_triangle_mesh(path)
    # pcd = tm.sample_points_uniformly(700)
    #
    # print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([pcd])
    #
    # diagram = v.makeDiagram(path, max_edge_length=0.5)
    # heat_kernel = v.makeHeatDiagram(diagram, sigma=1, n_bins=60)
    # plt.imshow(heat_kernel[0][2], cmap="jet")
    # plt.show()
    #
    # path = "IntrA/generated/vessel/obj/ArteryObjAN168-10.obj"
    tm = o3d.io.read_triangle_mesh(path)
    pcd = tm.sample_points_uniformly(700)

    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    diagram = v.makeDiagram(path, max_edge_length=0.5)
    heat_kernel = v.makeHeatDiagram(diagram, sigma=1, n_bins=60)
    plt.imshow(heat_kernel[0][2], cmap="jet")
    plt.show()
