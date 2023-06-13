import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import visualization as v

if __name__ == "__main__":
    path = "IntrA/generated/aneurysm/obj/ArteryObjAN168-29.obj"
    tm = o3d.io.read_triangle_mesh(path)
    pcd = tm.sample_points_uniformly(700)

    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    diagram = v.makeDiagram(path, max_edge_length=0.5)
    heat_kernel = v.makeHeatDiagram(diagram, sigma=1, n_bins=60)
    plt.imshow(heat_kernel[0][2], cmap="jet")
    plt.show()

    path = "IntrA/generated/vessel/obj/ArteryObjAN168-10.obj"
    tm = o3d.io.read_triangle_mesh(path)
    pcd = tm.sample_points_uniformly(700)

    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    diagram = v.makeDiagram(path, max_edge_length=0.5)
    heat_kernel = v.makeHeatDiagram(diagram, sigma=1, n_bins=60)
    plt.imshow(heat_kernel[0][2], cmap="jet")
    plt.show()
