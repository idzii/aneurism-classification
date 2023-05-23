import visualization as v
import os
import numpy as np


def makeData(max_edge_length=1):
    # Create heat kernel tensors (x) and corresponding labels (y) (0 for healthy vessels/ 1 for aneurysm)
    heat_diagrams_aneurysm1 = v.makeHeatDiagrams(os.path.join("IntrA", "annotated", "obj"), max_edge_length=max_edge_length)
    heat_diagrams_aneurysm2 = v.makeHeatDiagrams(os.path.join("IntrA", "generated", "aneurysm", "obj"), max_edge_length=max_edge_length)

    x = np.append(heat_diagrams_aneurysm1, heat_diagrams_aneurysm2)
    y = np.ones(len(heat_diagrams_aneurysm1)+len(heat_diagrams_aneurysm2))

    heat_diagrams_healthy = v.makeHeatDiagrams(os.path.join("IntrA", "generated", "vessel", "obj"), max_edge_length=max_edge_length)

    x = np.append(x, heat_diagrams_healthy)
    x = np.reshape(x, (-1,) + heat_diagrams_aneurysm1[0].shape[1:])
    y = np.append(y, np.zeros(len(heat_diagrams_healthy)))

    if not os.path.exists("data"):
        os.makedirs("data")

    np.save(os.path.join("data", "heat_kernel_tensors.npy"), x)
    np.save(os.path.join("data", "labels.npy"), y)


def loadData():
    x = np.load(os.path.join("data", "heat_kernel_tensors.npy"))
    y = np.load(os.path.join("data", "labels.npy"))
    return x, y
