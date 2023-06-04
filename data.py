import visualization as v
import os
import numpy as np

def makeHeatData(max_edge_length=1):
    # Create heat kernel tensors (x) and corresponding labels (y) (0 for healthy vessels/ 1 for aneurysm)
    heat_diagrams_aneurysm = v.makeHeatDiagrams(os.path.join("IntrA", "generated", "aneurysm", "obj"),
                                                max_edge_length=max_edge_length)

    x = np.array(heat_diagrams_aneurysm)
    y = np.ones(len(heat_diagrams_aneurysm))

    heat_diagrams_healthy = v.makeHeatDiagrams(os.path.join("IntrA", "generated", "vessel", "obj"),
                                               max_edge_length=max_edge_length)

    x = np.append(x, heat_diagrams_healthy)
    x = np.reshape(x, (-1,) + heat_diagrams_aneurysm[0].shape[1:])
    y = np.append(y, np.zeros(len(heat_diagrams_healthy)))

    if not os.path.exists("data"):
        os.makedirs("data")

    np.save(os.path.join("data", "heat", "heat_kernel_tensors.npy"), x)
    np.save(os.path.join("data", "heat", "labels.npy"), y)


def makeImageData(max_edge_length=1):
    # Create heat kernel tensors (x) and corresponding labels (y) (0 for healthy vessels/ 1 for aneurysm)
    persistence_images_aneurysm = v.makePersistenceImages(os.path.join("IntrA", "generated", "aneurysm", "obj"),
                                                max_edge_length=max_edge_length)

    x = np.array(persistence_images_aneurysm)
    y = np.ones(len(persistence_images_aneurysm))

    persistence_images_healthy = v.makePersistenceImages(os.path.join("IntrA", "generated", "vessel", "obj"),
                                               max_edge_length=max_edge_length)

    x = np.append(x, persistence_images_healthy)
    x = np.reshape(x, (-1,) + persistence_images_aneurysm[0].shape[1:])
    y = np.append(y, np.zeros(len(persistence_images_healthy)))

    if not os.path.exists("data"):
        os.makedirs("data")

    np.save(os.path.join("data", "images", "persistence_images.npy"), x)
    np.save(os.path.join("data", "images", "labels.npy"), y)


def resample(X, y):
    aneurysm_indices = np.where(y == 1)[0]
    healthy_indices = np.where(y == 0)[0]
    majority_count = len(healthy_indices)
    minority_count = len(aneurysm_indices)

    factor = majority_count // minority_count
    new_X = X[healthy_indices]
    new_y = y[healthy_indices]

    for i in range(factor):
        new_X = np.concatenate((new_X, X[aneurysm_indices]), axis=0)
        new_y = np.concatenate((new_y, y[aneurysm_indices]), axis=0)

    indices = np.arange(len(new_y))
    np.random.shuffle(indices)
    new_X.reshape((-1, 60, 60))

    return new_X[indices], new_y[indices]


def loadData():
    x = np.load(os.path.join("data", "heat_kernel_tensors.npy"))
    y = np.load(os.path.join("data", "labels.npy"))
    return x, y
