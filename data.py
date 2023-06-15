import visualization as v
import os
import numpy as np


def makeHeatData(max_edge_length=0.5):
    # Create heat kernel tensors (x) and corresponding labels (y) (0 for healthy vessels/ 1 for aneurysm)
    heat_diagrams_aneurysm1 = v.makeHeatDiagrams(os.path.join("IntrA", "generated", "aneurysm", "obj"), max_edge_length=max_edge_length)
    heat_diagrams_aneurysm2 = v.makeHeatDiagrams(os.path.join("IntrA", "annotated", "obj"), max_edge_length=max_edge_length)

    x = np.concatenate((heat_diagrams_aneurysm1, heat_diagrams_aneurysm2))
    y = np.ones(len(heat_diagrams_aneurysm1)+len(heat_diagrams_aneurysm2))

    heat_diagrams_healthy = v.makeHeatDiagrams(os.path.join("IntrA", "generated", "vessel", "obj"), max_edge_length=max_edge_length)

    x = np.append(x, heat_diagrams_healthy)
    x = np.reshape(x, (-1,) + heat_diagrams_aneurysm1[0].shape[1:])
    y = np.append(y, np.zeros(len(heat_diagrams_healthy)))

    if not os.path.exists(os.path.join("data", "heat")):
        os.makedirs(os.path.join("data", "heat"))

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

    if not os.path.exists(os.path.join("data", "image")):
        os.makedirs(os.path.join("data", "image"))

    np.save(os.path.join("data", "image", "persistence_images.npy"), x)
    np.save(os.path.join("data", "image", "labels.npy"), y)


def saveImage(image, folder_path, file_name):
    """
    Saves given persistence image in given folder, with given name.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig = image.plot()
    path = os.path.join(folder_path, file_name + ".png")
    fig.write_image(path)



def loadData(data_type="heat"):
    if data_type == "heat":
        x = np.load(os.path.join("data", data_type, "heat_kernel_tensors.npy"))
        y = np.load(os.path.join("data", data_type, "labels.npy"))
        return x, y


