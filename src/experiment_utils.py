import os
import open3d
import numpy as np

def ply_binary2ascii(file_full_name):
    pointcloud_in = open3d.io.read_point_cloud(file_full_name)
    # write_point_cloud(filename, pointcloud, write_ascii=False, compressed=False,
    #                   print_progress=False):  # real signature unknown; restored from __doc__
    open3d.io.write_point_cloud(filename=file_full_name, pointcloud=pointcloud_in, write_ascii=True)
    # print("transformed " + file_full_name + " from binary to ascii")


def ply_double2float(file_full_name):
    with open(file_full_name, 'r', encoding="utf-8") as f:
        lines = []
        for line in f.readlines():
            if "float" in line:
                print("the datatype of " + file_full_name + " is already float")
                return
            if line != '\n':
                lines.append(line)
        f.close()

    count = 0
    n = len(lines)
    for i in range(n):
        if count == 3:
            break
        # Replace "double" with "float" for x, y, z property lines
        str_split = lines[i].split(" ")
        if len(str_split) == 3:
            if str_split[1] == "double":
                if str_split[2] == "x\n" or str_split[2] == "y\n" or str_split[2] == "z\n":
                    lines[i] = "property float " + str_split[2]
                    count = count + 1

    f = open(file_full_name, "w")
    for i in range(n):
        f.write(lines[i])
    f.close()


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            ext_str = f.split(".")[1]
            if ext_str != "ply":
                print(f + " is not a ply file")
                yield fullname
                continue
            fullname = os.path.join(root, f)
            yield fullname

def set_label(label,points):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
        raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == points.shape[0]:
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16    # instance id in upper half
    else:
        print("Points shape: ", points.shape)
        print("Label shape: ", label.shape)
        raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((sem_label + (inst_label << 16) == label).all())
    return sem_label, inst_label

def load_data_labels(label_path):

    labels = np.fromfile(label_path, dtype=np.uint32)
    labels = labels.reshape((-1))
    return labels

def load_data_labels_KITTI360(label_path):

    labels = np.fromfile(label_path, dtype=np.int16)
    labels = labels.reshape((-1))
    return labels

def load_data_labels_nuSences(label_path):

    labels = np.fromfile(label_path, dtype=np.uint32)
    labels = labels.reshape((-1))
    return labels

def load_data_labels_waymo(label_path):

    labels = np.fromfile(label_path, dtype=np.float32).reshape((-1, 2))[:,1]
    labels = labels.reshape((-1))
    return labels

def load_data_labels_prediction(label_path):

    labels = np.fromfile(label_path, dtype=np.int32)
    labels = labels.reshape((-1))
    return labels

def load_data_points(points_path):
    cloud = np.fromfile(points_path, dtype=np.float32).reshape((-1, 4))
    # cloud = np.fromfile(points_path, dtype=np.float32)
    return cloud[:,0:3]

