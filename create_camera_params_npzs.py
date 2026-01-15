import torch
import math
import os
import sys
import json
import struct
from PIL import Image
import numpy as np
import collections
from typing import NamedTuple
import matplotlib.pyplot as plt


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    focal_length_x: float
    focal_length_y: float

    image_path: str
    image_name: str
    width: int
    height: int
    time: float

    znear:float = 0.01
    zfar:float = 100.0
    radial_distortion : float = 0.0
    scale_value: int = 1


    def __str__(self):
        return f"""
                uid : {self.uid},
                R : {self.R},
                T : {self.T},
                FovY : {self.FovY},
                FovX : {self.FovX},
                image_path : {self.image_path},
                image_name : {self.image_name},
                width : {self.width},
                height : {self.height},
                timestamp : {self.time}
                znear:{self.znear}
                zfar:{self.zfar}
                radial_distortion : {self.radial_distortion}
                scale_value: {self.scale_value}

               """

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

# def read_extrinsics_binary(path_to_model_file):
#     """
#     see: src/base/reconstruction.cc
#         void Reconstruction::ReadImagesBinary(const std::string& path)
#         void Reconstruction::WriteImagesBinary(const std::string& path)
#     """
#     images = {}
#     with open(path_to_model_file, "rb") as fid:
#         num_reg_images = read_next_bytes(fid, 8, "Q")[0]
#         for _ in range(num_reg_images):
#             binary_image_properties = read_next_bytes(
#                 fid, num_bytes=64, format_char_sequence="idddddddi")
#             image_id = binary_image_properties[0]
#             qvec = np.array(binary_image_properties[1:5])
#             tvec = np.array(binary_image_properties[5:8])
#             camera_id = binary_image_properties[8]
#             image_name = ""
#             current_char = read_next_bytes(fid, 1, "c")[0]
#             while current_char != b"\x00":   # look for the ASCII 0 entry
#                 image_name += current_char.decode("utf-8")
#                 current_char = read_next_bytes(fid, 1, "c")[0]
#             num_points2D = read_next_bytes(fid, num_bytes=8,
#                                            format_char_sequence="Q")[0]
            
#             x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
#                                        format_char_sequence="ddq"*num_points2D)
            
#             xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
#                                    tuple(map(float, x_y_id_s[1::3]))])
#             point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
#             images[image_id] = Image(
#                 id=image_id, qvec=qvec, tvec=tvec,
#                 camera_id=camera_id, name=image_name,
#                 xys=xys, point3D_ids=point3D_ids)
#     return images

def read_extrinsics_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]

            # # Read raw bytes (24 bytes per point)
            # raw_data = fid.read(24 * num_points2D)
            # triplets = list(struct.iter_unpack("ddq", raw_data))

            # Initialize lists to collect the data
            xys = []
            point3D_ids = []

            # Each point is 24 bytes: 8 + 8 + 8 (double, double, int64)
            for _ in range(num_points2D):
                raw = fid.read(24)
                x, y, id3D = struct.unpack("ddq", raw)
                xys.append([x, y])
                point3D_ids.append(id3D)

            xys = np.array(xys, dtype=np.float64)
            point3D_ids = np.array(point3D_ids, dtype=np.int64)

            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids
            )
    return images


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))

                # qvec, tvec = apply_rotation(qvec, tvec)
                
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_intrinsics_text(path, scale_value = 1):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(round(int(elems[2]) * scale_value))
                height = int(round(int(elems[3]) * scale_value))
                params = np.concatenate([
                                np.array(list(map(float, elems[4:-2]))) * scale_value,
                                np.array([float(width)/2, float(height)/2])
                            ])
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

# def change_axes_of_qvec(qvec:np.array, rotmat:np.array):
#     q = list(qvec)
#     res = rotmat @ qvec[1:]
#     return np.array([q[0],*list(res)])

def readColmapCameras(cam_extrinsics, cam_intrinsics, 
                      colmap_images_folder=None):
    cam_infos = {}
    time_length = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id

        R = qvec2rotmat(extr.qvec) # np.transpose()
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            radial_distortion = 0.0
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            radial_distortion = intr.params[3] # this is the radius of distortion actually
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_y, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(colmap_images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path)

        znear = 0.01
        zfar = 100.0

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, 
                              focal_length_x = focal_length_x, focal_length_y = focal_length_y, radial_distortion = radial_distortion,
                              znear = znear, zfar = zfar,
                              width=width, height=height, time=idx/time_length,
                              image_path=image_path, image_name=image_name, 
                              )
        """
        cam_info = CameraInfo(
                    depth = depth, crisp_image = crisp_image,
                    means = xyz,
                    )
        
        """
        cam_infos[image_name] = cam_info
    sys.stdout.write('\n')
    return cam_infos

#=========================================================================================

def get_dgmarbles_cam_infos(camera_names, cam_infos_colmap):
    dg_marbles_cam_infos = {}
    # print(camera_names)
    # print(cam_infos_colmap)
    for dg_image_name in camera_names.keys():
        if camera_names[dg_image_name] in cam_infos_colmap.keys():
            cam_info = cam_infos_colmap[camera_names[dg_image_name]]
            dg_marbles_cam_infos[dg_image_name] = cam_info
        else:...
            # print(f"Warning: {dg_image_name} not found in colmap cameras.")
    return dg_marbles_cam_infos

def create_camera_parameters_jsons(colmap_params_path, \
        camera_names_json = None, \
        colmap_images_folder = None,
        dgmarbles_images_folder = None,
        output_folder = None
    ):

    # motion mask : parts in motion are masked and background is preserved.
    # background mask : background will be masked leaving foreground

    highest_timestamp = len(os.listdir( colmap_images_folder))/2

    # try:
    # cameras_extrinsic_file = os.path.join(colmap_params_path, "sparse/0", "images.bin")
    # cameras_intrinsic_file = os.path.join(colmap_params_path, "sparse/0", "cameras.bin")
    # cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    # cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # except:
    cameras_extrinsic_file = os.path.join(colmap_params_path, "sparse/0", "images.txt") # poses_vggt
    cameras_intrinsic_file = os.path.join(colmap_params_path, "sparse/0", "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    #make changes here - folder names adjust
    cam_infos_colmap = readColmapCameras(cam_extrinsics=cam_extrinsics, 
                                           cam_intrinsics=cam_intrinsics,
                                           colmap_images_folder = colmap_images_folder
                                           )

    # camera_names = json.loads(camera_names_json)
    with open(camera_names_json, 'r') as f:
        camera_names = json.load(f)

    dg_marbles_cam_infos = get_dgmarbles_cam_infos(camera_names, cam_infos_colmap)

    # print(dg_marbles_cam_infos.keys())

    xs = []
    ys = []
    zs = []

    uids = []

    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    # B = A @ A

    genmojo_c2ws = np.zeros((len(dg_marbles_cam_infos), 4,4))
    fys = []

    for dg_image_name in dg_marbles_cam_infos.keys():
        fys.append(dg_marbles_cam_infos[dg_image_name].focal_length_y)
        # create poses json in dycheck camera format.
        # camera = {
        #     "focal_length":dg_marbles_cam_infos[dg_image_name].focal_length_x,
        #     "fx": dg_marbles_cam_infos[dg_image_name].focal_length_x,
        #     "fy": dg_marbles_cam_infos[dg_image_name].focal_length_y,
        #     "image_size": [dg_marbles_cam_infos[dg_image_name].width, dg_marbles_cam_infos[dg_image_name].height],
            
        #     "orientation":(A @ dg_marbles_cam_infos[dg_image_name].R).tolist(), #
        #     "pixel_aspect_ratio": 1.0,
        #     "position": ((-1 * dg_marbles_cam_infos[dg_image_name].R.T @ dg_marbles_cam_infos[dg_image_name].T)  ).tolist(), # (np.diag([1,-1,-1]) @ ) *(1880*518/196/712)
            
        #     "principal_point": [dg_marbles_cam_infos[dg_image_name].width/2, dg_marbles_cam_infos[dg_image_name].height/2], # this is cx, cy
        #     "radial_distortion": [dg_marbles_cam_infos[dg_image_name].radial_distortion, 0.0, 0.0],
        #     "skew": 0.0,
        #     "tangential_distortion": [0.0, 0.0]
        # }
        # cam_path = os.path.join(output_folder, 'camera', dg_image_name.replace('.png', '.json'))
        # with open(cam_path, "w") as f:
        #     json.dump(camera, f)

        position = -1 * dg_marbles_cam_infos[dg_image_name].R.T @ dg_marbles_cam_infos[dg_image_name].T
        R = dg_marbles_cam_infos[dg_image_name].R

        colmap_c2w = np.eye(4)
        colmap_c2w[0:3,0:3] = R.T
        colmap_c2w[0:3,3] = position

        # here we have colmap convention i.e. x-right, y-down, z-forward
        # but we need x-right, y-forward, z-up
        # is x: right, y: in, z: up
        world_change_axes = np.array([[1, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, 0, 1]]).astype(np.float32)
        
        camera_change_axes = np.array([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]]).astype(np.float32)

        genmojo_c2w = world_change_axes @ colmap_c2w @ camera_change_axes

        # print("uid, dg_image_name:", dg_marbles_cam_infos[dg_image_name].uid, dg_image_name )
        genmojo_c2ws[int(dg_image_name.split('.')[0])-1] = genmojo_c2w

        # temp = -1 * dg_marbles_cam_infos[dg_image_name].R.T @ dg_marbles_cam_infos[dg_image_name].T # *(1880*518/196/712)
        # xs.append(temp[0])
        # ys.append(temp[1])
        # zs.append(temp[2])
        # uids.append(dg_marbles_cam_infos[dg_image_name].uid)
    # print("xs lenght", len(xs))

    # save the genmojo_c2ws as npz. it's reading should happen liek this: loaded_poses = np.load(file)['cam_c2w']
    np.savez_compressed(os.path.join(output_folder,'cam_c2w.npz'), cam_c2w=genmojo_c2ws)

    # print average fy and fovy
    avg_fy = sum(fys)/len(fys)
    avg_fovy = 2*math.atan( (dg_marbles_cam_infos[dg_image_name].height/2) / avg_fy )
    print(f"Average fy: {avg_fy}, Average fovy: {avg_fovy} radians, {math.degrees(avg_fovy)} degrees")
    
    print(f"Camera parameters npzs created in {os.path.join(output_folder,'cam_c2w.npz')}")
    # plot_the_values(uids, xs, ys, zs)

    # fig = plt.figure()
    # ax = plt.axes(projection = '3d')

    # ax.plot3D(uids, xs, ys)
    # ax.set_xlabel('uid')
    # ax.set_ylabel('xs')
    # ax.set_zlabel('ys')
    # plt.show()
    # assuming xs, ys, zs, uids are already defined

def plot_the_values(uids, xs, ys, zs):
    def move_axis_to_zero(ax):
        # Move left and bottom spines to zero
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        # Hide the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add arrows
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # assuming xs, ys, zs, uids are already defined
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # x vs uid
    axes[0].plot(uids, xs, marker='o', linestyle='-')
    axes[0].set_ylabel("X")
    axes[0].set_title("Camera Coordinates vs UID")
    move_axis_to_zero(axes[0])

    # y vs uid
    axes[1].plot(uids, ys, marker='o', linestyle='-', color='orange')
    axes[1].set_ylabel("Y")
    move_axis_to_zero(axes[1])

    # z vs uid
    axes[2].plot(uids, zs, marker='o', linestyle='-', color='green')
    axes[2].set_ylabel("Z")
    axes[2].set_xlabel("UID")
    move_axis_to_zero(axes[2])

    plt.tight_layout()
    # plt.show(block=True)
    plt.savefig("camera_coords_translations.png", dpi=300)
    print("Camera coordinates plot saved as 'camera_coords_translations.png'")


def replace_depth_maps(vggt_depths_folder, dgmarbles_depth_folder,  camera_names_json):
    # Ensure destination folder exists
    os.makedirs(dgmarbles_depth_folder, exist_ok=True)

    # Load mapping JSON
    with open(camera_names_json, 'r') as f:
        mapping = json.load(f)

    for dest_name, src_name in mapping.items():
        # Replace extension with .npy
        src_file = os.path.splitext(src_name)[0] + '.npy'
        dest_file = os.path.splitext(dest_name)[0] + '.npy'

        # Construct full paths
        src_path = os.path.join(vggt_depths_folder, src_file)
        dest_path = os.path.join(dgmarbles_depth_folder, dest_file)

        # Load depth and save it to destination
        if not os.path.exists(src_path):
            print(f"Warning: Source file {src_path} does not exist.")
            continue

        depth = np.load(src_path)
        np.save(dest_path, depth)


if __name__ == "__main__":
    camera_params_path = "/media/dharma/dharma_folder/sai_folder/datasets/dynamic_marbles/experiments/vggt_camp2_masked_10_depth_upscaled/712_1880_multi_cam_opt"

    camera_names_json = "/media/dharma/dharma_folder/sai_folder/datasets/dynamic_marbles/experiments/vggt_camp2_camera_names.json"

    colmap_images_folder = "/media/dharma/dharma_folder/sai_folder/datasets/dynamic_marbles/experiments/vggt_camp2_masked_10_depth_upscaled/712_1880_multi_cam_opt/images_aif"
    dgmarbles_images_folder = "/media/dharma/dharma_folder/sai_folder/genmojo/data/camp/JPEGImages"

    output_folder = "data/camp/cam_path"
    os.makedirs(output_folder, exist_ok=True)
    
    # os.makedirs(os.path.join(output_folder, 'depth'), exist_ok=True)
    

    create_camera_parameters_jsons(camera_params_path, \
        camera_names_json = camera_names_json, \

        colmap_images_folder = colmap_images_folder, \
        dgmarbles_images_folder = dgmarbles_images_folder, \
        
        output_folder = output_folder
    )
    # vggt_depths_folder = 'vggt_camp2/dpt_712_1880'
    # dgmarbles_depth_folder  = 'vggt_camp2_final/depth/1x'

    # replace_depth_maps(vggt_depths_folder, dgmarbles_depth_folder,  camera_names_json)









