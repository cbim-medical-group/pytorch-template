import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords


def rotate_matrix_from_z(z_new):
    z_new = z_new / np.linalg.norm(z_new)
    if np.all(z_new == np.array([1, 0, 0])):
        y_new = np.array([0, 0, -1])
        x_new = np.array([0, -1, 0])
    elif np.all(z_new == np.array([-1, 0, 0])):
        y_new = np.array([0, 1, 0])
        x_new = np.array([0, 0, 1])
    else:
        y_new = np.cross(z_new, [1, 0, 0])
        x_new = np.cross(y_new, z_new)
        y_new = y_new / np.linalg.norm(y_new)
        x_new = x_new / np.linalg.norm(x_new)
    # rot_matrix = np.stack((x_new, y_new, z_new), axis=0)
    rot_matrix = np.stack((x_new, y_new, z_new), axis=1)
    return rot_matrix


def rotate_z(coords, z_new):
    rot_matrix = rotate_matrix_from_z(z_new)
    # coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    coords = np.dot(rot_matrix, coords.reshape(len(coords), -1)).reshape(coords.shape)
    return coords, rot_matrix


def interpolate_img(img, coords, order=3, mode='constant', cval=0.0):
    return ndimage.map_coordinates(img, coords, order=order, mode=mode, cval=cval)


## data: assume data has been resampled to be isotropic volume
## center is voxel coordinates
## angles are *pi
## patch_size is of size 2 (2D plane)
def MPR_plane(data, center_xyz, z_new, patch_size, order=1):
    coords = create_zero_centered_coordinate_mesh(patch_size)
    coords, _ = rotate_z(coords, z_new)
    for d in range(len(center_xyz)):
        coords[d] += center_xyz[d]
    MPR = interpolate_img(data, coords, order, mode='constant', cval=np.min(data))
    # MPR = interpolate_img(data, coords[::-1], order, mode='constant', cval=np.min(data))
    return MPR, coords


def interpolate_img_itk(itkimg, vcenter, rot_matrix, new_size, order=1):
    if order == 0:
        method = sitk.sitkNearestNeighbor
    elif order == 1:
        method = sitk.sitkLinear
    elif order == 2:
        method = sitk.sitkBSpline

    center = itkimg.TransformContinuousIndexToPhysicalPoint(np.round(vcenter).tolist())

    new_spacing = itkimg.GetSpacing()

    new_origin = center - np.array(new_size) // 2 * new_spacing

    resampler = sitk.ResampleImageFilter()

    T = sitk.VersorTransform()
    T.SetMatrix(rot_matrix.flatten().astype(float))
    T.SetCenter(center)
    resampler.SetTransform(T.GetInverse())

    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetSize(new_size)
    resampler.SetInterpolator(method)

    imgResampled = resampler.Execute(sitk.Cast(itkimg, sitk.sitkFloat32))

    return imgResampled


def MPR_plane_itk(itkimg, center, z_new, patch_size, order=1):
    rot_matrix = rotate_matrix_from_z(z_new).transpose()
    itkMPR = interpolate_img_itk(itkimg, center, rot_matrix, patch_size, order=order)
    coords = np.zeros((3, patch_size[0], patch_size[1], patch_size[2]))
    for i in range(patch_size[0]):
        for j in range(patch_size[1]):
            for k in range(patch_size[2]):
                wpt = itkMPR.TransformIndexToPhysicalPoint((i, j, k))
                coords[:, i, j, k] = itkimg.TransformPhysicalPointToContinuousIndex(wpt)
    return itkMPR, coords


# for example
patch_size = [101, 101, 1]
order = 1
rotationCenter_xyz = [10, 20, 30]
normal_direction = [1, 1, 1]
normal_direction = normal_direction / np.linalg.norm(normal_direction)


# img_resamp / itkimg_resamp should be isotropic
# mpr_img, coords = MPR_plane(img_resamp, rotationCenter_xyz, normal_direction, patch_size, order=order)

# mpr, coords = MPR_plane_itk(itkimg_resamp, rotationCenter_xyz, normal_direction, patch_size, order=order)
# mpr_img = sitk.GetArrayFromImage(mpr)
