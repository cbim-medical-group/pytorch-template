import h5py
import cv2
import numpy as np
# import dicom
import json
import os
import shutil
import sys
import random
from matplotlib import image
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_erosion
from scipy.fftpack import fftn, ifftn
from scipy.signal import argrelmin, correlate
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# number of bins to use in histogram for gaussian regression
NUM_BINS = 100
# number of standard deviations past which we will consider a pixel an outlier
STD_MULTIPLIER = 2
# number of points of our interpolated dataset to consider when searching for
# a threshold value; the function by default is interpolated over 1000 points,
# so 250 will look at the half of the points that is centered around the known
# myocardium pixel
THRESHOLD_AREA = 250
# number of pixels on the line within which to search for a connected component
# in a thresholded image, increase this to look for components further away
COMPONENT_INDEX_TOLERANCE = 20
# number of angles to search when looking for the correct orientation
ANGLE_SLICES = 36

def calc_rois(images):
    (num_slices, _, _, _) = images.shape
    # print("Calculating mean...", 2)
    dc = np.mean(images, 1)

    def get_H1(i):
        # print("Fourier transforming on slice %d..." % i, 3)
        ff = fftn(images[i])
        first_harmonic = ff[1, :, :]
        # print("Inverse Fourier transforming on slice %d..." % i, 3)
        result = np.absolute(ifftn(first_harmonic))
        # print("Performing Gaussian blur on slice %d..." % i, 3)
        result = cv2.GaussianBlur(result, (5, 5), 0)
        return result

    print("Performing Fourier transforms...", 2)
    h1s = np.array([get_H1(i) for i in range(num_slices)])
    m = np.max(h1s) * 0.05
    h1s[h1s < m] = 0

    # print("Applying regression filter...", 2)
    regress_h1s = regression_filter(h1s)
    # print("Post-processing filtered images...", 2)
    proc_regress_h1s, coords = post_process_regression(regress_h1s)
    # print("Determining ROIs...", 2)
    rois, circles = get_ROIs(dc, proc_regress_h1s, coords)
    return rois, circles

def regression_filter(imgs):
    condition = True
    iternum = 0
    while(condition):
        # print("Beginning iteration %d of regression..." % iternum, 3)
        iternum += 1
        imgs_filtered = regress_and_filter_distant(imgs)
        c1 = get_centroid(imgs)
        c2 = get_centroid(imgs_filtered)
        dc = np.linalg.norm(c1 - c2)
        imgs = imgs_filtered
        condition = (dc > 1.0)  # because python has no do-while loops
    return imgs

def regress_and_filter_distant(imgs):
    centroids = np.array([get_centroid(img) for img in imgs])
    raw_coords = np.transpose(np.nonzero(imgs))
    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    (coords, dists, weights) = get_weighted_distances(imgs, raw_coords, xslope,
                                                      xintercept, yslope,
                                                      yintercept)
    outliers = get_outliers(coords, dists, weights)
    imgs_cpy = np.copy(imgs)
    for c in outliers:
        (z, x, y) = c
        imgs_cpy[z, x, y] = 0
    return imgs_cpy

def get_centroid(img):
    nz = np.nonzero(img)
    pxls = np.transpose(nz)
    weights = img[nz]
    avg = np.average(pxls, axis=0, weights=weights)
    return avg

def regress_centroids(cs):
    num_slices = len(cs)
    y_centroids = cs[:, 0]
    x_centroids = cs[:, 1]
    z_values = np.array(range(num_slices))

    (xslope, xintercept, _, _, _) = linregress(z_values, x_centroids)
    (yslope, yintercept, _, _, _) = linregress(z_values, y_centroids)

    return (xslope, xintercept, yslope, yintercept)

def get_weighted_distances(imgs, coords, xs, xi, ys, yi):
    a = np.array([0, yi, xi])
    n = np.array([1, ys, xs])

    zeros = np.zeros(3)

    def dist(p):
        to_line = (a - p) - (np.dot((a - p), n) * n)
        d = euclidean(zeros, to_line)
        return d

    def weight(p):
        (z, y, x) = p
        return imgs[z, y, x]

    dists = np.array([dist(c) for c in coords])
    weights = np.array([weight(c) for c in coords])
    return (coords, dists, weights)

def get_outliers(coords, dists, weights):
    fivep = int(len(weights) * 0.05)
    ctr = 1
    while True:
        (mean, std, fn) = gaussian_fit(dists, weights)
        low_values = dists < (mean - STD_MULTIPLIER*np.abs(std))
        high_values = dists > (mean + STD_MULTIPLIER*np.abs(std))
        outliers = np.logical_or(low_values, high_values)
        if len(coords[outliers]) == len(coords):
            weights[-fivep*ctr:] = 0
            ctr += 1
        else:
            return coords[outliers]

def gaussian_fit(dists, weights):
    # based on http://stackoverflow.com/questions/11507028/fit-a-gaussian-function
    (x, y) = histogram_transform(dists, weights)
    fivep = int(len(x) * 0.05)
    xtmp = x
    ytmp = y
    fromFront = False
    while True:
        if len(xtmp) == 0 and len(ytmp) == 0:
            if fromFront:
                # well we failed
                idx = np.argmax(y)
                xmax = x[idx]
                p0 = [max(y), xmax, xmax]
                (A, mu, sigma) = p0
                return mu, sigma, lambda x: gauss(x, A, mu, sigma)
            else:
                fromFront = True
                xtmp = x
                ytmp = y

        idx = np.argmax(ytmp)
        xmax = xtmp[idx]

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        p0 = [max(ytmp), xmax, xmax]
        try:
            coeff, var_matrix = curve_fit(gauss, xtmp, ytmp, p0=p0)
            (A, mu, sigma) = coeff
            return (mu, sigma, lambda x: gauss(x, A, mu, sigma))
        except RuntimeError:
            if fromFront:
                xtmp = xtmp[fivep:]
                ytmp = ytmp[fivep:]
            else:
                xtmp = xtmp[:-fivep]
                ytmp = ytmp[:-fivep]


def histogram_transform(values, weights):
    hist, bins = np.histogram(values, bins=NUM_BINS, weights=weights)
    bin_width = bins[1] - bins[0]
    bin_centers = bins[:-1] + (bin_width / 2)

    return (bin_centers, hist)


def post_process_regression(imgs):
    (numimgs, _, _) = imgs.shape
    centroids = np.array([get_centroid(img) for img in imgs])
    # print("Performing final centroid regression...", 3)
    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    imgs_cpy = np.copy(imgs)

    def filter_one_img(zlvl):
        points_on_zlvl = np.transpose(imgs[zlvl].nonzero())
        points_on_zlvl = np.insert(points_on_zlvl, 0, zlvl, axis=1)
        (coords, dists, weights) = get_weighted_distances(imgs, points_on_zlvl,
                                                          xslope, xintercept,
                                                          yslope, yintercept)
        outliers = get_outliers(coords, dists, weights)
        for c in outliers:
            (z, x, y) = c
            imgs_cpy[z, x, y] = 0

    # print("Final image filtering...", 3)
    for z in range(numimgs):
        # print("Filtering image %d of %d..." % (z+1, numimgs), 4)
        filter_one_img(z)

    return (imgs_cpy, (xslope, xintercept, yslope, yintercept))

def get_ROIs(originals, h1s, regression_params):
    (xslope, xintercept, yslope, yintercept) = regression_params
    (num_slices, _, _) = h1s.shape
    results = []
    circles = []
    for i in range(num_slices):
        # print("Getting ROI in slice %d..." % i, 3)
        o = originals[i]
        h = h1s[i]
        ctr = (xintercept + xslope * i, yintercept + yslope * i)
        r = circle_smart_radius(h, ctr)
        tmp = np.zeros_like(o)
        floats_draw_circle(tmp, ctr, r, 1, -1)
        results.append(tmp * o)
        circles.append((ctr, r))

    return (np.array(results), np.array(circles))

def circle_smart_radius(img, center):
    domain = np.arange(1, 100)
    (xintercept, yintercept) = center

    def ratio(r):
        return filled_ratio_of_circle(img, (xintercept, yintercept), r)*r

    y = np.array([ratio(d) for d in domain])
    most = np.argmax(y)
    return domain[most]

def filled_ratio_of_circle(img, center, r):
    mask = np.zeros_like(img)
    floats_draw_circle(mask, center, r, 1, -1)
    masked = mask * img
    (x, _) = np.nonzero(mask)
    (x2, _) = np.nonzero(masked)
    if x.size == 0:
        return 0
    return float(x2.size) / x.size

def floats_draw_circle(img, center, r, color, thickness):
    (x, y) = center
    x, y = int(np.round(x)), int(np.round(y))
    r = int(np.round(r))
    cv2.circle(img, center=(x, y), radius=r, color=color, thickness=thickness)







