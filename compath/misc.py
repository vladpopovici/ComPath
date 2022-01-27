# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__all__ = ['rgb2ycbcr', 'ycbcr2rgb',
           'R_', 'G_', 'B_', 'RGBA2RGB']


"""
TIAX.UTILS.MISC: miscellanious utility functions.
@author: vlad
"""

import numpy as np
from skimage.util import img_as_uint
from compath.mask import binary_mask
import matplotlib
import matplotlib.pyplot
import numpy as np
import skimage.draw as skd


def rgb2ycbcr(im: np.ndarray) -> np.ndarray:
    """
    RGB2YCBCR: converts an RGB image into YCbCr (YUV) color space.

    :param im: numpy.ndarray
      [m x n x 3] image
    """

    if im.ndim != 3:
        raise ValueError('Input image must be RGB.')
    h, w, c = im.shape
    if c != 3:
        raise ValueError('Input image must be a 3-channel (RGB) image.')

    if im.dtype != np.uint8:
        im = img_as_uint(im)

    ycc = np.array([[0.257, 0.439, -0.148],
                    [0.504, -0.368, -0.291],
                    [0.098, -0.071, 0.439]])

    im = im.reshape((h * w, c))

    r = np.dot(im, ycc).reshape((h, w, c))
    r[:, :, 0] += 16
    r[:, :, 1:3] += 128

    im_res = np.array(np.round(r), dtype=im.dtype)

    return im_res
## end rgb2ycbcr


def ycbcr2rgb(im: np.ndarray) -> np.ndarray:
    """
    YCBCR2RGB: converts an YCbCr (YUV) in RGB color space.

    :param im: numpy.ndarray
      [m x n x 3] image
    """

    if im.ndim != 3:
        raise ValueError('Input image must be YCbCr.')
    h, w, c = im.shape
    if c != 3:
        raise ValueError('Input image must be a 3-channel (YCbCr) image.')

    if im.dtype != np.uint8:
        im = img_as_uint(im)

    iycc = np.array([[1.164, 1.164, 1.164],
                     [0, -0.391, 2.018],
                     [1.596, -0.813, 0]])

    r = im.reshape((h * w, c))

    r[:, 0] -= 16.0
    r[:, 1:3] -= 128.0
    r = np.dot(r, iycc)
    r[r < 0] = 0
    r[r > 255] = 255
    r = np.round(r)
    # x = r[:,2]; r[:,2] = r[:,0]; r[:,0] = x

    im_res = np.array(r.reshape((h, w, c)), dtype=np.uint8)

    return im_res
## end ycbcr2rgb


def R_(_img: np.ndarray) -> np.ndarray:
    return _img[:, :, 0]


def G_(_img: np.ndarray) -> np.ndarray:
    return _img[:, :, 1]


def B_(_img: np.ndarray) -> np.ndarray:
    return _img[:, :, 2]


def RGBA2RGB(img: np.ndarray,
        with_masking: bool=True,
        nonzero_alpha_is_foreground:bool=True,
        alpha_cutoff: np.uint8=127,
        background_level: np.uint8 = 255) -> np.ndarray:
    """Removes the alpha channel with eventual masking. Many WSI use the
    alpha channel to store the mask for foreground.

    :param img: (nmupy.ndarray) a NumPy array with image data (m x n x 4),
        with channel ordering R,G,B,A.
    :param with_masking: (bool) if True, the masking from the alpha channel
        is applied to the rest of the channles
    :param nonzero_alpha_is_foreground: (bool) indicates that nonzero alpha
        values are indicating pixels of the foreground
    :param alpha_cutoff: (uint8) cutoff for alpha mask
    :param backgound_level: (uint8) sets all the pixels in background (according
        to alpha channel) to this value. Typically, is set to 255 indicating
        that background is white.
    :return: a new RGB image
    """
    new_img = img[...,:3].copy()
    if with_masking:
        mask = binary_mask(img[...,3].squeeze(),
            level=alpha_cutoff,
            mode = 'below' if nonzero_alpha_is_foreground else 'above')
        for k in np.arange(0,3):
            new_img[mask==1, k] = background_level

    return new_img


#
# Coordinate conversion tools between magnifications/levels.
#
class CoordUtils:
    @staticmethod
    def xy2xy(point: tuple[float,float], 
              src_resolution: float, 
              dst_resolution: float, 
              units="power") -> tuple[float,float]:

        x, y = point
        f = 1.0
        if units.lower == "power":
            f = dst_resolution / src_resolution
        elif units.lower == "mpp":
            f = src_resolution / dst_resolution
        elif units.lower == "level":
            # assumes "natural" ordering of levels from highest resolution 
            # (level 0) to lowest resolution (level > 0)
            f = 2.0**(src_resolution - dst_resolution)
        
        return f*x, f*y


def array_to_image(filename, X, cmap=matplotlib.cm.plasma, dpi=120.0,
                   invert_y=True):
    """Produce a visual representation of a data matrix.

    Parameters:
        :param filename: str
            name of the file to save the image to
        :param X: numpy.array (2D)
            the data matrix to be converted to a raster image
        :param cmap:  matplotlib.cm
            color map
        :param dpi: float
            image resolution (DPI)
        :param invert_y: bool
            should the y-axis (rows) be inverted, such that the top
            of the matrix (low row counts) would correspond to low
            y-values?
    """

    # From SciPy cookbooks, https://scipy-cookbook.readthedocs.io/items/Matplotlib_converting_a_matrix_to_a_raster_image.html

    figsize = (np.array(X.shape) / float(dpi))[::-1]
    matplotlib.rcParams.update({'figure.figsize': figsize})
    fig = matplotlib.pyplot.figure(figsize=figsize)
    matplotlib.pyplot.axes([0, 0, 1, 1])  # Make the plot occupy the whole canvas
    matplotlib.pyplot.axis('off')
    fig.set_size_inches(figsize)

    matplotlib.pyplot.imshow(X, origin='upper' if invert_y else 'lower',
                             aspect='equal', cmap=cmap)

    matplotlib.pyplot.savefig(filename, facecolor='white', edgecolor='black', dpi=dpi)
    matplotlib.pyplot.close(fig)

    return


def mark_points(image: np.array, points: np.array, radius: int, color: tuple,
                in_situ: bool = True) -> np.array:
    """Mark a series of points in an image.

    Parameters:
        :param image: an array in which to mark the points. May be multi-channel.
        :param points: a N X 2 array with (row, col) coords for the N points
        :param radius: the radius of the disc to be drawn at the positions
        :param color: a tuple with R,G,B color specifications for the marks. If the
            image is single channel, only the first value will be used (R)
        :param in_situ: (bool) if True, the points are marked directly in the image,
            otherwise a copy will be used

    Return:
        an array with the points marked in the image
    """

    if in_situ:
        res = image
    else:
        res = image.copy()

    if res.ndim == 2:
        col = color[0]
    else:
        col = color[:3]

    for p in points:
        r, c = skd.circle(p[0], p[1], radius, shape=res.shape)
        res[r,c,...] = col

    return res
    