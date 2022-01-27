# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

# TISSUE: methods for tissue segmentation, stain normalization etc

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__all__ = [ 'SimpleHETissueMasker' ]

import numpy as np
import mahotas as mh
from skimage.util import img_as_bool, img_as_ubyte
import skimage.morphology as skm
from sklearn.cluster import MiniBatchKMeans
from compath.misc import G_
from compath.stain import rgb2he
from tiatoolbox.tools.tissuemask import TissueMasker
from joblib import Parallel, delayed

#
# SimpleHETissueMasker
#
class SimpleHETissueMasker(TissueMasker):
    """A simple method for H&E foreground segmentation"""

    def __init__(self, min_area: int=150) -> None:
        super().__init__()
        self.g_thresholds = None
        self.min_aread = min_area


    def _find_threshold(img: np.ndarray) -> int:
        # Apply vector quantization to remove the "white" background - work in the
        # green channel:
        vq = MiniBatchKMeans(n_clusters=2)
        if len(img.shape == 2):
            # just one channel,
            th = int(np.round(0.95 * np.max(vq.fit(img.reshape((-1, 1))).cluster_centers_.squeeze())))
        else:
            # RGB:
            th = int(np.round(0.95 * np.max(vq.fit(G_(img).reshape((-1, 1))).cluster_centers_.squeeze())))

        return th

    def _get_mask(img: np.ndarray, th: int, min_area: int) -> np.ndarray:
        mask = img[:,:,1] < th  # G-plane

        skm.binary_closing(mask, skm.disk(3), out=mask)
        mask = img_as_bool(mask)
        mask = skm.remove_small_objects(mask, min_size=min_area, in_place=True)

        # Some hand-picked rules:
        # -at least 5% H and E
        # -at most 50% background
        # for a region to be considered tissue

        h, e, b, _ = rgb2he(img)

        mask &= (h > np.percentile(h, 5)) | (e > np.percentile(e, 5))
        mask &= (b < np.percentile(b, 50))  # at most at 50% of "other components"

        mask = mh.close_holes(mask)

        return img_as_bool(mask)


    def fit(self, images: np.ndarray, masks=None) -> None:
        """ Finds suitable threshold for basic segmentation in G(reen) plane."""
        n = 1 if len(images.shape) == 3 else images.shape[0]   # number of images

        th = Parallel(n_jobs=-1)(delayed(self._find_threshold)(G_(img)) for img in images)
        self.g_thresholds = np.array(th, dtype=np.int)
        
        self.fitted = True
        
        return
    

    def transform(self, images: np.ndarray) -> np.ndarray:
        """Create masks."""

        super().transform(images)
        n = 1 if len(images.shape) == 3 else images.shape[0]   # number of images
        masks = Parallel(n_jobs=-1)(delayed(self._get_mask)(images[i,...], self.g_thresholds[i], self.min_area) \
            for i in np.arange(n))

        return [masks]
## end class SimpleHETissueMasker

    

