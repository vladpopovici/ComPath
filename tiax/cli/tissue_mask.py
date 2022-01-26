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
TIAX.CLI.TISSUE-MASK: CLI for tissue mask extraction. This implements different
methods than those in TIAtoolbox.
@author: vlad
"""

from genericpath import exists
import numpy as np
import click
from pathlib import Path
from PIL import Image
from shapely.affinity import translate

from tiatoolbox import utils, wsicore
from tiatoolbox.wsicore.slide_info import slide_info
from tiatoolbox.wsicore.wsimeta import WSIMeta
from tiax.tools.tissuemask import SimpleHETissueMasker, detect_foreground
from tiax.utils.mask import mask_to_external_contours


# minimum object sizes (areas, in px^2) for different magnifications to be considered as "interesting"
min_obj_size = {'0.3125': 1500, '1.25': 50000, '2.5': 100000, '5.0': 500000}
WORK_MAG_1 = 0.3125
WORK_MAG_2 = 2.5

@click.group()
def main():  # pragma: no cover
    """Define tissue_mask click group."""
    return 0


@main.command()
@click.option("--img-input", help="Path to WSI file")
@click.option(
    "--output-path",
    help="Path to output file to save the image region in save mode,"
    " default=tissue_mask",
    default="tissue_mask",
)
@click.option(
    "--method",
    help="Tissue masking method to use. Choose from 'generic', 'h&e',"
    " default=generic",
    default="generic",
)
@click.option(
    "--min-area",
    type=int,
    default=None,
    help="minimum area of a region to be kept in the mask",
)
@click.option(
    "--mode",
    default="save",
    help="'show' to display tissue mask or 'save' to save at the output path"
    ", default=save",
)
@click.option(
    "--file-types",
    help="file types to capture from directory, "
    "default='*.svs, *.ndpi, *.jp2, *.png', '*.jpg', '*.tif', '*.tiff'",
    default="*.svs, *.ndpi, *.jp2, *.png, *.jpg, *.tif, *.tiff",
)
def tissue_mask(img_input: str, output_path: str, 
    method: str, min_area: int, mode: str, file_types: str):

    file_types = utils.misc.string_to_tuple(in_str=file_types)
    output_path = Path(output_path).expanduser().absolute()
    input_path = Path(img_input).expanduser().absolute()

    if not input_path.exists():
        raise FileNotFoundError

    files_all = [
        img_input,
    ]

    if input_path.is_dir():
        files_all = utils.misc.grab_files_from_dir(
            input_path=img_input, file_types=file_types
        )

    if mode == "save":
        output_path.mkdir(parents=True, exist_ok=True)

    if method.lower() not in ["generic", "h&e"]:
        raise utils.exceptions.MethodNotSupported

    if min_area is None:
        min_area = min_obj_size[str(WORK_MAG_2)]
    else:
        min_obj_size[str(WORK_MAG_2)] = min_area

    for curr_file in files_all:
        # Use a two-pass strategy:
        # -first, lowest res for detecting the bounding box for all tissue parts
        # -second, higher res for the details

        # - first pass, common to all methods
        wsi = wsicore.wsireader.get_wsireader(input_img=curr_file)
        img = wsi.slide_thumbnail(resolution=WORK_MAG_1, units="power")
        mask, _ = detect_foreground(img, method='fesi', min_area=min_obj_size[str(WORK_MAG_1)])
        contours = mask_to_external_contours(mask, approx_factor=0.0001)
        # find the bounding box of the contours:
        xmin, ymin = img.shape[:2]
        xmax, ymax = 0, 0
        for c in contours:
            minx, miny, maxx, maxy = c.bounds
            xmin = min(xmin, minx)
            ymin = min(ymin, miny)
            xmax = max(xmax, maxx)
            ymax = max(ymax, maxy)

        # slightly larger bounds:
        xmin = max(0, xmin - 5)
        ymin = max(0, ymin - 5)
        xmax = min(img.shape[1] - 1, xmax + 5)
        ymax = min(img.shape[0] - 1, ymax + 5)

        # get extent at level 0:
        bounds_at_0 = wsi._bounds_at_resolution_to_baseline((xmin, ymin, xmax, ymax), 
            WORK_MAG_1, units="power")

        # get shape at WORK_MAG_2
        f = WORK_MAG_2 / WORK_MAG_1
        width = int(f * (xmax - xmin))
        height = int(f * (ymax - ymin))
        
        img = wsi.read_rect(bounds_at_0[:2], (width, height), resolution=WORK_MAG_2, units="power")
        if method.lower() == 'h&e':
            mask, _ = detect_foreground(img, method='simple-he', min_area=min_obj_size[str(WORK_MAG_2)])
        elif method.lower() == 'generic':
            mask, _ = detect_foreground(img, method='fesi',
                                    laplace_ker=15, gauss_ker=17, gauss_sigma=25.0,
                                    morph_open_ker=5, morph_open_iter=7, morph_blur=17,
                                    min_area=min_obj_size[str(WORK_MAG_2)])
        else:
            # should not have gottent this far...
            raise utils.exceptions.MethodNotSupported

        if mode == "show":
            im_region = Image.fromarray(mask[0])
            im_region.show()

        if mode == "save":
            utils.misc.imwrite(
                output_path / Path(curr_file).with_suffix(".png").name,
                mask.astype(np.uint8) * 255,
            )

    return
    