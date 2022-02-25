# -*- coding: utf-8 -*-

# Converts a whole slide image to ZARR (OME) format.

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
from datetime import datetime
import hashlib

_time = datetime.now()
__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "1.0"
__description__ = {
    'name': 'nuclei_dectection',
    'unique_id' : hashlib.md5(str.encode('nuclei_detection' + __version__)).hexdigest(),
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': ['None'],
    'output': ['None'],
    'params': dict()
}

import simplejson as json
import geojson as gjson
import configargparse as opt
import pathlib
import zarr
import numpy as np
from tqdm import tqdm, trange
from tinydb import TinyDB, Query
from csbdeep.utils.tf import keras_import
from csbdeep.data import Normalizer, normalize_mi_ma

from stardist.models import StarDist2D

from compath.core import WSIInfo
from compath.utils import NumpyJSONEncoder
from compath.annot import WSIAnnotation, Polygon

class IntensityNormalizer(Normalizer):
    def __init__(self, mi=0, ma=255):
        self.mi, self.ma = mi, ma

    def before(self, x, axes):
        return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)

    def after(*args, **kwargs):
        assert False

    @property
    def do_after(self):
        return False


def main():
    p = opt.ArgumentParser(description="Detect nuclei in a slide image.")
    p.add_argument("--mri_path", action="store", help="root folder for the multiresolution image (ZARR format)",
                   required=True)
    p.add_argument("--level", action="store", help="pyramid level to process", required=False, default=0, type=int)
    p.add_argument("--out", action="store",
                   help="JSON file for storing the resulting annotation (will be saved to ../annot/ relative to ZARR path)",
                   required=True)
    p.add_argument("--annotation_name", action="store", help="name of the resulting annotation",
                   default="nuclei", required=False)
    p.add_argument("--min_area", action="store", type=int, default=None,
                   help="minimum area of a nuclei", required=False)
    p.add_argument("--max_area", action="store", type=int, default=None,
                   help="maximum area of a nuclei", required=False)
    p.add_argument("--min_prob", action="store", type=float, default=0.5,
                   help="all candidate dections below this minimum probability will be discarded")
    p.add_argument("--track_processing", action="store_true",
                   help="should this action be stored in the <-RUN-detect_tissue.json> file for the slide?")

    args = p.parse_args()

    __description__['params'] = vars(args)
    __description__['input'] = [args.mri_path]
    __description__['output'] = [args.out]

    in_path = pathlib.Path(args.mri_path)
    slide_name = in_path.name
    out_path = pathlib.Path(args.out) / slide_name

    if not out_path.exists():
        out_path.mkdir()

    args.min_prob = max(0, min(args.min_prob, 1.0))

    # (out_path/'.run').mkdir(exist_ok=True)
    # if args.track_processing:
    #     with open(out_path/'.run'/f'run-{__description__["name"]}.json', 'w') as f:
    #         json.dump(__description__, f, indent=2)

    keras = keras_import()
    nrm = IntensityNormalizer(0, 255)
    wsi = WSIInfo(str(in_path))
    if args.level < 0 or args.level >= wsi.level_count():
        raise RuntimeError("Specified level is outside the pyramid.") from None
    img_shape = wsi.get_extent_at_level(args.level)
    sz = min(math.floor(math.sqrt(min(img_shape['width'], img_shape['height'])))**2, 2*4096)
    sz = int(sz)

    model = StarDist2D.from_pretrained('2D_versatile_he')

    img = zarr.open(str(in_path), mode='r')['/'+str(args.level)]
    _, polys = model.predict_instances_big(img, axes='YXC',
                                           block_size=sz,
                                           min_overlap=128, context=128,
                                           normalizer=nrm,
                                           n_tiles=(4, 4, 1),
                                           labels_out=False,
                                           show_progress=True)
    with open("/home/vlad/tmp/nuclei.json" , 'w') as f:
        json.dump(polys, f, cls=NumpyJSONEncoder)

    (idx,) = np.where(np.array(polys['prob']) >= args.min_prob)
    n = len(polys['prob'])
    annot = WSIAnnotation('nuclei',
                          wsi.get_extent_at_level(args.level),
                          mpp=wsi.get_mpp_for_level(args.level))
    for k in idx:
        p = Polygon([xy for xy in zip(polys['coord'][k][0], polys['coord'][k][1])])
        annot.add_annotation_object(p)

    with open("/home/vlad/tmp/nuclei.geojson" , 'w') as f:
        gjson.dump(annot.asGeoJSON(), f, cls=NumpyJSONEncoder)

    # annot_idx = out_path.parent / '.annot_idx.json'
    # with TinyDB(annot_idx) as db:
    #     q = Query()
    #     r = db.search(q.unique_id == __description__['unique_id'])
    #     if len(r) == 0:
    #         # empty DB or no such record
    #         db.insert({'unique_id' : __description__['unique_id'],
    #                    'annotator': __description__['name'], 'parameters': __description__['params']})
    #     else:
    #         db.update({'annotator': __description__['name'], 'parameters': __description__['params']},
    #                   q.unique_id == __description__['unique_id'])

    return
##


if __name__ == '__main__':
    main()
