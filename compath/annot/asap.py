
#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

"""Import annotations from ASAP (XML) files. See also https://github.com/computationalpathologygroup/ASAP"""

import xmltodict
import tiax.annot as ann
from pathlib import Path


def annotation_from_ASAP(infile: str, wsi_extent: tuple[int,int], magnification: float) -> ann.WSIAnnotation:
    infile = Path(infile)
    with open(infile, 'r') as input:
        annot_dict = xmltodict.parse(input.read(), xml_attribs=True)

    if 'ASAP_Annotations' not in annot_dict:
        raise RuntimeError('Syntax error in ASAP XML file')

    annot_dict = annot_dict['ASAP_Annotations']
    if 'Annotations' not in annot_dict:
        raise RuntimeError('Syntax error in ASAP XML file')

    try:
        group_list = [ g['@Name'] for g in annot_dict['AnnotationGroups']['Group'] ]
    except KeyError:
        group_list = []

    annot_list = annot_dict['Annotations']['Annotation']  # many nested levels...

    wsi_annotation = ann.WSIAnnotation(infile.name, wsi_extent, magnification, group_list)
    for annot in annot_list:
        if annot['@Type'].lower() == 'dot':
            coords = [float(annot["Coordinates"]["Coordinate"]["@X"]), float(annot["Coordinates"]["Coordinate"]["@Y"]) ]
            obj = ann.Point(coords, annot['@Name'], in_group=annot["@PartOfGroup"])
        elif annot['@Type'].lower() == 'pointset':
            coords = [(float(o["@X"]), float(o["@Y"])) for o in annot["Coordinates"]["Coordinate"]]
            obj = ann.PointSet(coords, annot['@Name'], in_group=annot["@PartOfGroup"])
        elif annot['@Type'].lower() == 'polygon':
            coords = [(float(o["@X"]), float(o["@Y"])) for o in annot["Coordinates"]["Coordinate"]]
            obj = ann.Polygon(coords, annot['@Name'], in_group=annot["@PartOfGroup"])
        else:
            raise RuntimeError(f"Unknown annotation type {annot['@Type']}")
        wsi_annotation.add_annotation_object(obj)

    return wsi_annotation
