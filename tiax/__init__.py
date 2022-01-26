# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""This package contains various extensions for __TIAtoolbox__ 
[https://github.com/TissueImageAnalytics/tiatoolbox]."""

__version__ = 0.1
__author__ = "Vlad Popovici <popovici@bioxlab.org>"

from ensurepip import version
import sys
from tiatoolbox import __version__ as tbver


def tiax_info():
    i = {
        'version': __version__,
        'tiatoolbox.version': tbver,
    }

    return i

def version_msg():
    v1 = tiax_info()['version']
    v2 = tiax_info()['tiatoolbox.version']
    v3 = sys.version[:3]
    
    return f"TIAX Ver. {v1} based on TIAtoolbox Ver. {v2} on Python {v3}"


if __name__ == "__main__":
    print(version_msg())
