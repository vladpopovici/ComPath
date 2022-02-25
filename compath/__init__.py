# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__version__= "0.2.5"

from termcolor import cprint

def version_msg():
    return f"COMPATH v.{__version__} (c) 2020-2022 Vlad Popovici"

def ComPathIntro():
    cprint("""ComPath: Computational Pathology""", 'blue', attrs=['bold'])
    cprint("""================================""", 'blue', attrs=['bold'])
    cprint("""Copyright (c) 2020-2022 Vlad Popovici <popovici@bioxlab.org>""", 'blue')
    cprint("""""")


class Error(Exception):
    """Basic error exception for COMPATH.

    Args:
        msg (str): Human-readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human-readable string describing the exception.
        code (int): Error code.
    """

    def __init__(self, msg, code=1, *args):
        self.message = "COMPATH: " + msg
        self.code = code
        super(Error, self).__init__(msg, code, *args)
