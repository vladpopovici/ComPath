# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

import click
import sys
from tiax.cli.tissue_mask import tissue_mask
from tiax import __version__, version_msg


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(
    __version__, "--version", "-V", help="Version", message=version_msg()
)
def main():
    """Extensions to the TIAtoolbox."""
    return 0


main.add_command(tissue_mask)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover