"""
PegasusIO is the IO component of Pegasus and a multi-modality single-cell genomics IO package. It is a command line tool, a python package and a base for Cloud-based analysis workflows.

Usage:
  pegasusio <command> [<args>...]
  pegasusio -h | --help
  pegasusio -v | --version

Sub-commands:
  aggregate_matrix        Aggregate multi-modality single-cell genomics data into one data object. It also enables users to import metadata into the aggregated data.

Options:
  -h, --help          Show help information.
  -v, --version       Show version.

Description:
  This is PegasusIO, the IO component of Pegasus.
"""

from docopt import docopt
from docopt import DocoptExit
from . import __version__ as VERSION
from . import commands


def main():
    args = docopt(__doc__, version=VERSION, options_first=True)

    command_name = args["<command>"]
    command_args = args["<args>"]
    command_args.insert(0, command_name)

    try:
        command_class = getattr(commands, command_name)
    except AttributeError:
        print("Unknown command {cmd}!".format(cmd=command_name))
        raise DocoptExit()

    command = command_class(command_args)
    command.execute()


if __name__ == "__main__":
    main()
