"""Console script for resoterre."""

import sys
from collections.abc import Sequence
from typing import Any

import click


@click.command()
def main(args: Sequence[Any] | None = None) -> int:
    """
    Console script for resoterre.

    Parameters
    ----------
    args : Sequence[Any] | None, optional
        Command line arguments, by default None. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code, 0 for success.
    """
    click.echo(
        "Replace this message by putting your code into resoterre.cli.main",
    )
    click.echo("See click documentation at https://click.palletsprojects.com/")

    click.echo(f"Called with arguments: {args}")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
