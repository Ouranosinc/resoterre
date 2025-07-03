"""Console script for resoterre."""
import sys

import click


@click.command()
def main(args=None) -> int:
    """Console script for resoterre."""
    click.echo(
        "Replace this message by putting your code into resoterre.cli.main",
    )
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
