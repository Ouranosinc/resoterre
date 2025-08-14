#!/usr/bin/env python
"""Tests for `resoterre` package."""

import pathlib
from importlib.util import find_spec

from click.testing import CliRunner

import resoterre.cli as cli
from resoterre import resoterre  # noqa: F401


# import pytest

# @pytest.fixture
# def response():
#     """Sample pytest fixture.
#
#     See more at: https://doc.pytest.org/en/latest/explanation/fixtures.html
#     """
#     # import requests
#     # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     # from bs4 import BeautifulSoup
#     # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "resoterre.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


def test_package_metadata():
    """Test the package metadata."""
    project = find_spec("resoterre")

    assert project is not None
    assert project.submodule_search_locations is not None
    location = project.submodule_search_locations[0]

    metadata = pathlib.Path(location).resolve().joinpath("__init__.py")

    with metadata.open() as f:
        contents = f.read()
        assert """Blaise Gauvin St-Denis""" in contents
        assert '__email__ = "gauvin-st-denis.blaise@ouranos.ca"' in contents
        assert '__version__ = "0.1.2"' in contents
