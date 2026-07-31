"""General utilities."""

import datetime
import hashlib
import os
from collections.abc import Callable
from string import Template
from typing import Any


class TemplateStore:
    """
    Collection of string templates with substitution capabilities.

    Parameters
    ----------
    templates : dict[str, str | Template]
        Dictionary of templates.
    substitutes : dict[str, str]
        Dictionary of substitution values.
    substitute_timestamp : bool
        Whether to automatically substitute the current timestamp.
    substitute_pid : bool
        Whether to automatically substitute the current process ID.
    """

    templates: dict[str, Template]
    substitutes: dict[str, str]

    def __init__(
        self,
        templates: dict[str, str | Template] | None = None,
        substitutes: dict[str, str] | None = None,
        substitute_timestamp: bool = True,
        substitute_pid: bool = True,
    ) -> None:
        self.templates = {}
        if templates is not None:
            d: dict[str, Template] = {}
            for key, value in templates.items():
                if isinstance(value, Template):
                    d[key] = value
                else:
                    d[key] = Template(value)
            self.templates = d
        if substitutes is None:
            self.substitutes = {}
        else:
            self.substitutes = substitutes
        self.substitute_timestamp = substitute_timestamp
        self.substitute_pid = substitute_pid

    def __copy__(self) -> "TemplateStore":
        """
        Create a copy of the TemplateStore.

        Returns
        -------
        TemplateStore
            A copy of the current TemplateStore.
        """
        templates_copy: dict[str, str | Template] = {}
        for key, value in self.templates.items():
            templates_copy[key] = Template(value.template)
        return TemplateStore(
            templates=templates_copy,
            substitutes=self.substitutes.copy(),
            substitute_timestamp=self.substitute_timestamp,
            substitute_pid=self.substitute_pid,
        )

    def __getitem__(self, template_name: str) -> str:
        """
        Get the template string with substitutions applied.

        Parameters
        ----------
        template_name : str
            Name of the template to retrieve.

        Returns
        -------
        str
            The template string with substitutions applied.
        """
        delete_timestamp = False
        if self.substitute_timestamp and ("timestamp" not in self.substitutes):
            self.substitutes["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            delete_timestamp = True
        delete_pid = False
        if self.substitute_pid and ("pid" not in self.substitutes):
            self.substitutes["pid"] = str(os.getpid())
            delete_pid = True
        substituted_str = self.templates[template_name].substitute(self.substitutes)
        if delete_timestamp:
            del self.substitutes["timestamp"]
        if delete_pid:
            del self.substitutes["pid"]
        return substituted_str

    def complete(self, template_name: str) -> str:
        """
        Complete the template by performing substitutions.

        Parameters
        ----------
        template_name : str
            Name of the template to complete.

        Returns
        -------
        str
            The completed template string.
        """
        template_str = self[template_name]
        # ToDo: template.get_identifiers() is available in Python 3.11
        if "$" in template_str:
            raise ValueError("Template not completely substituted.")
        return template_str

    def __contains__(self, key: str) -> bool:
        """
        Check if a template exists.

        Parameters
        ----------
        key : str
            Name of the template.

        Returns
        -------
        bool
            True if the template exists, False otherwise.
        """
        return key in self.templates

    def add(self, key: str, value: str | Template) -> None:
        """
        Add a new template.

        Parameters
        ----------
        key : str
            Name of the template.
        value : str | Template
            The template string or Template object.
        """
        if not isinstance(value, Template):
            self.templates[key] = Template(value)
        else:
            self.templates[key] = value

    def add_substitutes(self, **kwargs: str) -> None:
        r"""
        Add new substitution values.

        Parameters
        ----------
        \*\*kwargs : dict[str, str]
            Key-value pairs for substitution.
        """
        for key, value in kwargs.items():
            self.substitutes[key] = value


def unique_hex_digest(unique_elements: Any, length: int = 8) -> str:
    """
    Generate a unique hexadecimal digest based on the input elements.

    Parameters
    ----------
    unique_elements : Any
        The elements to generate a unique digest for. They must have a unique string representation.
    length : int
        The length of the hexadecimal digest truncation to return.

    Returns
    -------
    str
        A hexadecimal digest string of the specified length.
    """
    return hashlib.sha256(str(unique_elements).encode()).hexdigest()[0:length]


class ActionScheduler:
    """
    Class to schedule actions based on steps.

    Parameters
    ----------
    steps : list[int] | None
        List of specific steps to trigger the action.
    every : int | None
        Trigger the action every 'every' steps.
    every_progression : list[tuple[int, int]] | list[list[int]] | None
        List of tuples specifying step progressions for triggering the action.
    """

    def __init__(
        self,
        steps: list[int] | None = None,
        every: int | None = None,
        every_progression: list[tuple[int, int]] | list[list[int]] | None = None,
    ) -> None:
        self.steps = steps
        self.every = every
        self.every_progression = every_progression

    def __call__(self, step: int) -> bool:
        """
        Check if the action should be triggered for the given step.

        Parameters
        ----------
        step : int
            The current step.

        Returns
        -------
        bool
            True if the action should be triggered, False otherwise.
        """
        if (self.steps is not None) and (step in self.steps):
            return True
        if (self.every is not None) and (step % self.every == 0):
            return True
        if self.every_progression is not None:
            every = None
            for progression in self.every_progression:
                if len(progression) != 2:
                    raise ValueError("Each progression must be a tuple or list of length 2.")
                step_from, step_every = progression
                if (step < step_from) and (every is not None):
                    continue
                if step >= step_from:
                    every = step_every
            if (every is not None) and (step % every == 0):
                return True
        return False


def load_from_str(input_str: str, convert_fn: Callable[[str], Any] | None = None, allow_none: bool = False) -> Any:
    """
    Load a value from a string, optionally converting it using a provided function.

    Parameters
    ----------
    input_str : str
        The input string to load the value from.
    convert_fn : Callable[[str], Any] | None
        A function to convert the input string to the desired type. If None, no conversion is performed.
    allow_none : bool
        If True, the string "none" (case-insensitive) will be converted to None.

    Returns
    -------
    Any
        The loaded value, either converted or as the original string.
    """
    if convert_fn is None:
        return input_str
    if allow_none and (input_str.lower() == "none"):
        return None
    return convert_fn(input_str)
