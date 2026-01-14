"""General utilities."""

import datetime
import os
from string import Template


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

    def __init__(
        self,
        templates: dict[str, str | Template] | None = None,
        substitutes: dict[str, str] | None = None,
        substitute_timestamp: bool = True,
        substitute_pid: bool = True,
    ) -> None:
        if templates is None:
            self.templates = {}
        else:
            d = {}
            for key, value in templates.items():
                if not isinstance(value, Template):
                    if not isinstance(value, str):
                        raise ValueError("TemplateStore: value must be a string or a Template.")
                    d[key] = Template(value)
                else:
                    if not isinstance(value.template, str):
                        raise ValueError("TemplateStore: value.template must be a string.")
                    d[key] = value
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
        templates = {}
        for key, value in self.templates.items():
            if not isinstance(value.template, str):
                raise ValueError("TemplateStore: value.template must be a string for copy operation.")
            templates[key] = Template(value.template)
        substitutes = {}
        for key, value in self.substitutes.items():
            if not isinstance(value, str):
                raise ValueError("TemplateStore: substitute value must be a string for copy operation.")
            substitutes[key] = value
        return TemplateStore(
            templates=templates,
            substitutes=substitutes,
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

    def add_substitutes(self, **kwargs) -> None:
        """
        Add new substitution values.

        Parameters
        ----------
        **kwargs
            Key-value pairs for substitution.
        """
        for key, value in kwargs.items():
            self.substitutes[key] = value
