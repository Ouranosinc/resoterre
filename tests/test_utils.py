import copy
import time

import pytest

from resoterre import utils


def test_template_store_defaults():
    templates = utils.TemplateStore({"log_file": "/some/path/${timestamp}_test_${pid}.log"})
    log_file = templates["log_file"]
    assert "$" not in log_file
    time.sleep(1.1)
    log_file_2 = templates["log_file"]
    assert log_file_2 != log_file


def test_template_store_substitutes():
    templates = utils.TemplateStore({"figure_file": "/some/path/${experiment_name}/${figure_name}.png"})
    with pytest.raises(KeyError):
        _ = templates["figure_file"]
    templates.add_substitutes(experiment_name="experiment1", figure_name="figure1")
    figure_file = templates["figure_file"]
    assert figure_file == "/some/path/experiment1/figure1.png"


def test_template_store_copy():
    templates = utils.TemplateStore({"figure_file": "/some/path/${experiment_name}/${figure_name}.png"})
    templates.add_substitutes(experiment_name="experiment1", figure_name="figure1")
    templates_copy = copy.copy(templates)
    templates_copy.add_substitutes(experiment_name="experiment2", figure_name="figure2")
    figure_file_original = templates["figure_file"]
    figure_file_copy = templates_copy["figure_file"]
    assert figure_file_original == "/some/path/experiment1/figure1.png"
    assert figure_file_copy == "/some/path/experiment2/figure2.png"
