import pytest

from madewithml.data import CustomPreprocessor

# to run test in "code" module:
# python3 -m pytest                                          # all tests
# python3 -m pytest tests/code                               # tests under a directory
# python3 -m pytest tests/code/test_predict.py               # tests for a single file
# python3 -m pytest tests/code -m "not training"             # run tests besides those marked with `training` (@pytest.mark.training)

# to run tests with Coverage library
# python3 -m pytest tests/code --cov madewithml --cov-report html --disable-warnings                        # all
# python3 -m pytest tests/code -m "not training" --cov madewithml --cov-report html --disable-warnings      # exlude training marker

# to visualize test coverages
# coverage report -m            # through terminal
# htmlcov/index.html            # through html file

@pytest.fixture
def dataset_loc():
    return "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"


@pytest.fixture
def preprocessor():
    return CustomPreprocessor()
