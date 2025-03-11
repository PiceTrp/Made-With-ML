import pytest

from madewithml import predict
from madewithml.predict import TorchPredictor

# Model tests
# export EXPERIMENT_NAME="llm-1741243372"           # get it in mlflow dashboard
# export RUN_ID=$(python madewithml/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
# pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings


def pytest_addoption(parser):
    parser.addoption("--run-id", action="store", default=None, help="Run ID of model to use.")


@pytest.fixture(scope="module")
def run_id(request):
    return request.config.getoption("--run-id")


@pytest.fixture(scope="module")
def predictor(run_id):
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    return predictor
