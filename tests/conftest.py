import pytest

from rvm.fixture import create_fixture


@pytest.fixture
def fixture_bundle(tmp_path):
    return create_fixture(tmp_path / "synthetic")
