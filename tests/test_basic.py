import pytest
from flask import Flask
from acctech_api.acctech_api import AcctechApi
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config["API_SERVICE"] = "/v0:openapi.yaml"

    # Create API instance with correct specification directory
    specification_dir = os.path.join(os.path.dirname(__file__), "..", "specifications")
    api = AcctechApi(import_name=__name__, specification_dir=specification_dir)

    # Initialize the app and register routes
    api.init_app(app)

    with api.test_client() as client:
        yield client


def test_post_greeting(client):
    response = client.post("/v0/greeting/World")
    assert response.status_code == 200
    # assert response.data.decode() == '"Hello, World!"'


# def test_get_pet(client):
#     response = client.get("/v0")
#     assert response.status_code == 200
#     assert response.json == {"id": 1, "name": "Rex", "species": "dog"}
