from flask import jsonify


def post_greeting(name):
    """
    Handle POST request for greeting endpoint
    """
    return f"Hello, {name}!"


def get_pet():
    return {"id": 1, "name": "Rex", "species": "dog"}
