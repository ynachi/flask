# from connexion import FlaskApp

# app = FlaskApp(__name__, specification_dir="specifications")


# app.add_api("openapi.yaml")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)

from flask import Flask
from acctech_api.acctech_api import AcctechApi

app = Flask(__name__)

app.config["API_SERVICE"] = "/v0:openapi.yaml"
api = AcctechApi(specification_dir="specifications")
api.init_app(app)

if __name__ == "__main__":
    print("************** Registered URLs: ***************", app.url_map)
    app.run(host="0.0.0.0", port=8080)
