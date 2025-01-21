import pathlib
from flask import Flask
from connexion.frameworks import flask as flask_utils
from a2wsgi import WSGIMiddleware
from connexion import FlaskApp, utils
from typing import Callable, Optional, Sequence, Union
from connexion.options import SwaggerUIOptions
from flask_cors import CORS
import re


class AcctechApi(FlaskApp):
    def __init__(
        self,
        import_name: str = "",
        specification_dir: str = "specifications",
        arguments: Optional[dict] = None,
        resolver: Callable = None,
        extra_files: Optional[Sequence[Union[str, pathlib.Path]]] = None,
        options: Optional[dict] = None,
        skip_error_handlers: bool = False,
        modules: Optional[Sequence[str]] = None,
    ):
        self.swagger_ui_options = SwaggerUIOptions(
            swagger_ui_path="/docs",
        )
        self._specification_dir = specification_dir
        super().__init__(
            import_name=import_name,
            specification_dir=specification_dir,
            arguments=arguments,
            resolver=resolver,
            swagger_ui_options=self.swagger_ui_options,
        )
        self.modules = modules
        # to stisfy the type checker
        self.extra_files = extra_files
        self.options = options
        self.skip_error_handlers = skip_error_handlers

    def init_app(self, flask_app: Flask):
        self.app = flask_app

        # Setup the user flask app to match the one in connexion
        self._setup_user_flask_app()

        # checks Cors
        if self.app.config.get("FLASK_ENABLE_CORS", True):
            CORS(self.app)

        # Register the services
        self._register_services()

        def wrapped_run(*args, **kwargs):
            return self.run(*args, **kwargs)

        flask_app.run = wrapped_run

    def add_api(self, specification: str, **kwargs):
        super().add_api(specification, **kwargs)
        if self.modules:
            for module in self.modules:
                utils.import_module_from_spec(module)

    def _setup_user_flask_app(self):
        if self.app is None:
            raise ValueError("Flask app is not initialized")

        # Reconfigure the flask app as we override it with the above self.app = flask_app
        # These settings comes from connexion v3 class FlaskASGIApp(SpecMiddleware)
        self.app.json = flask_utils.FlaskJSONProvider(self.app)
        self.app.json = flask_utils.FlaskJSONProvider(self.app)
        self.app.url_map.converters["float"] = flask_utils.NumberConverter
        self.app.url_map.converters["int"] = flask_utils.IntegerConverter
        self.app.config["PROPAGATE_EXCEPTIONS"] = True
        self.app.config["TRAP_BAD_REQUEST_ERRORS"] = True
        self.app.config["TRAP_HTTP_EXCEPTIONS"] = True
        self.asgi_app = WSGIMiddleware(self.app.wsgi_app)

        self.app.extensions = getattr(self.app, "extensions", {})
        self.app.extensions["acctech_api"] = {}
        self.app.extensions["connexion"] = self

    def _register_services(self):
        # Try to register the swagger file
        if "API_SERVICE" not in self.app.config:
            print(
                "Warning, API_SERVICE not found in the app config. You will need to call add_api manually."
            )
            return
        if not isinstance(self.app.config["API_SERVICE"], str):
            raise ValueError(
                "API_SERVICE must be a dictionary, for instance /v0:example.yaml,/v1:example2.yaml"
            )
        # TODO: add something to properly validate the API_SERVICE
        for service in self.app.config["API_SERVICE"].split(","):
            if not self.is_valid_spec_path(service):
                raise ValueError(
                    f"API_SERVICE must be in the format /v0:example.yaml,/v1:example2.yaml. {service} is invalid"
                )

            # Load and parse specification file
            version, spec = service.split(":")
            spec_path = pathlib.Path(self._specification_dir) / spec
            with open(spec_path) as f:
                import yaml

                specification = yaml.safe_load(f)

            # Extract version from specification
            api_version = specification.get("info", {}).get("version")
            print(f"API Version: {api_version}")

            # maybe just do self.add_api(spec, base_path=version)
            self.app.extensions["connexion"].add_api(
                spec, base_path=version, swagger_ui_options=self.swagger_ui_options
            )

    @staticmethod
    def is_valid_spec_path(path: str) -> bool:
        pattern = r"^/v\d+:[\w-]+\.yaml$"
        return bool(re.match(pattern, path))
