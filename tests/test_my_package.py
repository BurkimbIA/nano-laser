from importlib.metadata import version

from python_package_template import main


def test_version():
    assert version("python_package_template") == "0.1.0"


def test_main():
    assert main() == "Hello from python-package-template!"
