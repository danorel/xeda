from setuptools import find_packages, setup

setup(
    name="xeda",
    packages=find_packages(exclude=["xeda_tests"]),
    install_requires=["dagster", "dagster-cloud"],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
