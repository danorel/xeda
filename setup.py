from setuptools import find_packages, setup

setup(
    name="offline_pipeline_annotation",
    packages=find_packages(exclude=["offline_pipeline_annotation_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
