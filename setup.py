"""Build configuration for Cython-compiled momentum modules."""

from Cython.Build import cythonize
from setuptools import Extension, setup

cython_extensions = [
    Extension(
        name="momentum._assessments_cy",
        sources=["momentum/assessments_cy.pyx"],
        language="c",
    ),
    Extension(
        name="momentum._timer_cy",
        sources=["momentum/timer_cy.pyx"],
        language="c",
    ),
    Extension(
        name="momentum._charts_cy",
        sources=["momentum/charts_cy.pyx"],
        language="c",
    ),
]

setup(
    ext_modules=cythonize(
        cython_extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)
