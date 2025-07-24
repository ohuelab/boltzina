from setuptools import setup, find_packages

setup(
    name="boltzina",
    version="0.1.0",
    description="Vina docking + Boltz scoring pipeline",
    packages=find_packages(),
    install_requires=[
        "rdkit",
        "pandas",
        "numpy",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "boltzina=boltzina_main:main",
        ],
    },
)