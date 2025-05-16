from setuptools import setup, find_packages

setup(
    name="crypto-prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'stable-baselines3',
        'ta',
        'pyyaml'
    ]
) 