from setuptools import find_packages, setup

setup(
    name='Agent-Smith',
    packages=find_packages(include=['AgentSmith']),
    version='0.1.0',
    description='Python Reinforcement Learning Library with PyTorch',
    author='Jason Pul',
    license='MIT',
    install_requires=['torch==1.9.0', 'plotly==5.1.0', 'matplotlib==3.4.2'],
)