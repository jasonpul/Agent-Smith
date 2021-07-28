from setuptools import find_packages, setup

setup(
    name='Agent-Smith',
    version='0.1.0',
    description='Python Reinforcement Learning Library with PyTorch',
    url='https://github.com/jasonpul/Agent-Smith',
    author='Jason Pul',
    license='MIT',
    packages=['AgentSmith'],
    include_package_data=True,
    install_requires=['torch==1.9.0', 'matplotlib==3.4.2'],
    keywords=['pytorch', 'dqn', 'a2c', 'reinforcement'],
)
