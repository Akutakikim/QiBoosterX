from setuptools import setup, find_packages

setup(
    name='qiboosterx',
    version='0.1.0',
    author='Akik Forazi',
    author_email='akikforaziinchaos@gmail.com',
    description='Quantum-inspired Super AI Booster for low-end devices.',
    url_git='
    url_Facebook=
    url_Instagram=
    url_X=
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'rich',
    ],
    entry_points={
        'console_scripts': [
            'qiboosterx=qiboosterx.qiboostx_cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)