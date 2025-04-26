from setuptools import setup, find_packages

setup(
    name='qiboosterx',
    version='0.1.0',
    author='Akik Forazi',
    author_email='akikforaziinchaos@gmail.com',
    description='Quantum-inspired Super AI Booster for low-end devices.',
    long_description="""
QIBOOSTERX - Quantum-Inspired Booster for AI Projects.

Developed as part of the QIST AI Project Family, specifically for the 'MINDWISE' prototype (QIST-NANO).
Designed to empower low-end devices with highly efficient and scalable AI capabilities.
Upcoming projects include QAIT - Quantum Artificial Intelligence Toolkit. Project will include QIST AI Model Family.
Many more projects are in bound by me.

Thank you for using QIBOOSTERX. — Akik Forazi
    """,
    long_description_content_type="text/markdown",
    url_author_facebook="https://www.facebook.com/share/16BEFdtR3y/"
    url='https://github.com/Akik-Forazi/QiBoosterX',  # You can update this when you have GitHub repo
    project_urls={
        'Documentation': 'https://github.com/Akik-Forazi/QiBoosterX/wiki',
        'Source': 'https://github.com/Akik-Forazi/QiBoosterX',
        'Bug Tracker': 'https://github.com/Akik-Forazi/QiBoosterX/issues',
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    keywords='quantum AI, AI booster, low-end optimization, deep learning, efficient AI, quantum-inspired, qist, mindwise, qait',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'rich',
    ],
    include_package_data=True,
)
