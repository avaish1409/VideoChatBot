from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='VideoChatBot',
    version='0.0.2',
    author='Anirudh Vaish',
    author_email='anirudhvaish147@gmail.com',
    url='https://github.com/avaish1409/VideoChatBot',
    description='A python based command line tool to compare Github Users or Repositories.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GNU General Public License v3 (GPLv3)',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    project_urls={
        "Bug Tracker": "https://github.com/avaish1409/VideoChatBot/issues",
        'Source': 'https://github.com/avaish1409/VideoChatBot'
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'license :: GNU General Public License v3 (GPLv3)'
    ],
    keywords='bot videochatbot videobot chatbot python package git github',
    install_requires=requirements,
    python_requires=">=3.6",
    zip_safe=False
)