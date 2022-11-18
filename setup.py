from distutils.core import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name="pytorch2tikz",
    version="1.0",
    packages=["pytorch2tikz"],
    install_requires=["torch", "torchvision"],
    author="Jannes Magnusson",
    author_email="jannes@magnusson.berlin",
    description="Automatically plots pytorch modules",
    long_description=readme(),
    url="https://github.com/"
)
