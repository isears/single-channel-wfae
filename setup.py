import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scwfae",
    version="0.0.1",
    author="Isaac Sears",
    author_email="isaac.j.sears@gmail.com",
    description="Single Channel Waveform Autoencoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"scwfae": "scwfae"},
    url="https://github.com/isears/scwfae",
    project_urls={
        "Bug Tracker": "https://github.com/isears/scwfae/issues",
    },
)
