import setuptools

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

with open("requirements.txt", encoding="utf-8") as file:
    install_requires = file.read()

setuptools.setup(
    name="fastlabel",
    version="0.9.1",
    author="eisuke-ueta",
    author_email="eisuke.ueta@fastlabel.ai",
    description="The official Python SDK for FastLabel API, the Data Platform for AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    include_package_data=True,
)
