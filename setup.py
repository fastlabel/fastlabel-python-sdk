import setuptools

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

with open("requirements.txt", encoding="utf-8") as file:
    install_requires = file.read()

setuptools.setup(
    name="fastlabel",
    author="eisuke-ueta",
    author_email="eisuke.ueta@fastlabel.ai",
    description="The official Python SDK for FastLabel API, the Data Platform for AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
    include_package_data=True,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
