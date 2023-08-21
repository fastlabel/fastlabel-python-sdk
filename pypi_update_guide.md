# PyPI Update Guide

_Creating and deploying a new package version is easy_

## Prerequisites

1. Ensure you have a PyPI account created and are added as a Collaborator

2. Store PyPI API Token to GitHub Secrets  
   If you have already created GitHub Secret `PYPI_API_TOKEN`, skip this step.

   1. Get PyPI API Token

      1. Go to [PyPI Account Settings Page](https://pypi.org/manage/account/)
      2. Click `Add API Token` button in API Token section
      3. Enter the following
         - Token name: `GitHub Actions Token`
         - Scope: `Project: fastlabel`
      4. Click `Add Token` button and get API Token

   2. Store Token to GitHub Secrets
      1. Go to GitHub `fastlabel-python-sdk` repository
      2. Go to Settings > Secrets > Actions
      3. Click `New repository secret` and enter the following
         - Name: `PYPI_API_TOKEN`
         - Value: PyPI API Token

## Deployment Steps

**Step 1: Create a new release**

1. Click `Releases` label in `Code` tab and go to Releases page

2. Click `Draft a new release` button

3. Enter the following

   - Tag

     - Click `Choose a tag` select box
     - input [version](#version) (ex: `1.12.0`)
     - Click `Create new tag: x.x.x`

   - Target: main

   - Release title: `Release x.x.x` (ex: `Release 1.12.0`)

   - Fill in the description with reference to past releases

4. Click `Publish release` button

**Step 2: (Automatically) Execute GitHub Actions Workflow**

After creating a release, GitHub Actions Workflow will be triggered automatically.  
This workflow builds the SDK distribution and uploads it to PyPI.

If the workflow fails, follow these steps:

1. Fix the cause of the error
2. Remove release created in Step 1
3. Remove tag created in Step 1
4. Repeat from Step 1

**Step 3: Check out the PyPI page to ensure all looks good**

[https://pypi.org/project/fastlabel/](https://pypi.org/project/fastlabel/)

---

### Version

We use [semantic versioning](https://packaging.python.org/guides/distributing-packages-using-setuptools/#semantic-versioning-preferred).  
If you are adding a meaningful feature, bump the minor version.  
If you are fixing a bug, bump the incremental version.
