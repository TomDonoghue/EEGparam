# EEG FOOOF

Project repository, part of the `Parameterizing Neural Power Spectra` project.

[![Preprint](https://img.shields.io/badge/preprint-10.1101/299859-informational.svg)](https://doi.org/10.1101/299859)

## Overview

This repository applies the [FOOOF](http://github.com/fooof-tools/fooof) algorithm to EEG data.

Analyses include:
- comparing resting state measures between young and old subjects
- task analyses of a working memory task, predicting behaviour using FOOOF outputs

## Project Guide

You can follow along with this project by looking through everything in the `notebooks`.

## Reference

The analyses in this repository were done as part of the
[`Parameterizing Neural Power Spectra`](https://doi.org/10.1101/299859).

A full guide to analyses including in this paper is availabble
[here](https://github.com/fooof-tools/Paper).

## Requirements

This project was written in Python 3 and requires Python >= 3.7 to run.

Dependencies include 3rd party scientific Python packages:
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [pandas](https://github.com/pandas-dev/pandas)
- [statsmodels](https://github.com/statsmodels/statsmodels)
- [matplotlib](https://github.com/matplotlib/matplotlib)

In addition to general scientific Python packages, this analysis requires:

- [mne](https://github.com/mne-tools/mne-python)
- [autoreject](https://github.com/autoreject/autoreject)
- [fooof](https://github.com/fooof-tools/fooof)

## Repository Layout

This project repository is set up in the following way:

- `code/` contains custom code for this analysis
- `figures/` contains all the figures produced from during this analysis
- `notebooks/` is a collection of Jupyter notebooks that step through the project and create the figures
- `scripts/` contains stand alone scripts that run parts of the project

## Data

This project uses and electroencephalography (EEG) dataset, with a working memory task.
