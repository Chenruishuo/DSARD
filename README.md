# DSARD: DBSCAN-based Solar Active Regions Detection
This is the codebase of the paper *Statistical Analyses of Solar Active Regions in SDO/HMI Magnetograms detected by Unsupervised Machine Learning Method DSARD*.

We propose DSARD (DBSCAN-based Solar Active Regions Detection), an unsupervised machine learning method for the efficient, pixel-level automatic identification of solar active regions (ARs) in magnetograms. It involves an initial thresholding of magnetic field intensities, a two-stage DBSCAN clustering process to identify and refine ARs, and a final integration step that merges regions based on magnetic polarity and proximity while filtering out noise.<br>

<div align="center">
<img src="pictures/HMI20221112/after_merge.png" width="80%" alt="Solar Active Regions">
</div>

## How to use the code
- AR_detection_model.py: The basic model for detecting AR using SDO's HMI data. You can utilize it as a demo by changing 'data_item'.<br>
- AR_Statistic_get.py: You can run this code to get AR statistics. Settings are define at the beginning of the code.<br>
- AR_region_growth.py: We reproduce the previous results of using the region growth algorithm to identify the solar active regions.([Jie Zhang et al.](https://iopscience.iop.org/article/10.1088/0004-637X/723/2/1006))<br>
- AR_csv_data_2010_2023.csv records the information of solar active regions we detected with our algorithm during [2010,2023].

### Installation:<br>
```
conda env create --file environment.yaml
conda activate DSARD
pip install -r requirements.txt
cd DSARD
```

### Run demo:<br>
```
python -u AR_detection_model.py
```

## Citing
This repository is archived on Zenodo with [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14222292.svg)](https://doi.org/10.5281/zenodo.14222292)