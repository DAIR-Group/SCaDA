# Statistical Inference for k-means Clustering after Domain Adaptation
This package provides a statistical inference framework for k-means clustering after domain adaptation (DA). It leverages the SI framework and employs a divide-and-conquer strategy to efficiently compute the p-value of selected features. Our method ensures reliable feature selection by controlling the false positive rate (FPR) while simultaneously maximizing the true positive rate (TPR), effectively reducing the false negative rate (FNR).

## Environment Setup
```bash
pip install -r requirements.txt
```

## Usage
We provide several Jupyter notebooks demonstrating how to use the SCaDA.
- Example for computing _p_-values for _k_-means clustering after DA: [`ex1_compute_pvalue.ipynb`](ex1_compute_pvalue.ipynb)
- Check the uniformity of the pivot: [`ex2_validity_of_pvalue.ipynb`](ex2_validity_of_pvalue.ipynb)

## PyPI package
The `SCaDA` is available on the PyPI and can be installed as follows:
```bash
pip install PySCaDA
```