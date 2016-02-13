# IRS - Image Recognition System

Developed as main part of the proseminar (pratical part) of the "Advanced Image
Processing & Computer Vision" class at the Department of Computer Sciences,
University of Salzburg, Austria.

## Requirements

The code was tested using the following setup:

* 4x3.60 GHz AMD FX-4100 CPU, 12GB main memory, Linux Mint 17.
* `python` in version 2.7.6

### Remarks

The k-means++ clustering uses 4 cores in parallel (by argument), so better make
sure to change this parameter (in `src/kmeanspp/kmeanspp.py`) or have 4 cores
available (I did not investigate how the scikit-learn kMeans behaves if there
are less than 4 cores available).
