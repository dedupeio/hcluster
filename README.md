# hcluster

[![Tests Passing](https://github.com/dedupeio/dedupe/workflows/tests/badge.svg)](https://github.com/dedupeio/dedupe/actions?query=workflow%3Atests)

This library provides Python functions for hierarchical clustering. Its features
include
* generating hierarchical clusters from distance matrices
* computing distance matrices from observation vectors
* computing statistics on clusters
* cutting linkages to generate flat clusters
* and visualizing clusters with dendrograms.
The interface is very similar to MATLAB's Statistics Toolbox API to make code
easier to port from MATLAB to Python/Numpy. The core implementation of this
library is in C for efficiency.

It is a fork of clustering and distance functions from the [scipy](http://www.scipy.org/) that removes all the
dependencies on scipy. It [preserves the API of hcluster 0.2](http://hcluster.damianeads.com/cluster.html).

