#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(../LinearResponseVariationalBayes.py)
jupyter notebook
