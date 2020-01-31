#!/bin/bash

for file in simulations/*.json; do python acprmodel.py "$file"; done
