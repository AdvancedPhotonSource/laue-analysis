#!/bin/bash
# Helper script to clean and rebuild during development

echo "Cleaning C binaries..."
cd src/laueanalysis/indexing/src/peaksearch && make clean
cd ../euler && make clean
cd ../pixels2qs && make clean
cd ../../../reconstruct/source/recon_cpu && make clean

echo "Reinstalling package..."
cd /net/s34data/export/s34data1/LauePortal/src/laue-analysis
pip install .

echo "Done!"
