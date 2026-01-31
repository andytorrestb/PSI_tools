#!/bin/bash
python3 scripts/quarter_cylinder_mesh.py -c case/scripts/quarter_cylinder_mesh_config.json -o case/constant/polyMesh/
trash constant/polyMesh
blockMesh
snappyHexMesh -overwrite
mkdir 0/
cp -r 0.orig/* 0/
dsmcInitialise
# blockMesh
# snappyHexMesh -overwrite
# # trash constant/polyMesh/
# # mv 2-e-6/polyMesh constant/
# # trash -rf *-e*
# mkdir 0/
# cp -r 0.orig/* 0/
# dsmcInitialise

