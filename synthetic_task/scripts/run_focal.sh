#!/bin/bash

a_s=(0.01 0.1 1 10 100 1000)

for a in "${a_s[@]}"
do
    echo "Running focal_mp.py with a=$a"
    python focal_mp.py --embed_dim=100 --device=0,1,2,3 --start_a=$a 
    echo "Finished running for a=$a"
done

echo "All a runs completed!"
