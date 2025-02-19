#!/bin/bash

a_s=(0.01 0.1 1 10 100 1000)

for a in "${a_s[@]}"
do
    echo "Running jointdisen_mp.py with a=$a"
    python jointdisen_mp.py --embed_dim=100 --device=0,1,2,3,4,5,6,7 --start_a=$a 
done

echo "All a runs completed!"
