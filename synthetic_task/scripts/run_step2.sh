#!/bin/bash

# Define the different values of beta you want to run
# betas=(0 0.001 0.01 0.1 0.5 1.0)
betas=(5.0 10.0 50.0 100.0 300.0 500.0 1000.0)

# Loop through each beta value
for beta in "${betas[@]}"
do
    echo "Running step2disen.py with beta=$beta"
    python step2disen.py --embed_dim=100 --device=0,1,2,3,4,5,6,7 --num_epoch_s2=30 --beta_start=$beta
    echo "Finished running for beta=$beta"
done

echo "All beta runs completed!"
