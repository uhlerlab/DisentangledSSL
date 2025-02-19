#!/bin/bash

echo "Running dmvae_mp.py"
python dmvae_mp.py --device=0,1,2,3,4,5,6,7
echo "Finished running"
