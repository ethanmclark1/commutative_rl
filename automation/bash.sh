#!/bin/bash

SEED=(42 23 6)
N_INSTANCES=5
APPROACHES=("QTable" "TripleDataQTable" "SuperActionQTable" "CombinedRewardQTable" "HashMapQTable")

ALPHA=(0.01)
EPSILON=(0.25) 
GAMMA=(0.99)

conda activate commutative_rl

total_jobs=$(( ${#SEED[@]} * (N_INSTANCES) * ${#APPROACHES[@]} * ${#ALPHA[@]} * ${#EPSILON[@]} * ${#GAMMA[@]} ))

current_job=0
for seed in "${SEED[@]}"; do
    for problem_instance in $(seq 0 $((N_INSTANCES - 1))); do
        for approach in "${APPROACHES[@]}"; do
            for alpha in "${ALPHA[@]}"; do
                for epsilon in "${EPSILON[@]}"; do
                    for gamma in "${GAMMA[@]}"; do
                        current_job=$((current_job + 1))

                        echo "Starting job $current_job of $total_jobs"

                        args="--approaches $approach --seed $seed --problem_instances instance_$problem_instance --alpha $alpha --epsilon $epsilon --gamma $gamma"                                    
                        python commutative_rl/main.py $args 
                        echo "Completed job $current_job: $job_name"
                    done
                done
            done
        done
    done
done

echo "All jobs completed successfully!"
