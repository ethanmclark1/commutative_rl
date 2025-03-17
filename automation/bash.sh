#!/bin/bash

SEED=(42)
N_INSTANCES=5
APPROACHES=("QTable" "TripleDataQTable" "SuperActionQTable" "CombinedRewardQTable" "HashMapQTable")

STEP_VALUE=(1)
OVER_PENALTY=(2)
MAX_NOISE=(10)

ALPHA=(0.2)
EPSILON=(0.25) 
GAMMA=(0.99)

conda activate commutative_rl

total_jobs=$(( ${#SEED[@]} * (N_INSTANCES) * ${#APPROACHES[@]} * ${#STEP_VALUE[@]} * ${#OVER_PENALTY[@]} *
             ${#MAX_NOISE[@]} * ${#ALPHA[@]} * ${#EPSILON[@]} * ${#GAMMA[@]} ))

current_job=0
for seed in "${SEED[@]}"; do
    for problem_instance in $(seq 0 $((N_INSTANCES - 1))); do
        for approach in "${APPROACHES[@]}"; do
            for step_value in "${STEP_VALUE[@]}"; do
                for over_penalty in "${OVER_PENALTY[@]}"; do
                    for max_noise in "${MAX_NOISE[@]}"; do
                        for alpha in "${ALPHA[@]}"; do
                            for epsilon in "${EPSILON[@]}"; do
                                for gamma in "${GAMMA[@]}"; do
                                    current_job=$((current_job + 1))

                                    echo "Starting job $current_job of $total_jobs"

                                    args="--approaches $approach --seed $seed --problem_instances instance_$problem_instance --step_value $step_value --over_penalty $over_penalty --max_noise $max_noise --alpha $alpha --epsilon $epsilon --gamma $gamma"                                    
                                    python commutative_rl/main.py $args 
                                    echo "Completed job $current_job: $job_name"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All jobs completed successfully!"
