#!/bin/bash

SEED=(42)
N_INSTANCES=8
APPROACHES=("TraditionalDQN" "CommutativeDQN")

GRANULARITY=(0.25)
FAILED_PATH_COST=(0)
SAFE_AREA_MULTIPLIER=(2)
N_EPISODE_STEPS=(10)
CONFIGS_TO_CONSIDER=(10)

ALPHA=(0.0005)
EPSILON=(0.25) 
GAMMA=(0.99)
BATCH_SIZE=(4)
HIDDEN_DIMS=(128)
BUFFER_SIZE=(5000)
TARGET_UPDATE_FREQ=(10)
DROPOUT=(0.2)

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

generate_job_name() {
    local approach=$1
    local seed=$2
    local problem_instance=$3
    local granularity=$4
    local failed_path_cost=$5
    local safe_area_multiplier=$6
    local n_episode_steps=$7
    local configs_to_consider=$8
    local alpha=$9
    local epsilon=${10}
    local gamma=${11}
    local batch_size=${12}
    local hidden_dims=${13}
    local buffer_size=${14}
    local target_update_freq=${15}
    local dropout=${16}
    echo "${approach}_${seed}_${problem_instance}_${granularity}_${failed_path_cost}_${safe_area_multiplier}_${n_episode_steps}_${configs_to_consider}_${alpha}_${epsilon}_${gamma}_${batch_size}_${hidden_dims}_${buffer_size}_${target_update_freq}_${dropout}.sh"
}

for seed in "${SEED[@]}"; do
    for problem_instance in $(seq 0 $((N_INSTANCES - 1))); do
        for approach in "${APPROACHES[@]}"; do
            for granularity in "${GRANULARITY[@]}"; do
                for failed_path_cost in "${FAILED_PATH_COSTS[@]}"; do
                    for safe_area_multiplier in "${SAFE_AREA_MULTIPLIER[@]}"; do
                        for n_episode_steps in "${N_EPISODE_STEPS[@]}"; do
                            for configs_to_consider in "${CONFIGS_TO_CONSIDER[@]}"; do
                                for alpha in "${ALPHA[@]}"; do
                                    for epsilon in "${EPSILON[@]}"; do
                                        for gamma in "${GAMMA[@]}"; do
                                            for batch_size in "${BATCH_SIZE[@]}"; do
                                                for hidden_dims in "${HIDDEN_DIMS[@]}"; do
                                                    for buffer_size in "${BUFFER_SIZE[@]}"; do
                                                        for target_update_freq in "${TARGET_UPDATE_FREQ[@]}"; do
                                                            for dropout in "${DROPOUT[@]}"; do
                                                                args="--approaches $approach --seed $seed --problem_instances instance_$problem_instance --granularity $granularity --failed_path_cost $failed_path_cost --safe_area_multiplier $safe_area_multiplier --n_episode_steps $n_episode_steps --configs_to_consider $configs_to_consider --alpha $alpha --epsilon $epsilon --gamma $gamma --batch_size $batch_size --hidden_dims $hidden_dims --buffer_size $buffer_size --target_update_freq $target_update_freq --dropout $dropout"
                                                                job_name=$(generate_job_name "$approach" "$seed" "$problem_instance" "$granularity" "$failed_path_cost" "$safe_area_multiplier" "$n_episode_steps" "$configs_to_consider" "$alpha" "$epsilon" "$gamma" "$batch_size""$hidden_dims" "$buffer_size" "$target_update_freq" "$dropout")
                                                                cat <<EOT > "$TMP_DIR/$job_name"
#!/bin/bash

#SBATCH -N 1                               # Number of nodes
#SBATCH -c 16                              # Number of cores
#SBATCH -t 1-00:00:00                      # time in d-hh:mm:ss
#SBATCH -p general                         # partition
#SBATCH -q public                          # QOS
#SBATCH -o artifacts/${job_name%.sh}.out   # file to save job's STDOUT (%j = JobID)
#SBATCH -e artifacts/${job_name%.sh}.err   # file to save job's STDERR (%j = JobID)
#SBATCH --mail-type=None                   # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=None                      # Purge a job-submitting shell environment

module load mamba/latest
source activate commutative_rl
python commutative_rl/main.py $args

EOT
                                                            done
                                                        done        
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

for job_script in "$TMP_DIR"/*.sh; do
    sbatch "$job_script"
    sleep 0.5
done
