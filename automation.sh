#!/bin/bash

SEED=(42)
N_INSTANCES=8
APPROACHES=("TraditionalDQN" "CommutativeDQN")

GRID_DIMS=("32x32")
N_STARTS=(5)
N_GOALS=(5)
N_BRIDGES=(40)
N_EPISODE_STEPS=(75)
CONFIGS_TO_CONSIDER=(25)
ACTION_SUCCESS_RATE=(0.50)

ALPHA=(0.0002)
EPSILON=(0.325) 
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
    local grid_dims=$4
    local n_starts=$5
    local n_goals=$6
    local n_bridges=$7
    local n_episode_steps=$9
    local configs_to_consider=${10}
    local action_success_rate=${11}
    local alpha=${12}
    local epsilon=${13}
    local gamma=${14}
    local batch_size=${15}
    local hidden_dims=${16}
    local buffer_size=${17}
    local target_update_freq=${18}
    local dropout=${19}
    echo "${approach}_${seed}_${problem_instance}_${grid_dims}_${n_starts}_${n_goals}_${n_bridges}_${n_episode_steps}_${configs_to_consider}_${action_success_rate}_${alpha}_${epsilon}_${gamma}_${batch_size}_${hidden_dims}_${buffer_size}_${target_update_freq}_${dropout}.sh"
}

for seed in "${SEED[@]}"; do
    for problem_instance in $(seq 0 $((N_INSTANCES - 1))); do
        for approach in "${APPROACHES[@]}"; do
            for grid_dims in "${GRID_DIMS[@]}"; do
                for n_starts in "${N_STARTS[@]}"; do
                    for n_goals in "${N_GOALS[@]}"; do
                        for n_bridges in "${N_BRIDGES[@]}"; do
                            for n_episode_steps in "${N_EPISODE_STEPS[@]}"; do
                                for configs_to_consider in "${CONFIGS_TO_CONSIDER[@]}"; do
                                    for action_success_rate in "${ACTION_SUCCESS_RATE[@]}"; do
                                        for alpha in "${ALPHA[@]}"; do
                                            for epsilon in "${EPSILON[@]}"; do
                                                for gamma in "${GAMMA[@]}"; do
                                                    for batch_size in "${BATCH_SIZE[@]}"; do
                                                        for hidden_dims in "${HIDDEN_DIMS[@]}"; do
                                                            for buffer_size in "${BUFFER_SIZE[@]}"; do
                                                                for target_update_freq in "${TARGET_UPDATE_FREQ[@]}"; do
                                                                    for dropout in "${DROPOUT[@]}"; do
                                                                        args="--approaches $approach --seed $seed --problem_instances instance_$problem_instance --grid_dims $grid_dims --n_starts $n_starts --n_goals $n_goals --n_bridges $n_bridges --n_episode_steps $n_episode_steps --configs_to_consider $configs_to_consider --action_success_rate $action_success_rate --alpha $alpha --epsilon $epsilon --gamma $gamma --batch_size $batch_size --hidden_dims $hidden_dims --buffer_size $buffer_size --target_update_freq $target_update_freq --dropout $dropout"
                                                                        job_name=$(generate_job_name "$approach" "$seed" "$problem_instance" "$grid_dims" "$n_starts" "$n_goals" "$n_bridges" "$n_episode_steps" "$configs_to_consider" "$action_success_rate" "$alpha" "$epsilon" "$gamma" "$batch_size""$hidden_dims" "$buffer_size" "$target_update_freq" "$dropout")
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
    done
done

for job_script in "$TMP_DIR"/*.sh; do
    sbatch "$job_script"
    sleep 0.5
done
