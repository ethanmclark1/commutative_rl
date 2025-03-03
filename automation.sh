#!/bin/bash

SEED=(42)
N_INSTANCES=8
APPROACHES=("TraditionalDQN" "CommutativeDQN")

GRID_DIMS=("32x32")
N_STARTS=(3)
N_GOALS=(3)
N_BRIDGES=(20)
N_EPISODE_STEPS=(25)
ACTION_SUCCESS_RATE=(0.65)
UTILITY_SCALE=(2)
TERMINAL_REWARD=(30)
BRIDGE_COST_LB=(2)
BRIDGE_COST_UB=(10)
DUPLICATE_BRIDGE_PENALTY=(250)

N_WARMUP_EPISODES=(25 50)
ALPHA=(0.001 0.0005)   
DROPOUT=(0.2)
EPSILON=(0.25 0.3)
GAMMA=(0.99)
BATCH_SIZE=(4 8)
HIDDEN_DIMS=(128)
BUFFER_SIZE=(5000)
TARGET_UPDATE_FREQ=(10)

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
    local n_episode_steps=$8
    local action_success_rate=$9
    local utility_scale=${10}
    local terminal_reward=${11}
    local bridge_cost_lb=${12}
    local bridge_cost_ub=${13}
    local duplicate_bridge_penalty=${14}
    local n_warmup_episodes=${15}
    local alpha=${16}
    local dropout=${17}
    local epsilon=${18}
    local gamma=${19}
    local batch_size=${20}
    local hidden_dims=${21}
    local buffer_size=${22}
    local target_update_freq=${23}
    echo "${approach}_${seed}_${problem_instance}_${grid_dims}_${n_starts}_${n_goals}_${n_bridges}_${n_episode_steps}_${action_success_rate}_${utility_scale}_${terminal_reward}_${bridge_cost_lb}_${bridge_cost_ub}_${duplicate_action_cost}_${n_warmup_episodes}_${alpha}_${dropout}_${epsilon}_${gamma}_${batch_size}_${hidden_dims}_${buffer_size}_${target_update_freq}.sh"
}

for seed in "${SEED[@]}"; do
    for problem_instance in $(seq 0 $((N_INSTANCES - 1))); do
        for approach in "${APPROACHES[@]}"; do
            for grid_dims in "${GRID_DIMS[@]}"; do
                for n_starts in "${N_STARTS[@]}"; do
                    for n_goals in "${N_GOALS[@]}"; do
                        for n_bridges in "${N_BRIDGES[@]}"; do
                            for n_episode_steps in "${N_EPISODE_STEPS[@]}"; do
                                for action_success_rate in "${ACTION_SUCCESS_RATE[@]}"; do
                                    for utility_scale in "${UTILITY_SCALE[@]}"; do
                                        for terminal_reward in "${TERMINAL_REWARD[@]}"; do
                                            for bridge_cost_lb in "${BRIDGE_COST_LB[@]}"; do
                                                for bridge_cost_ub in "${BRIDGE_COST_UB[@]}"; do
                                                    for duplicate_bridge_penalty in "${DUPLICATE_BRIDGE_PENALTY[@]}"; do
                                                        for n_warmup_episodes in "${N_WARMUP_EPISODES[@]}"; do
                                                            for alpha in "${ALPHA[@]}"; do
                                                                for dropout in "${DROPOUT[@]}"; do
                                                                    for epsilon in "${EPSILON[@]}"; do
                                                                        for gamma in "${GAMMA[@]}"; do
                                                                            for batch_size in "${BATCH_SIZE[@]}"; do
                                                                                for hidden_dims in "${HIDDEN_DIMS[@]}"; do
                                                                                    for buffer_size in "${BUFFER_SIZE[@]}"; do
                                                                                        for target_update_freq in "${TARGET_UPDATE_FREQ[@]}"; do
                                                                                            args="--approaches $approach --seed $seed --problem_instances instance_$problem_instance --grid_dims $grid_dims --n_starts $n_starts --n_goals $n_goals --n_bridges $n_bridges --n_episode_steps $n_episode_steps --action_success_rate $action_success_rate --utility_scale $utility_scale --terminal_reward $terminal_reward --bridge_cost_lb $bridge_cost_lb --bridge_cost_ub $bridge_cost_ub --duplicate_bridge_penalty $duplicate_bridge_penalty --n_warmup_episodes $n_warmup_episodes --alpha $alpha --dropout $dropout --epsilon $epsilon --gamma $gamma --batch_size $batch_size --hidden_dims $hidden_dims --buffer_size $buffer_size --target_update_freq $target_update_freq"
                                                                                            job_name=$(generate_job_name "$approach" "$seed" "$problem_instance" "$grid_dims" "$n_starts" "$n_goals" "$n_bridges" "$n_episode_steps" "$action_success_rate" "$utility_scale" "$terminal_reward" "$bridge_cost_lb" "$bridge_cost_ub" "$duplicate_bridge_penalty" "$n_warmup_episodes" "$alpha" "$dropout" "$epsilon" "$gamma" "$batch_size" "$hidden_dims" "$buffer_size" "$target_update_freq")
                                                                                            cat <<EOT > "$TMP_DIR/$job_name"
#!/bin/bash

#SBATCH -N 1                               # Number of nodes
#SBATCH -c 16                              # Number of cores
#SBATCH -t 2-00:00:00                      # time in d-hh:mm:ss
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
                done
            done
        done
    done
done

for job_script in "$TMP_DIR"/*.sh; do
    sbatch "$job_script"
    sleep 0.5
done
