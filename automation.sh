#!/bin/bash

SEED=(42)
N_INSTANCES=5
APPROACHES=("QTable" "TripleDataDQN" "DoubleTableQTable" "CombinedRewardQTable" "HashMapQTable")

GRID_DIMS=("12x12")
N_STARTS=(3)
N_GOALS=(3)
N_BRIDGES=(15)
N_EPISODE_STEPS=(100)
ACTION_SUCCESS_RATE=(0.65)
UTILITY_SCALE=(25)
TERMINAL_REWARD=(50)
BRIDGE_COST_LB=(10)
BRIDGE_COST_UB=(30)
DUPLICATE_BRIDGE_PENALTY=(150)

ALPHA=(0.001 0.01 0.1)   
EPSILON=(0.25 0.30)
GAMMA=(0.99)

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
    local duplicate_bridge_penalty=${12}
    local alpha=${13}
    local epsilon=${14}
    local gamma=${15}
    echo "${approach}_${seed}_${problem_instance}_${grid_dims}_${n_starts}_${n_goals}_${n_bridges}_${n_episode_steps}_${action_success_rate}_${utility_scale}_${terminal_reward}_${duplicate_action_cost}_${alpha}_${epsilon}_${gamma}.sh"
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
                                            for duplicate_bridge_penalty in "${DUPLICATE_BRIDGE_PENALTY[@]}"; do
                                                for alpha in "${ALPHA[@]}"; do
                                                    for epsilon in "${EPSILON[@]}"; do
                                                        for gamma in "${GAMMA[@]}"; do
                                                            args="--approaches $approach --seed $seed --problem_instances instance_$problem_instance --grid_dims $grid_dims --n_starts $n_starts --n_goals $n_goals --n_bridges $n_bridges --n_episode_steps $n_episode_steps --action_success_rate $action_success_rate --utility_scale $utility_scale --terminal_reward $terminal_reward --duplicate_bridge_penalty $duplicate_bridge_penalty --alpha $alpha --epsilon $epsilon --gamma $gamma"
                                                            job_name=$(generate_job_name "$approach" "$seed" "$problem_instance" "$grid_dims" "$n_starts" "$n_goals" "$n_bridges" "$n_episode_steps" "$action_success_rate" "$utility_scale" "$terminal_reward" "$duplicate_bridge_penalty" "$alpha" "$epsilon" "$gamma")
                                                            cat <<EOT > "$TMP_DIR/$job_name"
#!/bin/bash

#SBATCH -N 1                               # Number of nodes
#SBATCH -c 32                              # Number of cores
#SBATCH -t 0-01:00:00                      # time in d-hh:mm:ss
#SBATCH -p htc                             # partition
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

for job_script in "$TMP_DIR"/*.sh; do
    sbatch "$job_script"
    sleep 0.5
done
