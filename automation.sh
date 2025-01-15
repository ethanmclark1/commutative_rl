#!/bin/bash

SEED=(42)
NUM_INSTANCES=10
#APPROACHES=("TraditionalQTable" "CommutativeQTable" "TripleTraditionalQTable")
#ALPHA=(0.1)
APPROACHES=("TraditionalDQN" "CommutativeDQN" "TripleTraditionalDQN")
ALPHA=(0.0003)
EPSILON=(0.25) 
GAMMA=(0.99)
BATCH_SIZE=(64)
LEARNING_START_STEP=(512)
HIDDEN_DIMS=(32)
ACTIVATION_FN=("ReLU")
BUFFER_SIZE=(100000)
TARGET_UPDATE_FREQ=(150)
GRAD_CLIP_NORM=(1)
LOSS_FN=("Huber")
LAYER_NORM=(1)

STEP_SCALE=(10)
OVER_PENALTY=(100)
UNDER_PENALTY=(5)
COMPLETION_REWARD=(1)
MAX_NOISE=(10)

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

generate_job_name() {
    local approach=$1
    local seed=$2
    local problem_instance=$3
    local alpha=$4
    local epsilon=$5
    local gamma=$6
    local batch_size=$7
    local learning_start_step=$8
    local hidden_dims=$9
    local activation_fn=${10}
    local buffer_size=${11}
    local target_update_freq=${12}
    local grad_clip_norm=${13}
    local loss_fn=${14}
    local layer_norm=${15}
    local step_scale=${17}
    local over_penalty=${18}
    local under_penalty=${19}
    local completion_reward=${20}
    local max_noise=${21}
    echo "${approach}_${seed}_${problem_instance}_${alpha}_${epsilon}_${gamma}_${batch_size}_${learning_start_step}_${hidden_dims}_${activation_fn}_${buffer_size}_${target_update_freq}_${grad_clip_norm}_${loss_fn}_${layer_norm}_${step_scale}_${over_penalty}_${under_penalty}_${completion_reward}_${max_noise}.sh"
}

for seed in "${SEED[@]}"; do
    for problem_instance in $(seq 0 $((NUM_INSTANCES - 1))); do
        for approach in "${APPROACHES[@]}"; do
	        for alpha in "${ALPHA[@]}"; do
		        for epsilon in "${EPSILON[@]}"; do
	                for gamma in "${GAMMA[@]}"; do
  	                    for batch_size in "${BATCH_SIZE[@]}"; do
                            for learning_start_step in "${LEARNING_START_STEP[@]}"; do
                                for hidden_dims in "${HIDDEN_DIMS[@]}"; do
                                    for activation_fn in "${ACTIVATION_FN[@]}"; do
                                        for buffer_size in "${BUFFER_SIZE[@]}"; do
                                            for target_update_freq in "${TARGET_UPDATE_FREQ[@]}"; do
                                                for grad_clip_norm in "${GRAD_CLIP_NORM[@]}"; do
                                                    for loss_fn in "${LOSS_FN[@]}"; do
                                                        for layer_norm in "${LAYER_NORM[@]}"; do
                                                            for step_scale in "${STEP_SCALE[@]}"; do
                                                                for over_penalty in "${OVER_PENALTY[@]}"; do
                                                                    for under_penalty in "${UNDER_PENALTY[@]}"; do
                                                                        for completion_reward in "${COMPLETION_REWARD[@]}"; do
                                                                            for max_noise in "${MAX_NOISE[@]}"; do
                                                                                args="--approaches $approach --seed $seed --problem_instances instance_$problem_instance --alpha $alpha --epsilon $epsilon --gamma $gamma --batch_size $batch_size --learning_start_step $learning_start_step --hidden_dims $hidden_dims --activation_fn $activation_fn --buffer_size $buffer_size --target_update_freq $target_update_freq --grad_clip_norm $grad_clip_norm --loss_fn $loss_fn --layer_norm $layer_norm --step_scale $step_scale --over_penalty $over_penalty --under_penalty $under_penalty --completion_reward $completion_reward --max_noise $max_noise"
                                                                                job_name=$(generate_job_name "$approach" "$seed" "$problem_instance" "$alpha" "$epsilon" "$gamma" "$batch_size" "$learning_start_step" "$hidden_dims" "$activation_fn" "$buffer_size" "$target_update_freq" "$grad_clip_norm" "$loss_fn" "$layer_norm" "$step_scale" "$over_penalty" "$under_penalty" "$completion_reward" "$max_noise")
                                                                                cat <<EOT > "$TMP_DIR/$job_name"
#!/bin/bash

#SBATCH -N 1                               # Number of nodes
#SBATCH -c 16                              # Number of cores
#SBATCH -t 0-04:00:00                      # time in d-hh:mm:ss
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
                done
            done
        done
    done
done

for job_script in "$TMP_DIR"/*.sh; do
    sbatch "$job_script"
    sleep 0.5
done
