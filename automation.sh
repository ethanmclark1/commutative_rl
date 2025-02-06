#!/bin/bash

SEED=(42)
N_INSTANCES=10
APPROACHES=("TraditionalQTable" "CommutativeQTable" "TripleTraditionalQTable")
ALPHA=(0.2)
# APPROACHES=("TraditionalDQN" "CommutativeDQN" "TripleTraditionalDQN")
# ALPHA=(0.001)
EPSILON=(0.25) 
GAMMA=(0.99)
BATCH_SIZE=(4)
HIDDEN_DIMS=(128)
BUFFER_SIZE=(5000)
TARGET_UPDATE_FREQ=(10)
DROPOUT=(0.2)

STEP_VALUE=(1)
OVER_PENALTY=(2)
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
    local hidden_dims=$8
    local buffer_size=$9
    local target_update_freq=${10}
    local dropout=${11}
    local step_value=${12}
    local over_penalty=${13}
    local max_noise=${14}
    echo "${approach}_${seed}_${problem_instance}_${alpha}_${epsilon}_${gamma}_${batch_size}_${hidden_dims}_${buffer_size}_${target_update_freq}_${dropout}_${step_value}_${over_penalty}_${max_noise}.sh"
}

for seed in "${SEED[@]}"; do
    for problem_instance in $(seq 0 $((N_INSTANCES - 1))); do
        for approach in "${APPROACHES[@]}"; do
	        for alpha in "${ALPHA[@]}"; do
		        for epsilon in "${EPSILON[@]}"; do
	                for gamma in "${GAMMA[@]}"; do
  	                    for batch_size in "${BATCH_SIZE[@]}"; do
                            for hidden_dims in "${HIDDEN_DIMS[@]}"; do
                                for buffer_size in "${BUFFER_SIZE[@]}"; do
                                    for target_update_freq in "${TARGET_UPDATE_FREQ[@]}"; do
                                        for dropout in "${DROPOUT[@]}"; do
                                            for step_value in "${STEP_VALUE[@]}"; do
                                                for over_penalty in "${OVER_PENALTY[@]}"; do
                                                    for max_noise in "${MAX_NOISE[@]}"; do
                                                        args="--approaches $approach --seed $seed --problem_instances instance_$problem_instance --alpha $alpha --epsilon $epsilon --gamma $gamma --batch_size $batch_size --hidden_dims $hidden_dims --buffer_size $buffer_size --target_update_freq $target_update_freq --dropout $dropout --step_value $step_value --over_penalty $over_penalty --max_noise $max_noise"
                                                        job_name=$(generate_job_name "$approach" "$seed" "$problem_instance" "$alpha" "$epsilon" "$gamma" "$batch_size" "$hidden_dims" "$buffer_size" "$target_update_freq" "$dropout" "$step_value" "$over_penalty" "$max_noise")
                                                        cat <<EOT > "$TMP_DIR/$job_name"
#!/bin/bash

#SBATCH -N 1                               # Number of nodes
#SBATCH -c 32                              # Number of cores
#SBATCH -t 0-00:30:00                      # time in d-hh:mm:ss
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

for job_script in "$TMP_DIR"/*.sh; do
    sbatch "$job_script"
    sleep 0.5
done
