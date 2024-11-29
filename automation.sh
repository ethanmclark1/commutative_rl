#!/bin/bash

SEED=(42)
NUM_INSTANCES=10
APPROACHES=("TripleTraditional")
ALPHA=(0.0003)
EPSILON=(0.275) 
GAMMA=(0.995)
BATCH_SIZE=(512)
HIDDEN_DIMS=(128)
BUFFER_SIZE=(250000)
TARGET_UPDATE_FREQ=(1000)
GRAD_CLIP_NORM=(1)
LOSS_FN=("MSE")
LAYER_NORM=(1)
AGGREGATION_TYPE=("trace_back")

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
    local grad_clip_norm=${11}
    local loss_fn=${12}
    local layer_norm=${13}
    local aggregation_type=${14}
    echo "${approach}_${seed}_${problem_instance}_${alpha}_${epsilon}_${gamma}_${batch_size}_${hidden_dims}_${buffer_size}_${target_update_freq}_${grad_clip_norm}_${loss_fn}_${layer_norm}_${aggregation_type}.sh"
}

for seed in "${SEED[@]}"; do
    for problem_instance in $(seq 0 $((NUM_INSTANCES - 1))); do
    #for problem_instance in "9"; do
        for approach in "${APPROACHES[@]}"; do
	        for alpha in "${ALPHA[@]}"; do
		        for epsilon in "${EPSILON[@]}"; do
	                for gamma in "${GAMMA[@]}"; do
  	                    for batch_size in "${BATCH_SIZE[@]}"; do
		                    for hidden_dims in "${HIDDEN_DIMS[@]}"; do
	                            for buffer_size in "${BUFFER_SIZE[@]}"; do
		                            for target_update_freq in "${TARGET_UPDATE_FREQ[@]}"; do
	                                    for grad_clip_norm in "${GRAD_CLIP_NORM[@]}"; do
				                            for loss_fn in "${LOSS_FN[@]}"; do
				                                for layer_norm in "${LAYER_NORM[@]}"; do
						                            for aggregation_type in "${AGGREGATION_TYPE[@]}"; do
				                                        args="--approaches $approach --seed $seed --problem_instances instance_$problem_instance --alpha $alpha --epsilon $epsilon --gamma $gamma --batch_size $batch_size --hidden_dims $hidden_dims --buffer_size $buffer_size --target_update_freq $target_update_freq --grad_clip_norm $grad_clip_norm --loss_fn $loss_fn --layer_norm $layer_norm --aggregation_type $aggregation_type"
                                                        job_name=$(generate_job_name "$approach" "$seed" "$problem_instance" "$alpha" "$epsilon" "$gamma" "$batch_size" "$hidden_dims" "$buffer_size" "$target_update_freq" "$grad_clip_norm" "$loss_fn" "$layer_norm" "$aggregation_type")
                                                        cat <<EOT > "$TMP_DIR/$job_name"
#!/bin/bash

#SBATCH -N 1                               # Number of nodes
#SBATCH -c 16                              # Number of cores
#SBATCH -t 0-16:00:00                      # time in d-hh:mm:ss
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

for job_script in "$TMP_DIR"/*.sh; do
    sbatch "$job_script"
    sleep 0.5
done
