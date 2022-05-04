#! /bin/bash
#SBATCH --output=/home/%u/slurm_logs/slurm-%x-%A-%a.out
#SBATCH --error=/home/%u/slurm_logs/slurm-%x-%A-%a.out
#SBATCH --job-name=bert_0.0.7.0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0

# Use randomly initialized bert (not pretrained)

model_name=bertnet
model_version=0.0.7.0

output_path=/home/s1946695/RDP/outputs/bertnet_0.0.7.0
model_path=/home/s1946695/RDP/models/bertnet_0.0.7.0
pretrained_model_path=/home/s1946695/RDP/models/bertnet_0.0.7.0/ckpt-e8.pt

python main.py\
  --model_name=${model_name}\
  --model_version=${model_version}\
  --model_path=${model_path}\
  --output_path=${output_path}\
  --device=cuda\
  --gpu_id=0\
  --dataset=20news\
  --batch_size=50\
  --print_log_per_nbatch=500\
  --num_state=2000\
  --encoder_type=bert_random\
  --transition_init_scale=0.01\
  --exact_rsample=false\
  --sum_size=50\
  --sample_size=1\
  --learning_rate=0.001\
  --log_print_to_file=true\
  --x_lambd_warm_end_epoch=20\
  --x_lambd_warm_n_epoch=10\
  --num_epoch=50\
  --tau_anneal_start_epoch=42\
  --tau_anneal_n_epoch=8\
  --ent_approx=softmax\
  --z_beta_init=1.\
  --z_beta_final=0.001\
  --anneal_beta_with_lambd=true\
  --word_dropout_decay=false\
  --save_mode=state_matrix\
  --save_checkpoints=multiple\
  --use_tensorboard=false\
  --inspect_grad=first\
  --inspect_model=True\
  --pretrained_model_path=${pretrained_model_path}\
  --is_test=True\
        2>&1 | tee ${output_path}/test_log.txt