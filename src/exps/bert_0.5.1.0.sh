#! /bin/bash
#SBATCH --output=/home/%u/slurm_logs/slurm-%A-%a.out
#SBATCH --error=/home/%u/slurm_logs/slurm-%A-%a.out
#SBATCH --job-name=bert_0.5.1.0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --array=0

python main.py\
  --model_name=bertnet\
  --model_version=0.5.1.0\
  --device=cuda\
  --gpu_id=0\
  --dataset=20news\
  --batch_size=50\
  --print_log_per_nbatch=500\
  --num_state=2000\
  --transition_init_scale=0.01\
  --exact_rsample=false\
  --sum_size=1\
  --sample_size=49\
  --potential_normalization=minmax\
  --potential_scale=10\
  --learning_rate=0.001\
  --validate_start_epoch=0\
  --validation_criteria=p_log_prob_x\
  --log_print_to_file=true\
  --x_lambd_warm_end_epoch=20\
  --x_lambd_warm_n_epoch=10\
  --num_epoch=20\
  --tau_anneal_start_epoch=42\
  --tau_anneal_n_epoch=8\
  --ent_approx=softmax\
  --ent_approx=softmax\
  --z_beta_init=1.\
  --z_beta_final=0.1\
  --anneal_beta_with_lambd=true\
  --word_dropout_decay=false\
  --save_mode=state_matrix\
  --save_checkpoints=best\
  --use_tensorboard=false\
  --inspect_grad=first