import argparse
import torch
import sys

from data_utils import News20Data, BBNData, MSCOCOData
from bertnet import BertNetModel, BertNet
from gpt2net import GPT2NetModel, GPT2Net

from pprint import pprint
from frtorch import torch_model_utils as tmu
from frtorch import str2bool, set_arguments, Controller


def define_argument():
  ## add commandline arguments
  parser = argparse.ArgumentParser()

  # general 
  parser.add_argument(
    "--model_name", default='seq2seq', type=str)
  parser.add_argument(
    "--model_version", default='0.1.0.0', type=str)
  parser.add_argument(
    "--model_path", default='../models/', type=str)
  parser.add_argument(
    "--output_path", default='../outputs/', type=str)
  parser.add_argument(
    "--output_path_fig", default='', type=str)
  parser.add_argument(
    "--tensorboard_path", default='../tensorboard/', type=str)

  # data 
  parser.add_argument(
    "--dataset", default='', type=str)
  parser.add_argument('--data_path', type=str, default='.')
  parser.add_argument(
    "--subsample_data", type=str2bool, default=False)

  # hardware
  parser.add_argument(
    "--device", default='cpu', type=str)
  parser.add_argument(
    "--gpu_id", default='0', type=str)

  # batch, epoch 
  parser.add_argument(
    "--num_epoch", default=10, type=int)
  parser.add_argument(
    "--batch_size", default=64, type=int)
  parser.add_argument(
    "--test_batch_size", default=64, type=int)
  parser.add_argument(
    "--start_epoch", default=0, type=int)

  # saving, logging
  parser.add_argument(
    "--load_ckpt", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--pretrained_model_path", default='', type=str)
  parser.add_argument(
    "--print_log_per_nbatch", default=50, type=int)
  parser.add_argument(
    "--print_var", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--use_tensorboard", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--save_checkpoints", choices=['none', 'multiple', 'best'], default='none')
  parser.add_argument(
    "--inspect_model", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--inspect_grad", choices=['first', 'second', 'none'], default='none')
  parser.add_argument(
    "--log_print_to_file", type=str2bool, 
    nargs='?', const=True, default=False)


  # Validation, Test
  parser.add_argument(
    "--is_test", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--fast_test_pipeline", type=str2bool, 
    nargs='?', const=True, default=False,
    help='run a test before beginning of training'
    )
  parser.add_argument(
    "--fast_train_pipeline", type=str2bool, 
    nargs='?', const=True, default=False,
    help='only train 100 batches per epoch'
    )
  parser.add_argument(
    "--validate_start_epoch", default=0, type=int)
  parser.add_argument(
    "--write_output", type=str2bool, 
    nargs='?', const=True, default=False)
  parser.add_argument(
    "--write_output_full", type=str2bool, 
    nargs='?', const=True, default=False)
  

  # optimization
  parser.add_argument(
    "--optimizer", default='adam', type=str)
  parser.add_argument(
    "--learning_rate", default=1e-4, type=float)


  # model 
  parser.add_argument(
    "--state_size", type=int, default=100)
  parser.add_argument(
    "--embedding_size", type=int, default=100)
  parser.add_argument(
    "--dropout", type=float, default=0.0)


  # BertNet
  parser.add_argument(
    "--encoder_type", type=str, default='bert')
  parser.add_argument(
    "--num_state", type=int, default=20)
  parser.add_argument(
    "--transition_init_scale", type=float, default=0.01)
  parser.add_argument(
    "--exact_rsample", type=str2bool, default=True)
  parser.add_argument(
    "--potential_normalization", type=str, default='none',
    help='none, minmax, zscore')
  parser.add_argument(
    "--potential_scale", type=float, default=1.0)
  parser.add_argument(
    "--sum_size", type=int, default=10)
  parser.add_argument(
    "--sample_size", type=int, default=10)
  parser.add_argument(
    "--proposal", type=str, default='softmax',
    help='softmax, uniform')
  parser.add_argument(
    "--transition_proposal", type=str, default='none',
    help='none, prod, l1norm')
  parser.add_argument(
    "--use_bow", type=str2bool, default=False)
  parser.add_argument(
    "--use_copy", type=str2bool, default=True)
  parser.add_argument(
    "--task", type=str, default='density',
    help='density, paraphrase')
  parser.add_argument(
    "--emb_type", type=str, default='contextualized', 
    choices=['static', 'static_pos', 'positional', 'contextualized'])
  parser.add_argument(
    "--z_beta_init", type=float, default=1.0)
  parser.add_argument(
    "--z_beta_final", type=float, default=1.0)
  parser.add_argument(
    "--x_lambd_warm_end_epoch", type=int, default=1)
  parser.add_argument(
    "--x_lambd_warm_n_epoch", type=int, default=1)
  parser.add_argument(
    "--tau_anneal_start_epoch", type=int, default=18)
  parser.add_argument(
    "--tau_anneal_n_epoch", type=int, default=2)
  parser.add_argument(
    "--use_latent_proj", type=str2bool, default=False)
  parser.add_argument(
    "--anneal_z_prob", type=str2bool, default=False)
  parser.add_argument(
    "--latent_type", type=str, default='sampled_gumbel_crf')
  # parser.add_argument( # OBSOLETE, use crf_weight_norm instead
  #   "--use_sphere_norm", type=str2bool, default=True)
  parser.add_argument(
    "--ent_approx", type=str, default='softmax', choices=['softmax', 'rdp'])
  parser.add_argument(
    "--crf_weight_norm", type=str, default='none')
  parser.add_argument( 
    "--word_dropout_decay", type=str2bool, default=False)
  parser.add_argument( 
    "--anneal_beta_with_lambd", type=str2bool, default=False)
  parser.add_argument(
    "--save_mode", choices=['state_matrix', 'full'], default='state_matrix')

  parser.add_argument( 
    "--mask_z", type=str2bool, default=False)
  parser.add_argument( 
    "--word_dropout", type=str2bool, default=True)

  parser.add_argument( 
    "--z_st", type=str2bool, default=False)

  parser.add_argument( 
    "--freeze_z_at_epoch", type=int, default=10000000)

  parser.add_argument( 
    "--topk_sum", type=str2bool, default=False)

  parser.add_argument( 
    "--change_opt_to_sgd_at_epoch", type=int, default=10000000)

  parser.add_argument(
    "--cache_dir", type=str, default='')

  # BertNet End

  
  # softmax, approx_crf, exact_crf
  parser.add_argument(
    "--validation_criteria", type=str, default='')

  return parser


def main():
  # arguments
  parser = define_argument()
  args = parser.parse_args()
  args = set_arguments(args)

  # dataset
  if(args.dataset == 'bbn'):
    dataset = BBNData(args.data_path, args.batch_size)
  elif(args.dataset == '20news'):
    print('Using dataset 20news')
    dataset = News20Data(batch_size=args.batch_size, is_test=args.is_test)
  elif(args.dataset == 'mscoco'):
    print('Using dataset mscoco')
    dataset = MSCOCOData(batch_size=args.batch_size, 
                         test_batch_size=args.test_batch_size, 
                         subsample=args.subsample_data,
                         cache_dir=args.cache_dir)
  else: 
    raise NotImplementedError('dataset %s not implemented!' % args.dataset)
 
  # model 
  if(args.model_name == 'bert_tag'):
    pass
    # Need update
    # model_ = CRFScaleModel(num_z_state=dataset.label_size,
    #                        approx_state_r=args.approx_state_r,
    #                        norm_scale_transition=args.norm_scale_transition,
    #                        norm_scale_emission=args.norm_scale_emission,
    #                        lambd_softmax=args.lambd_softmax,
    #                        loss_type=args.loss_type)
    # model = CRFScale(model_, args.learning_rate, args.validation_criteria)
  elif(args.model_name == 'bertnet'):
    print('Using model bertnet')
    model_ = BertNetModel(num_state=args.num_state, 
                          transition_init_scale=args.transition_init_scale,
                          encoder_type=args.encoder_type,
                          exact_rsample=args.exact_rsample,
                          sum_size=args.sum_size,
                          sample_size=args.sample_size,
                          device=args.device,
                          use_latent_proj=args.use_latent_proj,
                          latent_type=args.latent_type, 
                          ent_approx=args.ent_approx,
                          crf_weight_norm=args.crf_weight_norm,
                          word_dropout_decay=args.word_dropout_decay,
                          potential_normalization=args.potential_normalization,
                          potential_scale=args.potential_scale,
                          topk_sum=args.topk_sum,
                          emb_type=args.emb_type,
                          # z_st=args.z_st
                          # pad_id=dataset.tokenizer.pad_token_id
                          ).to(args.device)
    model = BertNet(model=model_,
                    learning_rate=args.learning_rate,
                    validation_criteria=args.validation_criteria,
                    num_batch_per_epoch=dataset.num_batch_per_epoch,
                    x_lambd_warm_end_epoch=args.x_lambd_warm_end_epoch,
                    x_lambd_warm_n_epoch=args.x_lambd_warm_n_epoch,
                    tau_anneal_start_epoch=args.tau_anneal_start_epoch,
                    tau_anneal_n_epoch=args.tau_anneal_n_epoch,
                    tokenizer=dataset.tokenizer,
                    z_beta_init=args.z_beta_init,
                    z_beta_final=args.z_beta_final,
                    anneal_beta_with_lambd=args.anneal_beta_with_lambd,
                    save_mode=args.save_mode,
                    anneal_z_prob=args.anneal_z_prob,
                    data_path=dataset.data_path
                    )
  elif(args.model_name in ['gpt2net', 'gpt2net_lm', 'gpt2net_paraphrase']):
    print('Using model gptnet')
    model_ = GPT2NetModel(num_state=args.num_state,
                          transition_init_scale=args.transition_init_scale,
                          exact_rsample=args.exact_rsample,
                          sum_size=args.sum_size,
                          sample_size=args.sample_size,
                          proposal=args.proposal,
                          transition_proposal=args.transition_proposal,
                          device=args.device,
                          vocab_size=len(dataset.tokenizer),
                          pad_id=dataset.pad_id,
                          bos_id=dataset.bos_id,
                          max_dec_len=dataset.max_slen, 
                          use_bow=args.use_bow,
                          use_copy=args.use_copy,
                          task=args.task,
                          ent_approx=args.ent_approx,
                          word_dropout_decay=args.word_dropout_decay,
                          dropout=args.dropout,
                          potential_normalization=args.potential_normalization,
                          potential_scale=args.potential_scale,
                          mask_z=args.mask_z,
                          z_st=args.z_st,
                          topk_sum=args.topk_sum,
                          cache_dir=args.cache_dir
                          )
    model = GPT2Net(model=model_,
                    learning_rate=args.learning_rate,
                    validation_criteria=args.validation_criteria,
                    num_batch_per_epoch=dataset.num_batch_per_epoch,
                    word_dropout=args.word_dropout,
                    x_lambd_warm_end_epoch=args.x_lambd_warm_end_epoch,
                    x_lambd_warm_n_epoch=args.x_lambd_warm_n_epoch,
                    tau_anneal_start_epoch=args.tau_anneal_start_epoch,
                    tau_anneal_n_epoch=args.tau_anneal_n_epoch,
                    tokenizer=dataset.tokenizer,
                    z_beta_init=args.z_beta_init,
                    z_beta_final=args.z_beta_final,
                    anneal_beta_with_lambd=args.anneal_beta_with_lambd,
                    anneal_z_prob=args.anneal_z_prob,
                    save_mode=args.save_mode,
                    optimizer_type=args.optimizer,
                    space_token=dataset.space_token,
                    freeze_z_at_epoch=args.freeze_z_at_epoch,
                    change_opt_to_sgd_at_epoch=args.change_opt_to_sgd_at_epoch,
                    model_path=args.model_path
                    )
  else: 
    raise NotImplementedError('model %s not implemented!' % args.model_name)  
  tmu.print_params(model)

  # controller
  controller = Controller(args, model, dataset)

  if(not args.is_test):
    if(args.load_ckpt):
      print('Loading model from: %s' % args.pretrained_model_path)
      model.load_state_dict(torch.load(args.pretrained_model_path))
    model.to(args.device)
    controller.train(model, dataset)
  else:
    print('Loading model from: %s' % args.pretrained_model_path)
    # tmu.load_partial_state_dict(model, checkpoint['model_state_dict'])
    model.load(args.pretrained_model_path)
    model.to(args.device)
    _, scores = controller.validate(model, dataset, -1, -1)
    pprint(scores)
  return 


if __name__ == '__main__':
  main()
