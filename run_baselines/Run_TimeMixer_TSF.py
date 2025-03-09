import argparse
import os
import time

import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import json

def get_files(path):
    with open(path, mode='r', encoding='utf-8') as file:
        data = file.read()
        data_dict = json.loads(data)
    return data_dict

def main(seed_cur, pred_l):
    parser = argparse.ArgumentParser(description='TimeMixer for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    ## https://github.com/kwuking/TimeMixer
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                        help='whether to use future_temporal_feature; True 1 False 0')

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')


    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs') #
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data') #32
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    args.random_seed=seed_cur


    args.label_len = 18
    args.task_name = 'long_term_forecast'
    args.efficient_comp=False
    args.root_path='./dataset/'
    args.data_type='ETTh1'
    args.model='TimeMixer'
    args.pred_len=pred_l
    args.learning_rate = 0.0001
    args.batch_size = 128
    args.train_epochs = 30
    args.sample_num=5
    args.checkpoints = 'LongTermTSF_' + args.model+ '/' + args.data_type + '/random_seed_' + str(args.random_seed)
    args.gpu = 5
    args.device='cuda:'+str(args.gpu)
    args.scaleformers=['Autoformer_Scaleformer','NHits_Scaleformer','PatchTST_ScaleFormer']
    args.train_epochs = 30
    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.data_type=='ETTh1':
        args.data_path = 'ETTh1.csv'
        args.data = 'ETTh1'
        args.model_id = 'ETTh1'
        args.enc_in = 7
        args.e_layers = 2
        args.down_sampling_layers = 3
        args.down_sampling_window = 2
        args.d_model = 16
        args.d_ff = 32
        args.batch_size=64*2
        args.learning_rate=0.0001
        if args.pred_len==96 or args.pred_len==192:
            args.seq_len = 1280
        elif args.pred_len==336 or args.pred_len==720:
            args.seq_len=384
    elif args.data_type=='ETTh2':
        args.data_path = 'ETTh2.csv'
        args.data = 'ETTh2'
        args.model_id = 'ETTh2'
        args.enc_in = 7
        args.e_layers = 2
        args.down_sampling_layers = 3
        args.down_sampling_window = 2
        args.d_model = 16
        args.d_ff = 32
        args.batch_size=64*2
        args.learning_rate=0.0001
        if args.pred_len==96:
            args.seq_len = 1280
        elif args.pred_len==192 or args.pred_len==336:
            args.seq_len=1024
        elif args.pred_len==720:
            args.seq_len=768
    elif args.data_type=='ETTm1':
        args.data_path = 'ETTm1.csv'
        args.data = 'ETTm1'
        args.model_id = 'ETTm1'
        args.enc_in = 7
        args.e_layers = 2
        args.down_sampling_layers = 3
        args.down_sampling_window = 2
        args.d_model = 16
        args.d_ff = 32
        args.batch_size=64*2
        args.learning_rate=0.0001
        if args.pred_len==96:
            args.seq_len = 768
        elif args.pred_len==192 or args.pred_len==336:
            args.seq_len=1536
        elif args.pred_len==720:
            args.seq_len=1664
    elif args.data_type=='ETTm2':
        args.data_path = 'ETTm2.csv'
        args.data = 'ETTm2'
        args.model_id = 'ETTm2'
        args.enc_in = 7
        args.e_layers = 2
        args.down_sampling_layers = 3
        args.down_sampling_window = 2
        args.d_model = 32
        args.d_ff = 32
        args.batch_size=64*2
        args.learning_rate=0.0001
        if args.pred_len==96:
            args.seq_len = 768
        elif args.pred_len==192 or args.pred_len==336 or args.pred_len==720:
            args.seq_len=1664
    elif args.data_type=='weather':
        args.data_path = 'weather.csv'
        args.model_id = 'weather'
        args.data = 'custom'
        args.enc_in = 21
        args.dec_in = 21
        args.e_layers = 3
        args.down_sampling_layers = 3
        args.down_sampling_window = 2
        args.d_model = 16
        args.d_ff = 32
        args.batch_size = 64
        args.learning_rate=0.0001
        if args.pred_len==96 or args.pred_len==192 or args.pred_len==336 or args.pred_len==720:
            args.seq_len = 2048
    elif args.data_type=='electricity':
        args.data_path = 'electricity.csv'
        args.data = 'custom'
        args.model_id = 'electricity'
        args.enc_in = 321

        args.down_sampling_layers = 3
        args.down_sampling_window = 2
        args.d_model = 16
        args.d_ff = 32
        args.e_layers=3
        args.factor=3
        args.batch_size=32
        args.learning_rate=0.0001
        if args.pred_len==96 or args.pred_len==192 or args.pred_len==336 or args.pred_len==720:
            args.seq_len = 1664

    args.is_training=1


    args.des='Exp'
    args.itr=1


    args.record=True
    args.c_in=args.enc_in
    args.context_window=args.seq_len
    args.target_window=args.pred_len
    args.is_training = True
    print('Args in experiment:')
    if args.data_type == 'electricity':
        args.var_decomp = True
        args.var_sp_num = 15
    Exp = Exp_Main
    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_SeqLen{}_PredLen{}_HiddenDim_{}'.format(
                args.model_id,
                args.model,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.des,ii)

            path = os.path.join(args.checkpoints, setting)
            args.path=path
            if not os.path.exists(path):
                os.makedirs(path)
            args_dict = vars(args)
            json_record_args = json.dumps(args_dict, indent=4)
            if args.record:
                with open(path + '/record_args' + '.json', 'w') as json_file:
                    json_file.write(json_record_args)
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            torch.cuda.empty_cache()
            best_model_path = args.path + '/' + 'checkpoint.pth'

    else:
        ii = 0
        #{}_{}_{}_seed{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}
        setting = '{}_{}_SeqLen{}_PredLen{}_HiddenDim_{}'.format(
            args.model_id,
            args.model,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.des, ii)

        path = os.path.join(args.checkpoints, setting)
        args.path = path
        exp = Exp(args)  # set experiments
        print('>>>>>>>test_inference_time : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test_inference_time(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    Seeds_All = [0, 1, 2, 3, 4, 5]
    pred_len = [96, 192, 336, 720]
    for seed in Seeds_All:
        for pred_l in pred_len:
            main(seed,pred_l)


