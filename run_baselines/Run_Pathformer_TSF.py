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
    parser = argparse.ArgumentParser(description='Pathformer Multivariate Time Series Forecasting')
    #https://github.com/decisionintelligence/pathformer
    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='PathFormer',
                        help='model name, options: [PathFormer]')
    parser.add_argument('--model_id', type=str, default="ETT.sh")

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/weather', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S]; M:multivariate predict multivariate, S:univariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # model
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--num_nodes', type=int, default=21)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int,
                        default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2])  # 16 12 8 32 12 8 6 4 8 6 4 2
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--metric', type=str, default='mae')

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
    parser.add_argument('--devices', type=str, default='2', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    args.random_seed=seed_cur


    args.label_len = 18
    args.task_name = 'long_term_forecast'
    args.efficient_comp=False
    args.root_path='./dataset/'
    args.data_type='ETTh1'
    args.model='Pathformer'
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
    args.seq_len=94*4
    args.data_name=args.data_type
    args.model_id=args.data_type
    if args.data_name == 'ETTh1':
        args.data = args.data_name
        args.data_path_name = 'ETTh1.csv'
        args.model_id_name = 'ETTh1'
        args.data_path = args.data_path_name
        args.batch_size = 128
        args.residual_connection = 1
        args.num_nodes = 7
        args.layer_nums = 3
        args.k = 3
        args.d_model = 4
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    if args.data_name == 'ETTh2':
        args.data = args.data_name
        args.data_path_name = 'ETTh2.csv'
        args.model_id_name = 'ETTh2'
        args.data_path = args.data_path_name
        args.batch_size = 128

        args.residual_connection = 0
        args.num_nodes = 7
        args.layer_nums = 3
        args.k = 2
        args.d_model = 4
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    if args.data_name == 'ETTm1':
        args.data = args.data_name
        args.data_path_name = 'ETTm1.csv'
        args.model_id_name = 'ETTm1'
        args.data_path = args.data_path_name
        args.batch_size = 128

        args.num_nodes = 7
        args.layer_nums = 3
        args.k = 3
        args.d_model = 8
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 4, 12, 8, 6, 4, 8, 6, 2, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    if args.data_name == 'ETTm2':
        args.data = args.data_name
        args.data_path_name = 'ETTm2.csv'
        args.model_id_name = 'ETTm2'
        args.data_path = args.data_path_name
        args.batch_size = 128

        args.num_nodes = 7
        args.layer_nums = 3
        args.k = 2
        args.d_model = 16
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 32, 8, 6, 16, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
    if args.data_name == 'weather':
        args.data = 'custom'
        args.data_path_name = 'weather.csv'
        args.model_id_name = 'weather'
        args.data_path = args.data_path_name
        args.batch_size = 128 // 2  # 256

        args.num_nodes = 21
        args.layer_nums = 3
        args.k = 2
        args.d_model = 8
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 4, 12, 8, 6, 4, 8, 6, 2, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()
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

        args.num_nodes = 21
        args.layer_nums = 3
        args.k = 2
        args.d_model = 8
        args.d_ff = 64
        args.patch_size_list = [16, 12, 8, 4, 12, 8, 6, 4, 8, 6, 2, 12]
        args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

        if args.pred_len==96 or args.pred_len==192 or args.pred_len==336 or args.pred_len==720:
            args.seq_len = 1664


    args.is_training=1


    args.des='Exp'
    args.itr=1


    args.record=True
    args.c_in=args.num_nodes
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


