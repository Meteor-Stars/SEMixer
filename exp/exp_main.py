from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TimeMixer,PatchTST,PathFormer,PatchTST_ScaleFormer,NHits_Scaleformer,Autoformer_Scaleformer,FiLM,DLinear,NLinear,SEMixer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from transformers import PatchTSMixerConfig as CI_TSmixer_Config
from transformers import PatchTSMixerForPrediction as CI_TSmixer
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
# from robust_loss_pytorch import AdaptiveLossFunction
from torch.nn.parallel import DistributedDataParallel
warnings.filterwarnings('ignore')


def get_model(model):
    "Return the model maybe wrapped inside `model`."
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model

class moving_avg(nn.Module):
    def __init__(self):
        super(moving_avg, self).__init__()
    def forward(self, x, kernel_size):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            convert_numpy = True
            x = torch.tensor(x)
        else:
            convert_numpy = False
        x = nn.functional.avg_pool1d(x.permute(0, 2, 1), kernel_size, kernel_size)
        x = x.permute(0, 2, 1)
        if convert_numpy:
            x = x.numpy()
        return x

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.mv = moving_avg()
    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'NLinear': NLinear,

            'TimeMixer':TimeMixer,
            'CI_TSmixer':CI_TSmixer,
            'PatchTST':PatchTST,
            'FiLM':FiLM,
            'PatchTST_ScaleFormer':PatchTST_ScaleFormer,
            'Autoformer_Scaleformer':Autoformer_Scaleformer,
            'NHits_Scaleformer':NHits_Scaleformer,
            'Pathformer':PathFormer,
            'MEINet': MEINet,
        }

        if self.args.model =='CI_TSmixer':
            config = CI_TSmixer_Config(
                context_length=self.args.seq_len,
                prediction_length=self.args.pred_len,
                patch_length=self.args.patch_len, #16 self.args.patch_len
                num_input_channels=self.args.enc_in,
                patch_stride=self.args.stride, #8 self.args.stride
                d_model=self.args.d_model,
                num_layers=self.args.num_layers,
                expansion_factor=2,
                dropout=0.2,
                head_dropout=0.2,
                mode="common_channel",
                scaling="std",
            )
            model = CI_TSmixer(config)
        else:
            model = model_dict[self.args.model].Model(configs=self.args).float()
            # try:
            #     model = model_dict[self.args.model].Model(configs=self.args).float()
            # except:
            #     model = model_dict[self.args.model].Model(args=self.args).float()
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        loss_dict={}
        preds=[]
        trues=[]
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model=='CI_TSmixer':
                    outputs_all = self.model(batch_x)
                    outputs=outputs_all[0]
                elif self.args.model in self.args.scaleformers:
                    try:
                        outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    except:
                        outputs_all = self.model(batch_x)
                    outputs = outputs_all[-1]
                elif self.args.model == 'Pathformer':
                    outputs, balance_loss = self.model(batch_x)
                else:
                    try:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    except:
                        outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        loss_dict['mae']=float(mae)
        loss_dict['mse'] = float(mse)
        loss_dict['rmse'] = float(rmse)
        loss_dict['mape'] = float(mape)
        loss_dict['mspe'] = float(mspe)
        loss_dict['rse'] = float(rse)
        self.model.train()
        return loss_dict


    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,args=self.args)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        train_loss_all_dict={}
        val_loss_all_dict={}
        test_loss_all_dict = {}
        scale_loss_dict={}
        time_start= time.time()
        epoch_time_all=[]
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            scale_loss_all=[]
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                self.args.test = False

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                balance_loss=0
                if self.args.model=='CI_TSmixer':
                    outputs_all = self.model(batch_x)
                    outputs=outputs_all[0]
                elif self.args.model in  self.args.scaleformers:
                    try:
                        outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    except:
                        outputs_all = self.model(batch_x)
                    outputs = outputs_all[-1]
                elif self.args.model == 'Pathformer':
                    outputs, balance_loss = self.model(batch_x)
                else:
                    try:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    except:
                        outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self.args.model in  self.args.scaleformers:
                    loss=0
                    for li, (scale, output) in enumerate(zip(self.args.scales[:-1], outputs_all[:-1])):
                        tmp_y = self.mv(batch_y, scale)
                        tmp_loss = criterion(output, tmp_y)
                        loss += tmp_loss / scale
                    loss = loss / 2
                else:
                    loss = criterion(outputs, batch_y)
                if self.args.model=='Pathformer':
                    loss+=balance_loss
                train_loss.append(loss.item())
                scale_loss_all_batch=[]

                scale_loss_all.append(np.expand_dims(np.array(scale_loss_all_batch),axis=0))
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()




            epoch_time_all.append(time.time() - epoch_time)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            train_loss_all_dict[epoch] = train_loss
            self.args.test=True
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            val_loss_all_dict[epoch] = vali_loss
            test_loss_all_dict[epoch] = test_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss['mse'], test_loss['mse']))

            early_stopping.save_checkpoint(vali_loss['mse'], self.model, path)

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)


            json_record_loss_train = json.dumps(train_loss_all_dict, indent=4)
            json_record_loss_val = json.dumps(val_loss_all_dict, indent=4)
            json_record_loss_test = json.dumps(test_loss_all_dict, indent=4)

            json_record_loss_train_scale = json.dumps(scale_loss_dict, indent=4)
            if self.args.record:
                with open(path + '/record_all_loss_train' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_train)
                with open(path + '/record_all_loss_val' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_val)
                with open(path + '/record_all_loss_test' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_test)

                with open(path + '/record_all_loss_train_scale' + '.json', 'w') as json_file:
                    json_file.write(json_record_loss_train_scale)

        train_cost = time.time() - time_start
        train_loss_all_dict['train_cost_time'] = train_cost
        train_loss_all_dict['train_mean_epoch_time'] = np.mean(epoch_time_all)
        json_record_loss_train = json.dumps(train_loss_all_dict, indent=4)
        if self.args.record:
            with open(path + '/record_all_loss_train' + '.json', 'w') as json_file:
                json_file.write(json_record_loss_train)
        best_model_path = path + '/' + 'checkpoint.pth'
        return self.model
    def test_inference_time(self, setting, test=0):

        self.criterion_tmp = torch.nn.MSELoss(reduction='none')
        self.args.device='cuda:'+str(self.args.gpu)
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.args.path=path
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        self.test_loader = test_loader
        torch.cuda.empty_cache()
        path = os.path.join(self.args.checkpoints, setting)
        self.args.path=path

        best_model_path = path + '/' + 'checkpoint.pth'

        self.args.mode = 'test'
        self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        running_times = []
        test_mse=[]
        time_now=time.time()
        self.args.test=True
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark)in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                start_time = time.time()
                if self.args.model=='CI_TSmixer':
                    outputs_all = self.model(batch_x)
                    outputs=outputs_all[0]
                elif self.args.model in  self.args.scaleformers:
                    try:
                        outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    except:
                        outputs_all = self.model(batch_x)
                    outputs = outputs_all[-1]
                elif self.args.model == 'Pathformer':
                    outputs, balance_loss = self.model(batch_x)
                else:
                    try:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    except:
                        outputs = self.model(batch_x)
                running_times.append(time.time()-start_time)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs.numpy()
                true = batch_y.numpy()
                preds.append(pred)
                trues.append(true)
                # break
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark)in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                start_time = time.time()
                if self.args.model == 'CI_TSmixer':
                    outputs_all = self.model(batch_x)
                    outputs = outputs_all[0]
                elif self.args.model in self.args.scaleformers:
                    try:
                        outputs_all = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    except:
                        outputs_all = self.model(batch_x)
                    outputs = outputs_all[-1]
                elif self.args.model == 'Pathformer':
                    outputs, balance_loss = self.model(batch_x)
                else:
                    try:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    except:
                        outputs = self.model(batch_x)
                running_times.append(time.time()-start_time)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs.numpy()
                true = batch_y.numpy()
                preds.append(pred)
                trues.append(true)

        print('Inference time: ', time.time() - time_now)


