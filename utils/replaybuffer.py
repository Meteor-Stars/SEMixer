import torch
import numpy as np

from torch.utils.data import TensorDataset

class ReplayBuffer:
    def __init__(self, args):
        # self.s = np.zeros((args.batch_size, args.state_dim))
        # self.a = np.zeros((args.batch_size, args.action_dim))
        # self.a_logprob = np.zeros((args.batch_size, args.action_dim))

        # self.s = np.zeros((args.batch_size, args.seq_len,args.enc_in))
        # self.a = np.zeros((args.batch_size, args.enc_in,args.action_dim))
        # self.a_logprob = np.zeros((args.batch_size, args.enc_in,args.action_dim))
        # self.r = np.zeros((args.batch_size, 1))
        self.args=args
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []


        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0
        self.size=args.batch_size
    def store(self, s, a, r):
        # self.s[self.count] = s
        # self.a[self.count] = a
        # self.a_logprob[self.count] = a_logprob
        # self.r[self.count] = r
        # # self.s_[self.count] = s_
        # # self.dw[self.count] = dw
        # # self.done[self.count] = done
        # self.count += 1
        self.s.append(s)
        self.a.append(a)
        # self.a_logprob.append(a_logprob)
        self.r.append(r)
    def creat_loader(self):
        self.numpy_to_tensor()
        train_dataset = TensorDataset(torch.from_numpy(self.s_cat), torch.from_numpy(self.a_cat),
                                      torch.from_numpy(self.r_cat))


        data_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=self.args.batch_size,
                                                  shuffle=True)
        return data_loader
        # index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        # batch_s = torch.tensor(self.s[index], dtype=torch.float)
        # batch_a = torch.tensor(self.a[index], dtype=torch.float)
        # batch_r = torch.tensor(self.r[index], dtype=torch.float)
        # batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        # batch_dw = torch.tensor(self.dw[index], dtype=torch.float)
        #
        # return batch_s, batch_a, batch_r, batch_s_, batch_dw
    def numpy_to_tensor(self):
        self.s_cat=np.concatenate(self.s,axis=0)
        self.a_cat = np.concatenate(self.a, axis=0)
        self.r_cat = np.concatenate(self.r, axis=0)

        # self.s_cat = np.concatenate(self.s, axis=0)

        # s = torch.tensor(np.concatenate(self.s,axis=0), dtype=torch.float)
        # a = torch.tensor(np.concatenate(self.a,axis=0), dtype=torch.float)
        # a_logprob = torch.tensor(np.concatenate(self.a_logprob,axis=0), dtype=torch.float)
        # r = torch.tensor(np.concatenate(self.r,axis=0), dtype=torch.float)

        # s_ = torch.tensor(self.s_, dtype=torch.float)
        # dw = torch.tensor(self.dw, dtype=torch.float)
        # done = torch.tensor(self.done, dtype=torch.float)

        # return s, a, a_logprob, r, s_, dw, done

        # return s, a, a_logprob, r
