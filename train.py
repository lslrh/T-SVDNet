import os, sys
sys.path.append('./dataset')
sys.path.append('./model')
sys.path.append('./utils')
sys.path.append('./gcn')
import pdb
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import time
from model.build_gen import *
from dataset.dataset_read import dataset_read
from Sol import Solver
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
INIT_TGT_PORT = 0.2
MAX_TGT_PORT = 0.5
TGT_PORT_STEP = 0.05
EPR = 1


parser = argparse.ArgumentParser(description='Training for LtC-MSDA')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--use_target', action='store_true', default=True,
                    help='whether to use target domain')
parser.add_argument('--record_folder', type=str, default='record', metavar='N',
                    help='record folder')
parser.add_argument('--net', type=str, default='conv3_fc2', metavar='N',
                    help='backbone of the generator, lenet, resnet50, resnet101')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='direction to store checkpoints')
parser.add_argument('--load_checkpoint', type=str, default=None, metavar='N',
                    help='the checkpoint to load from')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')
parser.add_argument('--num_rounds', type=int, default=20, metavar='N',
                    help='the number of training round')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N',
                    help='which optimizer to use')
parser.add_argument('--save_epoch', type=int, default=20, metavar='N',
                    help='when to save the model')
parser.add_argument('--save_model', action='store_true', default=True,
                    help='save_model or not')
parser.add_argument('--max_epoch', type=int, default=500, metavar='N',
                    help='the number of training epoch')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='manually set seed')
parser.add_argument('--target', type=str, default='mnistm', metavar='N',
                    help='target domain dataset')
parser.add_argument('--entropy_thr', type=float, default=0.7, metavar='N',
                    help='the threshold for the entropy of prediction')
parser.add_argument('--sigma', type=float, default=0.05, metavar='N',
                    help='the variance parameter for Gaussian function')
parser.add_argument('--beta', type=float, default=0.5, metavar='N',
                    help='the decay ratio for moving average')
parser.add_argument('--Lambda_global', type=float, default=1000, metavar='N',
                    help='the trade-off parameter of losses')
parser.add_argument('--Lambda_KL', type=int, default=0.1, metavar='N',
                    help='the trade-off parameter of losses')
parser.add_argument('--Lambda_ST', type=int, default=0.1, metavar='N',
                    help='the trade-off parameter of losses')
parser.add_argument('--ratio', type=float, default=0.3, metavar='N',
                    help='the ratio of confident samples')
parser.add_argument('--mu', type=float, default=1e-5, metavar='N',
                    help='the scalar mu')
parser.add_argument('--max_mu', type=float, default=1, metavar='N',
                    help='the maximum of mu')
parser.add_argument('--pho', type=float, default=1.1, metavar='N',
                    help='the scalar pho')
parser.add_argument('--epoch_decay', type=int, default=100, metavar='N',
                    help='when to decay epoch')
parser.add_argument('--aux_iter', type=int, default=16, metavar='N',
                    help='when to update auxiliary variable')
parser.add_argument('--n_samples', type=int, default=10, metavar='N',
                    help='number of samples')
parser.add_argument('--init_tgt_port', default=INIT_TGT_PORT, type=float, dest='init_tgt_port',
                    help='The initial portion of target to determine kc')
parser.add_argument('--max_tgt_port', default=MAX_TGT_PORT, type=float, dest='max_tgt_port',
                    help='The max portion of target to determine kc')
parser.add_argument('--tgt_port_step', default=TGT_PORT_STEP, type=float, dest='tgt_port_step',
                    help='The portion step in target domain in every round of self-paced self-trained neural network')
parser.add_argument("--epr", type=int, default=EPR, help="Number of epochs per round for self-training.")
# args = parser.parse_args()
args, unknown = parser.parse_known_args()
# ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
# define task-specific parameters
args.nfeat = 2048
args.nclasses = 10
args.ndomain = 5

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
def main():
    # define the training solver
    solver = Solver(args, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, checkpoint_dir=args.checkpoint_dir, save_epoch=args.save_epoch)
    # define recording files
    record_num = 0
    record_train = '%s/%s_%s.txt' % (
        args.record_folder, args.target, record_num)
    record_test = '%s/%s_%s_test.txt' % (
        args.record_folder, args.target, record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = '%s/%s_%s.txt' % (
            args.record_folder, args.target, record_num)
        record_test = '%s/%s_%s_test.txt' % (
            args.record_folder, args.target, record_num)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.record_folder):
        os.mkdir(args.record_folder)

        
    # train the model
    for t in range(args.max_epoch):
        print('Epoch: ', t)
        # setting: Multi-Source Domain Adaptation
        if args.use_target:
            start = time.time()
            num = solver.train_adapt(t, record_file=record_train)
            print("\ntime/epoch:\t",time.time()-start)
        # setting: Domain Generalization
        else:
            num = solver.train_baseline(t, record_file=record_train)

        # test on target domain
        if t%5 == 0:
            solver.test(t, record_file=record_test, save_model=args.save_model)

if __name__ == '__main__':
    main()
    os.system('watch nvidia-smi')        # test on target domain