import pdb
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.build_gen import *
from dataset.dataset_read import dataset_read
from update_aux import update_aux
import scipy.io as io

class Solver(object):
    def __init__(self, args, batch_size=128,
                 target='mnistm', learning_rate=0.0002, interval=10, optimizer='adam',
                 checkpoint_dir=None, save_epoch=20):
        self.batch_size = batch_size
        self.target = target
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.interval = interval
        self.lr = learning_rate
        self.best_correct = 0
        self.args = args
        if self.args.use_target:
            self.ndomain = self.args.ndomain
        else:
            self.ndomain = self.args.ndomain - 1
        self.tgt_portion = self.args.init_tgt_port
        # load source and target domains
        self.datasets, self.dataset_test, self.dataset_size = dataset_read(target, self.batch_size)
        self.niter = self.dataset_size / self.batch_size
        print('Dataset loaded!')

        # define the feature extractor and GCN-based classifier
        self.G = Generator(self.args.net)
        self.C = Classifier(self.args.net, feat=args.nfeat, nclass=args.nclasses)
        self.U = Uncertainty(self.args.net, feat=args.nfeat, nclass=args.nclasses)
        #         pdb.set_trace()
        self.G.cuda()
        self.C.cuda()
        self.U.cuda()
        print('Model initialized!')

        if self.args.load_checkpoint is not None:
            self.state = torch.load(self.args.load_checkpoint)
            self.G.load_state_dict(self.state['G'])
            self.C.load_state_dict(self.state['C'])
            self.U.load_state_dict(self.state['U'])
            print('Model load from: ', self.args.load_checkpoint)

        # initialize statistics (prototypes and adjacency matrix)
        if self.args.load_checkpoint is None:
            self.mean = list()
            self.adj = list()
            self.aux = list()
            self.Y = list()
            for i in range(self.ndomain):
                self.mean.append(torch.zeros(args.nclasses, args.nfeat).cuda())
                self.adj.append(torch.zeros(args.nclasses, args.nclasses).cuda())
                self.aux.append(torch.zeros(args.nclasses, args.nclasses).cuda())
                self.Y.append(torch.zeros(args.nclasses, args.nclasses).cuda())
            print('Statistics initialized!')
        else:
            self.mean = self.state['mean'].cuda()
            self.adj = self.state['adj'].cuda()
            self.aux = self.state['aux'].cuda()
            self.Y = self.state['Y'].cuda()
            print('Statistics loaded!')

        # define the optimizer
        self.set_optimizer(which_opt=optimizer, lr=self.lr)
        print('Optimizer defined!')

    # optimizer definition
    def set_optimizer(self, which_opt='sgd', lr=0.001, momentum=0.9):
        if which_opt == 'sgd':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_c = optim.SGD(self.C.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
        elif which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)
#             self.sche_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_g, T_max=150, eta_min=0)
            self.opt_c = optim.Adam(self.C.parameters(),
                                    lr=lr, weight_decay=0.0005)
#             self.sche_c = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_c, T_max=150, eta_min=0)
            self.opt_u = optim.Adam(self.U.parameters(),
                                    lr=0.01, weight_decay=0.0005)
#             self.sche_u = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_u, T_max=150, eta_min=0)

    # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    def adjust_learning_rate(self, optimizer, epoch, lr):
        lr = lr * (0.1 ** (epoch // self.args.epoch_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    # empty gradients
    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c.zero_grad()
        self.opt_u.zero_grad()

    # compute the Euclidean distance between two tensors
    def euclid_dist(self, x, y):
        x_sq = (x ** 2).mean(-1)
        x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
        y_sq = (y ** 2).mean(-1)
        y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
        xy = torch.mm(x, y.t()) / x.size(-1)
        dist = x_sq_ + y_sq_ - 2 * xy
        return dist        

    
    # assign pseudo labels to target samples
    def pseudo_label(self, logit, feat, log_var):
        pred = F.softmax(logit, dim=1)
        entropy = (-pred * torch.log(pred)).sum(-1)
        label = torch.argmax(logit, dim=-1).long()

        mask = (entropy < self.args.entropy_thr).float()
        index = torch.nonzero(mask).squeeze(-1)
        feat_ = torch.index_select(feat, 0, index)
        label_ = torch.index_select(label, 0, index)
        log_var_ = torch.index_select(log_var, 0, index)
        
        return feat_, label_, log_var_
    
    
    # compute global relation alignment loss
    def prototype_align(self, logits):
        KL_loss = 0
        criterion_KL = nn.KLDivLoss()
        criterion_MSE = nn.MSELoss(size_average=True)
        for i in range(self.ndomain):
            for j in range(i,self.ndomain):
                KL_loss += criterion_KL(logits[i].log(), logits[j]) + criterion_KL(logits[j].log(), logits[i]) 
                KL_loss += criterion_MSE(self.mean[i], self.mean[j])
        return KL_loss
    
    
    # update prototypes and adjacency matrix
    def update_statistics(self, feats, labels, epsilon=1e-5):
        num_labels = 0
        loss_local = 0

        for domain_idx in range(self.ndomain):
            tmp_feat = feats[domain_idx]
            tmp_label = labels[domain_idx]
            num_labels += tmp_label.shape[0]

            if tmp_label.shape[0] == 0:
                break
                # tmp_mean = torch.zeros((self.args.nclasses, self.args.nfeat)).cuda()
            else:
                onehot_label = torch.zeros((tmp_label.shape[0], self.args.nclasses)).scatter_(1, tmp_label.unsqueeze(
                    -1).cpu(), 1).float().cuda()
                domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)
                tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)

                tmp_mask = (tmp_mean.sum(-1) != 0).float().unsqueeze(-1)
                self.mean[domain_idx] = self.mean[domain_idx].detach() * (1 - tmp_mask) + (
                        self.mean[domain_idx].detach() * self.args.beta + tmp_mean * (1 - self.args.beta)) * tmp_mask
                
                tmp_dist = self.euclid_dist(self.mean[domain_idx], self.mean[domain_idx])
                self.adj[domain_idx] = torch.exp(-tmp_dist / (2 * self.args.sigma ** 2))
                
                domain_feature_center = onehot_label.unsqueeze(-1) * self.mean[domain_idx].unsqueeze(0)
                tmp_mean_center = domain_feature_center.sum(1)
                # compute local relation alignment loss
                loss_local += (((tmp_mean_center - tmp_feat) ** 2).mean(-1)).sum()

        return self.adj, loss_local / num_labels
    
    #"""Create the model and start the evaluation process."""
    def val(self):
        conf_dict = {k: [] for k in range(self.args.nclasses)}
        pred_cls_num = torch.zeros(self.args.nclasses)
        with torch.no_grad():
            for batch_idx, data in enumerate(self.datasets):
                img = data['T'].cuda()
                output = F.softmax(self.C(self.G(img)))
                amax_output = torch.argmax(output, dim = -1)
                conf, _ = torch.max(output, dim =-1)

                # class-wise confidence maps
                for idx_cls in range(self.args.nclasses):
                    idx_temp = amax_output == idx_cls
                    pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + torch.sum(idx_temp)
                    if idx_temp.any():
                        conf_cls_temp = conf[idx_temp]
                        conf_dict[idx_cls].extend(conf_cls_temp)
        return conf_dict, pred_cls_num  

        
    # per epoch training in a Multi-Source Domain Adaptation setting    
    def train_adapt(self, epoch, record_file=None):  
        # evaluation & save confidence vectors
        criterion = nn.CrossEntropyLoss(reduce=False).cuda()
        self.adjust_learning_rate(self.opt_g, epoch, self.args.lr)
        self.adjust_learning_rate(self.opt_c, epoch, self.args.lr)
        self.adjust_learning_rate(self.opt_u, epoch, 0.005)
        self.G.train()
        self.C.train()
        self.U.train()
        for batch_idx, data in enumerate(self.datasets):           
            # get the source batches
            img_s = list()
            label_s = list()
            stop_iter = False
            for domain_idx in range(self.ndomain - 1):
                tmp_img = data['S' + str(domain_idx + 1)].cuda()
                tmp_label = data['S' + str(domain_idx + 1) + '_label'].long().cuda()
                img_s.append(tmp_img)
                label_s.append(tmp_label)

            # get the target batch
            img_t = data['T'].cuda()

            # get feature embeddings
            regularizer = 0
            feat_s = list()
            for domain_idx in range(self.ndomain - 1):
                tmp_img = img_s[domain_idx]
                tmp_feat = self.G(tmp_img)
                tmp_feat = F.normalize(tmp_feat, p=2, dim = 1)
                feat_s.append(tmp_feat)

            feat_t = self.G(img_t)
            feat_t = F.normalize(feat_t, p=2, dim = 1)

            # output classification logit
            logit_s = list()
            for domain_idx in range(self.ndomain - 1):
                tmp_logit = self.C(feat_s[domain_idx])
                logit_s.append(tmp_logit)
            logit_t = self.C(feat_t)
            
            # get uncertainty prediction
            log_var_s = list()
            for domain_idx in range(self.ndomain - 1):
                tmp_var = self.U(feat_s[domain_idx])
                log_var_s.append(tmp_var)
            log_var_t = self.U(feat_t)

            # predict the psuedo labels for target domain
            feat_t_, label_t_, log_var_t_ = self.pseudo_label(logit_t, feat_t, log_var_t)
            feat_s.append(feat_t_)
            label_s.append(label_t_)              
            log_var_s.append(log_var_t_)
            
            # update the statistics for source and target domains
            feat_var = list()
            for domain_idx in range(self.ndomain-1):
                feat_var.append(feat_s[domain_idx])
                #feat_var.append(feat_s[domain_idx] * (1 / (log_var_s[domain_idx].detach()**2 + 0.1)))
            feat_var.append(feat_s[domain_idx+1])
            self.adj, loss_local = self.update_statistics(feat_var, label_s)

            # ALM
            loss_alm = 0
            for domain_idx in range(self.ndomain):
                loss_alm += self.args.mu/2.0 * (torch.norm(self.adj[domain_idx]-self.aux[domain_idx]))**2

            # define classification losses
            loss_cls_dom = 0
            loss_cls_src = 0
            
            # get prototype embeddings
            prototype_logits = list()
            for domain_idx in range(self.ndomain-1):
                # domain
                domain_logit = self.C(self.mean[domain_idx])
                prototype_logits.append(F.softmax(domain_logit, dim = 1))
                domain_label = torch.arange(self.args.nclasses).long().cuda()
                loss_cls_dom += criterion(domain_logit, domain_label).mean()
                # source
                loss_cls_src += criterion(logit_s[domain_idx], label_s[domain_idx]).mean()
                #loss_cls_src += ((1 / (log_var_s[domain_idx]**2 + 0.1))* criterion(logit_s[domain_idx], label_s[domain_idx])+ 0.5 * torch.log(1 + log_var_s[domain_idx]**2)).mean()
            prototype_logits.append(F.softmax(self.C(self.mean[domain_idx+1]), dim =1))
            
            # target
            target_prob_ = F.softmax(self.C(feat_t_), dim=1)
            loss_cls_tgt = 0
            if len(label_t_.detach().cpu().numpy()) != 0:
                loss_cls_tgt = (-target_prob_ * torch.log(target_prob_ + 1e-8)).mean()
            
            
            loss_cls = loss_cls_dom + loss_cls_src + loss_cls_tgt
            
            # define total losses
            if torch.sum(self.mean[self.args.ndomain-1]) == 0:
                loss = loss_cls
            else:
                loss = loss_cls + loss_alm
            # back-propagation
            self.reset_grad()
            loss.backward(retain_graph=False)
            self.opt_c.step()
            self.opt_g.step()
            self.opt_u.step()
            
            
            for domain_idx in range(self.ndomain):
                self.adj[domain_idx] = self.adj[domain_idx].detach()
                self.aux[domain_idx] = self.aux[domain_idx].detach()
                self.Y[domain_idx] = self.Y[domain_idx].detach()
            if batch_idx%self.args.aux_iter == 0:
                # update auxiliary variable
                adj = torch.stack(self.adj, dim=2)
                aux = update_aux(adj, self.args.Lambda_global / self.args.mu)
                aux = list(torch.split(aux, 1, dim=2))
                for domain_idx in range(self.ndomain):
                    self.aux[domain_idx] = aux[domain_idx].squeeze().float().cuda()

                # update parameter mu
                self.args.mu = min(self.args.mu * self.args.pho, self.args.max_mu)

                
            # record training information
            if epoch == 0 and batch_idx == 0:
                record = open(record_file, 'a')
                record.write(str(self.args) + '\n')
                record.close()

            if batch_idx % self.interval == 0:
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                    '\tLoss_cls_target: {:.5f}\tLoss_local: {:.5f}\tLoss_ALM: {:.5f}\t mu: {:.5f}'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                        loss_cls_dom.item(), loss_cls_src.item(), loss_cls_tgt, loss_local.item(),
                        loss_alm.item(), self.args.mu))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\n  Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                        '\tLoss_cls_target: {:.5f}\tLoss_local: {:.5f}\tLoss_ALM: {:.5f}\t mu: {:.5f}'.format(
                            epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                            loss_cls_dom.item(), loss_cls_src.item(), loss_cls_tgt,
                            loss_local.item(), loss_alm.item(), self.args.mu))
                    record.close()
        print(torch.abs(log_var_s[0]).mean())
        print(torch.abs(log_var_s[1]).mean())
        print(torch.abs(log_var_s[2]).mean())
        return batch_idx

    
    
    # per epoch test on target domain
    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C.eval()

        test_loss = 0
        correct = 0
        size = 0
        for batch_idx, data in enumerate(self.dataset_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()

            feat = self.G(img)
            logit = self.C(feat)
            
            test_loss += -F.nll_loss(logit, label).item()
            pred = logit.max(1)[1]
            k = label.size()[0]
            correct += pred.eq(label).cpu().sum()
            size += k

        test_loss = test_loss / size

        if correct > self.best_correct:
            self.best_correct = correct
            if save_model:
                best_state = {'G': self.G.state_dict(), 'C': self.C.state_dict()}
                torch.save(best_state, os.path.join(self.checkpoint_dir, 'best_model.pth'))

        # save checkpoint
        if save_model and epoch % self.save_epoch == 0:
            state = {'G': self.G.state_dict(), 'C': self.C.state_dict()}
            torch.save(state, os.path.join(self.checkpoint_dir, 'epoch_' + str(epoch) + '.pth'))
            
            adj = list()
            for i in range(5):
                tmp = self.adj[i].detach().cpu().numpy()
                adj.append(tmp)
            io.savemat('checkpoint/adj_woMLR', {'adj': adj})

        # record test information
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Best Accuracy: {}/{} ({:.4f}%)  \n'.format(
                test_loss, correct, size, 100. * float(correct) / size, self.best_correct, size,
                                          100. * float(self.best_correct) / size))

        if record_file:
            if epoch == 0:
                record = open(record_file, 'a')
                record.write(str(self.args))
                record.close()

            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write(
                '\nEpoch {:>3}, Epoch {:>3} Average loss: {:.5f}, Accuracy: {:.5f}, Best Accuracy: {:.5f}'.format(
                   epoch ,epoch, test_loss, 100. * float(correct) / size, 100. * float(self.best_correct) / size))
            record.close()
