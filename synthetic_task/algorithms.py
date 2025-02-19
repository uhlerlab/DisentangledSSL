import torch.optim as optim
from models import *
from losses import *
import torch
import torch.nn as nn
from dataset import augment_data
import utils
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F


def mlp_head(dim_in, feat_dim):
    return nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )


class MVInfoMaxModel(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=2, activation='relu', initialization = 'xavier', distribution='normal', vmfkappa=1,
                 lr=1e-4, ratio=1, use_label=False, beta_start_value=1e-3, beta_end_value=1, beta_n_iterations=100000, beta_start_iteration=50000, split=50,
                 head='none', simclr=False):
        
        super().__init__()
        self.lr = lr
        self.ratio = ratio
        self.use_label = use_label
        self.iterations = 0
        self.split = split
        if beta_end_value > 0:
            self.beta_scheduler = utils.ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                    n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)
        self.beta_start_value = beta_start_value
        self.beta_end_value = beta_end_value
        self.distribution = distribution
        self.vmfkappa = vmfkappa
        self.simclr = simclr
        self.embed_dim = embed_dim

        self.encoder_x1 = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        self.encoder_x2 = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)

        if head == 'linear':
            self.head1 = nn.Linear(embed_dim, embed_dim)
            self.head2 = nn.Linear(embed_dim, embed_dim)
        elif head == 'mlp':
            self.head1 = mlp_head(embed_dim, embed_dim)
            self.head2 = mlp_head(embed_dim, embed_dim)
        elif head == 'none':
            self.head1 = nn.Identity()
            self.head2 = nn.Identity()
        else:
            raise NotImplementedError

        self.phead1 = ProbabilisticEncoder(self.head1, distribution=distribution, vmfkappa=vmfkappa)
        self.phead2 = ProbabilisticEncoder(self.head2, distribution=distribution, vmfkappa=vmfkappa)

        #critics
        self.critic = SupConLoss()
        
    def name(self):
        return 'Multiview IB'

    def forward(self, x1, x2):
        self.iterations += 1
        e1 = self.encoder_x1(x1)
        e2 = self.encoder_x2(x2)
        if self.simclr:
            mu1 = self.head1(e1)
            mu2 = self.head2(e2)
        else:
            p_z1_given_v1, mu1 = self.phead1(e1)
            p_z2_given_v2, mu2 = self.phead2(e2)
        
        if self.simclr:
            z1 = mu1
            z2 = mu2
        else:
            z1 = p_z1_given_v1.rsample()
            z2 = p_z2_given_v2.rsample()

        z1, z2 = nn.functional.normalize(z1, dim=-1), nn.functional.normalize(z2, dim=-1)
        concat_embed = torch.cat([z1.unsqueeze(dim=1), z2.unsqueeze(dim=1)], dim=1)
        joint_loss, loss_x, loss_y = self.critic(concat_embed)        
    
        if self.distribution == 'normal':
            skl = kl_divergence(mu1, mu2)
        elif self.distribution == 'vmf':
            skl = kl_vmf(mu1, mu2)

        if self.beta_end_value > 0:
            beta = self.beta_scheduler(self.iterations)
        else:
            beta = self.beta_start_value
        loss = joint_loss + beta * skl

        return loss, {'loss': loss.item(), 'clip': joint_loss.item(), 'skl': skl.item(), 'loss_x': loss_x.item(), 'loss_y': loss_y.item(), 'beta': beta}
        
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        e1 = self.encoder_x1(x1)
        e2 = self.encoder_x2(x2)
        return torch.cat([e1, e2], dim=1)

    def get_separate_embeddings(self, x):
        x1, x2 = x[0], x[1]
        e1 = self.encoder_x1(x1)
        e2 = self.encoder_x2(x2)
        return e1, e2


def train(model, train_loader, optimizer, train_dataset, test_dataset, num_epoch=50):
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
    for _iter in range(num_epoch):
        logs = {}
        logs.update({'Epoch': _iter})
        loss_meter = utils.AverageMeter('loss')
        clip_meter = utils.AverageMeter('clip')
        skl_meter = utils.AverageMeter('skl')
        loss_x_meter = utils.AverageMeter('loss_x')
        loss_y_meter = utils.AverageMeter('loss_y')
        beta_meter = utils.AverageMeter('beta')

        for i_batch, data_batch in enumerate(train_loader):
            model.train()
            x1 = data_batch[0].float().cuda()
            x2 = data_batch[1].float().cuda()
            x1 = augment_data(x1)
            x2 = augment_data(x2)
            loss, train_logs = model(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(train_logs['loss'])
            clip_meter.update(train_logs['clip'])
            skl_meter.update(train_logs['skl'])
            loss_x_meter.update(train_logs['loss_x'])
            loss_y_meter.update(train_logs['loss_y'])
            beta_meter.update(train_logs['beta'])
            
            if _iter == 0 and i_batch == 0:
                logs.update({'loss': 0, 'clip': 0, 'skl': 0, 'loss_x': 0, 'loss_y': 0, 'beta': 0})
                logs.update({'Test Acc 1': 0, 'Test Acc 2': 0, 'Test Acc 3': 0})
                utils.print_row([i for i in logs.keys()], colwidth=12)
            
            if i_batch == len(train_loader)-1:
                logs.update({'loss': loss_meter.avg, 'clip': clip_meter.avg, 'skl': skl_meter.avg, 'loss_x': loss_x_meter.avg, 'loss_y': loss_y_meter.avg, 'beta': beta_meter.avg})
                
        if _iter == num_epoch-1:
            test_acc = linearprobe(model, train_dataset, test_dataset)
            logs.update({'Test Acc 1': test_acc[0], 'Test Acc 2': test_acc[1], 'Test Acc 3': test_acc[2]})
        
        utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        

        lr_scheduler.step()
    return logs


def linearprobe(model, train_dataset, test_dataset):
    model.eval()
    train_embeds = model.get_embedding(torch.stack(train_dataset[:][:-3]).cuda()).detach().cpu().numpy()
    train_labels1 = train_dataset[:][-3]
    train_labels2 = train_dataset[:][-2]
    train_labels3 = train_dataset[:][-1]

    test_embeds = model.get_embedding(torch.stack(test_dataset[:][:-3]).cuda()).detach().cpu().numpy()
    test_labels1 = test_dataset[:][-3]
    test_labels2 = test_dataset[:][-2]
    test_labels3 = test_dataset[:][-1]

    clf = LogisticRegression(max_iter=200).fit(train_embeds, train_labels1)
    score1 = clf.score(test_embeds, test_labels1)

    clf = LogisticRegression(max_iter=200).fit(train_embeds, train_labels2)
    score2 = clf.score(test_embeds, test_labels2)

    clf = LogisticRegression(max_iter=200).fit(train_embeds, train_labels3)
    score3 = clf.score(test_embeds, test_labels3)

    return score1, score2, score3


class DisenModel(nn.Module):
    def __init__(self, zsmodel, x1_dim, x2_dim, hidden_dim, embed_dim, zs_dim=50, layers=2, activation='relu', initialization = 'xavier', 
                 lr=1e-4, lmd_start_value=1e-3, lmd_end_value=1, lmd_n_iterations=100000, lmd_start_iteration=50000,
                 ortho_norm=True, condzs=True, proj=False, usezsx=True, apdzs=True):
        super().__init__()
        self.lr = lr
        self.ortho_norm = ortho_norm
        self.condzs = condzs
        self.proj = proj
        self.usezsx = usezsx
        self.apdzs = apdzs
        self.iterations = 0
        if lmd_end_value > 0:
            self.lmd_scheduler = utils.ExponentialScheduler(start_value=lmd_start_value, end_value=lmd_end_value,
                                                    n_iterations=lmd_n_iterations, start_iteration=lmd_start_iteration)
        self.lmd_start_value = lmd_start_value
        self.lmd_end_value = lmd_end_value

        if self.condzs:
            self.encoder_x1 = mlp(x1_dim+zs_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
            self.encoder_x2 = mlp(x2_dim+zs_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        else:
            self.encoder_x1 = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
            self.encoder_x2 = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        
        if self.proj=='mlp':
            self.projection_x1 = mlp(embed_dim*2, embed_dim*2, embed_dim*2, 1, activation, initialization = initialization)
            self.projection_x2 = mlp(embed_dim*2, embed_dim*2, embed_dim*2, 1, activation, initialization = initialization)
        elif self.proj=='linear':
            self.projection_x1 = nn.Linear(embed_dim*2, embed_dim*2)
            self.projection_x2 = nn.Linear(embed_dim*2, embed_dim*2)
        else:
            self.projection_x1 = nn.Identity()
            self.projection_x2 = nn.Identity()

        self.critic = SupConLoss()

        self.zsmodel = zsmodel
        self.zsmodel.requires_grad = False
        self.embed_dim = embed_dim

    def forward(self, x1, x2, v1, v2):
        self.iterations += 1

        zsx1 = self.zsmodel.encoder_x1(x1).detach()
        zsx2 = self.zsmodel.encoder_x2(x2).detach()
        zsxv1 = self.zsmodel.encoder_x1(v1).detach()
        zsxv2 = self.zsmodel.encoder_x2(v2).detach()

        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, zsx1], dim=1))
            z1xv1 = self.encoder_x1(torch.cat([v1, zsxv1], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, zsx2], dim=1))
            z2xv2 = self.encoder_x2(torch.cat([v2, zsxv2], dim=1))
        else:
            z1x1 = self.encoder_x1(x1)
            z1xv1 = self.encoder_x1(v1)
            z2x2 = self.encoder_x2(x2)
            z2xv2 = self.encoder_x2(v2)

        if self.apdzs:
            if self.usezsx:
                zjointx1 = torch.cat([z1x1, zsx1], dim=1)
                zjointx2 = torch.cat([z2x2, zsx2], dim=1)
                zjointxv1 = torch.cat([z1xv1, zsxv1], dim=1)
                zjointxv2 = torch.cat([z2xv2, zsxv2], dim=1)
            else:
                zjointx1 = torch.cat([z1x1, zsx2], dim=1)
                zjointx2 = torch.cat([z2x2, zsx1], dim=1)
                zjointxv1 = torch.cat([z1xv1, zsxv2], dim=1)
                zjointxv2 = torch.cat([z2xv2, zsxv1], dim=1)

            zjointx1 = self.projection_x1(zjointx1)
            zjointx2 = self.projection_x2(zjointx2)
            zjointxv1 = self.projection_x1(zjointxv1)
            zjointxv2 = self.projection_x2(zjointxv2)

            zjointx1, zjointx2 = nn.functional.normalize(zjointx1, dim=-1), nn.functional.normalize(zjointx2, dim=-1)
            zjointxv1, zjointxv2 = nn.functional.normalize(zjointxv1, dim=-1), nn.functional.normalize(zjointxv2, dim=-1)
            concat_embed_x1 = torch.cat([zjointx1.unsqueeze(dim=1), zjointxv1.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([zjointx2.unsqueeze(dim=1), zjointxv2.unsqueeze(dim=1)], dim=1)
        else:
            z1x1_norm, z2x2_norm = nn.functional.normalize(z1x1, dim=-1), nn.functional.normalize(z2x2, dim=-1)
            z1xv1_norm, z2xv2_norm = nn.functional.normalize(z1xv1, dim=-1), nn.functional.normalize(z2xv2, dim=-1)
            concat_embed_x1 = torch.cat([z1x1_norm.unsqueeze(dim=1), z1xv1_norm.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([z2x2_norm.unsqueeze(dim=1), z2xv2_norm.unsqueeze(dim=1)], dim=1)

        specific_loss_x1, loss_x1, loss_y1 = self.critic(concat_embed_x1)
        specific_loss_x2, loss_x2, loss_y2 = self.critic(concat_embed_x2)

        loss_specific = specific_loss_x1 + specific_loss_x2

        if self.lmd_end_value > 0:
            lmd = self.lmd_scheduler(self.iterations)
        else:
            lmd = self.lmd_start_value

        loss_ortho = 0.5 * (ortho_loss(z1x1, zsx1, norm=self.ortho_norm) + ortho_loss(z2x2, zsx2, norm=self.ortho_norm)) + \
                    0.5 * (ortho_loss(z1xv1, zsxv1, norm=self.ortho_norm) + ortho_loss(z2xv2, zsxv2, norm=self.ortho_norm))
        
        loss = loss_specific + lmd * loss_ortho

        return loss, {'loss': loss.item(), 'specific': loss_specific.item(), 'ortho': loss_ortho.item(), 'lmd': lmd}
            
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        zsx1 = self.zsmodel.encoder_x1(x1).detach()
        zsx2 = self.zsmodel.encoder_x2(x2).detach()
        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, zsx1], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, zsx2], dim=1))
        else:
            z1x1 = self.encoder_x1(x1)
            z2x2 = self.encoder_x2(x2)
        return z1x1, z2x2


def train_Disen(model, train_loader, optimizer, train_dataset, test_dataset, num_epoch=50,
                noise_scale=0.01, drop_scale=10):
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
    for _iter in range(num_epoch):
        logs = {}
        logs.update({'Epoch': _iter})
        loss_meter = utils.AverageMeter('loss')
        specific_meter = utils.AverageMeter('specific')
        ortho_meter = utils.AverageMeter('ortho')
        lmd_meter = utils.AverageMeter('lmd')
        
        for i_batch, data_batch in enumerate(train_loader):
            model.train()
            x1 = data_batch[0].float().cuda()
            x2 = data_batch[1].float().cuda()
            x1 = augment_data(x1, noise_scale, drop_scale)
            x2 = augment_data(x2, noise_scale, drop_scale)
            v1 = augment_data(x1, noise_scale, drop_scale)
            v2 = augment_data(x2, noise_scale, drop_scale)
            loss, train_logs = model(x1, x2, v1, v2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(train_logs['loss'])
            specific_meter.update(train_logs['specific'])
            ortho_meter.update(train_logs['ortho'])
            lmd_meter.update(train_logs['lmd'])

            if _iter == 0 and i_batch == 0:
                logs.update({'loss': 0, 'specific': 0, 'ortho': 0, 'lmd': 0})
                logs.update({'Acc_spe1 1': 0, 'Acc_spe1 2': 0, 'Acc_spe1 3': 0,
                             'Acc_spe2 1': 0, 'Acc_spe2 2': 0, 'Acc_spe2 3': 0})
                utils.print_row([i for i in logs.keys()], colwidth=12)
            
            if i_batch == len(train_loader)-1:
                logs.update({'loss': loss_meter.avg, 'specific': specific_meter.avg, 'ortho': ortho_meter.avg, 'lmd': lmd_meter.avg})
        
        lr_scheduler.step()
        if _iter == num_epoch-1:
            test_acc = linearprobe_Disen(model, train_dataset, test_dataset)
            logs.update({'Acc_spe1 1': test_acc[0][0], 'Acc_spe1 2': test_acc[0][1], 'Acc_spe1 3': test_acc[0][2],
                             'Acc_spe2 1': test_acc[1][0], 'Acc_spe2 2': test_acc[1][1], 'Acc_spe2 3': test_acc[1][2]})   
        
        utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        

    return logs


def linearprobe_Disen(model, train_dataset, test_dataset):
    model.eval()
    train_spe1, train_spe2 = model.get_embedding(torch.stack(train_dataset[:][:-3]).cuda())
    train_spe1 = train_spe1.detach().cpu().numpy()
    train_spe2 = train_spe2.detach().cpu().numpy()
    train_labels1 = train_dataset[:][-3]
    train_labels2 = train_dataset[:][-2]
    train_labels3 = train_dataset[:][-1]

    test_spe1, test_spe2 = model.get_embedding(torch.stack(test_dataset[:][:-3]).cuda())
    test_spe1 = test_spe1.detach().cpu().numpy()
    test_spe2 = test_spe2.detach().cpu().numpy()
    test_labels1 = test_dataset[:][-3]
    test_labels2 = test_dataset[:][-2]
    test_labels3 = test_dataset[:][-1]

    clf = LogisticRegression(max_iter=200).fit(train_spe1, train_labels1)
    score1 = clf.score(test_spe1, test_labels1)
    clf = LogisticRegression(max_iter=200).fit(train_spe1, train_labels2)
    score2 = clf.score(test_spe1, test_labels2)
    clf = LogisticRegression(max_iter=200).fit(train_spe1, train_labels3)
    score3 = clf.score(test_spe1, test_labels3)
    score_spe1 = (score1, score2, score3)

    clf = LogisticRegression(max_iter=200).fit(train_spe2, train_labels1)
    score1 = clf.score(test_spe2, test_labels1)
    clf = LogisticRegression(max_iter=200).fit(train_spe2, train_labels2)
    score2 = clf.score(test_spe2, test_labels2)
    clf = LogisticRegression(max_iter=200).fit(train_spe2, train_labels3)
    score3 = clf.score(test_spe2, test_labels3)
    score_spe2 = (score1, score2, score3)

    return score_spe1, score_spe2


class JointDisenModel(nn.Module):
    """
    Joint optimization:
    max I(zsx;Y) + a * [I(zsx,z1x;X) - lmd*I(zsx;z1x)]
    zsx has the probablistic encoder; z1x has the deterministic encoder
    """
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=2, activation='relu', initialization='xavier', 
                 distribution='normal', vmfkappa=1, lr=1e-4,
                 lmd_start_value=1e-3, lmd_end_value=1, lmd_n_iterations=100000, lmd_start_iteration=50000,
                 a=1, ortho_norm=True, condzs=True, proj=False, usezsx=True, apdzs=True):
        super().__init__()
        self.lr = lr
        self.ortho_norm = ortho_norm
        self.condzs = condzs
        self.proj = proj
        self.usezsx = usezsx
        self.apdzs = apdzs
        self.vmfkappa = vmfkappa
        self.iterations = 0
        if lmd_end_value > 0:
            self.lmd_scheduler = utils.ExponentialScheduler(start_value=lmd_start_value, end_value=lmd_end_value,
                                                    n_iterations=lmd_n_iterations, start_iteration=lmd_start_iteration)
        self.lmd_start_value = lmd_start_value
        self.lmd_end_value = lmd_end_value
        self.a = a

        self.encoder_x1s = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        self.encoder_x2s = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)

        self.phead1 = ProbabilisticEncoder(nn.Identity(), distribution=distribution, vmfkappa=vmfkappa)
        self.phead2 = ProbabilisticEncoder(nn.Identity(), distribution=distribution, vmfkappa=vmfkappa)

        if self.condzs:
            self.encoder_x1 = mlp(x1_dim+embed_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
            self.encoder_x2 = mlp(x2_dim+embed_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        else:
            self.encoder_x1 = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
            self.encoder_x2 = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)

        if self.proj:
            self.projection_x1 = mlp(embed_dim*2, embed_dim*2, embed_dim*2, 1, activation, initialization = initialization)
            self.projection_x2 = mlp(embed_dim*2, embed_dim*2, embed_dim*2, 1, activation, initialization = initialization)

        self.critic = SupConLoss()
        self.embed_dim = embed_dim

    def forward(self, x1, x2, v1, v2):
        self.iterations += 1
        e1 = self.encoder_x1s(x1)
        e2 = self.encoder_x2s(x2)
        e1_v = self.encoder_x1s(v1)
        e2_v = self.encoder_x2s(v2)

        p_zs1_given_x1, mu1 = self.phead1(e1)
        p_zs2_given_x2, mu2 = self.phead2(e2)
        p_zsv1_given_v1, mu1_v = self.phead1(e1_v)
        p_zsv2_given_v2, mu2_v = self.phead2(e2_v)

        zs1 = p_zs1_given_x1.rsample()
        zs2 = p_zs2_given_x2.rsample()
        zsv1 = p_zsv1_given_v1.rsample()
        zsv2 = p_zsv2_given_v2.rsample()

        concat_embed = torch.cat([zs1.unsqueeze(dim=1), zs2.unsqueeze(dim=1)], dim=1)
        concat_embed_v = torch.cat([zsv1.unsqueeze(dim=1), zsv2.unsqueeze(dim=1)], dim=1)
        joint_loss, loss_x, loss_y = self.critic(concat_embed)
        joint_loss_v, loss_x_v, loss_y_v = self.critic(concat_embed_v)
        joint_loss = 0.5 * (joint_loss + joint_loss_v)
        loss_x = 0.5 * (loss_x + loss_x_v)
        loss_y = 0.5 * (loss_y + loss_y_v)
        loss_shared = joint_loss

        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, e1], dim=1))
            z1xv1 = self.encoder_x1(torch.cat([v1, e1_v], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, e2], dim=1))
            z2xv2 = self.encoder_x2(torch.cat([v2, e2_v], dim=1))
        else:
            z1x1 = self.encoder_x1(x1)
            z1xv1 = self.encoder_x1(v1)
            z2x2 = self.encoder_x2(x2)
            z2xv2 = self.encoder_x2(v2)

        if self.apdzs:
            if self.usezsx:
                zjointx1 = torch.cat([z1x1, e1], dim=1)
                zjointx2 = torch.cat([z2x2, e2], dim=1)
                zjointxv1 = torch.cat([z1xv1, e1_v], dim=1)
                zjointxv2 = torch.cat([z2xv2, e2_v], dim=1)
            else:
                zjointx1 = torch.cat([z1x1, e2], dim=1)
                zjointx2 = torch.cat([z2x2, e1], dim=1)
                zjointxv1 = torch.cat([z1xv1, e2_v], dim=1)
                zjointxv2 = torch.cat([z2xv2, e1_v], dim=1)

            if self.proj:
                zjointx1 = self.projection_x1(zjointx1)
                zjointx2 = self.projection_x2(zjointx2)
                zjointxv1 = self.projection_x1(zjointxv1)
                zjointxv2 = self.projection_x2(zjointxv2)

            zjointx1, zjointx2 = nn.functional.normalize(zjointx1, dim=-1), nn.functional.normalize(zjointx2, dim=-1)
            zjointxv1, zjointxv2 = nn.functional.normalize(zjointxv1, dim=-1), nn.functional.normalize(zjointxv2, dim=-1)
            concat_embed_x1 = torch.cat([zjointx1.unsqueeze(dim=1), zjointxv1.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([zjointx2.unsqueeze(dim=1), zjointxv2.unsqueeze(dim=1)], dim=1)
        else:
            z1x1_norm, z2x2_norm = nn.functional.normalize(z1x1, dim=-1), nn.functional.normalize(z2x2, dim=-1)
            z1xv1_norm, z2xv2_norm = nn.functional.normalize(z1xv1, dim=-1), nn.functional.normalize(z2xv2, dim=-1)
            concat_embed_x1 = torch.cat([z1x1_norm.unsqueeze(dim=1), z1xv1_norm.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([z2x2_norm.unsqueeze(dim=1), z2xv2_norm.unsqueeze(dim=1)], dim=1)

        specific_loss_x1, loss_x1, loss_y1 = self.critic(concat_embed_x1)
        specific_loss_x2, loss_x2, loss_y2 = self.critic(concat_embed_x2)

        loss_specific = specific_loss_x1 + specific_loss_x2

        if self.lmd_end_value > 0:
            lmd = self.lmd_scheduler(self.iterations)
        else:
            lmd = self.lmd_start_value

        loss_ortho = 0.5 * (ortho_loss(z1x1, e1, norm=self.ortho_norm) + ortho_loss(z2x2, e2, norm=self.ortho_norm)) + \
                    0.5 * (ortho_loss(z1xv1, e1_v, norm=self.ortho_norm) + ortho_loss(z2xv2, e2_v, norm=self.ortho_norm))
        
        loss = 2 * loss_shared/(1+self.a) + self.a * loss_specific/(1+self.a) + lmd * loss_ortho

        return loss, {'loss': loss.item(), 'shared': loss_shared.item(), 'clip': joint_loss.item(), 'loss_x': loss_x.item(), 'loss_y': loss_y.item(),
                       'specific': loss_specific.item(), 'ortho': loss_ortho.item(), 'lmd': lmd}#, 'beta': beta,
    
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        zsx1 = self.encoder_x1s(x1)
        zsx2 = self.encoder_x2s(x2)
        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, zsx1], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, zsx2], dim=1))
        else:
            z1x1 = self.encoder_x1(x1)
            z2x2 = self.encoder_x2(x2)
        return zsx1, zsx2, z1x1, z2x2
    
    def train_model(self, train_loader, train_dataset, test_dataset, optimizer, num_epoch=50, noise_scale=0.01, drop_scale=10):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
        for _iter in range(num_epoch):
            logs = {}
            logs.update({'Epoch': _iter})
            loss_meter = utils.AverageMeter('loss')
            shared_meter = utils.AverageMeter('shared')
            clip_meter = utils.AverageMeter('clip')
            loss_x_meter = utils.AverageMeter('loss_x')
            loss_y_meter = utils.AverageMeter('loss_y')
            specific_meter = utils.AverageMeter('specific')
            ortho_meter = utils.AverageMeter('ortho')
            lmd_meter = utils.AverageMeter('lmd')

            for i_batch, data_batch in enumerate(train_loader):
                self.train()
                x1 = data_batch[0].float().cuda()
                x2 = data_batch[1].float().cuda()
                x1 = augment_data(x1, noise_scale, drop_scale)
                x2 = augment_data(x2, noise_scale, drop_scale)
                v1 = augment_data(x1, noise_scale, drop_scale)
                v2 = augment_data(x2, noise_scale, drop_scale)
                loss, train_logs = self(x1, x2, v1, v2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meter.update(train_logs['loss'])
                shared_meter.update(train_logs['shared'])
                clip_meter.update(train_logs['clip'])
                loss_x_meter.update(train_logs['loss_x'])
                loss_y_meter.update(train_logs['loss_y'])
                specific_meter.update(train_logs['specific'])
                ortho_meter.update(train_logs['ortho'])
                lmd_meter.update(train_logs['lmd'])
                
                def getemb(dataset):
                    zsx, zsy, z1x, z2y = self.get_embedding(torch.stack(dataset[:][:-3]).cuda())
                    zs = torch.cat([zsx, zsy], dim=1).cpu().detach().numpy()
                    z1x = z1x.cpu().detach().numpy()
                    z2y = z2y.cpu().detach().numpy()
                    return zs, z1x, z2y
                
                def linearprobe(train_dataset, test_dataset):
                    train_zs, train_z1x, train_z2y = getemb(train_dataset)
                    test_zs, test_z1x, test_z2y = getemb(test_dataset)
                    score_s = linearprobe_acc(train_zs, test_zs, train_dataset, test_dataset)
                    score_spe1 = linearprobe_acc(train_z1x, test_z1x, train_dataset, test_dataset)
                    score_spe2 = linearprobe_acc(train_z2y, test_z2y, train_dataset, test_dataset)
                    return (score_s, score_spe1, score_spe2)
                
                def linearprobe_acc(train_z, test_z, train_dataset, test_dataset):
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-3])
                    score1 = clf.score(test_z, test_dataset[:][-3])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-2])
                    score2 = clf.score(test_z, test_dataset[:][-2])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-1])
                    score3 = clf.score(test_z, test_dataset[:][-1])
                    return (score1, score2, score3)

                if _iter == 0 and i_batch == 0:
                    logs.update({'loss': 0, 'shared': 0, 'clip': 0, 'loss_x': 0, 'loss_y': 0, 'specific': 0, 'ortho': 0, 'lmd': 0})
                    logs.update({'Acc_s 1': 0, 'Acc_s 2': 0, 'Acc_s 3': 0,
                                 'Acc_spe1 1': 0, 'Acc_spe1 2': 0, 'Acc_spe1 3': 0,
                                 'Acc_spe2 1': 0, 'Acc_spe2 2': 0, 'Acc_spe2 3': 0})
                    utils.print_row([i for i in logs.keys()], colwidth=12)
                
                if i_batch == len(train_loader)-1:
                    logs.update({'loss': loss_meter.avg, 'shared': shared_meter.avg, 'clip': clip_meter.avg,
                                    'loss_x': loss_x_meter.avg, 'loss_y': loss_y_meter.avg, 'specific': specific_meter.avg, 'ortho': ortho_meter.avg,
                                    'lmd': lmd_meter.avg})
                    
            if _iter == num_epoch-1:
                test_acc = linearprobe(train_dataset, test_dataset)
                logs.update({'Acc_s 1': test_acc[0][0], 'Acc_s 2': test_acc[0][1], 'Acc_s 3': test_acc[0][2],
                                'Acc_spe1 1': test_acc[1][0], 'Acc_spe1 2': test_acc[1][1], 'Acc_spe1 3': test_acc[1][2],
                                'Acc_spe2 1': test_acc[2][0], 'Acc_spe2 2': test_acc[2][1], 'Acc_spe2 3': test_acc[2][2]})                
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        
            else:
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)  

            lr_scheduler.step()
        return logs
    

class Focal(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=2, activation='relu', initialization='xavier',  
                a=1, lmd=1e-3):
        super().__init__()
        self.encoder_x1s = mlp(x1_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        self.encoder_x2s = mlp(x2_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        self.encoder_x1 = mlp(x1_dim+embed_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        self.encoder_x2 = mlp(x2_dim+embed_dim, hidden_dim, embed_dim, layers, activation, initialization = initialization)
        
        self.a = a
        self.lmd = lmd
        self.embed_dim = embed_dim

        self.critic_c = SupConLoss()
        self.critic_s = SupConLoss()

    def forward(self, x1, x2, v1, v2):
        e1_c = self.encoder_x1s(x1)
        e2_c = self.encoder_x2s(x2)
        e1_c_v = self.encoder_x1s(v1)
        e2_c_v = self.encoder_x2s(v2)
        e1_s = self.encoder_x1(torch.cat([x1, e1_c], dim=1))
        e2_s = self.encoder_x2(torch.cat([x2, e2_c], dim=1))
        e1_s_v = self.encoder_x1(torch.cat([v1, e1_c_v], dim=1))
        e2_s_v = self.encoder_x2(torch.cat([v2, e2_c_v], dim=1))

        e1_c, e2_c = F.normalize(e1_c, dim=-1), F.normalize(e2_c, dim=-1)
        e1_c_v, e2_c_v = F.normalize(e1_c_v, dim=-1), F.normalize(e2_c_v, dim=-1)
        e1_s, e2_s = F.normalize(e1_s, dim=-1), F.normalize(e2_s, dim=-1)
        e1_s_v, e2_s_v = F.normalize(e1_s_v, dim=-1), F.normalize(e2_s_v, dim=-1)

        concat_embed_c = torch.cat([e1_c.unsqueeze(1), e2_c.unsqueeze(1)], dim=1)
        concat_embed_c_v = torch.cat([e1_c_v.unsqueeze(1), e2_c_v.unsqueeze(1)], dim=1)
        joint_loss_c, _, _ = self.critic_c(concat_embed_c)
        joint_loss_c_v, _, _ = self.critic_c(concat_embed_c_v)
        joint_loss = joint_loss_c + joint_loss_c_v

        concat_embed_s1 = torch.cat([e1_s.unsqueeze(1), e1_s_v.unsqueeze(1)], dim=1)
        concat_embed_s2 = torch.cat([e2_s.unsqueeze(1), e2_s_v.unsqueeze(1)], dim=1)
        specific_loss_s1, _, _ = self.critic_s(concat_embed_s1)
        specific_loss_s2, _, _ = self.critic_s(concat_embed_s2)
        loss_specific = specific_loss_s1 + specific_loss_s2

        loss_ortho = ortho_loss_focal(e1_s, e1_c) + ortho_loss_focal(e2_s, e2_c) + \
                        ortho_loss_focal(e1_s_v, e1_c_v) + ortho_loss_focal(e2_s_v, e2_c_v) + \
                        ortho_loss_focal(e1_s, e2_s) + ortho_loss_focal(e1_s_v, e2_s_v)
        
        loss = joint_loss + self.a * loss_specific + self.lmd * loss_ortho
        return loss, {'loss': loss.item(), 'loss_shared': joint_loss.item(), 'loss_specific': loss_specific.item(), 'loss_ortho': loss_ortho.item()}
    
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        e1_c = self.encoder_x1s(x1)
        e2_c = self.encoder_x2s(x2)
        e1_s = self.encoder_x1(torch.cat([x1, e1_c], dim=1))
        e2_s = self.encoder_x2(torch.cat([x2, e2_c], dim=1))
        return e1_c, e2_c, e1_s, e2_s
    
    def train_model_focal(self, train_loader, train_dataset, test_dataset, optimizer, num_epoch=50, noise_scale=0.01, drop_scale=10):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
        for _iter in range(num_epoch):
            logs = {}
            logs.update({'Epoch': _iter})
            loss_meter = utils.AverageMeter('loss')
            shared_meter = utils.AverageMeter('shared')
            specific_meter = utils.AverageMeter('specific')
            ortho_meter = utils.AverageMeter('ortho')
            lmd_meter = utils.AverageMeter('lmd')

            for i_batch, data_batch in enumerate(train_loader):
                self.train()
                x1 = data_batch[0].float().cuda()
                x2 = data_batch[1].float().cuda()
                x1 = augment_data(x1, noise_scale, drop_scale)
                x2 = augment_data(x2, noise_scale, drop_scale)
                v1 = augment_data(x1, noise_scale, drop_scale)
                v2 = augment_data(x2, noise_scale, drop_scale)
                loss, train_logs = self(x1, x2, v1, v2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meter.update(train_logs['loss'])
                shared_meter.update(train_logs['loss_shared'])
                specific_meter.update(train_logs['loss_specific'])
                ortho_meter.update(train_logs['loss_ortho'])
                
                def getemb(dataset):
                    zsx, zsy, z1x, z2y = self.get_embedding(torch.stack(dataset[:][:-3]).cuda())
                    zs = torch.cat([zsx, zsy], dim=1).cpu().detach().numpy()
                    z1x = z1x.cpu().detach().numpy()
                    z2y = z2y.cpu().detach().numpy()
                    return zs, z1x, z2y
                
                def linearprobe(train_dataset, test_dataset):
                    train_zs, train_z1x, train_z2y = getemb(train_dataset)
                    test_zs, test_z1x, test_z2y = getemb(test_dataset)
                    score_s = linearprobe_acc(train_zs, test_zs, train_dataset, test_dataset)
                    score_spe1 = linearprobe_acc(train_z1x, test_z1x, train_dataset, test_dataset)
                    score_spe2 = linearprobe_acc(train_z2y, test_z2y, train_dataset, test_dataset)
                    return (score_s, score_spe1, score_spe2)
                
                def linearprobe_acc(train_z, test_z, train_dataset, test_dataset):
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-3])
                    score1 = clf.score(test_z, test_dataset[:][-3])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-2])
                    score2 = clf.score(test_z, test_dataset[:][-2])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-1])
                    score3 = clf.score(test_z, test_dataset[:][-1])
                    return (score1, score2, score3)

                if _iter == 0 and i_batch == 0:
                    logs.update({'loss': 0, 'shared': 0, 'specific': 0, 'ortho': 0})
                    logs.update({'Acc_s 1': 0, 'Acc_s 2': 0, 'Acc_s 3': 0,
                                 'Acc_spe1 1': 0, 'Acc_spe1 2': 0, 'Acc_spe1 3': 0,
                                 'Acc_spe2 1': 0, 'Acc_spe2 2': 0, 'Acc_spe2 3': 0})
                    utils.print_row([i for i in logs.keys()], colwidth=12)
                
                if i_batch == len(train_loader)-1:
                    logs.update({'loss': loss_meter.avg, 'shared': shared_meter.avg, 'specific': specific_meter.avg, 'ortho': ortho_meter.avg})
                    
            if _iter%5 ==0 or _iter == num_epoch-1:
                test_acc = linearprobe(train_dataset, test_dataset)
                logs.update({'Acc_s 1': test_acc[0][0], 'Acc_s 2': test_acc[0][1], 'Acc_s 3': test_acc[0][2],
                                'Acc_spe1 1': test_acc[1][0], 'Acc_spe1 2': test_acc[1][1], 'Acc_spe1 3': test_acc[1][2],
                                'Acc_spe2 1': test_acc[2][0], 'Acc_spe2 2': test_acc[2][1], 'Acc_spe2 3': test_acc[2][2]})                
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        
            else:
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)  

            lr_scheduler.step()
        return logs
    

class DMVAE(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=2, activation='relu', initialization='xavier', a=1):
        super().__init__()
        self.encoder_x1 = mlp(x1_dim, hidden_dim, 4*embed_dim, layers, activation, initialization = initialization)
        self.encoder_x2 = mlp(x2_dim, hidden_dim, 4*embed_dim, layers, activation, initialization = initialization)

        self.decoder_x1 = mlp(2*embed_dim, hidden_dim, x1_dim, layers, activation, initialization = initialization)
        self.decoder_x2 = mlp(2*embed_dim, hidden_dim, x2_dim, layers, activation, initialization = initialization)
        
        self.a = a
        self.embed_dim = embed_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def product_of_expert(self, mu1, logvar1, mu2, logvar2):
        logvar = - (1/logvar1.exp() + 1/logvar2.exp()).log()
        mu = logvar.exp() * (mu1 / logvar1.exp() + mu2 / logvar2.exp())
        return mu, logvar
    
    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x1, x2):
        emb1 = self.encoder_x1(x1)
        emb2 = self.encoder_x2(x2)
        mu_s1, logvar_s1, mu_1, logvar_1 = emb1.chunk(4, dim=1)
        mu_s2, logvar_s2, mu_2, logvar_2 = emb2.chunk(4, dim=1)
        z1 = self.reparameterize(mu_1, logvar_1)
        z2 = self.reparameterize(mu_2, logvar_2)
        z1_s = self.reparameterize(mu_s1, logvar_s1)
        z2_s = self.reparameterize(mu_s2, logvar_s2)
        mu_s, logvar_s = self.product_of_expert(mu_s1, logvar_s1, mu_s2, logvar_s2)
        z_s = self.reparameterize(mu_s, logvar_s)
        x1_recon = self.decoder_x1(torch.cat([z1, z_s], dim=1))
        x2_recon = self.decoder_x2(torch.cat([z2, z_s], dim=1))
        loss_recon = F.mse_loss(x1_recon, x1) + F.mse_loss(x2_recon, x2)
        loss_kl = self.kl_divergence(mu_1, logvar_1) + self.kl_divergence(mu_2, logvar_2) + 2 * self.kl_divergence(mu_s, logvar_s)

        x1_recon_cross = self.decoder_x1(torch.cat([z1, z2_s], dim=1))
        x2_recon_cross = self.decoder_x2(torch.cat([z2, z1_s], dim=1))
        loss_recon_cross = F.mse_loss(x1_recon_cross, x1) + F.mse_loss(x2_recon_cross, x2)
        loss_kl_cross = self.kl_divergence(mu_1, logvar_2) + self.kl_divergence(mu_2, logvar_1) + self.kl_divergence(mu_s1, logvar_s1) + self.kl_divergence(mu_s2, logvar_s2)

        loss = loss_recon + self.a * loss_kl + loss_recon_cross + self.a * loss_kl_cross
        return loss, {'loss': loss.item(), 'loss_recon': loss_recon.item(), 'loss_kl': loss_kl.item(), 'loss_recon_cross': loss_recon_cross.item(), 'loss_kl_cross': loss_kl_cross.item(), 'a': self.a}

    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        emb1 = self.encoder_x1(x1)
        emb2 = self.encoder_x2(x2)
        mu_s1, logvar_s1, mu_1, logvar_1 = emb1.chunk(4, dim=1)
        mu_s2, logvar_s2, mu_2, logvar_2 = emb2.chunk(4, dim=1)
        return mu_s1, mu_s2, mu_1, mu_2
    
    def train_model_dmvae(self, train_loader, train_dataset, test_dataset, optimizer, num_epoch=50, noise_scale=0.01, drop_scale=10):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
        for _iter in range(num_epoch):
            logs = {}
            logs.update({'Epoch': _iter})
            loss_meter = utils.AverageMeter('loss')
            recon_meter = utils.AverageMeter('recon')
            kl_meter = utils.AverageMeter('kl')
            recon_cross_meter = utils.AverageMeter('recon_cross')
            kl_cross_meter = utils.AverageMeter('kl_cross')
            a_meter = utils.AverageMeter('a')

            for i_batch, data_batch in enumerate(train_loader):
                self.train()
                x1 = data_batch[0].float().cuda()
                x2 = data_batch[1].float().cuda()
                x1 = augment_data(x1, noise_scale, drop_scale)
                x2 = augment_data(x2, noise_scale, drop_scale)
                loss, train_logs = self(x1, x2) #, v1, v2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meter.update(train_logs['loss'])
                recon_meter.update(train_logs['loss_recon'])
                kl_meter.update(train_logs['loss_kl'])
                recon_cross_meter.update(train_logs['loss_recon_cross'])
                kl_cross_meter.update(train_logs['loss_kl_cross'])
                a_meter.update(train_logs['a'])
                
                def getemb(dataset):
                    zsx, zsy, z1x, z2y = self.get_embedding(torch.stack(dataset[:][:-3]).cuda())
                    zs = torch.cat([zsx, zsy], dim=1).cpu().detach().numpy()
                    z1x = z1x.cpu().detach().numpy()
                    z2y = z2y.cpu().detach().numpy()
                    return zs, z1x, z2y
                
                def linearprobe(train_dataset, test_dataset):
                    train_zs, train_z1x, train_z2y = getemb(train_dataset)
                    test_zs, test_z1x, test_z2y = getemb(test_dataset)
                    score_s = linearprobe_acc(train_zs, test_zs, train_dataset, test_dataset)
                    score_spe1 = linearprobe_acc(train_z1x, test_z1x, train_dataset, test_dataset)
                    score_spe2 = linearprobe_acc(train_z2y, test_z2y, train_dataset, test_dataset)
                    return (score_s, score_spe1, score_spe2)
                
                def linearprobe_acc(train_z, test_z, train_dataset, test_dataset):
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-3])
                    score1 = clf.score(test_z, test_dataset[:][-3])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-2])
                    score2 = clf.score(test_z, test_dataset[:][-2])
                    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-1])
                    score3 = clf.score(test_z, test_dataset[:][-1])
                    return (score1, score2, score3)

                if _iter == 0 and i_batch == 0:
                    logs.update({'loss': 0, 'recon': 0, 'kl': 0, 'recon_cross': 0, 'kl_cross': 0, 'a': 0})
                    logs.update({'Acc_s 1': 0, 'Acc_s 2': 0, 'Acc_s 3': 0,
                                 'Acc_spe1 1': 0, 'Acc_spe1 2': 0, 'Acc_spe1 3': 0,
                                 'Acc_spe2 1': 0, 'Acc_spe2 2': 0, 'Acc_spe2 3': 0})
                    utils.print_row([i for i in logs.keys()], colwidth=12)
                
                if i_batch == len(train_loader)-1:
                    logs.update({'loss': loss_meter.avg, 'recon': recon_meter.avg, 'kl': kl_meter.avg, 'recon_cross': recon_cross_meter.avg, 'kl_cross': kl_cross_meter.avg, 'a': a_meter.avg})
                    
            if _iter%5 ==0 or _iter == num_epoch-1:
                test_acc = linearprobe(train_dataset, test_dataset)
                logs.update({'Acc_s 1': test_acc[0][0], 'Acc_s 2': test_acc[0][1], 'Acc_s 3': test_acc[0][2],
                                'Acc_spe1 1': test_acc[1][0], 'Acc_spe1 2': test_acc[1][1], 'Acc_spe1 3': test_acc[1][2],
                                'Acc_spe2 1': test_acc[2][0], 'Acc_spe2 2': test_acc[2][1], 'Acc_spe2 3': test_acc[2][2]})                
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)        
            else:
                utils.print_row([logs[key] for key in logs.keys()], colwidth=12)  

            lr_scheduler.step()
        return logs
