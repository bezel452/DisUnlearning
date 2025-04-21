import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Dict, Mapping
import os
from torch.autograd import Variable
from PIL import Image
import numpy as np
import math
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from kornia import augmentation
import time
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    def __init__(self, nz, ngf, img_size, nc):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.to(device)
    
class BaseSynthesis(ABC):
    def __init__(self, teacher, student):
        super(BaseSynthesis, self).__init__()
        self.teacher = teacher
        self.student = student
    
    @abstractclassmethod
    def synthesize(self) -> Dict[str, torch.Tensor]:
        """ take several steps to synthesize new images and return an image dict for visualization. 
            Returned images should be normalized to [0, 1].
        """
        pass
    
    @abstractclassmethod
    def sample(self, n):
        """ fetch a batch of training data. 
        """

        pass

class Metric(ABC):
    @abstractmethod
    def update(self, pred, target):
        """ Overridden by subclasses """
        raise NotImplementedError()
    
    @abstractmethod
    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack

class RunningLoss(Metric):
    def __init__(self, loss_fn, is_batch_average=False):
        self.reset()
        self.loss_fn = loss_fn
        self.is_batch_average = is_batch_average

    @torch.no_grad()
    def update(self, outputs, targets):
        self._accum_loss += self.loss_fn(outputs, targets)
        if self.is_batch_average:
            self._cnt += 1
        else:
            self._cnt += len(outputs)

    def get_results(self):
        return (self._accum_loss / self._cnt).detach().cpu()
    
    def reset(self):
        self._accum_loss = self._cnt = 0.0

class TopkAccuracy(Metric):
    def __init__(self, topk=(1, 5)):
        self._topk = topk
        self.reset()
    
    @torch.no_grad()
    def update(self, outputs, targets):
        for k in self._topk:
            _, topk_outputs = outputs.topk(k, dim=1, largest=True, sorted=True)
            correct = topk_outputs.eq( targets.view(-1, 1).expand_as(topk_outputs) )
            self._correct[k] += correct[:, :k].view(-1).float().sum(0).item()
        self._cnt += len(targets)

    def get_results(self):
        return tuple( self._correct[k] / self._cnt * 100. for k in self._topk )

    def reset(self):
        self._correct = {k: 0 for k in self._topk}
        self._cnt = 0.0

class MetricCompose(dict):
    def __init__(self, metric_dict: Mapping):
        self._metric_dict = metric_dict

    @property
    def metrics(self):
        return self._metric_dict
        
    @torch.no_grad()
    def update(self, outputs, targets):
        for key, metric in self._metric_dict.items():
            if isinstance(metric, Metric):
                metric.update(outputs, targets)
    
    def get_results(self):
        results = {}
        for key, metric in self._metric_dict.items():
            if isinstance(metric, Metric):
                results[key] = metric.get_results()
        return results

    def reset(self):
        for key, metric in self._metric_dict.items():
            if isinstance(metric, Metric):
                metric.reset()

    def __getitem__(self, name):
        return self._metric_dict[name]

class Evaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm.tqdm(self.dataloader, disable=not progress) ):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()
    
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

def classification_evaluator(dataloader):
    metric = MetricCompose({
        'Acc': TopkAccuracy(),
        'Loss': RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader)

def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0) )
            img.save(output_filename+'-%d.png'%(idx))

def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance( postfix, str):
        postfix = [ postfix ]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith( pos ):
                    images.append( os.path.join( dirpath, f ) )
    return images

class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(self.root) #[ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open( self.images[idx] )
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s'%(self.root, len(self), self.transform)

class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join( self.root, "%d.png"%(self._idx) ), pack=False)
        self._idx+=1

    def get_dataset(self, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform)

class DeepInversionHook():
    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = ( self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data,
                         self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data )

    def remove(self):
        self.hook.remove()

def reset_l0(model):
    for n,m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67)

class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)



class FastMetaSythesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size, 
                 init_dataset=None, iterations=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1, oh=1,
                 save_dir='data/fastdfkd', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, device='cpu', distributed =False, lr_z=0.01, 
                 warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,
                 is_maml=1):
        super(FastMetaSythesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.bn_mmt = bn_mmt
        self.ismaml = is_maml

        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.init_dataset = init_dataset
        self.use_fp16 = use_fp16
        self.autocast = autocast
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)

        self.transform = transform
        self.data_iter = None
        self.generator = generator.to(device).train()
        self.device = device
        self.hooks = []

        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None

        self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])

    def synthesize(self, targets=None):

        start = time.time()

        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        if (self.ep == 120 + self.ep_start) and self.reset_l0:
            reset_l0(self.generator)

        best_inputs = None
        z = torch.randn(size = (self.synthesis_batch_size, self.nz), device = self.device).requires_grad_()
        if targets is None:
            targets = torch.randint(low = 0, high = self.num_classes, size = (self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0]
        targets = targets.to(self.device)
        fast_generator = self.generator.clone()
        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr = self.lr_g, betas = [0.5, 0.999])

        for it in range(self.iterations):
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs)

            if it == 0:
                originalMeta = inputs

            t_out = self.teacher(inputs_aug)
            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.to(self.device)

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy(t_out, targets)
            if self.adv > 0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug)
                mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()

            if self.ismaml:
                if it == 0: self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations - 1): self.meta_optimizer.step()

            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            self.meta_optimizer.step()

        self.student.train()
        self.prev_z = (z, targets)
        end = time.time()

        self.data_pool.add(best_inputs)
        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler
        )
        self.data_iter = DataIter(loader)
        return {"synthetic": best_inputs}, end - start

    def sample(self):
        return self.data_iter.next()
    
class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)