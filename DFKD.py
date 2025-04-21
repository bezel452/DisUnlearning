import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from distillation.FastDFKD import save_image_batch, Generator, FastMetaSythesizer, Normalizer, KLDiv, TopkAccuracy, RunningLoss, classification_evaluator
from models.LeNet import LeNet
from models.ResNet import ResNet18, ResNet34
from train_teacher import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(synthesizer, model, criterion, optimizer, kd_steps, epoch):
    loss_metric = RunningLoss(KLDiv(reduction='sum'))
    acc_metric = TopkAccuracy(topk=(1, 5))
    student, teacher = model
    student.train()
    teacher.eval()
    for i in range(kd_steps):
        images = synthesizer.sample()
        images = images.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            t_out = teacher(images)
        s_out = student(images.detach())
        loss_s = criterion(s_out, t_out.detach())
        loss_s.backward()
        optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if i % 10: 
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            print(f"[Train] Epoch={epoch} | Iter={i + 1}/{kd_steps} | Acc Top1={train_acc1} | Acc Top5={train_acc5} | Loss={train_loss} | Lr={optimizer.param_groups[0]['lr']}")
            loss_metric.reset()
            acc_metric.reset()


if __name__ == '__main__':
    teacher = ResNet34(classes_num=10)
    student = ResNet18(classes_num=10).to(device)
    criterion = KLDiv(T=20)
    teacher.load_state_dict(torch.load('TEACHERpretrain_R34.pth'))
    teacher = teacher.to(device)
    nz = 256
    normalizer = Normalizer(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainloader, testloader = load_data('/home/data/datasets/cifar10', batch_size=256)
    evaluator = classification_evaluator(testloader)
    generator = Generator(nz = nz, ngf = 64, img_size=32, nc=3)
    generator = generator.to(device)
    synthesizer = FastMetaSythesizer(teacher, student, generator, 
                                     nz = nz, num_classes=10, img_size = (3, 32, 32), init_dataset=None,
                                     save_dir='data/fastdfkd', device=device,
                                     transform=transform, normalizer = normalizer,
                                     synthesis_batch_size = 256, sample_batch_size = 256,
                                     iterations = 5, warmup=20, lr_g = 3e-3, lr_z=0.01,
                                     adv = 1.33, bn = 10.0, oh = 0.5, reset_l0=1, reset_bn=0,
                                     bn_mmt=0.9, is_maml=1)
    
    optimizer = torch.optim.SGD(student.parameters(), lr=0.2, weight_decay=1e-4, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

    kd_steps = 400
    ep_steps = 400
    epochs = 200

    for epoch in range(epochs):
        for _ in range(ep_steps // kd_steps):
            vis_results, cost = synthesizer.synthesize()
#            time_cost += cost
            if epoch >= 20:
                train(synthesizer, [student, teacher], criterion, optimizer, kd_steps, epoch)
        for vis_name, vis_image in vis_results.items():
            save_image_batch(vis_image, 'data/fastdfkd_%s.png'%(vis_name))
        
        student.eval()

        eval_results = evaluator(student, device=device)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        print(f"[Eval] Epoch={epoch} | Acc Top1={acc1} | Acc Top5={acc5} | Loss={val_loss} | Lr={optimizer.param_groups[0]['lr']}")

        if epoch >= 20:
            scheduler.step()

    torch.save(student.state_dict(), "STUDENT_Res18.pth")