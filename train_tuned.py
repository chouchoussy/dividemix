from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import logging
from tqdm import tqdm
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_tuned as dataloader

# --- CẤU HÌNH LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),
        logging.StreamHandler()
    ]
)

parser = argparse.ArgumentParser(description='PyTorch General Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--image_size', default=224, type=int, help='image size for resize/crop')
parser.add_argument('--warm_up', default=10, type=int, help='number of warmup epochs')
# Dataset paths and columns
parser.add_argument('--train_csv_path', type=str, required=True)
parser.add_argument('--train_feather_path', type=str, required=True)
parser.add_argument('--train_data_column', type=str, required=True)
parser.add_argument('--train_label_column', type=str, required=True)
parser.add_argument('--train_image_dir', type=str, required=True)
parser.add_argument('--test_csv_path', type=str, required=True)
parser.add_argument('--test_data_column', type=str, required=True)
parser.add_argument('--test_label_column', type=str, required=True)
parser.add_argument('--test_image_dir', type=str, required=True)
parser.add_argument('--num_workers', default=4, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(tqdm(labeled_trainloader, desc=f"Train Epoch {epoch}")):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, args.warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"{epoch=}, {batch_idx=}, Lx={Lx.item():.2f}, Lu={Lu.item():.2f}")


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(tqdm(dataloader, desc=f"Warmup Epoch {epoch}")):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        penalty = conf_penalty(outputs)
        L = loss + penalty
        L.backward()
        optimizer.step()
        print(f"Warmup {epoch=}, {batch_idx=}, loss={loss.item():.4f}, penalty={penalty.item():.4f}")

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc=f"Test Epoch {epoch}")):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc)) 
    logging.info(f"Test Epoch {epoch} Accuracy: {acc:.2f}%")

def predict_testset(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    all_preds = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Predict Testset"):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100.*correct/total if total > 0 else 0.0
    print(f"\n| Predict Testset\t Accuracy: {acc:.2f}%\n")
    logging.info(f"Predict Testset Accuracy: {acc:.2f}%")
    return np.concatenate(all_preds)

def eval_train(model,all_loss):    
    model.eval()
    # Generalize losses size to match dataset
    dataset_size = len(eval_loader.dataset)
    losses = torch.zeros(dataset_size)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(tqdm(eval_loader, desc="Eval Train")):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
    #     history = torch.stack(all_loss)
    #     input_loss = history[-5:].mean(0)
    #     input_loss = input_loss.reshape(-1,1)
    # else:
    #     input_loss = losses.reshape(-1,1)
    input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

loader = dataloader.dataloader_tuned(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    image_size=args.image_size,
    train_csv_path=args.train_csv_path,
    train_feather_path=args.train_feather_path,
    train_data_column=args.train_data_column,
    train_label_column=args.train_label_column,
    train_image_dir=args.train_image_dir,
    test_csv_path=args.test_csv_path,
    test_data_column=args.test_data_column,
    test_label_column=args.test_label_column,
    test_image_dir=args.test_image_dir
)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True


criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

all_loss = [[],[]]

for epoch in tqdm(range(args.num_epochs+1), desc="Epochs"):
    lr=args.lr
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    if epoch < args.warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print('Warmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader)
    else:
        prob1, all_loss[0] = eval_train(net1, all_loss[0])
        prob2, all_loss[1] = eval_train(net2, all_loss[1])
        pred1 = (prob1 > args.p_threshold)
        pred2 = (prob2 > args.p_threshold)
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)
        print('Train Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)
    test(epoch, net1, net2)
    # Save test predictions at last epoch
    if epoch == args.num_epochs:
        preds = predict_testset(net1, net2, test_loader)
        np.save('test_predictions.npy', preds)
