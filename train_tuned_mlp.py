"""
DivideMix training with MLP on pre-computed embeddings
Modified from train_tuned.py to use embeddings instead of images
"""
from __future__ import print_function
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
from MLP import create_model
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pprint import pprint
import dataloader_tuned_mlp as dataloader

# ===== [A] START TIMER =====
start_wall = time.time()

parser = argparse.ArgumentParser(description='PyTorch DivideMix Training with MLP on Embeddings')
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
parser.add_argument('--embedding_dim', default=512, type=int, help='dimension of input embeddings')
parser.add_argument('--warm_up', default=10, type=int, help='number of warmup epochs')
# Dataset paths and columns
parser.add_argument('--dataset', default='fashion-mnist', type=str)
parser.add_argument('--noise_type', default='llm', type=str)
parser.add_argument('--train_csv_path', type=str, required=True, help='CSV with true labels')
parser.add_argument('--train_embedding_feather_path', type=str, required=True, help='Feather with embeddings')
parser.add_argument('--train_noisy_label_feather_path', type=str, required=True, help='Feather with noisy labels')
parser.add_argument('--train_label_column', type=str, required=True, help='Column name for true labels')
parser.add_argument('--test_csv_path', type=str, required=True)
parser.add_argument('--test_embedding_feather_path', type=str, required=True)
parser.add_argument('--test_label_column', type=str, required=True)
parser.add_argument('--num_workers', default=4, type=int)
args = parser.parse_args()

pprint(vars(args))

# Device setup - support for CUDA, MPS (Apple Silicon), and CPU
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpuid}')
    torch.cuda.set_device(args.gpuid)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Using CUDA device: {torch.cuda.get_device_name(args.gpuid)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS (Apple Silicon) device")
else:
    device = torch.device('cpu')
    print("Using CPU device")

random.seed(args.seed)
torch.manual_seed(args.seed)

# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()  # fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + 
                  torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T)  # temperature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T)  # temperature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch - for embeddings, we do linear interpolation
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
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], 
                                  logits_u, mixed_target[batch_size*2:], 
                                  epoch+batch_idx/num_iter, args.warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        L = loss
        L.backward()
        optimizer.step()


def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))


def predict_testset(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            predicted = outputs.argmax(dim=1)
            all_preds.append(predicted.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    if all_targets:
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"Accuracy: {(acc * 100):.2f}%")
        print(f"F1-macro: {(f1_macro * 100):.2f}%")
        print(f"F1-weighted: {(f1_weighted * 100):.2f}%")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

        return y_pred
    else:
        print("Accuracy: 0.00%\nF1-macro: 0.00%\nF1-weighted: 0.00%\n\nClassification report:\n(none)")
        return np.array([], dtype=np.int64)


def eval_train(model, all_loss):    
    model.eval()
    # Generalize losses size to match dataset
    dataset_size = len(eval_loader.dataset)
    losses = torch.zeros(dataset_size)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:, gmm.means_.argmin()]         
    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


# Initialize dataloader
loader = dataloader.dataloader_tuned_mlp(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    train_csv_path=args.train_csv_path,
    train_embedding_feather_path=args.train_embedding_feather_path,
    train_noisy_label_feather_path=args.train_noisy_label_feather_path,
    train_label_column=args.train_label_column,
    test_csv_path=args.test_csv_path,
    test_embedding_feather_path=args.test_embedding_feather_path,
    test_label_column=args.test_label_column
)

print('| Building MLP nets')
net1 = create_model(in_dim=args.embedding_dim, num_classes=args.num_class)
net2 = create_model(in_dim=args.embedding_dim, num_classes=args.num_class)
net1 = net1.to(device)
net2 = net2.to(device)
if device.type == 'cuda':
    cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

all_loss = [[], []]

for epoch in tqdm(range(args.num_epochs+1), desc="Epochs"):
    lr = args.lr
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
        out_name = f"{args.dataset}_{args.noise_type}_mlp_test-predictions.npy"
        np.save(out_name, preds)
        print(f"Saved predictions to {out_name}")

# ===== [B] AFTER TRAIN =====
end_wall = time.time()
wall_sec = end_wall - start_wall
print(f"[TIME] Total wall time: {wall_sec:.2f}s ({wall_sec/3600:.4f}h)")
