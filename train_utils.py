import torch
import matplotlib.pyplot as plt

def train(teacher_model,student_model, train_loader, optimizer, criterion, device,mode):
    if mode == "distil":
        teacher_model.eval()
        student_model.train()
        total_loss = 0
        correct_pred = 0
        total_pred = 0
        for batch in train_loader:
            optimizer.zero_grad()
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            with torch.no_grad():
                teacher_output = teacher_model(X, mask)
            student_output = student_model(X, mask)
           
            T = 2
            distill_loss = torch.nn.KLDivLoss()(torch.nn.functional.softmax(student_output, dim=1), torch.nn.functional.softmax(teacher_output , dim=1)) / (X.size(0) * T * T)
            alpha = 0.8
            cross_entropy_loss = criterion(student_output, y)
            loss = alpha * distill_loss + (1 - alpha) * cross_entropy_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(student_output, 1)
            correct_pred += (predicted == y).sum().item()
            total_pred += len(y)
            
    elif mode == "LoRA":
        teacher_model.train()
        total_loss = 0
        correct_pred = 0
        total_pred = 0
        for batch in train_loader:
            optimizer.zero_grad()
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            output = teacher_model(X, mask)
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_pred += (predicted == y).sum().item()
            total_pred += len(y)
            
    elif mode == "rnn":
        student_model.train()
        total_loss = 0
        correct_pred = 0
        total_pred = 0
        for batch in train_loader:
            optimizer.zero_grad()
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            output = student_model(X, mask)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_pred += (predicted == y).sum().item()
            total_pred += len(y)
    
    return total_loss / len(train_loader), correct_pred / total_pred 
  
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_pred = 0
    total_pred = 0
    with torch.no_grad():
        for batch in val_loader:
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            output = model(X, mask)
            loss = criterion(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_pred += (predicted == y).sum().item()
            total_pred += len(y)
             
    return total_loss / len(val_loader) , correct_pred / total_pred 

    
def plot_losses(train_losses, val_losses, mode, args):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{mode} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/{mode}_{args.sr_no}_loss.png')
    plt.close()
    print(f"Plots saved at plots/{mode}_{args.sr_no}_loss.png")
    
def plot_metrics(train_accs, val_accs, mode, args):
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'{mode} Accuracy')
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'plots/{mode}_{args.sr_no}_acc.png')
    plt.close()
    print(f"Plots saved at plots/{mode}_{args.sr_no}_acc.png")