import argparse
from transformers import AutoTokenizer

from utils import *
from train_utils import *
from model import *

import matplotlib.pyplot as plt # added by me for plotting

def main(args):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_data_loader(
        'data/in_domain_train.tsv', args.batch_size, tokenizer)
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)
    

    if args.mode == "gen":
        model = GPT(args.gpt_variant, is_gen=True).to(args.device)
        model.eval()

        # TODO: You can add your super creative prompt here
        prompt = "My name is Inigo Montoya. You killed my father. Prepare to die. "
        # prompt = "Once upon a time, in a land far far away, there was a"

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "LoRA":    
        model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(args.epochs):
            train_loss,train_acc = train(model,model, train_loader, optimizer, criterion, args.device, args.mode)
            val_loss,val_acc = evaluate(model, val_loader, criterion, args.device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
        
        # TODO: Also plot the training losses and metrics
        # save the plot in plots folder 
        plot_losses(train_losses, val_losses, args.mode, args)
        plot_metrics(train_accs, val_accs, args.mode, args)
        model.save_trainable_params(args.model_path)
        
    elif args.mode == "distil":
        teacher_model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        teacher_model.load_trainable_params(args.model_path)
        teacher_model.eval()

        student_model = DistilRNN().to(args.device)  # TODO: Implement the student model class
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        # HINT: You can use an additional parameter in train function to differentiate LoRA and distillation training, no changes in evaluate function required.
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        for epoch in range(args.epochs):
            train_loss,train_acc = train(teacher_model, student_model, train_loader, optimizer, criterion, args.device, args.mode)
            val_loss , val_acc = evaluate(student_model, val_loader, criterion, args.device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")    
        
        plot_losses(train_losses, val_losses, args.mode, args)
        plot_metrics(train_accs, val_accs, args.mode, args)
        
    elif args.mode == "rnn":
        model = DistilRNN().to(args.device)
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        for epoch in range(args.epochs):
            train_loss,train_acc = train(model, model, train_loader, optimizer, criterion, args.device, args.mode)
            val_loss , val_acc = evaluate(model, val_loader, criterion, args.device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}") 
            
        plot_losses(train_losses, val_losses, args.mode, args)
        plot_metrics(train_accs, val_accs, args.mode, args)
           
        
    else:
        print("Invalid mode")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2")
    parser.add_argument("mode", type=str, choices=["gen", "LoRA", "distil", "rnn"], help="Mode to run the program in")
    parser.add_argument("sr_no", type=int, help="5 digit SR number")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--gpt_variant", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"], help="Model to use")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/LoRA.pth", help="Path to save the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--LoRA_rank", type=int, default=4, help="Low rank matrix bottleneck")
    # TODO: Add more arguments as needed
    
    args = parser.parse_args()
    args.device = torch.device(
        "cuda:4" if torch.cuda.is_available() and args.gpu_id >= 0 else\
        "mps" if torch.backends.mps.is_available() else "cpu")
    
    seed_everything(args.sr_no)

    main(args)
