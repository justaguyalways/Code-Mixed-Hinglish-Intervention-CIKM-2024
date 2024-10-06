import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorchtools import EarlyStopping  
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


def save_to_json(filename, pred_list, gold_list, complete_test_prompts):
        data = {
            "predictions": pred_list,
            "gold_references": gold_list,
            "complete_gold_prompts" : complete_test_prompts,
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

def load_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    predictions = data["predictions"]
    gold_references = data["gold_references"]
    complete_test_prompts = data["complete_gold_prompts"]
    return predictions, gold_references, complete_test_prompts

def train(model, train_dataloader, val_dataloader, test_dataloader, config, num_epochs, patience, devices=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
    
    model.to(config.device)
    model.train()

    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    best_val_loss = 9999.99
    
    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        complete_test_prompts = []
        # Training
        model.train()
        
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            batch["video"] = batch["video"].to(config.device)
            batch["audio"] = batch["audio"].to(config.device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss/len(train_dataloader)
        # scheduler.step(avg_train_loss)
        train_losses.append(avg_train_loss)

        torch.cuda.empty_cache()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                complete_test_prompts.extend(list(batch["prompt"]))

                batch["video"] = batch["video"].to(config.device)
                batch["audio"] = batch["audio"].to(config.device)

                val_outputs = model(batch)
                val_loss += val_outputs.loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        filename = config.results_directory + "test_result_" + config.file_name + "_" + str(epoch) + ".json" 
        pred_text, gold_text = generate_text(model, test_dataloader, device=config.device)
        save_to_json(filename, pred_text, gold_text, complete_test_prompts)

        if epoch%3 == 0:
            save_name = config.directory + "checkpoint_"  + config.file_name + "_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), save_name)
        
        if avg_val_loss < best_val_loss:
            #save results for test
            best_val_loss = avg_val_loss

            filename = config.results_directory + "test_result_" + config.file_name + "_best.json" 
            # pred_text, gold_text = generate_text(model, test_dataloader, device=config.device)
            save_to_json(filename, pred_text, gold_text, complete_test_prompts)
            
            #save model
            save_name = config.directory + "checkpoint_"  + config.file_name + "_best.pth"
            torch.save(model.state_dict(), save_name)
       

        # Check for early stopping
        # early_stopping(avg_val_loss, model)

        # if early_stopping.early_stop:

        #     save_name = config.directory + "checkpoint_"  + config.file_name + "_" + str(epoch) + ".pth"
        #     torch.save(model.state_dict(), save_name)

        #     print("Early stopping")
        #     break

        # Plotting the training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(config.results_directory + "loss_figure_" + config.file_name + ".png")

    return model
    
    
def generate_text(model, test_dataloader, device):
    model.to(device)
    gold_reference_text = []
    generated_text = []
    model.eval()
    model.config.train = False
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch["video"] = batch["video"].to(device)
            batch["audio"] = batch["audio"].to(device)
            
            test_outputs = model(batch)
            
            test_outputs = model.tokenizer.batch_decode(test_outputs, skip_special_tokens=True)
            
            generated_text.extend(list(test_outputs))
            gold_reference_text.extend(list(batch["prompt"]))

    model.config.train = True
          
    return generated_text, gold_reference_text