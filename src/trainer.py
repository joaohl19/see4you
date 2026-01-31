import torch
from torch import nn
from tqdm import tqdm
from src.utils import write_loss_graph
from torch.nn.utils import clip_grad_norm_
import os
import PIL

# Integrando tudo em uma classe
class ImageCaptionTrainer:
    def __init__(self, model, optimizer, loss_function, clip_norm, device=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = loss_function
        self.clip_norm=clip_norm
        
        self.model.to(self.device)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="  Training Batches", leave=False)
        
        for images, captions, captions_lengths in pbar:
            images = images.to(self.device)
            captions = captions.to(self.device)

            # Flattening for CrossEntropyLoss:
            # We want to treat every word in every sequence as an individual sample.
            self.optimizer.zero_grad(set_to_none=True)

            # Logits shape: [Batch, Seq_Len + 1, Vocab]
            logits = self.model(images, captions, captions_lengths)

            logits_flattened = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            captions_flattened = captions.contiguous().view(-1)

            loss = self.loss_function(logits_flattened, captions_flattened)
            
            loss.backward()

            clip_grad_norm_(self.model.parameters(), self.clip_norm)

            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())
            
        return total_loss / len(dataloader.dataset)

    def eval_one_epoch(self, dataloader, run_inference):
        self.model.eval()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="  Evaluating Batches", leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, captions, captions_lengths) in enumerate(pbar):
                images = images.to(self.device)
                captions = captions.to(self.device)

                logits = self.model(images, captions, captions_lengths)

                logits_flattened = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                captions_flattened = captions.contiguous().view(-1)

                loss = self.loss_function(logits_flattened, captions_flattened)
            
                total_loss += loss.item() * images.size(0)

                # Run inference every N epochs
                if run_inference and batch_idx == 0:
                    for i in range(min(images.size(0), 10)):
                        predicted_tokenized_caption = self.model.infer(images[i])
                
        return total_loss / len(dataloader.dataset)

    def fit(self, train_loader, val_loader, epochs, patience, epsilon=1e-4, checkpoint_dir='checkpoints'):
            """
            Training loop with Early Stopping logic.
            """
            print(f"Training is beginning with device: {self.device}")
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            
            min_val_loss = float('inf')
            epochs_no_improve = 0
            train_losses_list = list()
            val_losses_list = list()
            
            # Use tqdm for a visual progress bar over epochs
            epoch_pbar = tqdm(range(epochs), desc="Epochs", leave = True)
            
            for epoch in epoch_pbar:
                # Training phase
                train_loss = self.train_one_epoch(train_loader)
                train_losses_list.append(train_loss)
                # Validation phase
                val_loss = self.eval_one_epoch(val_loader, run_inference=((epoch+1)%5==0))
                val_losses_list.append(val_loss)
                # Check if the improvement is greater than epsilon
                if val_loss < min_val_loss - epsilon:
                    min_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': val_loss,
                        }, checkpoint_path)
                else:
                    epochs_no_improve += 1
                
                #Update progress bar description
                epoch_pbar.set_postfix({
                    'Train Loss': f"{train_loss:.4f}",
                    'Val Loss': f"{val_loss:.4f}",
                    'Patience': f"{epochs_no_improve}/{patience}"
                })
                
                # Check if patience limit is reached
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
            
            write_loss_graph(train_losses_list, val_losses_list)
