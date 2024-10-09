import torch
from tqdm import tqdm

def evaluate_loss(model, criterion, dataloader, device):
    # Set model to eval mode
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Load inputs and labels to device
            inputs = inputs.to(device)
            labels = move_to_cuda(labels, device)  # Use the move_to_cuda function for labels
            
            # Call model and get outputs
            outputs = model(inputs, labels)
            
            # Calculate the loss using the loss function
            loss = outputs['bbox_regression']
            
            total_loss += loss.item()
            
    average_loss = total_loss / len(dataloader)
    return average_loss

def evaluate_accuracy(model, dataloader, device):
    # Set model to eval mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Load inputs and labels to device
            inputs = inputs.to(device)
            labels = move_to_cuda(labels, device)  # Use the move_to_cuda function for labels
            
            # Call model and get outputs
            outputs = model(inputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100
    return accuracy

def move_to_cuda(data, device):
    if isinstance(data, dict):
        return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_cuda(item, device) for item in data]
    else:
        return data.to(device)

def train(model, optimizer_class, optimizer_reg, criterion, trainloader, testloader, epochs, device):
    """
    Part 1.a: complete the training loop
    """
    train_losses = []  # For recording train losses
    test_losses = []  # For recording test losses
    
    # Move model to device here
    model.to(device)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_reg, step_size=10, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda')  # Create a GradScaler for mixed precision training
    
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    for epoch in range(epochs):
        running_loss = 0.0
        # Set the model to train mode    
        model.train()
        num_iters = 0
        for i, data in enumerate(tqdm(trainloader)):
            inputs, labels = data

            # Load inputs and labels to device
            inputs = inputs.to(device, non_blocking=True)
            labels = [{'boxes': labels['boxes'][i], 'labels': labels['labels'][i]} for i in range(len(labels['boxes']))]
            labels = move_to_cuda(labels, device)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs, labels)
            
                loss_reg = outputs['bbox_regression']
                loss_class = outputs['classification']
                
            
            if (i + 1) % 1000 == 0: 
                total_loss = loss_reg + loss_class 
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer_reg) 
                scaler.step(optimizer_class) 
                scaler.update() 
                
                optimizer_reg.zero_grad() 
                optimizer_class.zero_grad() 

                running_loss += loss_reg.item()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}")

        # test_loss = evaluate_loss(model, criterion, testloader, device)
        # test_losses.append(test_loss)
        # print(f"Epoch {epoch+1}/{epochs} - Test Loss: {test_loss}")

        # test_accuracy = evaluate_accuracy(model, testloader, device)
        # print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_accuracy:.2f}%")
        
        scheduler.step()

        if epoch % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_reg.state_dict(),
                'train_loss': train_loss,
                'test_loss': train_loss
            }
            torch.save(checkpoint, f"./model_checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch+1}")

    return train_losses, test_losses
