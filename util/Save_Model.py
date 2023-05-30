import os
import torch
def save_model(model, optimizer,config,scheduler=None, accuracy_list=None):
    folder = f'checkpoints/{config["model"]}_{config["dataset"]}'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    if scheduler !=None:
        torch.save({
        #'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
    else:
        torch.save({
        #'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)