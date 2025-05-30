# %%
# %%
from datetime import datetime
from os.path import join as opj
import os
import dataloading
from dataloading import CustomDataset
import policy_model
base_path=os.environ.get("project_path")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from base_model import Module 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
import math
import os
import torch.optim.lr_scheduler as lr_scheduler
from transformers import Adafactor
from hydra.utils import instantiate
import hydra
from omegaconf import OmegaConf,DictConfig

def add_l2_regularization(loss:torch.Tensor, model:nn.Module, l2_lambda:float ,device:torch.device):
        l2_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        return loss + l2_lambda * l2_reg
# %%
def train_epoch(model: Module,model_name:str,optimizer: torch.optim.Optimizer, loss_fn: nn.Module, trainloader: DataLoader,
                device: torch.device,lamb:float,input_mode:str=None,lamb2:float=None) -> float:
    total_loss = 0
    assert input_mode is not None,"input_mode must be set"
    model.train()
    for i, traindata in tqdm(enumerate(trainloader), total=len(trainloader), desc="Training"):
        traj_evs_ims = torch.squeeze(traindata["traj_evs_img"], dim=0).to(device).float()
        traj_dvs = torch.squeeze(traindata["traj_depths"], dim=0).unsqueeze(1).to(device).float()
        traj_vels = torch.squeeze(traindata["traj_vels"], dim=0).to(device).float()
        traj_state = torch.squeeze(traindata["traj_state"],dim=0).to(device).float()
        if input_mode == 'dvs':
            inputs = (traj_evs_ims,)
        elif input_mode == 'depth':
            inputs = (traj_dvs,)
        else:
            inputs = (traj_evs_ims,traj_dvs)
        loss = 0
        vel_pred,extras,others = model(*inputs,traj_state,extras=None)
        vx = vel_pred[:,0]
        vy = vel_pred[:,1]
        vel = torch.stack([vx,vy,torch.zeros_like(vx)],dim=-1)
        constraint = lamb * torch.mean((vx**2 + vy**2 - 1)**2)
        if model_name == 'OrigUnet_lstm':
            depth_predict = others
            loss = loss_fn(vel, traj_vels)+loss_fn(traj_dvs,depth_predict)*0.1 + lamb * constraint
        else:
            loss = loss_fn(vel, traj_vels) + lamb * constraint
        if lamb2 is not None:
            loss = add_l2_regularization(loss, model, lamb2, device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / len(trainloader)
    return avg_loss


def validate_epoch(model: Module,model_name:str,loss_fn: nn.Module, valloader: DataLoader, device: torch.device,lamb:float,input_mode:str=None,lamb2:float=None) -> float:
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, valdata in tqdm(enumerate(valloader), total=len(valloader), desc="Validation"):
            traj_evs_ims = torch.squeeze(valdata["traj_evs_img"], dim=0).to(device).float()
            traj_dvs = torch.squeeze(valdata["traj_depths"], dim=0).unsqueeze(1).to(device).float()
            traj_vels = torch.squeeze(valdata["traj_vels"], dim=0).to(device).float()
            obs_dis = torch.squeeze(valdata["obs_dis"],dim=0).to(device).float()
            traj_state = torch.squeeze(valdata["traj_state"],dim=0).to(device).float()
            #with autocast(device_type="cuda"):
            loss = 0
            if input_mode == 'dvs':
                inputs = (traj_evs_ims,)
            elif input_mode == 'depth':
                inputs = (traj_dvs,)
            else:
                inputs = (traj_evs_ims,traj_dvs)
            loss = 0
            vel_pred,extras,others = model(*inputs,traj_state,extras=None)
            vx = vel_pred[:,0]
            vy = vel_pred[:,1]
            vel = torch.stack([vx,vy,torch.zeros_like(vx)],dim=-1)
            constraint = lamb * torch.mean((vx**2 + vy**2 - 1)**2)
            if model_name == 'OrigUnet_lstm':
                depth_predict = others
                loss = loss_fn(vel, traj_vels)+loss_fn(traj_dvs,depth_predict)*0.1 + lamb * constraint
            else:
                loss = loss_fn(vel, traj_vels) + lamb * constraint
            if lamb2 is not None:
                loss = add_l2_regularization(loss, model, lamb2, device)
            total_loss += loss.item()
        avg_loss = total_loss / len(valloader)
    return avg_loss


@hydra.main(config_path=opj(base_path,"configs"), config_name="train", version_base="1.3")
def main(cfg:DictConfig):
    if cfg.device =='cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dataset_name = cfg.dataset.dataset_name
    epochs = cfg.epochs
    val_freq = cfg.val_freq
    batch_size = cfg.batch_size
    lamb = cfg.lamb
    lamb2 = cfg.lamb2
    input_mode = cfg.input_mode
    assert input_mode in ['dvs','depth','fusion'],"input_mode must be in dvs,depth or fusion"
    ## create workspace
    model_name = cfg.model._target_.split('.')[-1]
    if 'Fusion' in model_name:
        assert input_mode == 'fusion',"input_mode must be fusion"
    model = instantiate(cfg.model).to(device)
    path = f'{model_name}_{input_mode}_{dataset_name}_b{batch_size}'
    rnn_type = cfg.model.get('rnn_type') if 'rnn_type' in cfg.model and cfg.model.rnn_type is not None else ''
    if rnn_type != '':
        assert rnn_type in ['lstm','rnn','gru'],'rnn must be implemented'
        path += f'_{rnn_type}'
    has_state = cfg.model.get('has_state') if 'rnn_type' in cfg.model else ''
    if has_state:
        path += f'_state'
    cross_mode = cfg.model.get('cross_mode') if 'cross_mode' in cfg.model else ''
    if cross_mode != '':
        path +=f'_{cross_mode}'
    expname = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    path +=f'_{expname}'
    base_path = os.environ.get("project_path")
    model_path = opj(base_path,"learner/models",path)
    workspace_path = opj(base_path, "learner/logs",path)
    print(path)
    os.makedirs(workspace_path, exist_ok=True)
    os.makedirs(model_path,exist_ok=True)
    writer = SummaryWriter(workspace_path)
    config_path = opj(workspace_path, 'config.yaml')
    OmegaConf.save(config=cfg, f=config_path)
    # dataset
    dataset = instantiate(cfg.dataset)
    dataset.dataloading()
    train_dataset = CustomDataset(dataset=dataset.train_data, batch_size=batch_size)
    val_dataset = CustomDataset(dataset=dataset.val_data, batch_size=batch_size)

    trainloader = DataLoader(train_dataset,shuffle=True, batch_size=1)
    valloader = DataLoader(val_dataset,shuffle=True,batch_size=1)
    checkpoint_path = cfg.checkpoint_path
    # loss
    loss_fn = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer,params=model.parameters())
    if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            min_val_loss = checkpoint['val_loss']
            print(f"Resuming training from epoch {start_epoch} with min_val_loss {min_val_loss}")
    else:
        min_val_loss = math.inf
        start_epoch = 1

    for epoch in range(start_epoch, start_epoch + epochs + 1):
        train_loss = train_epoch(model=model,model_name=model_name,optimizer=optimizer, loss_fn=loss_fn, trainloader=trainloader, device=device,lamb=lamb,input_mode=input_mode,lamb2=lamb2)
        writer.add_scalar('Loss/train', train_loss, epoch)
        if epoch % val_freq == 0 or epoch == start_epoch + epochs:
            val_loss = validate_epoch(model=model,model_name=model_name,loss_fn=loss_fn, valloader=valloader, device=device,lamb=lamb,input_mode=input_mode,lamb2=lamb2)
            writer.add_scalar('Loss/val', val_loss, epoch)
            #scheduler.step(val_loss)
            print(f"Epoch: {epoch}, Train loss:{train_loss:.4f},Val loss: {val_loss: .4f}")    
            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    #'schduler_state_dict':scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                checkpoint_path = opj(model_path,f"checkpoint_epoch_best_{epoch}.pth")
                torch.save(checkpoint, checkpoint_path)
            else:
                checkpoint_path = opj(model_path,f"checkpoint_epoch_{epoch}.pth")
                torch.save(checkpoint, checkpoint_path)
        writer.close()

# %%
if __name__ == '__main__':
    main()