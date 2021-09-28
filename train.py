import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from dataset_tool import TestDataSet
from torch.utils.data import DataLoader
# from test_model.network import Generator,PatchGan,VGGPerceptualLoss,WarpLoss
from model.network import PatchGan,VGGPerceptualLoss,WarpLoss
from model.network_new import Generator
import torch.optim as optim
import argparse
# import warnings
# warnings.filterwarnings("error")
def save_checkpoint(model_G,model_kp, model_D):
    os.makedirs(f'checkpoint/{args.out_checkpoint}', exist_ok=True)
    G_model_path = os.path.join("checkpoint",args.out_checkpoint, "frG.pth")
    KP_model_path=os.path.join("checkpoint",args.out_checkpoint,"frKp.pth")
    D_model_path = os.path.join("checkpoint",args.out_checkpoint, "frD.pth")
    torch.save(model_G.state_dict(), G_model_path)
    torch.save(model_kp.state_dict(),KP_model_path)
    torch.save(model_D.state_dict(), D_model_path)
    G_optim_path=os.path.join("checkpoint",args.out_checkpoint,"opt_G.pth")
    D_optim_path=os.path.join("checkpoint",args.out_checkpoint,"opt_D.pth")
    torch.save(g_optimizer.state_dict(),G_optim_path)
    torch.save(d_optimizer.state_dict(),D_optim_path)
def load_checkpoint(model_G,model_kp, model_D):
    checkpoint_root = args.load_checkpoint
    G_model_path = os.path.join("checkpoint",checkpoint_root, "frG.pth")
    D_model_path = os.path.join("checkpoint",checkpoint_root, "frD.pth")
    kp_model_path=os.path.join('checkpoint',checkpoint_root,'frKp.pth')
    model_G.load_state_dict(torch.load(G_model_path))
    model_D.load_state_dict(torch.load(D_model_path))
    model_kp.load_state_dict(torch.load(kp_model_path))
    G_optim_path=os.path.join("checkpoint",checkpoint_root,"opt_G.pth")
    D_optim_path=os.path.join("checkpoint",checkpoint_root,"opt_D.pth")
    Kp_optim_path=os.path.join('checkpoint',checkpoint_root,'opt_Kp.pth')
    g_optimizer.load_state_dict(torch.load(G_optim_path))
    d_optimizer.load_state_dict(torch.load(D_optim_path))   

def train(train_dataloader,val_dataloader):
    step_epoch = len(train_dataloader)
    if args.load_checkpoint:
        load_checkpoint(generator,discriminator)
    for epoch in range(args.max_epoch):
        generator.train()
        discriminator.train()
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        prog_bar=tqdm(train_dataloader,position=0,leave=True)
        # running_l1_loss,
        running_wrap_loss,running_p_loss,running_g_loss,running_d_loss=0.,0.,0.,0.#,0.
        for step,data in enumerate(prog_bar):
            now_step=(epoch*step_epoch)+(step+1)
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            source_img=data[0].to(device)
            drive_img=data[1].to(device)
            out,warp,W,M,phat,qhat=generator(source_img,drive_img)
            real=discriminator(drive_img)
            fake=discriminator(out.detach())
            fake_loss=bceloss(fake,torch.zeros(fake.shape,device=device))
            real_loss=bceloss(real,torch.ones(real.shape,device=device))
            d_loss=fake_loss+real_loss
            d_loss.backward()
            d_optimizer.step()
            #g_loss
            g_d=discriminator(out)
            g_loss=bceloss(g_d,torch.ones(fake.shape,device=device))
            warp_loss=Warploss(W,M,phat,qhat)
            p_loss=perceptualloss(out,drive_img)
            loss=1*g_loss+1*p_loss+1*warp_loss
            loss.backward()
            # running_l1_loss+=l1_loss.item()
            running_wrap_loss+=warp_loss
            running_p_loss+=p_loss.item()
            running_g_loss+=g_loss.item()
            running_d_loss=d_loss.item()
            g_optimizer.step()
            next_step = step+1
            if step%100==0:
                # writer.add_scalar("TrainLoss/Generator/L1Loss",l1_loss,now_step)
                writer.add_scalar("TrainLoss/Generator/WarpLoss",warp_loss,now_step)
                writer.add_scalar("TrainLoss/Generator/PerceptualLoss",p_loss,now_step)
                writer.add_scalar("TrainLoss/Generator/AdversarialLoss",g_loss,now_step)
                writer.add_scalar("TrainLoss/DiscriminatorLoss/real",real_loss,now_step)
                writer.add_scalar("TrainLoss/DiscriminatorLoss/fake",fake_loss,now_step)
                writer.add_scalar("TrainLoss/DiscriminatorLoss/all",d_loss,now_step)
                grid_input_x1=make_grid(source_img[:,[2,1,0],:,:]*0.5+0.5)
                grid_input_x2=make_grid(drive_img[:,[2,1,0],:,:]*0.5+0.5)
                grid_output=make_grid(out[:,[2,1,0],:,:]*0.5+0.5)
                # grid_warp=make_grid(warp[:,[2,1,0],:,:]*0.5+0.5)
                # grid_wi=make_grid(wi[:,[2,1,0],:,:]*0.5+0.5)
                writer.add_image("input/x1",grid_input_x1,now_step)
                writer.add_image("input/x2",grid_input_x2,now_step)
                writer.add_image("output/image",grid_output,now_step)
                # writer.add_image("output/warp",grid_warp,now_step)
                # writer.add_image("output/warp_image",grid_wi,now_step)
            prog_bar.set_description('W:{:0.4f},P:{:0.4f},AD:{:0.4f},D:{:0.4f}'.format(
                running_wrap_loss/next_step,running_p_loss / next_step,running_g_loss/next_step,running_d_loss / next_step))
            save_checkpoint(generator, discriminator)
        val(epoch,val_dataloader)



def val(epoch,val_dataloader):
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        # running_l1_loss, 
        running_warp_loss,running_perceptual_loss, running_ad_loss,running_d_loss =0., 0., 0., 0.
        all_step = len(val_dataloader)
        pro_bar=tqdm(val_dataloader)
            # for step, data in enumerate(val_dataloader):
        for step, data in enumerate(pro_bar):
            now_step=(epoch*all_step)+(step+1)
            x = data[0].to(device)
            y = data[1].to(device)
            y_pred,warp,M,W,phat,qhat= generator(x,y)
            perceptual_loss = perceptualloss(y_pred, y)
            fake=discriminator(y_pred)
            real=discriminator(y)
            ad_loss = bceloss(fake,torch.ones(fake.shape,device=device))
            warp_loss=Warploss(M,W,phat,qhat)
            real_loss,fake_loss = bceloss(real,torch.ones(real.shape,device=device)),bceloss(fake,torch.zeros(fake.shape,device=device))
            d_loss = (real_loss+fake_loss)
            # running_l1_loss += l1loss.item()
            running_warp_loss+=warp_loss.item()
            running_perceptual_loss += perceptual_loss.item()
            running_ad_loss+=ad_loss.item()

            running_d_loss += d_loss.item()
            if step%100==0:
                # writer.add_scalar("ValLoss/Generator/L1Loss",l1loss,now_step)
                writer.add_scalar("ValLoss/Generator/WarpLoss",warp_loss,now_step)
                writer.add_scalar("ValLoss/Generator/PerceptualLoss",perceptual_loss,now_step)
                writer.add_scalar("ValLoss/Generator/AdversarialLoss",ad_loss,now_step)
                writer.add_scalar("ValLoss/Discriminator/real",real_loss,now_step)
                writer.add_scalar("ValLoss/Discriminator/fake",fake_loss,now_step)
                writer.add_scalar("ValLoss/Discriminator/all",d_loss,now_step)
                # writer.add_scalar("ValLoss/Generator/TVLoss",tvloss,now_step)
        print('EVAL| warp:{:0.4f},PLoss:{:0.4f},ADLoss:{:0.4f},DLoss:{:0.4f}'.format(
            running_warp_loss / all_step, running_perceptual_loss / all_step, running_ad_loss/ all_step,
            running_d_loss / all_step))
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', dest='input_dir', type=str)
    parser.add_argument('--batch_size', dest='batch_size',default=8, type=int)
    parser.add_argument("--max_epoch", dest="max_epoch",default=200, type=int)
    parser.add_argument("--learning_rate", dest="lr", default=2e-4, type=float)
    parser.add_argument("--out_checkpoint", dest="out_checkpoint", type=str)
    parser.add_argument("--load_checkpoint", dest="load_checkpoint", type=str)
    parser.add_argument("--log",dest="log",default=None,type=str)
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator=Generator().to(device)
    discriminator=PatchGan(3).to(device)
    g_optimizer = optim.Adam(generator.parameters(),
                             lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=args.lr, betas=(0.5, 0.999))
    kp_optimzer=optim.Adam(optim.Adam(discriminator.parameters(),
                             lr=args.lr, betas=(0.5, 0.999)))
    Warploss=WarpLoss().to(device)
    perceptualloss=VGGPerceptualLoss().to(device)
    bceloss=nn.BCELoss().to(device)
    train_dataset=TestDataSet(args.input_dir,"train")
    val_dataset=TestDataSet(args.input_dir,"val")
    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,num_workers=4,drop_last=True,shuffle=True,pin_memory=True)
    val_dataloader=DataLoader(val_dataset,batch_size=args.batch_size,num_workers=4,drop_last=True,shuffle=True,pin_memory=True)
    if args.log is None:
        writer=SummaryWriter()
    else:
        writer=SummaryWriter(f"runs/{args.log}")
    train(train_dataloader,val_dataloader)


