import cv2
import torch
import os
from model.network_new import Generator
import argparse
from torchvision import transforms
def load_checkpoint(model_G):
    checkpoint_root = args.load_checkpoint
    G_model_path = os.path.join("checkpoint",checkpoint_root, "frG.pth")
    model_G.load_state_dict(torch.load(G_model_path))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", dest="load_checkpoint", type=str)
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator=Generator().to(device)
    generator.eval()
    source=cv2.imread("img/1.png")
    video=cv2.VideoCapture("test1.mp4")
    load_checkpoint(generator)
    t=transforms.Compose([transforms.ToTensor()])
    s_i=t(source).unsqueeze(0).cuda()
    with torch.no_grad():
        while video.isOpened():
            ret,frame=video.read()
            if not ret:
                break
            d_i=t(frame).unsqueeze(0).cuda()
            out=generator(s_i,d_i)

            c_i=torch.cat([s_i,d_i,out[0]],dim=3)
            new=c_i[0].permute(1,2,0).cpu().numpy()
            cv2.imshow("frame",new)
            cv2.waitKey(1)
        