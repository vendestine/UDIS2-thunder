# coding: utf-8
import argparse
import torch
from network import build_model, Network
import os
import numpy as np
import cv2
import glob



last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')

def loadSingleData(data_path):

    # load image1
    warp1 = cv2.imread(data_path+"warp1.jpg")
    warp1 = warp1.astype(dtype=np.float32)
    warp1 = (warp1 / 127.5) - 1.0
    warp1 = np.transpose(warp1, [2, 0, 1])

    # load image2
    warp2 = cv2.imread(data_path+"warp2.jpg")
    warp2 = warp2.astype(dtype=np.float32)
    warp2 = (warp2 / 127.5) - 1.0
    warp2 = np.transpose(warp2, [2, 0, 1])

    # load mask1
    mask1 = cv2.imread(data_path+"mask1.jpg")
    mask1 = mask1.astype(dtype=np.float32)
    mask1 = mask1 / 255
    mask1 = np.transpose(mask1, [2, 0, 1])

    # load mask2
    mask2 = cv2.imread(data_path+"mask2.jpg")
    mask2 = mask2.astype(dtype=np.float32)
    mask2 = mask2 / 255
    mask2 = np.transpose(mask2, [2, 0, 1])

    # convert to tensor
    warp1_tensor = torch.tensor(warp1).unsqueeze(0)
    warp2_tensor = torch.tensor(warp2).unsqueeze(0)
    mask1_tensor = torch.tensor(mask1).unsqueeze(0)
    mask2_tensor = torch.tensor(mask2).unsqueeze(0)

    return warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor


def test_other(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 确定设备：如果GPU可用且用户没有指定-1，则使用GPU，否则使用CPU
    use_cuda = torch.cuda.is_available() and args.gpu != '-1'
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"使用设备: {device}")

    # define the network
    net = Network()
    net = net.to(device)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        
        # 根据设备加载模型
        if use_cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')
        return


    # load dataset(only one pair of images)
    warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor = loadSingleData(data_path=args.path)
    # 将张量移动到正确的设备上
    warp1_tensor = warp1_tensor.to(device)
    warp2_tensor = warp2_tensor.to(device)
    mask1_tensor = mask1_tensor.to(device)
    mask2_tensor = mask2_tensor.to(device)

    net.eval()
    with torch.no_grad():
        batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
    stitched_image = batch_out['stitched_image']
    learned_mask1 = batch_out['learned_mask1']
    learned_mask2 = batch_out['learned_mask2']

    # (optional) draw composition images with different colors like our paper
    s1 = ((warp1_tensor[0]+1)*127.5 * learned_mask1[0]).cpu().detach().numpy().transpose(1,2,0)
    s2 = ((warp2_tensor[0]+1)*127.5 * learned_mask2[0]).cpu().detach().numpy().transpose(1,2,0)
    fusion = np.zeros((warp1_tensor.shape[2],warp1_tensor.shape[3],3), np.uint8)
    fusion[...,0] = s2[...,0]
    fusion[...,1] = s1[...,1]*0.5 +  s2[...,1]*0.5
    fusion[...,2] = s1[...,2]
    path = args.path + "composition_color.jpg"
    cv2.imwrite(path, fusion)


    # save learned masks and final composition
    stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
    learned_mask1 = (learned_mask1[0]*255).cpu().detach().numpy().transpose(1,2,0)
    learned_mask2 = (learned_mask2[0]*255).cpu().detach().numpy().transpose(1,2,0)

    path = args.path + "learn_mask1.jpg"
    cv2.imwrite(path, learned_mask1)
    path = args.path + "learn_mask2.jpg"
    cv2.imwrite(path, learned_mask2)
    path = args.path + "composition.jpg"
    cv2.imwrite(path, stitched_image)

    print(f"处理完成！结果已保存到: {args.path}")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID，设置为-1表示使用CPU')
    parser.add_argument('--path', type=str, default='../../Carpark-DHW/')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)

    test_other(args)