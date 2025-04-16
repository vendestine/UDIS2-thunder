# coding: utf-8
# 导入所需的库和模块
import argparse  # 用于解析命令行参数
import torch  # PyTorch深度学习框架
from torch.utils.data import DataLoader  # 数据加载器
import torch.nn as nn  # 神经网络模块
import imageio  # 用于图像I/O操作，如GIF创建
from network import build_model, Network  # 导入自定义网络模型
from dataset import *  # 导入数据集处理相关函数
import os  # 操作系统接口
import numpy as np  # 数值计算库
import skimage  # 图像处理库，用于计算PSNR和SSIM
import cv2  # OpenCV库，用于图像处理
import glob  # 用于文件路径匹配


# 获取上级目录路径
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')  # 模型保存目录

# 创建GIF动画的辅助函数
def create_gif(image_list, gif_name, duration=0.35):
    """
    将图像列表转换为GIF动画
    
    参数：
    image_list - 图像文件路径列表
    gif_name - 输出GIF文件名
    duration - 每帧持续时间
    """
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def test(args):
    """
    主测试函数，用于评估模型性能
    
    参数：
    args - 命令行参数
    """
    # 设置GPU环境
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 加载测试数据集
    test_data = TestDataset(data_path=args.test_path)  # 创建测试数据集实例
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)  # 创建数据加载器

    # 定义网络模型
    net = Network()  # 创建网络实例
    use_cuda = torch.cuda.is_available() and args.gpu != '-1'
    if use_cuda:
        net = net.cuda()  # 如果可用，将模型移至GPU
        print("使用GPU进行测试")
    else:
        print("使用CPU进行测试")

    # 加载已有的模型权重（如果存在）
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")  # 获取所有.pth模型文件
    ckpt_list.sort()  # 按名称排序
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]  # 获取最新的模型文件
        
        # 根据是否使用GPU选择加载方式
        if use_cuda:
            checkpoint = torch.load(model_path)  # 加载模型权重
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # 在CPU上加载模型权重
            
        net.load_state_dict(checkpoint['model'])  # 将权重加载到模型中
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found! 请确保model目录中有预训练模型文件(.pth)')
        return  # 如果没有找到模型文件，直接返回

    print("##################start testing#######################")
    psnr_list = []  # 存储所有测试样本的PSNR值
    ssim_list = []  # 存储所有测试样本的SSIM值
    net.eval()  # 设置模型为评估模式
    for i, batch_value in enumerate(test_loader):
        # 获取输入图像对
        inpu1_tesnor = batch_value[0].float()  # 第一张图像
        inpu2_tesnor = batch_value[1].float()  # 第二张图像

        if use_cuda:
            # 将输入数据移至GPU
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        with torch.no_grad():  # 关闭梯度计算，减少内存消耗
            # 通过模型生成输出
            batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor, is_training=False)

        # 获取变形后的网格掩码和网格
        warp_mesh_mask = batch_out['warp_mesh_mask']  # 变形后的掩码
        warp_mesh = batch_out['warp_mesh']  # 变形后的图像


        # 将张量转换为NumPy数组，用于计算评估指标
        warp_mesh_np = ((warp_mesh[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)  # 将[-1,1]范围转换为[0,255]
        warp_mesh_mask_np = warp_mesh_mask[0].cpu().detach().numpy().transpose(1,2,0)
        inpu1_np = ((inpu1_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)

        try:
            # 计算PSNR和SSIM质量指标
            # 只在有效区域（掩码区域）计算指标
            psnr = skimage.measure.compare_psnr(inpu1_np*warp_mesh_mask_np, warp_mesh_np*warp_mesh_mask_np, 255)
            ssim = skimage.measure.compare_ssim(inpu1_np*warp_mesh_mask_np, warp_mesh_np*warp_mesh_mask_np, data_range=255, multichannel=True)

            print('i = {}, psnr = {:.6f}'.format( i+1, psnr))  # 打印当前样本的PSNR值

            # 保存结果
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        except Exception as e:
            print(f"处理第{i+1}个样本时出错: {e}")
            
        if use_cuda:
            torch.cuda.empty_cache()  # 清空GPU缓存

    # 结果分析部分
    print("=================== Analysis ==================")
    print("psnr")
    psnr_list.sort(reverse = True)  # 对PSNR值降序排序
    # 将结果分为三组：前30%、30%-60%、60%-100%
    psnr_list_30 = psnr_list[0 : 331]
    psnr_list_60 = psnr_list[331: 663]
    psnr_list_100 = psnr_list[663: -1]
    # 输出各组平均PSNR值
    print("top 30%", np.mean(psnr_list_30))
    print("top 30~60%", np.mean(psnr_list_60))
    print("top 60~100%", np.mean(psnr_list_100))
    print('average psnr:', np.mean(psnr_list))  # 总平均PSNR

    # 对SSIM值进行类似的分析
    ssim_list.sort(reverse = True)  # 对SSIM值降序排序
    ssim_list_30 = ssim_list[0 : 331]
    ssim_list_60 = ssim_list[331: 663]
    ssim_list_100 = ssim_list[663: -1]
    # 输出各组平均SSIM值
    print("top 30%", np.mean(ssim_list_30))
    print("top 30~60%", np.mean(ssim_list_60))
    print("top 60~100%", np.mean(ssim_list_100))
    print('average ssim:', np.mean(ssim_list))  # 总平均SSIM
    print("##################end testing#######################")


if __name__=="__main__":
    # 程序入口点
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('--gpu', type=str, default='-1')  # 指定使用的GPU，默认-1表示使用CPU
    parser.add_argument('--batch_size', type=int, default=1)  # 批处理大小
    parser.add_argument('--test_path', type=str, default='../../UDIS-D/testing/')  # 测试数据路径，修改为您的数据集路径

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()  # 解析命令行参数
    print(args)  # 打印参数信息
    test(args)  # 执行测试函数
