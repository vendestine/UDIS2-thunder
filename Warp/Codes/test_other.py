import argparse  # 导入参数解析模块，用于命令行参数处理
import torch  # 导入PyTorch深度学习框架

import numpy as np  # 导入NumPy数值计算库
import os  # 导入操作系统接口模块
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块

import cv2  # 导入OpenCV图像处理库
#from torch_homography_model import build_model
from network import get_stitched_result, Network, build_new_ft_model  # 从network.py导入必要的函数和类

import glob  # 导入文件路径模式匹配模块
from loss import cal_lp_loss2  # 从loss.py导入损失函数
import torchvision.transforms as T  # 导入图像变换模块

#import PIL
resize_512 = T.Resize((512,512))  # 创建调整图像大小为512x512的变换


def loadSingleData(data_path, img1_name, img2_name):
    """
    加载单对图像数据
    
    参数:
        data_path: 图像数据路径
        img1_name: 第一张图像的文件名
        img2_name: 第二张图像的文件名
    
    返回:
        两张图像的张量表示
    """
    # 加载图像1
    input1 = cv2.imread(data_path+img1_name)
    input1 = input1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0  # 归一化到[-1, 1]范围
    input1 = np.transpose(input1, [2, 0, 1])  # 调整维度顺序为[通道,高,宽]

    # 加载图像2
    input2 = cv2.imread(data_path+img2_name)
    input2 = input2.astype(dtype=np.float32)
    input2 = (input2 / 127.5) - 1.0  # 归一化到[-1, 1]范围
    input2 = np.transpose(input2, [2, 0, 1])  # 调整维度顺序为[通道,高,宽]

    # 转换为张量
    input1_tensor = torch.tensor(input1).unsqueeze(0)  # 添加批次维度
    input2_tensor = torch.tensor(input2).unsqueeze(0)  # 添加批次维度
    return (input1_tensor, input2_tensor)



# 项目路径
#nl: os.path.dirname("__file__") ----- 当前绝对路径
#nl: os.path.pardir ---- 上一级路径
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))


#nl: 保存模型文件的路径
MODEL_DIR = os.path.join(last_path, 'model')

#nl: 如果模型目录不存在则创建
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def train(args):
    """
    训练和优化拼接模型函数
    
    参数:
        args: 包含训练参数的命名空间
    """
    # 检查GPU是否可用
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        # 设置CUDA设备顺序和可见设备
        os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda')
        print("使用GPU进行计算")
    else:
        device = torch.device('cpu')
        print("使用CPU进行计算")
    
    # 定义网络模型
    net = Network()
    net = net.to(device)  # 将模型移至设备（GPU或CPU）

    # 定义优化器和学习率
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # 默认学习率为0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)  # 学习率指数衰减

    # 如果存在已训练的模型则加载
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]  # 获取最新的模型文件
        # 根据当前设备加载模型
        checkpoint = torch.load(model_path, map_location=device)

        net.load_state_dict(checkpoint['model'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
        
        # 确保优化器状态中的张量在正确的设备上
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        start_epoch = checkpoint['epoch']  # 获取起始轮次
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        print('training from stratch!')  # 从头开始训练

    # 加载数据集(只有一对图像)
    input1_tensor, input2_tensor = loadSingleData(data_path=args.path, img1_name = args.img1_name, img2_name = args.img2_name)
    # 将图像数据移至设备（GPU或CPU）
    input1_tensor = input1_tensor.to(device)
    input2_tensor = input2_tensor.to(device)

    # 调整图像大小为512x512，用于网络输入
    input1_tensor_512 = resize_512(input1_tensor)
    input2_tensor_512 = resize_512(input2_tensor)

    loss_list = []  # 用于存储每次迭代的损失值

    print("##################开始迭代优化#######################")
    for epoch in range(start_epoch, start_epoch + args.max_iter):
        net.train()  # 设置网络为训练模式

        optimizer.zero_grad()  # 梯度清零

        # 通过网络获取输出
        batch_out = build_new_ft_model(net, input1_tensor_512, input2_tensor_512)
        warp_mesh = batch_out['warp_mesh']  # 变形后的图像
        warp_mesh_mask = batch_out['warp_mesh_mask']  # 变形后的掩码
        rigid_mesh = batch_out['rigid_mesh']  # 刚性网格
        mesh = batch_out['mesh']  # 变形网格

        # 计算损失并反向传播
        total_loss = cal_lp_loss2(input1_tensor_512, warp_mesh, warp_mesh_mask)
        total_loss.backward()
        # 裁剪梯度，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
        optimizer.step()  # 更新参数

        current_iter = epoch-start_epoch+1
        print("训练: 迭代[{:0>3}/{:0>3}] 总损失: {:.4f} 学习率={:.8f}".format(current_iter, args.max_iter, total_loss, optimizer.state_dict()['param_groups'][0]['lr']))

        # 记录损失值
        loss_list.append(total_loss)

        # 第一次迭代后保存优化前的结果
        if current_iter == 1:
            with torch.no_grad():
                output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)
                
            # 保存图像结果
            stitched_img = output['stitched'][0].cpu().detach().numpy().transpose(1,2,0) if use_gpu else output['stitched'][0].detach().numpy().transpose(1,2,0)
            cv2.imwrite(args.path + 'before_optimization.jpg', stitched_img)
            cv2.imwrite(args.path + 'before_optimization_mesh.jpg', output['stitched_mesh'])

        # 如果连续四次迭代损失变化很小，则提前结束训练
        if current_iter >= 4:
            if torch.abs(loss_list[current_iter-4]-loss_list[current_iter-3]) <= 1e-4 and torch.abs(loss_list[current_iter-3]-loss_list[current_iter-2]) <= 1e-4 \
            and torch.abs(loss_list[current_iter-2]-loss_list[current_iter-1]) <= 1e-4:
                with torch.no_grad():
                    output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)

                # 保存结果图像
                path = args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + ".jpg"
                
                # 根据设备类型处理图像数据
                if use_gpu:
                    stitched_img = output['stitched'][0].cpu().detach().numpy().transpose(1,2,0)
                    warp1_img = output['warp1'][0].cpu().detach().numpy().transpose(1,2,0)
                    warp2_img = output['warp2'][0].cpu().detach().numpy().transpose(1,2,0)
                    mask1_img = output['mask1'][0].cpu().detach().numpy().transpose(1,2,0)
                    mask2_img = output['mask2'][0].cpu().detach().numpy().transpose(1,2,0)
                else:
                    stitched_img = output['stitched'][0].detach().numpy().transpose(1,2,0)
                    warp1_img = output['warp1'][0].detach().numpy().transpose(1,2,0)
                    warp2_img = output['warp2'][0].detach().numpy().transpose(1,2,0)
                    mask1_img = output['mask1'][0].detach().numpy().transpose(1,2,0)
                    mask2_img = output['mask2'][0].detach().numpy().transpose(1,2,0)
                
                cv2.imwrite(path, stitched_img)
                cv2.imwrite(args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
                cv2.imwrite(args.path + 'warp1.jpg', warp1_img)
                cv2.imwrite(args.path + 'warp2.jpg', warp2_img)
                cv2.imwrite(args.path + 'mask1.jpg', mask1_img)
                cv2.imwrite(args.path + 'mask2.jpg', mask2_img)
                break

        # 达到最大迭代次数时保存结果
        if current_iter == args.max_iter:
            with torch.no_grad():
                output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)

            # 保存最终结果图像
            path = args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + ".jpg"
            
            # 根据设备类型处理图像数据
            if use_gpu:
                stitched_img = output['stitched'][0].cpu().detach().numpy().transpose(1,2,0)
                warp1_img = output['warp1'][0].cpu().detach().numpy().transpose(1,2,0)
                warp2_img = output['warp2'][0].cpu().detach().numpy().transpose(1,2,0)
                mask1_img = output['mask1'][0].cpu().detach().numpy().transpose(1,2,0)
                mask2_img = output['mask2'][0].cpu().detach().numpy().transpose(1,2,0)
            else:
                stitched_img = output['stitched'][0].detach().numpy().transpose(1,2,0)
                warp1_img = output['warp1'][0].detach().numpy().transpose(1,2,0)
                warp2_img = output['warp2'][0].detach().numpy().transpose(1,2,0)
                mask1_img = output['mask1'][0].detach().numpy().transpose(1,2,0)
                mask2_img = output['mask2'][0].detach().numpy().transpose(1,2,0)
            
            cv2.imwrite(path, stitched_img)
            cv2.imwrite(args.path + "iter-" + str(epoch-start_epoch+1).zfill(3) + "_mesh.jpg", output['stitched_mesh'])
            cv2.imwrite(args.path + 'warp1.jpg', warp1_img)
            cv2.imwrite(args.path + 'warp2.jpg', warp2_img)
            cv2.imwrite(args.path + 'mask1.jpg', mask1_img)
            cv2.imwrite(args.path + 'mask2.jpg', mask2_img)

        scheduler.step()  # 更新学习率

    print("##################迭代优化结束#######################")


if __name__=="__main__":
    """
    主程序入口
    """

    print('<==================== 设置参数 ===================>\n')

    # 创建参数解析器
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--gpu', type=str, default='0')  # 使用的GPU编号
    parser.add_argument('--max_iter', type=int, default=50)  # 最大迭代次数
    parser.add_argument('--path', type=str, default='../../Carpark-DHW/')  # 输入图像路径
    parser.add_argument('--img1_name', type=str, default='input1.jpg')  # 第一张图像文件名
    parser.add_argument('--img2_name', type=str, default='input2.jpg')  # 第二张图像文件名

    # 解析参数
    args = parser.parse_args()
    print(args)

    # 开始训练
    train(args)


