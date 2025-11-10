import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torchvision.transforms as T
import os
import datetime
import matplotlib.pyplot as plt
from adversarial_qwen import QwenAdversarialBase
import numpy as np


# =====================
# 子类：Patch 优化
# =====================
class QwenAdversarialCrypto(QwenAdversarialBase):
    def __init__(self, model_dir, target_image_path, lr=1e-1, steps=300):
        super().__init__(model_dir, target_image_path, lr, steps)
        
    def train_crypto(self, background_image_path1, background_image_path2):
        # 加载第一张背景图像
        bg1 = Image.open(background_image_path1).convert("RGB")
        # 加载第二张背景图像
        bg2 = Image.open(background_image_path2).convert("RGB")
        
        # 统一尺寸为目标图像大小
        target_size = (self.target_image.width, self.target_image.height)
        bg1 = bg1.resize(target_size, Image.BICUBIC)
        bg2 = bg2.resize(target_size, Image.BICUBIC)
        print(f"背景图像1尺寸: {bg1.size}, 背景图像2尺寸: {bg2.size}")

        # 将两张图像转换为张量并叠加（像素值取平均，避免溢出）
        bg1_tensor = torch.tensor(np.array(bg1), dtype=torch.float32).to(self.device)  # [H, W, 3]
        bg2_tensor = torch.tensor(np.array(bg2), dtype=torch.float32).to(self.device)  # [H, W, 3]
        final_bg_tensor = (bg1_tensor + bg2_tensor) / 2.0  # 像素叠加（平均）
        final_bg_tensor = final_bg_tensor.permute(2, 0, 1).contiguous()  # 转换为[3, H, W]
        final_bg_tensor.requires_grad_(True)  # 启用梯度
        print(f"叠加后背景张量形状: {final_bg_tensor.shape}")

        # 获取图像尺寸
        _, H, W = final_bg_tensor.shape

        # 初始化噪声
        noise = torch.randn((3, H, W), device=self.device, requires_grad=True)
        optimizer = optim.Adam([noise], lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        loss_fn = nn.MSELoss()

        self.loss_history = []

        for step in range(self.steps):
            optimizer.zero_grad()

            # 将噪声添加到叠加后的背景图像
            noisy_image = final_bg_tensor.clone()
            noisy_image += noise.clamp(0, 255)  # 限制噪声范围
            noisy_image = torch.clamp(noisy_image, 0, 255)  # 确保像素值合法

            # 处理输入并获取图像特征
            cur_inputs = self.processor(
                text=["Describe this image"],
                images=[noisy_image],
                return_tensors="pt"
            ).to(self.device)

            cur_embeds = self.model.get_image_features(
                cur_inputs['pixel_values'],
                cur_inputs['image_grid_thw']
            )

            # 计算损失
            loss = loss_fn(cur_embeds[0], self.target_embeds[0])
            self.loss_history.append(loss.item())

            # 反向传播与优化
            loss.backward()
            if (step + 1) % 10 == 0:
                grad_norm = torch.norm(noise.grad).item() if noise.grad is not None else 0
                print(f"梯度范数: {grad_norm:.6f}")
                if noise.grad is None:
                    print("警告：noise梯度为None！")
                else:
                    print(f"梯度均值: {noise.grad.mean().item():.6f}")
            
            optimizer.step()
            scheduler.step()

            # 限制噪声范围
            with torch.no_grad():
                noise.clamp_(0, 255)

            # 打印日志与保存中间结果
            if (step + 1) % 1 == 0:
                print(f"[noise] Step {step+1}/{self.steps}, Loss={loss.item():.6f}")
                
                if step == 0:
                    self.visualize_embeddings_2d(cur_embeds[0], self.target_embeds[0], step)
                if (step + 1) % 10 == 0:
                    self.visualize_embeddings_2d(cur_embeds[0], self.target_embeds[0], step)
                    img_path = os.path.join(self.output_dir, f"noise_step_{step+1}.png")
                    T.ToPILImage()(noisy_image.to(torch.uint8).cpu()).save(img_path)
                    desc = self._generate_description(noisy_image, step, prompt="Describe this image")
                    print(f"图像描述: {desc}")
                    self._plot_loss()

        # 保存最终结果
        final_img_path = os.path.join(self.output_dir, "noiseed_final.png")
        T.ToPILImage()(noisy_image.to(torch.uint8).cpu()).save(final_img_path)
        self._plot_loss()
        print(f"\n✅ noise 优化完成，结果保存到 {final_img_path}")
        return final_img_path
    
    def train_noise_multi_background(self, bg_image_path1, bg_image_path2):
        """
        使用两张背景图像作为输入，一起生成扰动，并对模型 embedding 与目标 embedding 做对齐优化。
        参数:
            bg_image_path1: str — 背景图像路径1
            bg_image_path2: str — 背景图像路径2
        返回:
            final_img_path: str — 优化后的最终图像保存路径
        """
        # 加载两张背景图像（宿主图）并转换
        img1 = Image.open(bg_image_path1).convert("RGB")
        img2 = Image.open(bg_image_path2).convert("RGB")
        # 统一尺寸为目标图像尺寸（假设 self.target_image 已设置）
        img1 = img1.resize((self.target_image.width//2, self.target_image.height), Image.BICUBIC)
        img2 = img2.resize((self.target_image.width//2, self.target_image.height), Image.BICUBIC)
        print(f"bg1.size: {img1.size}, bg2.size: {img2.size}")
        print("开始 noise 优化（多图输入）")

        # 转换为张量 [C, H, W]
        tensor1 = torch.tensor(np.array(img1), dtype=torch.float32).to(self.device).permute(2,0,1).contiguous()
        tensor2 = torch.tensor(np.array(img2), dtype=torch.float32).to(self.device).permute(2,0,1).contiguous()
        tensor1.requires_grad_(True)
        tensor2.requires_grad_(True)
        print(f"tensor1.shape: {tensor1.shape}, tensor2.shape: {tensor2.shape}")

        _, H, W = tensor1.shape

        # 初始化 noise，针对两个背景图叠加一个共同扰动或者分别扰动（这里示例为两个图共享同一 noise）
        noise1 = torch.randn((3, H, W), device=self.device, requires_grad=True)
        noise2 = torch.randn((3, H, W), device=self.device, requires_grad=True)
        optimizer = optim.Adam([noise1,noise2], lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        loss_fn = nn.MSELoss()
        # target_emb = torch.cat([self.target_embeds[0], self.target_embeds[0]], dim=0)   
        target_emb = self.target_embeds[0]
        self.loss_history = []

        for step in range(self.steps):
            optimizer.zero_grad()

            # 给两张背景图加扰动
            noisy1 = tensor1.clone() + noise1.clamp(0,255)
            noisy2 = tensor2.clone() + noise2.clamp(0,255)

            # 构造模型输入：两张图像一起
            # 这里假设 processor 支持 list 多图像
            cur_inputs = self.processor(
                text=["Describe these two images."],
                images=[noisy1, noisy2],
                return_tensors="pt"
            ).to(self.device)

            cur_embeds = self.model.get_image_features(
                cur_inputs['pixel_values'],
                cur_inputs['image_grid_thw']
            )

            # 假设 cur_embeds 返回 batch 的 embedding，取第一个
            # print(f"cur_embeds: {cur_embeds}, len(cur_embeds):{len(cur_embeds)}")
            combined_emb = torch.cat([cur_embeds[0], cur_embeds[1]], dim=0)
            print(f"combined_emb.shape: {combined_emb.shape}, target_embeds.shape: {target_emb.shape}")
            loss = loss_fn(combined_emb, target_emb)

            self.loss_history.append(loss.item())
            loss.backward()

            if (step + 1) % 10 == 0:
                grad_norm = torch.norm(noise1.grad).item() if noise1.grad is not None else 0
                print(f"梯度范数: {grad_norm:.6f}")
                if noise1.grad is None:
                    print("警告：noise梯度为 None！")
                else:
                    print(f"梯度均值: {noise1.grad.mean().item():.6f}")

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                noise1.clamp_(0,255)
                noise2.clamp_(0,255)

            if (step + 1) % 1 == 0:
                print(f"[noise] Step {step+1}/{self.steps}, Loss={loss.item():.6f}")
                if step == 0 or (step + 1) % 10 == 0:
                    self.visualize_embeddings_2d(combined_emb, target_emb, step)
                    img_path = os.path.join(self.output_dir, f"noise_step_{step+1}.png")
                    # 保存其中一张扰动后的图像，这里保存第一张为示例
                    T.ToPILImage()(noisy1.to(torch.uint8).cpu()).save(img_path)
                    desc = self._generate_description([noisy1,noisy2], step, prompt="Describe these two images.")
                    print(f"图像描述: {desc}")
                    self._plot_loss()

        final_img_path = os.path.join(self.output_dir, "noiseed_multi_final.png")
        T.ToPILImage()(noisy1.to(torch.uint8).cpu()).save(final_img_path)

        self._plot_loss()
        print(f"\n✅ noise 优化完成（多图），结果保存到 {final_img_path}")
        return final_img_path

# 使用示例
if __name__ == "__main__":
    MODEL_DIR = "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct"
    # TARGET_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/attack/content1.png"
    TARGET_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/B.png"
    BACKGROUND_IMAGE_PATH1 = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/apple.png"
    BACKGROUND_IMAGE_PATH2 = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/dog.png"
    
    # 创建实例并运行优化
    adversarial = QwenAdversarialCrypto(
        model_dir=MODEL_DIR,
        target_image_path=TARGET_IMAGE_PATH,
        lr=5e-1,
        steps=1000
    )
    # adversarial.train_crypto(BACKGROUND_IMAGE_PATH1,BACKGROUND_IMAGE_PATH2)
    adversarial.train_noise_multi_background(BACKGROUND_IMAGE_PATH1,BACKGROUND_IMAGE_PATH2)
    
