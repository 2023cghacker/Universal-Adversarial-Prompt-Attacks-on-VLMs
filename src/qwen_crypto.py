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
    # adversarial.run_optimization(background_image_path=BACKGROUND_IMAGE_PATH)
    adversarial.train_crypto(BACKGROUND_IMAGE_PATH1,BACKGROUND_IMAGE_PATH2)
