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
class QwenAdversarialPatch(QwenAdversarialBase):
    def __init__(self, model_dir, target_image_path, background_image_path, lr=1e-1, steps=300):
        super().__init__(model_dir, target_image_path, lr, steps)

        # 加载背景图像（宿主图）
        self.background_image = Image.open(background_image_path).convert("RGB")
        # self.background_image = torch.tensor(np.array(self.background_image))  # 转换为张量
        self.background_image = self.background_image.resize((self.target_image.width, self.target_image.height), Image.BICUBIC)  
        
        # print(f"self.background_image.shape: {self.background_image.shape}")
        print(f"self.background_image.size: {self.background_image.size}")

    def train_patch(self, patch_size=(50, 50), position=(0, 0)):
        print(f"开始 Patch 优化，patch_size={patch_size}, position={position}")

        base_tensor = torch.tensor(np.array(self.background_image), dtype=torch.float32).to(self.device)  # 转换为张量[3,H,W]
        base_tensor = base_tensor.permute(2,0,1).contiguous()
        base_tensor.requires_grad_(True)  # 显式启用梯度
        print(f"base_tensor.shape: {base_tensor.shape}")
    
        _, H, W= base_tensor.shape
        ph, pw = patch_size
        y, x = position
        assert y+ph <= H and x+pw <= W, "Patch 超出了原图范围! ph={ph}, pw={pw}, H={H}, W={W}, x={x}, y={y}"

        # 初始化 patch``
        patch = torch.randn((3, ph, pw), device=self.device, requires_grad=True)
        optimizer = optim.Adam([patch], lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        loss_fn = nn.MSELoss()

        self.loss_history = []

        for step in range(self.steps):
            optimizer.zero_grad()

            patched_image = base_tensor.clone()
            patched_image[:, y:y+ph, x:x+pw] = patch.clamp(0, 255)
            # print(f"patched_image.shape: {patched_image.shape}, patch.shape: {patch.shape}")
            
            cur_inputs = self.processor(
                text=["Describe this image"],
                images=[patched_image],
                return_tensors="pt"
            ).to(self.device)

            cur_embeds = self.model.get_image_features(
                cur_inputs['pixel_values'],
                cur_inputs['image_grid_thw']
            )
                    
            # print(f"cur_embeds.shape: {cur_embeds[0].shape}, self.target_embeds.shape: {self.target_embeds[0].shape}")
            loss = loss_fn(cur_embeds[0], self.target_embeds[0])
            self.loss_history.append(loss.item())
            loss.backward()
            # 调试：检查梯度
            if (step + 1) % 10 == 0:
                grad_norm = torch.norm(patch.grad).item() if patch.grad is not None else 0
                print(f"梯度范数: {grad_norm:.6f}")
                if patch.grad is None:
                    print("警告：patch梯度为None！")
                else:
                    print(f"梯度均值: {patch.grad.mean().item():.6f}")
            
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                patch.clamp_(0, 255)

            if (step+1) % 1 == 0:
                print(f"[Patch] Step {step+1}/{self.steps}, Loss={loss.item():.6f}")
                
                # 保存中间结果和生成描述
                if (step + 1) % 10 == 0:
                    img_path = os.path.join(self.output_dir, f"patch_step_{step+1}.png")
                    T.ToPILImage()(patched_image.to(torch.uint8).cpu()).save(img_path)
                    # T.ToPILImage()(patched_image.cpu()).save(img_path)
                    desc = self._generate_description(patched_image,step)
                    print(f"图像描述: {desc}")
                    self._plot_loss()
         

        final_img_path = os.path.join(self.output_dir, "patched_final.png")
        T.ToPILImage()(patched_image.to(torch.uint8).cpu()).save(img_path)

        self._plot_loss()
        print(f"\n✅ Patch 优化完成，结果保存到 {final_img_path}")
        return final_img_path


# =====================
# main
# =====================
if __name__ == "__main__":
    MODEL_DIR = "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct"
    TARGET_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/attack/test1.png"
    BACKGROUND_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/attack/test1.png"

    # Patch 优化
    adv_patch = QwenAdversarialPatch(
        model_dir=MODEL_DIR,
        target_image_path=TARGET_IMAGE_PATH,      # 目标对齐图像
        background_image_path=BACKGROUND_IMAGE_PATH,  # 被攻击宿主图像
        lr=1,
        steps=500
    )
    adv_patch.train_patch(patch_size=(140, 200), position=(220, 30))
