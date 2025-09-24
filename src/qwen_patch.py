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


# =====================
# 子类：Patch 优化
# =====================
class QwenAdversarialPatch(QwenAdversarialBase):
    def __init__(self, model_dir, target_image_path, background_image_path, lr=1e-1, steps=300):
        super().__init__(model_dir, target_image_path, lr, steps)

        # 加载背景图像（宿主图）
        self.background_image = Image.open(background_image_path).convert("RGB")
        self.background_image = self.background_image.resize((self.target_image.width, self.target_image.height), Image.BICUBIC)  
        

    def train_patch(self, patch_size=(50, 50), position=(0, 0)):
        print(f"开始 Patch 优化，patch_size={patch_size}, position={position}")

        transform = T.ToTensor()
        base_tensor = transform(self.background_image).unsqueeze(0).to(self.device)  # [1,3,H,W]

        _, _, H, W = base_tensor.shape
        ph, pw = patch_size
        y, x = position
        assert y+ph <= H and x+pw <= W, "Patch 超出了原图范围！"

        # 初始化 patch
        patch = torch.randn((1, 3, ph, pw), device=self.device, requires_grad=True)

        optimizer = optim.Adam([patch], lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        loss_fn = nn.MSELoss()

        self.loss_history = []

        for step in range(self.steps):
            optimizer.zero_grad()

            patched_image = base_tensor.clone()
            patched_image[:, :, y:y+ph, x:x+pw] = patch.clamp(0, 1)
            
            cur_inputs = self.processor(
                text=["Describe this image"],
                # images=[patched_image.squeeze(0).clamp(0, 1)],
                images=[patched_image.squeeze(0)],
                return_tensors="pt"
            ).to(self.device)

            cur_embeds = self.model.get_image_features(
                cur_inputs['pixel_values'],
                cur_inputs['image_grid_thw']
            )
            
            loss = loss_fn(cur_embeds[0], self.target_embeds[0])
            self.loss_history.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                patch.clamp_(0, 1)

            if (step+1) % 1 == 0:
                print(f"[Patch] Step {step+1}/{self.steps}, Loss={loss.item():.6f}")
                img_path = os.path.join(self.output_dir, f"patch_step_{step+1}.png")
                T.ToPILImage()(patched_image.squeeze(0).cpu()).save(img_path)
                
             
                # 保存中间结果和生成描述
                if (step + 1) % 10 == 0:
                    img_path = os.path.join(self.output_dir, f"patch_step_{step+1}.png")
                    T.ToPILImage()(patched_image.squeeze(0).cpu()).save(img_path)
                    desc = self._generate_description(patched_image,step)
                    print(f"图像描述: {desc}")
         

        final_img_path = os.path.join(self.output_dir, "patched_final.png")
        T.ToPILImage()(patched_image.squeeze(0).cpu()).save(final_img_path)

        self._plot_loss()
        print(f"\n✅ Patch 优化完成，结果保存到 {final_img_path}")
        return final_img_path


# =====================
# main
# =====================
if __name__ == "__main__":
    MODEL_DIR = "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct"
    TARGET_IMAGE_PATH = "/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/apple.png"
    BACKGROUND_IMAGE_PATH = "/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/dog.png"

    # Patch 优化
    adv_patch = QwenAdversarialPatch(
        model_dir=MODEL_DIR,
        target_image_path=TARGET_IMAGE_PATH,      # 目标对齐图像
        background_image_path=BACKGROUND_IMAGE_PATH,  # 被攻击宿主图像
        lr=1e-1,
        steps=500
    )
    adv_patch.train_patch(patch_size=(70, 70), position=(20, 30))
