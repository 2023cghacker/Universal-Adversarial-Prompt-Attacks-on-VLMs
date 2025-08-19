import torch
import torch.optim as optim
import torch.nn.functional as F
from adversarial_base import CLIPAdversarialBase
import os
import sys


class CLIPAdversarialPerturbation(CLIPAdversarialBase):
    def __init__(self, model_path, device=None, num_steps=300, lr=1e-3):
        """
        全图对抗扰动训练类：仅实现train_perturbation方法
        继承自CLIPAdversarialBase，复用所有公共功能
        """
        super().__init__(model_path, device, num_steps, lr)

    def train_perturbation(
        self,
        background_image_path,
        target_text,
        save_names,
        epsilon=0.15,
        save_suffix=".png",
    ):
        """
        核心功能：训练全图对抗扰动（使背景图被CLIP误分类为目标文本）
        :param background_image_path: 背景图像路径（单张）
        :param target_text: 目标文本（如"a cat"）
        :param save_names: 保存名称列表（长度1）
        :param epsilon: 扰动幅度限制
        :param save_suffix: 保存文件后缀
        :return: 保存的图像路径列表
        """
        # 加载背景图像与目标文本嵌入
        img_tensor, _ = self.load_and_preprocess_image(background_image_path)
        target_emb = self.get_target_text_embedding(target_text)

        # 初始化扰动（小噪声）并限制幅度
        perturbation = torch.randn_like(img_tensor) * 0.01
        perturbation = perturbation.clamp(-epsilon, epsilon).to(self.device)
        perturbation.requires_grad_(True)

        # 优化器配置
        optimizer = optim.Adam([perturbation], lr=self.lr)

        # 训练循环
        for step in range(self.num_steps):
            optimizer.zero_grad()
            # 生成对抗图像
            adv_img = img_tensor + perturbation
            # 计算图像嵌入与损失（最小化余弦相似度→1 - 相似度）
            img_emb = self.model.get_image_features(pixel_values=adv_img)
            img_emb = F.normalize(img_emb, dim=-1)
            loss = 1 - F.cosine_similarity(img_emb, target_emb).mean()

            # 反向传播与优化
            loss.backward()
            optimizer.step()
            # 再次限制扰动幅度（防止超界）
            perturbation.data = torch.clamp(perturbation.data, -epsilon, epsilon)

            # 日志与中间保存（每100步）
            if step % 100 == 0:
                self.save_adversarial_image(
                    img_tensors=[adv_img.detach()],
                    base_names=save_names,
                    save_suffix=save_suffix,
                )
            if step % 20 == 0 or step == self.num_steps - 1:
                print(f"[全图扰动 Step {step}] Loss: {loss.item():.6f}")

        # 最终保存结果
        return self.save_adversarial_image(
            img_tensors=[adv_img.detach()],
            base_names=save_names,
            save_suffix=save_suffix,
        )
