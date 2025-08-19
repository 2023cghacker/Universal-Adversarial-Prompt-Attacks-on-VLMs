import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import os
import sys
from torchvision import transforms
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys


class CLIPAdversarialBase:
    def __init__(self, model_path, device=None, num_steps=300, lr=1e-3):
        """
        对抗训练基类：封装公共功能（模型加载、预处理、保存等）
        :param model_path: CLIP模型本地路径
        :param device: 计算设备（自动选择cuda/cpu）
        :param num_steps: 训练步数
        :param lr: 学习率
        """
        # 设备配置
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_steps = num_steps
        self.lr = lr

        # 输出目录与时间戳（所有子类共享）
        self.timestamp = datetime.now().strftime("%m%d_%H%M%S")
        self.output_dir = os.path.join("output", self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出文件保存路径: {self.output_dir}")

        # 加载CLIP模型与处理器
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        self.model = CLIPModel.from_pretrained(model_path).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_path)

        # CLIP默认图像标准化参数（ImageNet）
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def load_and_preprocess_image(self, image_path):
        """公共方法：加载并预处理单张图像（返回张量与原始图像）"""
        raw_img = Image.open(image_path).convert("RGB")
        image_inputs = self.processor(images=raw_img, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        return image_inputs.clone().detach(), raw_img

    def get_target_text_embedding(self, target_text):
        """公共方法：获取目标文本的归一化嵌入"""
        with torch.no_grad():
            text_inputs = self.processor(
                text=[target_text], return_tensors="pt", padding=True
            ).to(self.device)
            target_emb = self.model.get_text_features(**text_inputs)
            return F.normalize(target_emb, dim=-1)

    def get_target_image_embedding(self, target_img_path):
        """公共方法：获取目标图像的归一化嵌入"""
        with torch.no_grad():
            img_tensor, _ = self.load_and_preprocess_image(target_img_path)
            target_emb = self.model.get_image_features(pixel_values=img_tensor)
            return F.normalize(target_emb, dim=-1)

    def _denormalize(self, tensor):
        """内部公共方法：将标准化张量反转为0-255图像格式"""
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        tensor = tensor * std + mean
        return torch.clamp(tensor * 255.0, 0, 255).byte()

    def save_adversarial_image(
        self,
        img_tensors,
        base_names,
        patch=None,
        patch_size=80,
        positions=None,
        save_suffix=".png",
    ):
        """公共方法：批量保存对抗图像（支持单patch+多背景图）"""
        # 输入合法性校验
        if len(img_tensors) != len(base_names):
            raise ValueError("图像张量列表与基础名称列表长度必须一致")
        if positions is not None and len(img_tensors) != len(positions):
            raise ValueError("图像张量列表与位置列表长度必须一致")

        saved_paths = []

        # 保存补丁（仅保存1次）
        if patch is not None:
            patch_denorm = self._denormalize(patch.clone())
            patch_np = (
                patch_denorm.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            patch_path = os.path.join(
                self.output_dir, f"patch_{self.timestamp}{save_suffix}"
            )
            Image.fromarray(patch_np).save(patch_path)

        # 批量保存背景图像（含补丁应用）
        for i, (img_tensor, base_name) in enumerate(zip(img_tensors, base_names)):
            final_img = img_tensor.clone()
            # 应用补丁（若提供）
            if patch is not None and positions is not None:
                x, y = positions[i]
                final_img[:, :, y : y + patch_size, x : x + patch_size] = patch

            # 转换为PIL图像并保存
            final_denorm = self._denormalize(final_img.clone())
            final_np = (
                final_denorm.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            save_path = os.path.join(
                self.output_dir, f"{base_name}_{self.timestamp}{save_suffix}"
            )
            Image.fromarray(final_np).save(save_path)
            saved_paths.append(save_path)

        print(f"✅ 已保存 {len(saved_paths)} 张对抗图像")
        return saved_paths

    def test(self, img_path, target_text=None, target_img=None):
        """公共方法：测试图像与目标（文本/图像）的余弦相似度"""
        # 输入合法性校验
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"测试图像不存在: {img_path}")
        if not ((target_text is None) ^ (target_img is None)):
            raise ValueError("target_text与target_img必须二选一")

        # 计算图像嵌入与目标嵌入
        img_tensor, _ = self.load_and_preprocess_image(img_path)
        if target_text is not None:
            target_emb = self.get_target_text_embedding(target_text)
            target_info = f"文本: '{target_text}'"
        else:
            target_emb = self.get_target_image_embedding(target_img)
            target_info = f"图像: '{target_img}'"

        # 计算余弦相似度
        with torch.no_grad():
            img_emb = self.model.get_image_features(pixel_values=img_tensor)
            img_emb = F.normalize(img_emb, dim=-1)
            similarity = F.cosine_similarity(img_emb, target_emb).mean()

        print(f"📊 目标{target_info}与图像的余弦相似度: {similarity.item():.6f}")
        return similarity.item()

    @staticmethod
    def _format_timedelta(td):
        """内部静态方法：格式化时间（Xh Ym Zs）"""
        total_sec = int(td.total_seconds())
        hours, rem = divmod(total_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours}h {minutes}m {seconds}s"
