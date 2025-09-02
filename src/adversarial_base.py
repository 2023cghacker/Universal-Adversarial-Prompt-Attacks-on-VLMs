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
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954,0.26130258, 0.27577711]).view(1, 3, 1, 1)

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
            save_suffix=".png",  # 强制默认用PNG（无损格式）
        ):
        """
        公共方法：批量保存对抗图像（支持单patch+多背景图）- 无损版本
        核心：无压缩PNG保存 + 反归一化逻辑与预处理完全对齐
        """
        # 输入合法性校验
        if len(img_tensors) != len(base_names):
            raise ValueError("图像张量列表与基础名称列表长度必须一致")
        if positions is not None and len(img_tensors) != len(positions):
            raise ValueError("图像张量列表与位置列表长度必须一致")
        # 强制校验格式：仅允许PNG（其他格式如JPG为有损，不支持无损保存）
        if save_suffix.lower() != ".png":
            raise ValueError("无损保存仅支持PNG格式，请将save_suffix设为'.png'")

        saved_paths = []

        # ---------------------- 1. 无损保存补丁（仅1次）----------------------
        if patch is not None:
            # 反归一化：恢复为0~255范围（与图像预处理反向）
            patch_denorm = self._denormalize(patch.clone())
            # 张量→numpy：注意数据类型转换（uint8是图像标准格式，避免失真）
            patch_np = (
                patch_denorm.squeeze()  # 移除批次维度 (1,3,H,W) → (3,H,W)
                .cpu()  # 转移到CPU（numpy不支持CUDA张量）
                .permute(1, 2, 0)  # 维度转置 (3,H,W) → (H,W,3)（符合PIL图像格式）
                .numpy()
                .astype(np.uint8)  # 转为8位无符号整数（图像像素标准）
            )
            # 无压缩保存PNG（compress_level=0表示完全无损，无任何压缩）
            patch_path = os.path.join(
                self.output_dir, f"patch_{self.timestamp}{save_suffix}"
            )
            Image.fromarray(patch_np).save(patch_path, format="PNG", compress_level=0)
            # print(f"✅ 无损保存补丁: {patch_path}")

        # ---------------------- 2. 批量无损保存对抗图像 ----------------------
        for i, (img_tensor, base_name) in enumerate(zip(img_tensors, base_names)):
            final_img = img_tensor.clone()  # 复制张量，避免修改原数据

            # 应用补丁（若提供）：与训练时叠加逻辑完全一致
            if patch is not None and positions is not None:
                x, y = positions[i]
                # 确保补丁与图像张量设备一致
                final_img[:, :, y:y+patch_size, x:x+patch_size] = patch.to(final_img.device)

            # 打印【训练时的对抗张量】信息（关键对比基准）
            # print(f"\n===== 保存前 - 对抗张量 {i} 信息 =====")
            # print(f"训练时张量形状: {final_img.shape}")
            # print(f"训练时张量设备: {final_img.device}")
            # print(f"训练时张量范围: [{final_img.min().item():.6f}, {final_img.max().item():.6f}]")
            # print(f"训练时张量前3个值: {final_img.flatten()[:3].cpu().numpy()}")  # 打印前3个元素
        

            # 关键步骤：反归一化 + 张量→图像转换（无失真）
            final_denorm = self._denormalize(final_img.clone())  # 反归一化到0~255
            
            final_np = (
                final_denorm.squeeze()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                .astype(np.uint8)  # 必须用uint8，否则PIL保存时会自动截断导致失真
            )

            # 无压缩保存PNG
            save_path = os.path.join(
                self.output_dir, f"{base_name}_{self.timestamp}{save_suffix}"
            )
            Image.fromarray(final_np).save(save_path, format="PNG", compress_level=0)
            saved_paths.append(save_path)

        # print(f"✅ 已无损保存 {len(saved_paths)} 张对抗图像（无压缩PNG）")
        return saved_paths
    # def save_adversarial_image(
    #     self,
    #     img_tensors,
    #     base_names,
    #     patch=None,
    #     patch_size=80,
    #     positions=None,
    #     save_suffix=".png",
    # ):
    #     """公共方法：批量保存对抗图像（支持单patch+多背景图）"""
    #     # 输入合法性校验
    #     if len(img_tensors) != len(base_names):
    #         raise ValueError("图像张量列表与基础名称列表长度必须一致")
    #     if positions is not None and len(img_tensors) != len(positions):
    #         raise ValueError("图像张量列表与位置列表长度必须一致")

    #     saved_paths = []

    #     # 保存补丁（仅保存1次）
    #     if patch is not None:
    #         patch_denorm = self._denormalize(patch.clone())
    #         patch_np = (
    #             patch_denorm.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    #         )
    #         patch_path = os.path.join(
    #             self.output_dir, f"patch_{self.timestamp}{save_suffix}"
    #         )
    #         Image.fromarray(patch_np).save(patch_path)

    #     # 批量保存背景图像（含补丁应用）
    #     for i, (img_tensor, base_name) in enumerate(zip(img_tensors, base_names)):
    #         final_img = img_tensor.clone()
    #         # 应用补丁（若提供）
    #         if patch is not None and positions is not None:
    #             x, y = positions[i]
    #             final_img[:, :, y : y + patch_size, x : x + patch_size] = patch

    #         # 转换为PIL图像并保存
    #         final_denorm = self._denormalize(final_img.clone())
    #         final_np = (
    #             final_denorm.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    #         )
    #         save_path = os.path.join(
    #             self.output_dir, f"{base_name}_{self.timestamp}{save_suffix}"
    #         )
    #         Image.fromarray(final_np).save(save_path)
    #         saved_paths.append(save_path)

    #     # print(f"✅ 已保存 {len(saved_paths)} 张对抗图像")
    #     return saved_paths

    def test(self, img_path, target_text=None, target_img=None):
        """公共方法：测试图像与目标（文本/图像）的余弦相似度"""

        # 计算图像嵌入与目标嵌入
        img_tensor, _ = self.load_and_preprocess_image(img_path)
        # 打印【加载后】的张量信息（与训练时对比）
        # print(f"\n===== 加载后 - 图像张量信息 =====")
        # print(f"加载后张量形状: {img_tensor.shape}")
        # print(f"加载后张量范围: [{img_tensor.min().item():.6f}, {img_tensor.max().item():.6f}]")
        # print(f"加载后张量前3个值: {img_tensor.flatten()[:3].cpu().numpy()}")  # 打印前3个元素
        
        if target_text is not None:
            target_emb = self.get_target_text_embedding(target_text)
            target_info = f"'{target_text}'"
        else:
            target_emb = self.get_target_image_embedding(target_img)
            target_info = f"'{target_img}'"

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
