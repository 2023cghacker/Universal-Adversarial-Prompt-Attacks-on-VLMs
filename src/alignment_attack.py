import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import os
import sys
import shutil
from torchvision import transforms
import numpy as np
from datetime import datetime  # 用于时间戳记录
from tqdm import tqdm


class AdversarialTrainer:
    def __init__(self, model_path, device=None, num_steps=300, lr=1e-3):
        """
        初始化对抗性训练器
        :param model_path: CLIP模型本地路径
        :param device: 计算设备，默认自动选择cuda/cpu
        :param num_steps: 默认训练步数
        :param lr: 默认学习率
        """
        # 确定计算设备
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_steps = num_steps
        self.lr = lr

        # 记录初始化时间戳，所有保存文件都使用这个时间戳
        self.timestamp = datetime.now().strftime("%m%d_%H%M%S")  # 增加秒级精度
        self.output_dir = os.path.join("output", self.timestamp)  # 按时间戳创建输出目录

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出文件将保存到: {self.output_dir}")

        # 加载模型和处理器
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"本地模型路径不存在: {model_path}")
        self.model = CLIPModel.from_pretrained(model_path).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_path)

        # 图像标准化参数（CLIP默认使用ImageNet的均值和标准差）
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def load_and_preprocess_image(self, image_path):
        """加载并预处理图像"""
        raw_img = Image.open(image_path).convert("RGB")
        image_inputs = self.processor(images=raw_img, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        return image_inputs.clone().detach(), raw_img

    def get_target_text_embedding(self, target_text):
        """获取目标文本的特征嵌入"""
        with torch.no_grad():
            text_inputs = self.processor(
                text=[target_text], return_tensors="pt", padding=True
            ).to(self.device)
            target_embedding = self.model.get_text_features(**text_inputs)
            return F.normalize(target_embedding, dim=-1)

    def get_target_image_embedding(self, target_img_path):
        """获取目标图像的特征嵌入"""
        with torch.no_grad():
            img_tensor, _ = self.load_and_preprocess_image(target_img_path)
            target_embedding = self.model.get_image_features(pixel_values=img_tensor)
            return F.normalize(target_embedding, dim=-1)

    def _denormalize(self, tensor):
        """反标准化张量为0-255图像格式（内部方法）"""
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        tensor = tensor * std + mean
        return torch.clamp(tensor * 255.0, 0, 255).byte()

    def save_adversarial_image(
        self,
        img_tensors,  # 改为接受张量列表
        base_names,  # 基础名称列表，如["adv1", "adv2"]
        patch=None,
        patch_size=80,
        positions=None,  # 位置列表，与图像列表对应
        save_suffix=".png",
    ):
        """
        批量保存对抗性图像，一次性处理多张图像
        patch只保存一次，环境图像根据列表保存多张
        """
        # 确保输入列表长度匹配
        if len(img_tensors) != len(base_names):
            raise ValueError("图像张量列表和基础名称列表长度必须相同")

        if positions is not None and len(img_tensors) != len(positions):
            raise ValueError("图像张量列表和位置列表长度必须相同")

        saved_paths = []

        # 先保存patch（只保存一次）
        if patch is not None:
            patch_denorm = self._denormalize(patch.clone())
            patch_np = (
                patch_denorm.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            patch_filename = f"patch_{self.timestamp}{save_suffix}"
            patch_path = os.path.join(self.output_dir, patch_filename)
            Image.fromarray(patch_np).save(patch_path)
            print(f"📌 已保存Patch图像: {patch_path}")

        # 批量保存环境图像
        for i, (img_tensor, base_name) in enumerate(zip(img_tensors, base_names)):
            final_img = img_tensor.clone()

            # 应用patch（如果提供）
            if patch is not None and positions is not None:
                x, y = positions[i]
                final_img[:, :, y : y + patch_size, x : x + patch_size] = patch

            # 处理图像保存
            final_denorm = self._denormalize(final_img.clone())
            final_np = (
                final_denorm.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )

            # 构建文件名：基础名称 + 时间戳 + 后缀
            filename = f"{base_name}_{self.timestamp}{save_suffix}"
            save_path = os.path.join(self.output_dir, filename)
            Image.fromarray(final_np).save(save_path)
            print(f"📌 已保存图像: {save_path}")
            saved_paths.append(save_path)

        return saved_paths

    def train_perturbation(
        self,
        background_image_path,
        target_text,
        save_names,  # 改为保存名称
        epsilon=0.15,
        save_suffix=".png",
    ):
        """训练全图对抗扰动"""

        img_tensor, _ = self.load_and_preprocess_image(background_image_path)
        target_embedding = self.get_target_text_embedding(target_text)

        # 初始化扰动
        perturbation = torch.randn_like(img_tensor) * 0.01
        perturbation = perturbation.clamp(-epsilon, epsilon).to(self.device)
        perturbation.requires_grad_(True)

        optimizer = torch.optim.Adam([perturbation], lr=self.lr)

        for step in range(self.num_steps):
            optimizer.zero_grad()
            adv_img = img_tensor + perturbation
            image_embedding = self.model.get_image_features(pixel_values=adv_img)
            image_embedding = F.normalize(image_embedding, dim=-1)

            loss = 1 - F.cosine_similarity(image_embedding, target_embedding).mean()
            loss.backward()
            optimizer.step()

            # 限制扰动幅度
            perturbation.data = torch.clamp(perturbation.data, -epsilon, epsilon)

            # 每隔100轮保存一次（覆盖式）
            if step % 100 == 0:
                self.save_adversarial_image(
                    img_tensors=[adv_img.detach()],  # 传入列表
                    base_names=save_names,
                    save_suffix=save_suffix,
                )

            if step % 20 == 0 or step == self.num_steps - 1:
                print(f"[扰动训练 Step {step}] Loss: {loss.item():.6f}")
                sys.stdout.flush()

        # 最终保存
        return self.save_adversarial_image(
            img_tensors=[adv_img.detach()],  # 传入列表
            base_names=save_names,
            save_suffix=save_suffix,
        )

    def train_patch(
        self,
        background_image_paths,  # 背景图像路径列表
        target_text=None,
        target_img=None,
        patch_size=80,
        positions=[[30, 30]],  # 位置列表
        background_weight=0.1,
        initial_patch_path=None,
        save_names=None,  # 保存名称列表，如["adv1", "adv2"]
    ):
        """训练局部对抗性补丁，支持多背景图像和多位置"""
        # 检查target_text和target_img互斥
        if not ((target_text is None) ^ (target_img is None)):
            raise ValueError("target_text和target_img必须且只能有一个为非None")

        # 检查背景图像路径和位置列表长度是否匹配
        if len(background_image_paths) != len(positions):
            raise ValueError("背景图像路径列表和位置列表长度必须相同")

        # 检查保存名称是否提供且长度匹配
        if save_names is None or len(save_names) != len(background_image_paths):
            raise ValueError("保存名称列表必须提供且与背景图像数量相同")

        # 加载所有背景图像
        img_tensors = []
        for img_path in background_image_paths:
            img_tensor, _ = self.load_and_preprocess_image(img_path)
            img_tensors.append(img_tensor)

        # 检查所有位置的有效性
        for i, (img_tensor, pos) in enumerate(zip(img_tensors, positions)):
            x, y = pos
            B, C, H, W = img_tensor.shape
            if x < 0 or x + patch_size > W or y < 0 or y + patch_size > H:
                raise ValueError(f"第{i}个patch位置超出图像边界: {W}x{H}")

        # 获取目标嵌入（文本或图像）
        if target_text is not None:
            target_embedding = self.get_target_text_embedding(target_text)
        else:  # target_img is not None
            target_embedding = self.get_target_image_embedding(target_img)

        # 初始化补丁（根据初始图像或背景，使用第一个图像的背景初始化）
        if initial_patch_path is not None:
            transform = transforms.Compose(
                [
                    transforms.Resize((patch_size, patch_size)),
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: x.unsqueeze(0)
                    ),  # 增加批次维度，变为[1, 3, patch_size, patch_size]
                ]
            )
            patch_img = Image.open(initial_patch_path).convert("RGB")
            patch_tensor = transform(patch_img)
            patch = patch_tensor * 0.9 + torch.randn_like(patch_tensor) * 0.1
            patch = patch.to(self.device)  # 转移到设备

        else:
            # 使用第一个图像的背景初始化
            x, y = positions[0]
            background_patch = img_tensors[0][
                :, :, y : y + patch_size, x : x + patch_size
            ].clone()
            patch = background_patch * 0.8 + torch.randn_like(background_patch) * 0.2

        # 保存原始背景（用于内容损失计算）
        original_patches = []
        for img_tensor, pos in zip(img_tensors, positions):
            x, y = pos
            original_patch = img_tensor[
                :, :, y : y + patch_size, x : x + patch_size
            ].detach()  # 剥离计算图
            original_patches.append(original_patch)

        # 如果使用初始补丁，也保存初始补丁
        if initial_patch_path is not None:
            initial_original_patch = patch_tensor.detach().to(self.device)
            original_patches.append(initial_original_patch)

        # 准备优化
        patch.requires_grad_(True)
        optimizer = torch.optim.Adam([patch], lr=self.lr)

        for step in range(1, self.num_steps + 1):
            optimizer.zero_grad()

            total_adversarial_loss = 0.0
            total_background_loss = 0.0

            # 对每个背景图像和位置计算损失
            for img_tensor, pos, original_patch in zip(
                img_tensors, positions, original_patches[: len(img_tensors)]
            ):
                x, y = pos

                # 应用补丁
                adv_img = img_tensor.clone()
                adv_img[:, :, y : y + patch_size, x : x + patch_size] = patch

                # 计算损失
                image_embedding = self.model.get_image_features(pixel_values=adv_img)
                image_embedding = F.normalize(image_embedding, dim=-1)

                # 累加对抗损失
                adversarial_loss = (
                    1 - F.cosine_similarity(image_embedding, target_embedding).mean()
                )
                total_adversarial_loss += adversarial_loss

                # 累加背景损失
                background_loss = F.mse_loss(patch, original_patch)
                total_background_loss += background_loss

            # 计算平均损失（除以图像数量）
            num_images = len(img_tensors)
            avg_adversarial_loss = total_adversarial_loss / num_images
            avg_background_loss = total_background_loss / num_images
            total_loss = avg_adversarial_loss + background_weight * avg_background_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 每隔100轮保存一次并输出详细信息
            if step % 100 == 0:
                # 一次性保存所有图像和一个patch
                self.save_adversarial_image(
                    img_tensors=img_tensors,
                    base_names=save_names,
                    patch=patch.detach(),
                    patch_size=patch_size,
                    positions=positions,
                    save_suffix=".png",
                )
                # 输出详细信息
                print(
                    f"\n[补丁训练 Step {step}] 对抗损失: {avg_adversarial_loss.item():.6f}, "
                    f"背景损失: {avg_background_loss.item():.6f}, "
                    f"总损失: {total_loss.item():.6f}"
                )
                sys.stdout.flush()

            # 第一步和最后一步也输出详细信息（除了已经被100整除的情况）
            if (step == 1 or step == self.num_steps) and step % 100 != 0:
                print(
                    f"\n[补丁训练 Step {step}] 对抗损失: {avg_adversarial_loss.item():.6f}, "
                    f"背景损失: {avg_background_loss.item():.6f}, "
                    f"总损失: {total_loss.item():.6f}"
                )
                sys.stdout.flush()

        # 最终保存所有图像
        return self.save_adversarial_image(
            img_tensors=img_tensors,
            base_names=save_names,
            patch=patch.detach(),
            patch_size=patch_size,
            positions=positions,
            save_suffix=".png",
        )

    def test(self, img_path, target_text=None, target_img=None):
        """
        计算图像与目标（文本或图像）之间的余弦相似度
        """
        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        # 检查target_text和target_img互斥
        if not ((target_text is None) ^ (target_img is None)):
            raise ValueError("target_text和target_img必须且只能有一个为非None")

        # 加载并预处理图像
        img_tensor, _ = self.load_and_preprocess_image(img_path)

        # 获取目标嵌入（文本或图像）
        if target_text is not None:
            target_embedding = self.get_target_text_embedding(target_text)
            target_info = f"文本: '{target_text}'"
        else:  # target_img is not None
            target_embedding = self.get_target_image_embedding(target_img)
            target_info = f"图像: '{target_img}'"

        # 计算图像嵌入
        with torch.no_grad():
            image_embedding = self.model.get_image_features(pixel_values=img_tensor)
            image_embedding = F.normalize(image_embedding, dim=-1)

        # 计算余弦相似度
        similarity = F.cosine_similarity(image_embedding, target_embedding).mean()
        print(f"目标{target_info} 与当前图像的余弦相似度: {similarity.item():.6f}")
        return similarity.item()


if __name__ == "__main__":
    # 配置参数
    local_model_path = (
        "/HOME/paratera_xy/pxy480/HDD_POOL/Ling/downloads/clip-vit-large-patch14-336"
    )
    # 背景图像路径列表
    background_image_paths = [
        "images/pig.png",
        "images/another_background.png",
    ]
    # 保存名称列表（仅名称，不含路径和扩展名）
    save_names = ["adv_pig", "adv_another"]
    # 二选一：目标文本或目标图像
    target_text = None  # "an apple"
    target_img = "images/apple.png"  # 目标图像路径（与target_text互斥）
    train_mode = "patch"
    # 位置列表
    positions = [[30, 30], [50, 50]]

    # 初始化训练器
    trainer = AdversarialTrainer(model_path=local_model_path, num_steps=300, lr=1e-3)
    print(f"使用设备: {trainer.device}")
    print(f"保存文件时间戳: {trainer.timestamp}")

    # 训练并保存结果
    if train_mode == "perturbation":
        if target_text is None:
            raise ValueError("扰动训练需要target_text不为None")
        # 扰动训练只使用第一个图像
        trainer.train_perturbation(
            background_image_path=background_image_paths[0],
            target_text=target_text,
            save_names=save_names[:1],  # 只使用第一个名称
        )
    elif train_mode == "patch":
        trainer.train_patch(
            background_image_paths=background_image_paths,
            target_text=target_text,
            target_img=target_img,
            patch_size=80,
            positions=positions,
            background_weight=0.2,
            save_names=save_names,
        )
    else:
        print("❌ 无效的训练模式，请选择 'perturbation' 或 'patch'")
