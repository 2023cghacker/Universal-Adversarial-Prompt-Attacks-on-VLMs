import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import os
import sys
from torchvision import transforms
import numpy as np


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

    def train_perturbation(
        self, background_image_path, target_text, savepath, epsilon=0.15
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

            if step % 20 == 0 or step == self.num_steps - 1:
                print(f"[扰动训练 Step {step}] Loss: {loss.item():.6f}")
                sys.stdout.flush()

        self.save_adversarial_image(
            img_tensor, savepath, perturbation=perturbation.detach()
        )

    def train_patch(
        self,
        background_image_path,
        target_text=None,
        target_img=None,  # 新增：目标图像路径，与target_text互斥
        patch_size=80,
        position=[30, 30],
        background_weight=0.1,
        initial_patch_path=None,
        save_path=None,
    ):
        """
        训练局部对抗性补丁
        :param target_text: 目标文本描述（与target_img互斥）
        :param target_img: 目标图像路径（与target_text互斥）
        :param initial_patch_path: 初始patch图像的路径,为None则使用原图背景初始化
        """
        # 新增：检查target_text和target_img互斥
        if not ((target_text is None) ^ (target_img is None)):
            raise ValueError("target_text和target_img必须且只能有一个为非None")

        # 加载背景图像
        img_tensor, _ = self.load_and_preprocess_image(background_image_path)
        B, C, H, W = img_tensor.shape
        x, y = position

        # 检查位置有效性
        if x < 0 or x + patch_size > W or y < 0 or y + patch_size > H:
            raise ValueError("patch位置超出图像边界")

        # 新增：获取目标嵌入（文本或图像）
        if target_text is not None:
            target_embedding = self.get_target_text_embedding(target_text)
        else:  # target_img is not None
            target_embedding = self.get_target_image_embedding(target_img)

        # 初始化补丁（根据初始图像或背景）
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
            # 使用原图背景初始化
            background_patch = img_tensor[
                :, :, y : y + patch_size, x : x + patch_size
            ].clone()
            patch = background_patch * 0.8 + torch.randn_like(background_patch) * 0.2

        # 保存原始背景（用于损失计算）
        original_background = img_tensor[
            :, :, y : y + patch_size, x : x + patch_size
        ].detach()

        # 准备优化
        patch.requires_grad_(True)
        optimizer = torch.optim.Adam([patch], lr=self.lr)
        print(
            f"patch.shape={patch.shape},"
            f"original_background.shape={original_background.shape},"
            f"patch_position={position},"
            f"patch_size={patch_size}"
        )

        for step in range(1, self.num_steps + 1):
            optimizer.zero_grad()

            # 应用补丁
            adv_img = img_tensor.clone()
            adv_img[:, :, y : y + patch_size, x : x + patch_size] = patch

            # 计算损失（逻辑不变，目标嵌入来源已根据参数调整）
            image_embedding = self.model.get_image_features(pixel_values=adv_img)
            image_embedding = F.normalize(image_embedding, dim=-1)

            adversarial_loss = (
                1 - F.cosine_similarity(image_embedding, target_embedding).mean()
            )
            background_loss = F.mse_loss(patch, original_background)
            total_loss = adversarial_loss + background_weight * background_loss

            total_loss.backward()
            optimizer.step()

            if step % 20 == 0 or step == self.num_steps or step == 1:
                print(
                    f"[补丁训练 Step {step}] 对抗损失: {adversarial_loss.item():.6f}, "
                    f"背景损失: {background_loss.item():.6f}, "
                    f"总损失: {total_loss.item():.6f}"
                )
                sys.stdout.flush()

        self.save_adversarial_image(
            img_tensor,
            save_path,
            patch=patch.detach(),
            patch_size=patch_size,
            position=position,
        )

    def save_adversarial_image(
        self,
        img_tensor,
        save_path,
        perturbation=None,
        patch=None,
        patch_size=80,
        position=[30, 30],
    ):
        """保存对抗性图像、patch及其原始张量（默认保存所有内容）"""
        final_img = img_tensor.clone()

        # 应用扰动
        if perturbation is not None:
            final_img += perturbation

        # 应用并保存patch
        if patch is not None:
            x, y = position
            final_img[:, :, y : y + patch_size, x : x + patch_size] = patch

            # 构建patch保存路径
            patch_img_path = f"{save_path}_patch.png"
            patch_np_path = f"{save_path}_patch.npy"

            # 保存patch图像和张量
            patch_denorm = self._denormalize(patch.clone())
            # 修正图像数据格式（确保uint8和正确维度）
            patch_np = (
                patch_denorm.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            Image.fromarray(patch_np).save(patch_img_path)
            np.save(patch_np_path, patch.cpu().numpy())
            print(f"✅ 已保存Patch: {patch_img_path} 和 {patch_np_path}")

        # 保存对抗性图像和原始张量
        final_denorm = self._denormalize(final_img.clone())
        final_np = (
            final_denorm.squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        )
        Image.fromarray(final_np).save(f"{save_path}.png")
        np.save(f"{save_path}.npy", final_img.cpu().numpy())

        print(f"✅ 已保存对抗图像: {save_path}.png 和 {save_path}.npy")

    # def test(self, img_tensor_path, target_text, target_img):
    #     """计算图像张量与文本之间的余弦相似度"""
    #     if not os.path.exists(img_tensor_path):
    #         raise FileNotFoundError(f"图像张量文件不存在: {img_tensor_path}")

    #     img_tensor = np.load(img_tensor_path)
    #     img_tensor = torch.from_numpy(img_tensor).to(self.device)

    #     # 获取目标嵌入（文本或图像）
    #     if target_text is not None:
    #         target_embedding = self.get_target_text_embedding(target_text)
    #     else:  # target_img is not None
    #         target_embedding = self.get_target_image_embedding(target_img)

    #     with torch.no_grad():
    #         image_embedding = self.model.get_image_features(pixel_values=img_tensor)
    #         image_embedding = F.normalize(image_embedding, dim=-1)

    #     similarity = F.cosine_similarity(image_embedding, target_embedding).mean()
    #     print(f"文本: '{text}' 与图像的余弦相似度: {similarity.item():.6f}")
    #     return similarity.item()

    def test(self, img_path, target_text=None, target_img=None):
        """
        计算图像与目标（文本或图像）之间的余弦相似度
        :param img_path: 图像文件路径（如.png格式）
        :param target_text: 目标文本描述（与target_img互斥）
        :param target_img: 目标图像路径（与target_text互斥）
        :return: 余弦相似度值
        """
        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        # 检查target_text和target_img互斥
        if not ((target_text is None) ^ (target_img is None)):
            raise ValueError("target_text和target_img必须且只能有一个为非None")

        # 加载并预处理图像（从图像文件而非.npy）
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
    background_image_path = "images/pig.png"  # 背景图像路径
    save_path = "output/adv_patch"  # 保存路径（无需扩展名）
    # 二选一：目标文本或目标图像
    target_text = None  # "an apple"
    target_img = "images/apple.png"  # 目标图像路径（与target_text互斥）
    train_mode = "patch"

    # 初始化训练器
    trainer = AdversarialTrainer(model_path=local_model_path, num_steps=300, lr=1e-3)
    print(f"使用设备: {trainer.device}")

    # 训练并保存结果
    if train_mode == "perturbation":
        if target_text is None:
            raise ValueError("扰动训练需要target_text不为None")
        trainer.train_perturbation(
            background_image_path=background_image_path,
            target_text=target_text,
            savepath=save_path,
        )
    elif train_mode == "patch":
        trainer.train_patch(
            background_image_path=background_image_path,
            target_text=target_text,
            target_img=target_img,
            patch_size=80,
            position=[30, 30],
            background_weight=0.2,
            save_path=save_path,
        )
    else:
        print("❌ 无效的训练模式，请选择 'perturbation' 或 'patch'")
