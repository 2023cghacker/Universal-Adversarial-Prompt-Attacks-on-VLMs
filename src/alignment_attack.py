import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import os


def load_clip_model(model_path, device):
    """加载本地CLIP模型和处理器"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"本地模型路径不存在: {model_path}")

    model = CLIPModel.from_pretrained(model_path).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor


def load_and_preprocess_image(image_path, processor, device):
    """加载图像并进行预处理"""
    raw_img = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=raw_img, return_tensors="pt")["pixel_values"].to(device)
    return image_inputs.clone().detach(), raw_img


def initialize_perturbation(img_tensor, device, init_scale=0.01, epsilon=0.15):
    """初始化全图对抗扰动"""
    perturbation = (torch.randn_like(img_tensor) * init_scale).clamp(-epsilon, epsilon).to(device)
    perturbation.requires_grad_()
    return perturbation


def get_target_text_embedding(model, processor, target_text, device):
    """获取目标文本的特征嵌入"""
    with torch.no_grad():
        text_inputs = processor(text=[target_text], return_tensors="pt", padding=True).to(device)
        target_embedding = model.get_text_features(**text_inputs)
        return F.normalize(target_embedding, dim=-1)


def train_adversarial_perturbation(model, img_tensor, perturbation, target_embedding,
                                   device, num_steps=300, lr=1e-3, epsilon=0.15):
    """训练全图对抗扰动"""
    optimizer = torch.optim.Adam([perturbation], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        adv_img = torch.clamp(img_tensor + perturbation, 0, 1)
        image_embedding = model.get_image_features(pixel_values=adv_img)
        image_embedding = F.normalize(image_embedding, dim=-1)
        loss = 1 - F.cosine_similarity(image_embedding, target_embedding).mean()
        loss.backward()
        optimizer.step()
        perturbation.data = torch.clamp(perturbation.data, -epsilon, epsilon)

        if step % 20 == 0 or step == num_steps - 1:
            print(f"[perturbation Step {step}] Loss: {loss.item():.6f}")

    return perturbation


def train_adversarial_patch(model, img_tensor, target_embedding, patch_size=80,
                            position='center', device='cuda', num_steps=300, lr=1e-2):
    """训练局部对抗性patch（已修复non-leaf Tensor错误）"""
    B, C, H, W = img_tensor.shape

    # 修复：确保patch是叶子张量（可优化）
    patch = torch.randn((1, C, patch_size, patch_size), device=device)  # 初始随机张量（叶子张量）
    patch = patch * 0.01  # 缩放初始化（仍为叶子张量）
    patch.requires_grad_(True)  # 手动开启梯度

    optimizer = torch.optim.Adam([patch], lr=lr)  # 现在可正常优化

    def apply_patch(img, patch, position='center'):
        """将patch贴到图像指定位置"""
        patched = img.clone()
        if position == 'center':
            x = (W - patch_size) // 2
            y = (H - patch_size) // 2
        elif position == 'top_left':
            x, y = 0, 0
        elif position == 'top_right':
            x, y = W - patch_size, 0
        elif position == 'bottom_left':
            x, y = 0, H - patch_size
        elif position == 'bottom_right':
            x, y = W - patch_size, H - patch_size
        else:
            raise ValueError("不支持的patch位置，可选：center/top_left/top_right/bottom_left/bottom_right")

        patched[:, :, y:y + patch_size, x:x + patch_size] = patch
        return torch.clamp(patched, 0, 1)

    # 训练patch
    for step in range(num_steps):
        optimizer.zero_grad()
        adv_img = apply_patch(img_tensor, patch, position)
        image_embedding = model.get_image_features(pixel_values=adv_img)
        image_embedding = F.normalize(image_embedding, dim=-1)
        loss = 1 - F.cosine_similarity(image_embedding, target_embedding).mean()
        loss.backward()
        optimizer.step()
        patch.data = torch.clamp(patch.data, 0, 1)  # 保持patch像素在有效范围

        if step % 20 == 0 or step == num_steps - 1:
            print(f"[patch Step {step}] Loss: {loss.item():.6f}")

    return patch.detach()


def save_adversarial_image(img_tensor, perturbation=None, patch=None, patch_size=80,
                           save_path="../output/adv_output.jpg", position='center'):
    """保存对抗性图像（支持全图扰动或局部patch）"""
    final_img = img_tensor.clone()

    # 应用全图扰动
    if perturbation is not None:
        final_img = torch.clamp(final_img + perturbation, 0, 1)

    # 应用局部patch
    if patch is not None:
        B, C, H, W = final_img.shape
        if position == 'center':
            x = (W - patch_size) // 2
            y = (H - patch_size) // 2
        elif position == 'top_left':
            x, y = 0, 0
        elif position == 'top_right':
            x, y = W - patch_size, 0
        elif position == 'bottom_left':
            x, y = 0, H - patch_size
        elif position == 'bottom_right':
            x, y = W - patch_size, H - patch_size
        else:
            raise ValueError("不支持的patch位置")

        final_img[:, :, y:y + patch_size, x:x + patch_size] = patch
        final_img = torch.clamp(final_img, 0, 1)

    # 保存图像
    final_img = final_img.squeeze().cpu()
    adv_pil = transforms.ToPILImage()(final_img)
    adv_pil.save(save_path)
    print(f"✅ 对抗图像保存完毕: {save_path}")


def main():
    # 参数配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model_path = r"/root/lc/clip-vit-large-patch14-336"  # 本地CLIP模型路径
    image_path = "/root/lc/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/pig.png"  # 输入图像路径
    target_text = "an apple"  # 目标文本（希望图像被误分类为该文本）

    # 加载模型和图像
    model, processor = load_clip_model(local_model_path, device)
    img_tensor, raw_img = load_and_preprocess_image(image_path, processor, device)
    target_embedding = get_target_text_embedding(model, processor, target_text, device)

    # 选择训练模式（二选一）
    train_mode = "patch"  # 可选："perturbation"（全图扰动）或 "patch"（局部patch）
    print(f"🔧 训练模式: {train_mode}")
    if train_mode == "perturbation":
        # 1. 训练全图扰动
        perturbation = initialize_perturbation(img_tensor, device)
        perturbation = train_adversarial_perturbation(
            model, img_tensor, perturbation, target_embedding, device
        )
        save_adversarial_image(
            img_tensor, perturbation=perturbation, save_path="../output/adv_perturb.jpg"
        )
    elif train_mode == "patch":
        # 2. 训练局部对抗性patch
        patch = train_adversarial_patch(
            model, img_tensor, target_embedding,
            patch_size=80,  # patch尺寸
            position='center',  # patch位置（center/top_left等）
            device=device,
            num_steps=300  # 训练步数
        )
        save_adversarial_image(
            img_tensor, patch=patch, patch_size=80,
            save_path="../output/adv_patch.jpg", position='center'
        )
    else:
        print("❌ 无效的训练模式，请选择 'perturbation' 或 'patch'")


if __name__ == "__main__":
    main()