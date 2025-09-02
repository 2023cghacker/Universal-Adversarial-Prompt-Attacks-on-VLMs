import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

def verify_image_tensor_consistency(
    input_image_path: str,
    output_dir: str = "output",
    save_suffix: str = ".png"
) -> None:
    """
    验证图像在「加载→Tensor→保存→重新加载」流程中的数值一致性
    """
    # 初始化输出目录
    os.makedirs(output_dir, exist_ok=True)
    saved_image_path = os.path.join(output_dir, f"saved_image{save_suffix}")
    print(f"=== 图像张量一致性验证开始 ===")
    print(f"输入图像路径: {input_image_path}")
    print(f"保存图像路径: {saved_image_path}\n")

    # 定义标准化/反标准化流程
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - mean) / std

    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        denorm_tensor = tensor * std + mean
        return torch.clamp(denorm_tensor, 0.0, 1.0)

    # 步骤1：加载原始图像
    print(f"--- 步骤1：加载原始图像 ---")
    try:
        raw_image = Image.open(input_image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"输入图像不存在：{input_image_path}")
    
    print(f"原始图像尺寸: {raw_image.size} (宽x高)")
    raw_pixels = np.array(raw_image)[:2, :2, :]
    print(f"原始图像左上角2x2像素值:\n{raw_pixels}\n")

    # 步骤2：PIL图像 → 未标准化的Tensor
    print(f"--- 步骤2：PIL图像 → 未标准化Tensor (0~1) ---")
    to_tensor = transforms.ToTensor()
    tensor_01 = to_tensor(raw_image).unsqueeze(0)  # 形状: [1, 3, H, W]
    print(f"未标准化Tensor形状: {tensor_01.shape}")
    print(f"未标准化Tensor范围: [{tensor_01.min().item():.6f}, {tensor_01.max().item():.6f}]")
    
    tensor_01_pixels = tensor_01[0, :, :2, :2].permute(1, 2, 0).cpu().numpy()
    print(f"未标准化Tensor左上角2x2像素值:\n{np.round(tensor_01_pixels, 8)}\n")

    # 步骤3：标准化Tensor
    print(f"--- 步骤3：标准化Tensor ---")
    tensor_normalized = normalize(tensor_01)
    print(f"标准化Tensor范围: [{tensor_normalized.min().item():.6f}, {tensor_normalized.max().item():.6f}]\n")

    # 步骤4：反标准化
    print(f"--- 步骤4：反标准化 ---")
    tensor_denormed = denormalize(tensor_normalized)
    print(f"反标准化后Tensor范围: [{tensor_denormed.min().item():.6f}, {tensor_denormed.max().item():.6f}]")
    
    mse_denorm = torch.nn.functional.mse_loss(tensor_denormed, tensor_01).item()
    print(f"反标准化与原始Tensor的MSE: {mse_denorm:.8f}\n")

    # 步骤5：Tensor → PIL图像并保存（修复核心部分）
    print(f"--- 步骤5：Tensor → PIL图像并保存 ---")
    # 确保处理的是张量而不是标量
    if tensor_denormed.dim() == 4:  # [1, 3, H, W]
        # 正确的维度转换流程
        tensor_squeezed = tensor_denormed.squeeze(0)  # 移除批次维度 → [3, H, W]
        tensor_transposed = tensor_squeezed.permute(1, 2, 0)  # 转置为 [H, W, 3]
        tensor_np = tensor_transposed.cpu().numpy()  # 转为numpy数组
        
        # 缩放并转换为uint8
        tensor_np = (tensor_np * 255.0).clip(0, 255).astype(np.uint8)
        
        # 保存图像
        saved_image = Image.fromarray(tensor_np)
        saved_image.save(saved_image_path, format=save_suffix.strip('.'), compress_level=0)
        print(f"图像已保存: {saved_image_path}")
    else:
        raise ValueError(f"无效的张量维度: {tensor_denormed.dim()}，预期为4维张量 [1, 3, H, W]")

    # 步骤6：重新加载验证
    print(f"--- 步骤6：重新加载验证 ---")
    loaded_image = Image.open(saved_image_path).convert("RGB")
    loaded_tensor_01 = to_tensor(loaded_image).unsqueeze(0)
    print(f"Tensor形状: {loaded_tensor_01.shape}")
    print(f"Tensor范围: [{loaded_tensor_01.min().item():.6f}, {loaded_tensor_01.max().item():.6f}]")
    tensor_01_pixels = loaded_tensor_01[0, :, :2, :2].permute(1, 2, 0).cpu().numpy()
    print(f"Tensor左上角2x2像素值:\n{np.round(tensor_01_pixels, 8)}\n")
    
    mse_final = torch.nn.functional.mse_loss(loaded_tensor_01, tensor_01).item()
    print(f"\n=== 验证结果 ===")
    print(f"反标准化MSE: {mse_denorm:.8f}")
    print(f"加载后MSE: {mse_final:.8f}")
    
    if mse_final < 1e-6:
        print("✅ 验证通过：转换过程无显著失真")
    else:
        print("❌ 验证失败：存在明显数值差异")


if __name__ == "__main__":
    # 替换为你的测试图像路径
    TEST_IMAGE_PATH = "/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/environment3.png"  # 确保该图像存在
    verify_image_tensor_consistency(
        input_image_path=TEST_IMAGE_PATH,
        save_suffix=".png"  # 必须使用无损格式
    )
