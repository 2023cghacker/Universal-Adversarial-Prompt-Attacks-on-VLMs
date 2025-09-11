import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --------------------------
# 配置
# --------------------------
MODEL_DIR = "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct"
IMAGE_PATH = "/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/apple.png"
DEVICE = "cuda"
EPS = 8/255.0
LR = 1e-3
STEPS = 50

# --------------------------
# 1. 加载模型和 processor
# --------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR)

# --------------------------
# 2. 加载原始图像并生成 pixel_values
# --------------------------
pil_img = Image.open(IMAGE_PATH).convert("RGB")
print(f"\n> 原始PIL图像尺寸 (宽, 高): {pil_img.size}")
inputs = processor(text=["Describe this image"],images=pil_img, return_tensors="pt").to(DEVICE)
target_pixel_values = inputs["pixel_values"]  # (1,C,H,W)
image_grid_thw = inputs["image_grid_thw"]  # shape: (1,3)
print(f"> target_pixel_values shape: {target_pixel_values.shape}",
      f"> image_grid_thw: {image_grid_thw}")

# 获取目标潜在表示
with torch.no_grad():
    target_latent = model.get_image_features(target_pixel_values, image_grid_thw)  # (1, N_patch, D)
    print(f"> target_latent shape: {target_latent[0].shape}")

# --------------------------
# 3. 初始化 adversarial image
# --------------------------
adv_img = target_pixel_values.clone().detach()  # 先克隆原始像素值（不立即开启梯度）
random_noise = torch.normal(mean=0.0, std=EPS/3, size=adv_img.shape, device=DEVICE)
adv_img = torch.clamp(adv_img + random_noise, target_pixel_values - EPS, target_pixel_values + EPS)
adv_img = adv_img.requires_grad_(True)

print(f"> adv_img shape: {adv_img.shape}")
optimizer = torch.optim.Adam([adv_img], lr=LR)
loss_fct = torch.nn.MSELoss()

# --------------------------
# 4. 在视觉潜在空间中对齐
# --------------------------
for step in range(STEPS):
    optimizer.zero_grad()
    
    # 可限制扰动范围
    adv_img_clamped = torch.clamp(adv_img, target_pixel_values-EPS, target_pixel_values+EPS)
    
    # 获取 adversarial latent
    adv_latent = model.get_image_features(adv_img_clamped,image_grid_thw)
    print(f"> adv_latent shape: {adv_latent[0].shape}")
    
    # latent space 对齐
    loss = loss_fct(adv_latent[0], target_latent[0])
    loss.backward()
    optimizer.step()
    
    if step % 1 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")

# --------------------------
# 5. 保存对抗样本
# --------------------------
# 将 adv_img 转回 [0,1] 范围并保存
mean = processor.image_processor.image_mean
std = processor.image_processor.image_std

# 反归一化
adv_img_denorm = adv_img_clamped.squeeze(0).detach().cpu()
for c in range(3):
    adv_img_denorm[c] = adv_img_denorm[c] * std[c] + mean[c]

adv_img_denorm = torch.clamp(adv_img_denorm, 0, 1)
adv_pil = T.ToPILImage()(adv_img_denorm)
adv_pil.save("adv_latent_aligned.png")
print("Saved adversarial latent-aligned image as adv_latent_aligned.png")
