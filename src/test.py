import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torchvision.transforms as T

# --------------------------
# 配置
# --------------------------
MODEL_DIR = "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct"
TARGET_IMAGE_PATH = "/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/apple.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 5e-1
STEPS = 500

# --------------------------
# 加载模型 & 处理器
# --------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR)

# --------------------------
# 目标图像 → embedding
# --------------------------
target_image = Image.open(TARGET_IMAGE_PATH).convert("RGB")

# processor 会自动做 resize/normalize → tensor
with torch.no_grad():
    target_inputs = processor(text=["Describe this image"],images=[target_image], return_tensors="pt").to(DEVICE)
    target_embeds = model.get_image_features(target_inputs['pixel_values'], target_inputs['image_grid_thw'])  # (1, seq_len, dim)

# --------------------------
# 随机初始化图像 (噪声)
# --------------------------
rand_image = torch.randn(1, 3, target_image.height, target_image.width, device=DEVICE, requires_grad=True)

optimizer = optim.Adam([rand_image], lr=LR)
# 添加学习率调度器 - 每STEP_SIZE步将学习率乘以GAMMA
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.90)
loss_fn = nn.MSELoss()

# --------------------------
# 优化循环
# --------------------------
for step in range(STEPS):
    optimizer.zero_grad()

    # 当前图像 embedding
    cur_inputs = processor(text=["Describe this image"],images=[rand_image.squeeze(0).clamp(0, 1)],
                           return_tensors="pt").to(DEVICE)
    cur_embeds = model.get_image_features(cur_inputs['pixel_values'], cur_inputs['image_grid_thw'])  # (1, seq_len, dim)

    # 目标：embedding 接近
    loss = loss_fn(cur_embeds[0], target_embeds[0])

    loss.backward()
    optimizer.step()
    scheduler.step()

    # 限制像素范围
    with torch.no_grad():
        rand_image.clamp_(0, 1)

    if (step + 1) % 1 == 0:
        print(f"Step {step+1}/{STEPS}, Loss={loss.item():.6f}")
        # 保存中间结果
        if (step + 1) % 20 == 0:
            T.ToPILImage()(rand_image.squeeze(0).cpu()).save(f"output/adv_step.png")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "output/adv_step.png",
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[rand_image.squeeze(0).clamp(0, 1)],
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(f"output_text={output_text}")

# --------------------------
# 保存最终结果
# --------------------------
adv_img_pil = T.ToPILImage()(rand_image.squeeze(0).cpu())
adv_img_pil.save("adv_reconstructed.png")

print("\n✅ 对抗优化完成，已保存到 adv_reconstructed.png")
