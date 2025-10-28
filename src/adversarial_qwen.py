import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torchvision.transforms as T
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

class QwenAdversarialBase:
    def __init__(self, model_dir, target_image_path, lr=5e-1, steps=500):
        # 初始化时间戳和输出目录
        self.timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.output_dir = os.path.join("output", self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出文件保存路径: {self.output_dir}")
        
        # 配置参数
        self.model_dir = model_dir
        self.target_image_path = target_image_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.steps = steps
        
        # 存储训练过程中的loss
        self.loss_history = []
        
        # 加载模型和处理器
        self.model = None
        self.processor = None
        self._load_model()
        
        # 目标图像和其embedding
        self.target_image = None
        self.target_embeds = None
        self._process_target_image()
        
    
    def _load_model(self):
        """加载模型和处理器"""
        print(f"加载模型: {self.model_dir}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        print("模型加载完成")
    
    def _process_target_image(self):
        """处理目标图像并获取其embedding"""
        print(f"加载目标图像: {self.target_image_path}")
        self.target_image = Image.open(self.target_image_path).convert("RGB")
        
        with torch.no_grad():
            target_inputs = self.processor(
                text=["Describe this image"],
                images=[self.target_image], 
                return_tensors="pt"
            ).to(self.device)
            self.target_embeds = self.model.get_image_features(
                target_inputs['pixel_values'], 
                target_inputs['image_grid_thw']
            )  # (1, seq_len, dim)
        print("目标图像处理完成")
    
    def _init_optimizer(self):
        """初始化优化器、调度器和随机图像"""
        # 随机初始化图像 (噪声)
        self.rand_image = torch.randn(
            1, 3, self.target_image.height, self.target_image.width, 
            device=self.device, 
            requires_grad=True
        )
        
        self.optimizer = optim.Adam([self.rand_image], lr=self.lr)
        # 添加学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.90
        )
    
    def _save_intermediate_result(self, step):
        """保存中间结果图像"""
        img_path = os.path.join(self.output_dir, f"adv_step_{step+1}.png")
        T.ToPILImage()(self.rand_image.squeeze(0).cpu()).save(img_path)
        return img_path
    
    def _generate_description(self, img_tensor,step,prompt="Describe this image."):
        """生成当前图像的描述"""
        messages = [
            {
                "role": "system",
                "content": "You are a embodied AI Nova, please carefully observe the environmental images and answer user questions."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(self.output_dir, f"adv_step_{step+1}.png"),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 准备推理
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            # images=[img_tensor.squeeze(0).clamp(0, 1)],
            images=[img_tensor],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 生成输出
        generated_ids = self.model.generate(** inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    def _plot_loss(self):
        """绘制并保存loss曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history)+1), self.loss_history, label='MSE Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Loss During Adversarial Training')
        plt.legend()
        plt.grid(True)
        
        loss_plot_path = os.path.join(self.output_dir, f"loss_{self.timestamp}.png")
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss曲线已保存至: {loss_plot_path}")
    
    def run_optimization(self):
        """运行优化循环"""
        # 优化器和调度器
        self.rand_image = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = nn.MSELoss()
        self._init_optimizer()
        
        
        print(f"开始优化，共{self.steps}步，使用设备: {self.device}")
        
        
        for step in range(self.steps):
            self.optimizer.zero_grad()

            # 获取当前图像的embedding
            cur_inputs = self.processor(
                text=["Describe this image"],
                images=[self.rand_image.squeeze(0).clamp(0, 1)],
                return_tensors="pt"
            ).to(self.device)
            cur_embeds = self.model.get_image_features(
                cur_inputs['pixel_values'], 
                cur_inputs['image_grid_thw']
            )

            # 计算损失
            loss = self.loss_fn(cur_embeds[0], self.target_embeds[0])
            self.loss_history.append(loss.item())

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # 限制像素范围
            with torch.no_grad():
                self.rand_image.clamp_(0, 1)

            # 打印进度
            if (step + 1) % 1 == 0:
                print(f"Step {step+1}/{self.steps}, Loss={loss.item():.6f}")
                
                # 保存中间结果和生成描述
                if (step + 1) % 20 == 0:
                    self._save_intermediate_result(step)
                    desc = self._generate_description(self.rand_image,step)
                    print(f"图像描述: {desc}")
        
        # 保存最终结果
        final_img_path = os.path.join(self.output_dir, "adv_reconstructed.png")
        adv_img_pil = T.ToPILImage()(self.rand_image.squeeze(0).cpu())
        adv_img_pil.save(final_img_path)
        
        # 绘制并保存loss曲线
        self._plot_loss()
        
        print(f"\n✅ 对抗优化完成，最终结果已保存到 {final_img_path}")
        return final_img_path


# 使用示例
if __name__ == "__main__":
    MODEL_DIR = "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct"
    TARGET_IMAGE_PATH = "/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/apple.png"
    
    # 创建实例并运行优化
    adversarial = QwenAdversarialBase(
        model_dir=MODEL_DIR,
        target_image_path=TARGET_IMAGE_PATH,
        lr=5e-1,
        steps=30
    )
    adversarial.run_optimization()
