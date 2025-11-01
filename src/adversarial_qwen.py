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
from sklearn.manifold import TSNE

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
        print(f"目标图像处理完成, image_grid_thw={target_inputs['image_grid_thw']}")
    
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
    
    def run_optimization(self, current_image=None):
        """
        在 current_image 上添加可训练噪声，使其 embedding 对齐 target_image。
        若 current_image=None，则默认随机初始化。
        """
        self.loss_fn = nn.MSELoss()
        self.loss_history = []

        # ========== 初始化图像 ==========
        if current_image is not None:
            print(f"加载当前图像用于优化: {current_image}")
            base_img = Image.open(current_image).convert("RGB")
            base_img = base_img.resize((self.target_image.width, self.target_image.height), Image.BICUBIC)
            base_tensor = T.ToTensor()(base_img).unsqueeze(0).to(self.device)
        else:
            print("未提供 current_image，使用随机噪声初始化。")
            base_tensor = torch.rand(
                1, 3, self.target_image.height, self.target_image.width, 
                device=self.device
            )

        # 添加可训练噪声参数
        self.noise = torch.zeros_like(base_tensor, requires_grad=True)
        self.optimizer = optim.Adam([self.noise], lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

        print(f"开始优化，共 {self.steps} 步，设备: {self.device}")

        for step in range(self.steps):
            self.optimizer.zero_grad()

            # 叠加噪声
            cur_image = (base_tensor + self.noise).clamp(0, 1)

            # 获取当前图像 embedding
            cur_inputs = self.processor(
                text=["Describe this image"],
                images=[cur_image.squeeze(0)],
                return_tensors="pt"
            ).to(self.device)

            cur_embeds = self.model.get_image_features(
                cur_inputs['pixel_values'],
                cur_inputs['image_grid_thw']
            )

            # 计算 MSE loss
            loss = self.loss_fn(cur_embeds[0], self.target_embeds[0])
            self.loss_history.append(loss.item())

            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # 打印进度
            if (step + 1) % 1 == 0:
                print(f"Step {step+1}/{self.steps}, Loss={loss.item():.6f}")

            # 每 20 步保存中间结果
            if (step + 1) % 20 == 0:
                save_path = os.path.join(self.output_dir, f"adv_step_{step+1}.png")
                T.ToPILImage()(cur_image.squeeze(0).detach().cpu()).save(save_path)
                desc = self._generate_description(cur_image, step)
                print(f"图像描述: {desc}")

        # ========== 保存最终结果 ==========
        final_img_path = os.path.join(self.output_dir, "adv_reconstructed.png")
        adv_img_pil = T.ToPILImage()(cur_image.squeeze(0).detach().cpu())
        adv_img_pil.save(final_img_path)
        self._plot_loss()
        print(f"\n✅ 对抗优化完成，最终结果已保存到 {final_img_path}")
        return final_img_path


    def train_noise(self,background_image_path):
        # 加载背景图像（宿主图）
        self.background_image = Image.open(background_image_path).convert("RGB")
        self.background_image = self.background_image.resize((self.target_image.width, self.target_image.height), Image.BICUBIC)  
        print(f"self.background_image.size: {self.background_image.size}")
        print(f"开始 noise 优化")

        base_tensor = torch.tensor(np.array(self.background_image), dtype=torch.float32).to(self.device)  # 转换为张量[3,H,W]
        base_tensor = base_tensor.permute(2,0,1).contiguous()
        base_tensor.requires_grad_(True)  # 显式启用梯度
        print(f"base_tensor.shape: {base_tensor.shape}")
    
        _, H, W= base_tensor.shape
        # ph, pw = noise_size
        # y, x = position
        # assert y+ph <= H and x+pw <= W, "noise 超出了原图范围! ph={ph}, pw={pw}, H={H}, W={W}, x={x}, y={y}"

        # 初始化 noise``
        noise = torch.randn((3, H, W), device=self.device, requires_grad=True)
        optimizer = optim.Adam([noise], lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        loss_fn = nn.MSELoss()

        self.loss_history = []

        for step in range(self.steps):
            optimizer.zero_grad()

            noiseed_image = base_tensor.clone()
            noiseed_image += noise.clamp(0, 255)
            # print(f"noiseed_image.shape: {noiseed_image.shape}, noise.shape: {noise.shape}")
            
            cur_inputs = self.processor(
                text=["Describe this image"],
                images=[noiseed_image],
                return_tensors="pt"
            ).to(self.device)

            cur_embeds = self.model.get_image_features(
                cur_inputs['pixel_values'],
                cur_inputs['image_grid_thw']
            )
                    
            print(f"cur_embeds.shape: {cur_embeds[0].shape}, self.target_embeds.shape: {self.target_embeds[0].shape}")
            # print(f"cur_embeds.type: {type(cur_embeds[0])}, self.target_embeds.type: {type(self.target_embeds[0])}")
            loss = loss_fn(cur_embeds[0], self.target_embeds[0])


            self.loss_history.append(loss.item())
            loss.backward()
            # 调试：检查梯度
            if (step + 1) % 10 == 0:
                grad_norm = torch.norm(noise.grad).item() if noise.grad is not None else 0
                print(f"梯度范数: {grad_norm:.6f}")
                if noise.grad is None:
                    print("警告：noise梯度为None！")
                else:
                    print(f"梯度均值: {noise.grad.mean().item():.6f}")
            
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                noise.clamp_(0, 255)

            if (step+1) % 1 == 0:
                print(f"[noise] Step {step+1}/{self.steps}, Loss={loss.item():.6f}")
                
                # 保存中间结果和生成描述
                if step == 0:
                    self.visualize_embeddings_2d(cur_embeds[0], self.target_embeds[0],step)
                if (step + 1) % 10 == 0:
                    self.visualize_embeddings_2d(cur_embeds[0], self.target_embeds[0],step)
                    img_path = os.path.join(self.output_dir, f"noise_step_{step+1}.png")
                    T.ToPILImage()(noiseed_image.to(torch.uint8).cpu()).save(img_path)
                    # T.ToPILImage()(noiseed_image.cpu()).save(img_path)
                    desc = self._generate_description(noiseed_image,step, prompt="Describe this image")
                    print(f"图像描述: {desc}")
                    self._plot_loss()
         

        final_img_path = os.path.join(self.output_dir, "noiseed_final.png")
        T.ToPILImage()(noiseed_image.to(torch.uint8).cpu()).save(img_path)

        self._plot_loss()
        print(f"\n✅ noise 优化完成，结果保存到 {final_img_path}")
        return final_img_path
    
    def visualize_embeddings_2d(self, cur_embeds, target_embeds, step):
        """2D可视化对比两个嵌入向量集合（新增对应点连线功能）"""
        # 1. 张量转numpy数组（保持原逻辑，处理BFloat16转float32）
        cur_np = cur_embeds.cpu().detach().to(torch.float32).numpy()
        target_np = target_embeds.cpu().detach().to(torch.float32).numpy()
        
        # 2. 合并数据用于t-SNE降维
        combined = np.vstack([cur_np, target_np])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        combined_2d = tsne.fit_transform(combined)
        
        # 3. 分离两组2D数据（维度均为(N, 2)，N为样本数量）
        cur_2d = combined_2d[:len(cur_np)]
        target_2d = combined_2d[len(cur_np):]
        
        # 4. 绘图：先画散点，再画对应点连线（避免连线遮挡散点）
        plt.figure(figsize=(10, 8))
        
        # -------------------------- 新增：绘制对应点连线 --------------------------
        # 循环遍历每个索引，连接target_2d[i]与cur_2d[i]
        for i in range(len(cur_2d)):  # len(cur_2d) = len(target_2d) = N，确保索引匹配
            # 获取第i组对应点的坐标
            x_coords = [target_2d[i, 0], cur_2d[i, 0]]  # x轴：target点 → cur点
            y_coords = [target_2d[i, 1], cur_2d[i, 1]]  # y轴：target点 → cur点
            # 绘制连线：设置浅色（灰色）、细线条（linewidth=0.8）、低透明度（alpha=0.5）
            plt.plot(
                x_coords, 
                y_coords, 
                c='gray',        # 连线颜色（与散点区分，避免干扰）
                linewidth=0.8,   # 线条粗细（细线条避免遮挡散点）
                alpha=0.5,       # 透明度（降低连线视觉权重）
                linestyle='-'    # 线型（实线，清晰显示连接关系）
            )
        # --------------------------------------------------------------------------
        
        # 绘制散点（保持原逻辑，散点在连线上方，确保可见）
        plt.scatter(
            cur_2d[:, 0], cur_2d[:, 1], 
            c='blue', alpha=0.6, label='Current Embeddings', 
            s=60  # 适当调大散点，避免被连线覆盖
        )
        plt.scatter(
            target_2d[:, 0], target_2d[:, 1], 
            c='red', alpha=0.6, label='Target Embeddings', 
            s=60
        )
        
        # 5. 图样式配置（优化标题，明确包含连线信息）
        plt.title(f't-SNE 2D Visualization (Step: {step}) - With Matching Lines', fontsize=14)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        # plt.xlim(-45, 30)   # X轴范围：[-5, 5]
        # plt.ylim(-25, 28)   # Y轴范围：[-3, 3]
        plt.legend(fontsize=10)  # 图例区分散点（连线无需单独加图例，避免冗余）
        plt.grid(True, alpha=0.3)
        
        # 6. 保存与显示（文件名保留step，便于追溯训练阶段）
        save_path = os.path.join(self.output_dir, f"visualize_embeddings_{step}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # bbox_inches避免标签截断
        # plt.show()
    
    def visualize_embeddings_3d(self, cur_embeds, target_embeds, step):
        """3D可视化对比两个嵌入向量集合（带训练步数标记）"""
        # 1. 张量转numpy数组（保持原逻辑，确保数据格式正确）
        cur_np = cur_embeds.cpu().detach().to(torch.float32).numpy()
        target_np = target_embeds.cpu().detach().to(torch.float32).numpy()
        
        # 2. 合并数据（样本数量：357+357=714，特征维度：3584）
        combined = np.vstack([cur_np, target_np])
        
        # 3. t-SNE降维（维度=3，保持随机种子确保结果可复现）
        tsne = TSNE(n_components=3, random_state=42, perplexity=30)
        combined_3d = tsne.fit_transform(combined)  # 输出形状：(714, 3)
        
        # 4. 分离两组3D数据（样本数量不变，各357个，维度=3）
        cur_3d = combined_3d[:len(cur_np)]  # 形状：(357, 3)
        target_3d = combined_3d[len(cur_np):]  # 形状：(357, 3)
        
        # 5. 3D散点图绘制：核心修复——添加散点绘制逻辑
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')  # 启用3D投影（必须保留）
        
        # -------------------------- 关键修复：添加散点绘制代码 --------------------------
        # 绘制Current Embeddings（蓝色点）
        ax.scatter(
            cur_3d[:, 0],  # x轴坐标（3D数据第一列）
            cur_3d[:, 1],  # y轴坐标（3D数据第二列）
            cur_3d[:, 2],  # z轴坐标（3D数据第三列）
            c='blue',      # 点颜色
            alpha=0.6,     # 透明度（避免点重叠遮挡）
            label='Current Embeddings',  # 图例标签
            s=50           # 点大小（建议50-100，太小易看不见）
        )
        
        # 绘制Target Embeddings（红色点）
        ax.scatter(
            target_3d[:, 0],  # x轴坐标
            target_3d[:, 1],  # y轴坐标
            target_3d[:, 2],  # z轴坐标
            c='red',          # 点颜色（与Current区分）
            alpha=0.6,
            label='Target Embeddings',
            s=50
        )
        # --------------------------------------------------------------------------------
        
        # 6. 3D图样式配置（提升可读性，避免标签缺失）
        ax.set_title(f't-SNE 3D Visualization (Step: {step})', fontsize=14)  # 标题带训练步数
        ax.set_xlabel('Dimension 1', fontsize=12)  # x轴标签
        ax.set_ylabel('Dimension 2', fontsize=12)  # y轴标签
        ax.set_zlabel('Dimension 3', fontsize=12)  # z轴标签（3D图必须配置，否则无z轴标识）
        ax.legend(fontsize=10)  # 显示图例（区分两组点）
        ax.grid(True, alpha=0.3)  # 显示网格（辅助观察点的位置）
        
        # 7. 保存与显示（确保先保存再显示，避免空图）
        # 文件名带step，方便区分不同训练阶段的可视化结果
        save_path = os.path.join(self.output_dir, f"visualize_embeddings_step_{step}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # bbox_inches避免标签被截断
        # plt.show()
        
        
# 使用示例
if __name__ == "__main__":
    MODEL_DIR = "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct"
    # TARGET_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/attack/content1.png"
    TARGET_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/B.png"
    BACKGROUND_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/apple.png"
    # BACKGROUND_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/Nothing_white.png"
    
    # 创建实例并运行优化
    adversarial = QwenAdversarialBase(
        model_dir=MODEL_DIR,
        target_image_path=TARGET_IMAGE_PATH,
        lr=5e-1,
        steps=1000
    )
    # adversarial.run_optimization(background_image_path=BACKGROUND_IMAGE_PATH)
    adversarial.train_noise(background_image_path=BACKGROUND_IMAGE_PATH)
