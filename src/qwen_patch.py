import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torchvision.transforms as T
import os
import datetime
import matplotlib.pyplot as plt
from adversarial_qwen import QwenAdversarialBase
import numpy as np
from sklearn.manifold import TSNE


# =====================
# 子类：Patch 优化
# =====================
class QwenAdversarialPatch(QwenAdversarialBase):
    def __init__(self, model_dir, target_image_path, background_image_path, lr=1e-1, steps=300):
        super().__init__(model_dir, target_image_path, lr, steps)

        # 加载背景图像（宿主图）
        self.background_image = Image.open(background_image_path).convert("RGB")
        # self.background_image = torch.tensor(np.array(self.background_image))  # 转换为张量
        self.background_image = self.background_image.resize((self.target_image.width, self.target_image.height), Image.BICUBIC)  
        
        # print(f"self.background_image.shape: {self.background_image.shape}")
        print(f"self.background_image.size: {self.background_image.size}")

    def train_patch(self, patch_size=(50, 50), position=(0, 0)):
        print(f"开始 Patch 优化，patch_size={patch_size}, position={position}")

        base_tensor = torch.tensor(np.array(self.background_image), dtype=torch.float32).to(self.device)  # 转换为张量[3,H,W]
        base_tensor = base_tensor.permute(2,0,1).contiguous()
        base_tensor.requires_grad_(True)  # 显式启用梯度
        print(f"base_tensor.shape: {base_tensor.shape}")
    
        _, H, W= base_tensor.shape
        ph, pw = patch_size
        y, x = position
        assert y+ph <= H and x+pw <= W, "Patch 超出了原图范围! ph={ph}, pw={pw}, H={H}, W={W}, x={x}, y={y}"

        # 初始化 patch``
        patch = torch.randn((3, ph, pw), device=self.device, requires_grad=True)
        optimizer = optim.Adam([patch], lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        loss_fn = nn.MSELoss()

        self.loss_history = []

        for step in range(self.steps):
            optimizer.zero_grad()

            patched_image = base_tensor.clone()
            patched_image[:, y:y+ph, x:x+pw] = patch.clamp(0, 255)
            # print(f"patched_image.shape: {patched_image.shape}, patch.shape: {patch.shape}")
            
            cur_inputs = self.processor(
                text=["Describe this image"],
                images=[patched_image],
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
                grad_norm = torch.norm(patch.grad).item() if patch.grad is not None else 0
                print(f"梯度范数: {grad_norm:.6f}")
                if patch.grad is None:
                    print("警告：patch梯度为None！")
                else:
                    print(f"梯度均值: {patch.grad.mean().item():.6f}")
            
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                patch.clamp_(0, 255)

            if (step+1) % 1 == 0:
                print(f"[Patch] Step {step+1}/{self.steps}, Loss={loss.item():.6f}")
                
                # 保存中间结果和生成描述
                if step == 0:
                    self.visualize_embeddings_2d(cur_embeds[0], self.target_embeds[0],step)
                if (step + 1) % 10 == 0:
                    self.visualize_embeddings_2d(cur_embeds[0], self.target_embeds[0],step)
                    img_path = os.path.join(self.output_dir, f"patch_step_{step+1}.png")
                    T.ToPILImage()(patched_image.to(torch.uint8).cpu()).save(img_path)
                    # T.ToPILImage()(patched_image.cpu()).save(img_path)
                    desc = self._generate_description(patched_image,step,prompt="is there any apple in this image?")
                    print(f"图像描述: {desc}")
                    self._plot_loss()
         

        final_img_path = os.path.join(self.output_dir, "patched_final.png")
        T.ToPILImage()(patched_image.to(torch.uint8).cpu()).save(img_path)

        self._plot_loss()
        print(f"\n✅ Patch 优化完成，结果保存到 {final_img_path}")
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
        plt.legend(fontsize=10)  # 图例区分散点（连线无需单独加图例，避免冗余）
        plt.grid(True, alpha=0.3)
        
        # 6. 保存与显示（文件名保留step，便于追溯训练阶段）
        save_path = os.path.join(self.output_dir, f"visualize_embeddings_2d_with_lines_step_{step}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # bbox_inches避免标签截断
        plt.show()
    
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
        plt.show()
        
        
    
# =====================
# main
# =====================
if __name__ == "__main__":
    MODEL_DIR = "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct"
    TARGET_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/attack/test6.png"
    BACKGROUND_IMAGE_PATH = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/attack/test6.png"

    # Patch 优化
    adv_patch = QwenAdversarialPatch(
        model_dir=MODEL_DIR,
        target_image_path=TARGET_IMAGE_PATH,      # 目标对齐图像
        background_image_path=BACKGROUND_IMAGE_PATH,  # 被攻击宿主图像
        lr=1,
        steps=500
    )
    # adv_patch.train_patch(patch_size=(140, 200), position=(100, 30))
    adv_patch.train_patch(patch_size=(150, 400), position=(100, 30))
