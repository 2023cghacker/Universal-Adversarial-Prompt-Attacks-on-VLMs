import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from datetime import datetime
import matplotlib.pyplot as plt
from adversarial_base import CLIPAdversarialBase
import os
import sys
from tqdm import tqdm  # 导入tqdm


class CLIPAdversarialPatch(CLIPAdversarialBase):
    def __init__(self, model_path, device=None, num_steps=300, lr=1e-3):
        """
        局部对抗补丁训练类：仅实现train_patch方法
        继承自CLIPAdversarialBase，复用所有公共功能
        """
        super().__init__(model_path, device, num_steps, lr)

    def train_patch(
        self,
        background_image_paths,
        target_text=None,
        target_img=None,
        patch_size=80,
        positions=[[30, 30]],
        background_weight=0.1,
        initial_patch_path=None,
        save_names=None,
    ):
        """
        核心功能：训练局部对抗补丁（在多张背景图上叠加补丁，使CLIP误分类）
        :param background_image_paths: 背景图像路径列表
        :param target_text: 目标文本（与target_img二选一）
        :param target_img: 目标图像（与target_text二选一）
        :param patch_size: 补丁尺寸（正方形）
        :param positions: 补丁在背景图上的位置列表（与背景图数量一致）
        :param background_weight: 背景损失权重（控制补丁与原图相似度）
        :param initial_patch_path: 初始补丁图像路径（可选）
        :param save_names: 保存名称列表（与背景图数量一致）
        :return: 保存的图像路径列表
        """
        # 输入合法性校验
        if not ((target_text is None) ^ (target_img is None)):
            raise ValueError("target_text与target_img必须二选一")
        if len(background_image_paths) != len(positions):
            raise ValueError("背景图路径列表与位置列表长度必须一致")
        if save_names is None or len(save_names) != len(background_image_paths):
            raise ValueError("保存名称列表必须提供且与背景图数量一致")

        """1.获取目标嵌入（文本或图像）""" 
        if target_text is not None:
            target_emb = self.get_target_text_embedding(target_text)
        else:
            target_emb = self.get_target_image_embedding(target_img)


        """2.加载所有背景图像 """ 
        img_tensors = []
        for img_path in background_image_paths:
            img_tensor, _ = self.load_and_preprocess_image(img_path)
            img_tensors.append(img_tensor)

        """3.初始化对抗patch """
        # 校验所有补丁位置是否超出图像边界
        for i, (img_tensor, pos) in enumerate(zip(img_tensors, positions)):
            x, y = pos
            _, _, h, w = img_tensor.shape
            if x < 0 or x + patch_size > w or y < 0 or y + patch_size > h:
                raise ValueError(f"第{i}个补丁位置超出图像边界（图像尺寸：{w}x{h}）")
            
        # 创建patch
        if initial_patch_path is not None: # 从指定路径加载并Resize为补丁尺寸
            trans = transforms.Compose(
                [
                    transforms.Resize((patch_size, patch_size)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.unsqueeze(0)),  # 增加批次维度
                ]
            )
            patch_img = Image.open(initial_patch_path).convert("RGB")
            patch = trans(patch_img).to(self.device)
        else: # 从背景图的指定位置截取并添加噪声
            x_init, y_init = positions[0]
            background_patch = img_tensors[0][
                :, :, y_init : y_init + patch_size, x_init : x_init + patch_size
            ].clone()
            patch = background_patch * 0.8 + torch.randn_like(background_patch) * 0.2

        # 保存原始补丁（用于计算背景损失）
        original_patches = []
        for img_tensor, pos in zip(img_tensors, positions):
            x, y = pos
            original_patch = img_tensor[
                :, :, y : y + patch_size, x : x + patch_size
            ].detach()
            original_patches.append(original_patch)

        """4.训练patch """
        # 优化器配置
        patch.requires_grad_(True)
        optimizer = optim.Adam([patch], lr=self.lr)

        # 训练记录初始化
        start_time = datetime.now()
        loss_history = []
        similarity_history = []

        # 使用tqdm创建进度条
        print("\n\nStart patch training...")
        progress_bar = tqdm(range(1, self.num_steps + 1), desc="Progress")
        
        # 训练循环
        for step in progress_bar:
            optimizer.zero_grad()
            total_adv_loss = 0.0
            total_bg_loss = 0.0
            total_similarity = 0.0

            # 遍历所有背景图计算损失
            for img_tensor, pos, orig_patch in zip(
                img_tensors, positions, original_patches
            ):
                x, y = pos
                # 叠加补丁生成对抗图像
                adv_img = img_tensor.clone()
                clamp_patch = torch.clamp(patch, 0.0, 1.0)
                adv_img[:, :, y : y + patch_size, x : x + patch_size] = clamp_patch
                # print(f"patch 范围:{patch.min().item():.6f}, {patch.max().item():.6f}")

                # 计算对抗损失（余弦相似度损失）
                img_emb = self.model.get_image_features(pixel_values=adv_img)
                img_emb = F.normalize(img_emb, dim=-1)
                similarity = F.cosine_similarity(img_emb, target_emb).mean()
                total_similarity += similarity
                adv_loss = 1 - similarity
                total_adv_loss += adv_loss

                # 计算背景损失（补丁与原始背景的MSE）
                bg_loss = F.mse_loss(patch, orig_patch)
                total_bg_loss += bg_loss

            # 计算平均损失与总损失
            num_imgs = len(img_tensors)
            avg_adv_loss = total_adv_loss / num_imgs
            avg_bg_loss = total_bg_loss / num_imgs
            total_loss = avg_adv_loss + background_weight * avg_bg_loss
            loss_history.append(total_loss.item())

            # 计算平均相似度并记录
            avg_similarity = total_similarity / num_imgs
            similarity_history.append(avg_similarity.item())

            # 反向传播与优化
            total_loss.backward()
            optimizer.step()
            patch.data = torch.clamp(patch.data, 0.0, 1.0)

            # 时间计算与格式化
            elapsed = datetime.now() - start_time
            progress = step / self.num_steps
            est_total = elapsed / progress if progress > 0 else elapsed
            remaining = est_total - elapsed
            elapsed_str = self._format_timedelta(elapsed)
            remaining_str = self._format_timedelta(remaining)

            # 更新tqdm进度条的描述信息
            progress_bar.set_postfix({
                '对抗损失': f'{avg_adv_loss.item():.4f}',
                '背景损失': f'{avg_bg_loss.item():.4f}',
                '总损失': f'{total_loss.item():.4f}',
                '相似度': f'{avg_similarity.item():.4f}',
                # '已用时间': elapsed_str,
                # '剩余时间': remaining_str
            })

            # 每10步绘制并保存损失曲线和相似度曲线
            if step % 10 == 0:
                self.plot_curves(loss_history, similarity_history)

            # 批量保存当前补丁与所有对抗背景图（每100步）
            if step % 100== 0:
                self.save_adversarial_image(
                    img_tensors=img_tensors,
                    base_names=save_names,
                    patch=patch.detach(),
                    patch_size=patch_size,
                    positions=positions,
                )

        # 训练结束：最终保存所有结果
        final_saved_paths = self.save_adversarial_image(
            img_tensors=img_tensors,
            base_names=save_names,
            patch=patch.detach(),
            patch_size=patch_size,
            positions=positions,
        )
        print(f"\n🎉 补丁训练完成！最终文件保存路径: {final_saved_paths}")
        return final_saved_paths

    def plot_curves(self, loss_history, similarity_history):
        """辅助方法：绘制并保存损失曲线和相似度曲线"""
        
        # 绘制损失曲线
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, label="Total Loss", color="#1f77b4")
        plt.xlabel("Training Step", fontsize=10)
        plt.ylabel("Loss Value", fontsize=10)
        plt.title(f"Patch Training Loss Curve", fontsize=12)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        loss_path = os.path.join(
            self.output_dir, f"loss_curve_{self.timestamp}.png"
        )
        plt.savefig(loss_path, dpi=150)
        plt.close()

        
        # 绘制相似度曲线
        plt.figure(figsize=(8, 4))
        plt.plot(similarity_history, label="Cosine Similarity", color="#ff7f0e")
        plt.xlabel("Training Step", fontsize=10)
        plt.ylabel("Similarity Value", fontsize=10)
        plt.title(f"Embedding Similarity Curve", fontsize=12)
        plt.ylim(0, 1)  # 余弦相似度范围在0-1之间
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        similarity_path = os.path.join(
            self.output_dir, f"similarity_{self.timestamp}.png"
        )
        plt.savefig(similarity_path, dpi=150)
        plt.close()
