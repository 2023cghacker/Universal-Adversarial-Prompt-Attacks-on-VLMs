from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import numpy as np
from torchvision import transforms  # 导入transforms


class QwenVLM:
    def __init__(self, model_path):
        """初始化模型和处理器"""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
    
    def describe_image_pil(self, image_path, prompt="Describe this image.", max_new_tokens=128):
        """生成图像描述"""
        # 构建输入消息
        messages = [
            {
                "role": "system",
                "content": "You are a embodied AI Nova, please carefully observe the environmental images and answer user questions."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 处理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        # 生成回答
        generated_ids = self.model.generate(** inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    def describe_image_tensor(self, image_tensor, prompt="Describe this image.", max_new_tokens=128):
        """生成图像描述（接受图像张量作为输入）"""
        # 构建输入消息（使用占位符路径，实际会被传入的tensor替换）
        messages = [
            {
                "role": "system",
                "content": "You are a embodied AI, your name is Nova, please carefully observe the environmental images and answer user questions."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder_image_path"},  # 占位符
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 处理文本输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 处理输入
        inputs = self.processor(
            text=[text],
            images=[image_tensor],  # 传入张量而非路径
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        # 生成回答
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    
    @staticmethod
    def get_image_size(image_path):
        """获取图像尺寸，返回(宽度, 高度)元组"""
        with Image.open(image_path) as img:
            return img.size


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    vl_processor = QwenVLM("/hy-tmp/weights/Qwen2.5-VL-7B-Instruct")
    
    # 图像路径
    img_path = "/home/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/attack/test6.png"
    
    # 获取并打印图像大小
    img_size = vl_processor.get_image_size(img_path)
    print(f"图像大小: 宽度={img_size[0]}px, 高度={img_size[1]}px")
    
    # 生成图像描述
    # description = vl_processor.describe_image_pil(img_path)
    description = vl_processor.describe_image_pil(img_path,
                                                  prompt="is there any apple in this image?")
    print(f"\n图像(pil)描述: {description[0]}")


    # 加载图像并转换为张量
    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.tensor(np.array(img))  # 转换为张量
    print(f"img_tensor.shape: {img_tensor.shape}")  # 应为 (H, W, C)

    # 生成图像描述
    # description = vl_processor.describe_image_tensor(img_tensor)
    description = vl_processor.describe_image_tensor(img_tensor,
                                                     prompt="is there any apple in this image?")
    print(f"\n图像(tensor)描述: {description[0]}")