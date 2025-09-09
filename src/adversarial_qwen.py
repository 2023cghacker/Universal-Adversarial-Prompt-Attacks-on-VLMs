import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from torch.utils.data import DataLoader

# 1. 加载模型和处理器
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 准备单条训练数据
def create_single_data(image_path, user_text, assistant_text):
    image = Image.open(image_path).convert("RGB")
    return Dataset.from_dict({
        "conversations": [
            [
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": assistant_text}
                ]}
            ]
        ]
    })

# 3. 数据预处理
def preprocess(examples):
    texts = [processor.apply_chat_template(conv, tokenize=False) 
            for conv in examples["conversations"]]
    images = [conv[0]["content"][1]["image"] 
             for conv in examples["conversations"]]
    
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    )
    # 移动数据到模型设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # 标签与输入一致，但忽略pad_token的损失
    inputs["labels"] = inputs["input_ids"].clone()
    inputs["labels"][inputs["input_ids"] == processor.tokenizer.pad_token_id] = -100
    return inputs

# 4. 加载并处理数据
dataset = create_single_data(
    image_path="apple.png",
    user_text="Describe this picture",
    assistant_text="this is an apple"
)
tokenized_data = dataset.map(preprocess, batched=True, remove_columns=["conversations"])
dataloader = DataLoader(tokenized_data, batch_size=1)

# 5. 定义优化器（手动指定）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.01
)

# 6. 手动训练循环
model.train()  # 设置训练模式
num_epochs = 3

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    total_loss = 0.0
    
    for batch in dataloader:
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播计算损失（手动计算loss）
        outputs = model(** batch)
        loss = outputs.loss  # 获取模型计算的损失
        
        # 反向传播
        loss.backward()  # 计算梯度
        
        # 参数更新
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Batch loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch average loss: {avg_loss:.4f}")

# 7. 保存微调后的模型
model.save_pretrained("./qwen_manual_finetuned")
processor.save_pretrained("./qwen_manual_finetuned")
print("模型保存完成")
    