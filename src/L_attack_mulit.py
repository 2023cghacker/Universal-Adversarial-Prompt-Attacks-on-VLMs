import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime


def extract_model_response(text):
    parts = text.strip().split("\n", 1)
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def load_model(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,   # 用 dtype 替换 torch_dtype
        device_map="auto"
    )
    model.eval()
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False
    # 开启梯度 checkpoint 减少显存占用
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return tokenizer, model


def evaluate(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def adversarial_suffix_optimization(model, tokenizer, prompt, target_text, 
                                    suffix_length, num_steps, lr, plot_filename=None):
    """
    对抗后缀优化（float16安全版本）
    """
    device = model.device
    dtype = torch.float16  # 与模型一致

    # 编码目标文本
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids[0].to(device)
    target_len = target_ids.shape[0]
    if target_len == 0:
        raise ValueError("目标文本编码后为空，请检查输入")
    if suffix_length < target_len:
        raise ValueError(f"后缀长度 ({suffix_length}) 必须 >= 目标长度 ({target_len})")

    # 嵌入层
    embedding_layer = model.base_model.embed_tokens
    vocab_size, hidden_size = embedding_layer.weight.shape

    # 初始化后缀嵌入，float16 + 极小值
    suffix_embeds = torch.randn(suffix_length, hidden_size, device=device, dtype=dtype) * 1e-4
    suffix_embeds.requires_grad_(True)

    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([suffix_embeds], lr=lr)
    loss_list = []

    # prompt 嵌入
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
    prompt_embeds = embedding_layer(prompt_ids).unsqueeze(0).half()  # float16

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        # 拼接 prompt + suffix
        input_embeds = torch.cat([prompt_embeds, suffix_embeds.unsqueeze(0)], dim=1)  # float16

        # 前向传播
        outputs = model(inputs_embeds=input_embeds)

        # 取后缀部分最后 target_len logits
        logits = outputs.logits[:, -suffix_length:, :]      # [1, suffix_len, vocab_size]
        logits = logits[:, -target_len:, :]                # [1, target_len, vocab_size]

        # loss 使用 float32 避免 NaN
        loss = F.cross_entropy(
            logits.float().reshape(-1, vocab_size),
            target_ids.repeat(logits.shape[0], 1).reshape(-1)
        )

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_([suffix_embeds], max_norm=1.0)

        optimizer.step()

        loss_value = loss.item()
        loss_list.append(loss_value)

        if (step + 1) % 10 == 0:
            print(f"[Step {step + 1}/{num_steps}] Loss: {loss_value:.8f}")
            if plot_filename:
                plot_loss(loss_list, plot_filename)

    return suffix_embeds, loss_list

def decode_suffix(suffix_embeds, embedding_layer, tokenizer):
    final_tokens = []
    with torch.no_grad():
        for i in range(suffix_embeds.shape[0]):
            token_logits = torch.matmul(embedding_layer.weight, suffix_embeds[i])
            token_id = torch.argmax(token_logits).item()
            final_tokens.append(token_id)
    suffix = tokenizer.decode(final_tokens, skip_special_tokens=True)
    suffix = re.sub(r'[^\w\s.,!?]', '', suffix)
    return suffix.strip()


def plot_loss(loss_list, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label="Optimization Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Adversarial Suffix Optimization Loss Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale("log")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def llava_adversarial_attack():
    model_path = "/hy-tmp/weights/llava-v1.5-7b"
    original_prompt = "Count how many goods are on the shelves."
    target_output = "I didn't see any goods"
    suffix_length = 15
    num_steps = 600
    lr = 1e-8

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"output/{timestamp}_llava_attack_loss.png"

    print("加载模型中...")
    tokenizer, model = load_model(model_path)
    print(f"模型加载完成，设备: {model.device}")

    print("\n=== 原始输出 ===")
    original_output = evaluate(model, tokenizer, original_prompt)
    print(original_output)

    print("\n开始对抗后缀优化...")
    suffix_embeds, loss_list = adversarial_suffix_optimization(
        model, tokenizer, original_prompt, target_output,
        suffix_length, num_steps, lr, plot_filename=plot_filename
    )

    final_suffix = decode_suffix(suffix_embeds, model.base_model.embed_tokens, tokenizer)
    print(f"\n=== 优化后的对抗后缀 ===")
    print(repr(final_suffix))

    attacked_prompt = original_prompt + final_suffix
    print(f"\n=== 攻击后的提示 ===")
    print(attacked_prompt)

    print("\n=== 攻击后的输出 ===")
    attacked_output = evaluate(model, tokenizer, attacked_prompt)
    print(attacked_output)

    result_filename = f"output/{timestamp}_attack_result.txt"
    with open(result_filename, "w", encoding="utf-8") as f:
        f.write(f"原始提示: {original_prompt}\n\n")
        f.write(f"原始输出: {original_output}\n\n")
        f.write(f"对抗后缀: {final_suffix}\n\n")
        f.write(f"攻击后提示: {attacked_prompt}\n\n")
        f.write(f"攻击后输出: {attacked_output}\n")
    print(f"\n结果已保存至: {result_filename}")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    llava_adversarial_attack()
