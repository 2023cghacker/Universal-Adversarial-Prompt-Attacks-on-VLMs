from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("/hy-tmp/weights/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/output/0911_172820/adv_step.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
# print(f"image_inputs.shape={image_inputs[0].shape}")
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
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
print(f"\n\noutput_text={output_text}")


# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# import torch
# import torchvision.transforms as T
# from PIL import Image

# # 加载模型
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/hy-tmp/weights/Qwen2.5-VL-7B-Instruct",
#     dtype=torch.bfloat16,
#     device_map="auto",
# )

# # 加载处理器
# processor = AutoProcessor.from_pretrained("/hy-tmp/weights/Qwen2.5-VL-7B-Instruct")

# # 读取PNG并转成tensor (C,H,W)，范围 [0,1]
# image_path = "/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/adv_reconstructed.png"
# image = Image.open(image_path).convert("RGB")
# image_tensor = T.ToTensor()(image)  # (3,H,W)

# # 消息必须有 image 类型
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},   # 占位符，告诉 processor 有图像
#             {"type": "text", "text": "Describe this image."}
#         ],
#     }
# ]

# # 构建文本输入
# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# # text + image 一起送进 processor
# inputs = processor(
#     text=[text],
#     images=[image_tensor],   # 传 tensor
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to("cuda")

# # 推理
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )

# print(f"\n\noutput_text={output_text}")
