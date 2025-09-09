from PIL import Image
import os

def combine_images(image_a_path, image_b_path, output_path, target_size):
    """
    将图片a缩小到指定尺寸，放置在图片b的左下角（图片a背景透明）
    
    参数:
        image_a_path: 图片a的路径（需要放置的小图，支持透明背景）
        image_b_path: 图片b的路径（底图）
        output_path: 输出图片的保存路径
        target_size: 元组 (width, height)，图片a缩小后的尺寸
    """
    # 打开图片
    try:
        # 打开图片a并保持透明通道
        img_a = Image.open(image_a_path).convert("RGBA")
        # 打开图片b
        img_b = Image.open(image_b_path).convert("RGBA")
    except Exception as e:
        raise IOError(f"无法打开图片: {e}")
    
    # 缩小图片a到目标尺寸
    img_a_resized = img_a.resize(target_size, Image.Resampling.LANCZOS)  # 使用高质量缩放
    
    # 计算放置位置（左下角）
    # x坐标为0，y坐标为底图高度减去小图高度
    position = (200, img_b.height - img_a_resized.height)
    
    # 创建一个新的图像作为结果（基于图片b）
    result = Image.new('RGBA', img_b.size)
    # 先绘制底图
    result.paste(img_b, (0, 0))
    # 再绘制缩小后的图片a（使用mask参数保留透明度）
    result.paste(img_a_resized, position, mask=img_a_resized)
    
    # 保存结果（如果输出格式是JPG，需要转换为RGB）
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        result = result.convert('RGB')
    result.save(output_path)
    
    # 关闭图片
    img_a.close()
    img_b.close()
    result.close()
    
    print(f"图片已成功保存至: {os.path.abspath(output_path)}")


# 使用示例
if __name__ == "__main__":
    # 示例：将logo缩小到(100, 100)放在背景图左下角
    combine_images(
        image_a_path="/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/trashbag_text_english.png",       # 带透明背景的图片a
        image_b_path="/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/env/environment5.jpg", # 底图b
        output_path="/root/lingchen/Universal-Adversarial-Prompt-Attacks-on-VLMs/images/env+text/environment5.png",    # 输出图片
        target_size=(150, 150)         # 图片a缩小后的尺寸
    )
