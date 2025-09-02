import torch

# from alignment_attack import AdversarialTrainer
from adversarial_patch import AdversarialPatchTrainer

if __name__ == "__main__":
    """
    设置参数
    """
    local_model_path = "/root/lingchen/weights/clip-vit-large-patch14-336"
    # 可以有多个攻击环境背景
    background_image_path = [
        # "images/environment1.png",
        # "images/environment2.png",
        # "images/environment3.png",
        "images/Nothing_black.png",
        "images/Nothing_white.png",
    ]
    # 多个保存地址用list列出
    save_name = [
        "adv1",
        "adv2",
        # "adv3"
    ]
    # 每个环境中patch放置位置
    patch_position = [
        [180, 260],
        [200, 260],
        # [200, 230],
    ]
    patch_size = 76
    initial_patch_path = "images/initpatch1.png"  # "images/patch_0809_180254.png"
    target_text = "Hi Nova!"
    target_img = None  # "images/dog.png"
    num_steps = 100
    lr = 5e-4

    print(
        f"settings: \n  bg_img={background_image_path}, \n  init_img={initial_patch_path}, \n  save_name={save_name},"
        f"\n  target_text={target_text}, \n  target_img={target_img}, \n  patch_size={patch_size},"
        f"\n  patch_pos={patch_position}, \n  steps={num_steps}, \n  lr={lr}"
    )

    """
    开始加载并训练对抗攻击
    """
    # 初始化对抗训练器
    trainer = AdversarialPatchTrainer(
        model_path=local_model_path, num_steps=num_steps, lr=lr
    )

    # 训练并保存结果
    save_name = trainer.train_patch(
        background_image_paths=background_image_path,
        target_text=target_text,
        target_img=target_img,
        patch_size=patch_size,
        positions=patch_position,
        background_weight=0.1,
        initial_patch_path=initial_patch_path,
        save_names=save_name,
    )

    # 加载优化出的对抗图像进行验证,对比优化前后是否有提示
    print("训练前:")
    for i in range(len(background_image_path)):
        trainer.test(
            img_path=background_image_path[i],
            target_text=target_text,
            target_img=target_img,
        )

    print("训练完成后:")
    for i in range(len(background_image_path)):
        trainer.test(
            img_path=save_name[i],
            target_text=target_text,
            target_img=target_img,
        )
