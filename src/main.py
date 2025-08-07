import torch
from alignment_attack import AdversarialTrainer

if __name__ == "__main__":
    local_model_path = (
        "/HOME/paratera_xy/pxy480/HDD_POOL/Ling/downloads/clip-vit-large-patch14-336"
    )
    background_image_path = "images/environment.png"
    initial_patch_path = "images/initpatch1.png"
    save_path = "output/adv_trashbag1"  # 不需要后缀
    target_text = None  # "an apple"
    target_img = "images/apple.png"

    # 初始化对抗训练器
    trainer = AdversarialTrainer(model_path=local_model_path, num_steps=100000, lr=3e-3)

    # 训练并保存结果
    patch_size = 60
    patch_position = [100, 240]
    trainer.train_patch(
        background_image_path=background_image_path,
        target_text=target_text,
        target_img=target_img,
        patch_size=patch_size,
        position=patch_position,
        background_weight=0.2,
        initial_patch_path=initial_patch_path,
        save_path=save_path,
    )

    # 加载优化出的对抗图像进行验证,对比优化前后是否有提示
    print("初始图像:")
    trainer.test(
        img_path=background_image_path,
        target_text=target_text,
        target_img=target_img,
    )

    print("加上对抗贴片的图像:")
    trainer.test(
        img_path=save_path + ".png",
        target_text=target_text,
        target_img=target_img,
    )
