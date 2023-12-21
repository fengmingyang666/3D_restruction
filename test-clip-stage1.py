import os
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format
from elevation_estimate.estimate_wild_imgs import estimate_elev
import clip

if __name__ == "__main__":
    img_root = '/mnt/petrelfs/zhangbeichen/One-2-3-45/exp/'
    class_name = 'calculator'
    total_image = os.listdir(img_root)
    total_clip = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    with torch.no_grad():
        for img in total_image:
            if class_name in img:
                target_img_name = img_root + img + '/stage1_8/0.png'
                target_img = Image.open(target_img_name)
                target_img = clip_preprocess(target_img).unsqueeze(0).to(device)
                target_img_feature = clip_model.encode_image(target_img)
                target_img_feature /= target_img_feature.norm(dim=-1, keepdim=True)
                stage_2_root = img_root + img + '/stage1_8/'
                reconstruct_img_list = os.listdir(stage_2_root)
                for reconstruct_img in reconstruct_img_list:
                    if reconstruct_img == '0.png': #skip the target image
                        continue
                    query_img = Image.open(stage_2_root + reconstruct_img)
                    query_img = clip_preprocess(query_img).unsqueeze(0).to(device)
                    query_img_feature = clip_model.encode_image(query_img)
                    query_img_feature /= query_img_feature.norm(dim=-1, keepdim=True)
                    sim = query_img_feature @ target_img_feature.T
                    total_clip.append(sim.item())
    print("average_clip_clip")
    print(class_name)
    print(sum(total_clip)/len(total_clip))
                    
                    


    

    # utilize cost volume-based 3D reconstruction to generate textured 3D mesh
    #mesh_path = reconstruct(shape_dir, output_format=args.output_format, device_idx=args.gpu_idx, resolution=args.mesh_resolution)
    #print("Mesh saved to:", mesh_path)
