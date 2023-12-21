import os
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format
from elevation_estimate.estimate_wild_imgs import estimate_elev
import clip

is_init = False
def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256

def stage1_run(model, device, exp_dir,
               input_im, scale, ddim_steps):
    # folder to save the stage 1 images
    stage1_dir = os.path.join(exp_dir, "stage1_8")
    os.makedirs(stage1_dir, exist_ok=True)

    # stage 1: generate 4 views at the same elevation as the input
    output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)
    i = 0
    
    # stage 2 for the first image
    # infer 4 nearby views for an image to estimate the polar angle of the input
    stage2_steps = 50 # ddim_steps
    zero123_infer(model, exp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)
    # estimate the camera pose (elevation) of the input image.
    try:
        polar_angle = int(estimate_elev(exp_dir))
    except:
        print("Failed to estimate polar angle")
        polar_angle = 90
    print("Estimated polar angle:", polar_angle)
    gen_poses(exp_dir, polar_angle)
    
    # stage 1: generate another 4 views at a different elevation
    if polar_angle <= 75:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4,8)), device=device, ddim_steps=ddim_steps, scale=scale)
    else:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(8,12)), device=device, ddim_steps=ddim_steps, scale=scale)
    torch.cuda.empty_cache()
    total_img_list = output_ims+output_ims_2[1:]
    
    clip_similarity = get_clip_featrue(total_img_list, output_ims[0])
    torch.cuda.empty_cache()
    #print("clip_similarity")
    #print(clip_similarity)
    return clip_similarity
    return 90-polar_angle, output_ims+output_ims_2
    
def stage2_run(model, device, exp_dir,
               elev, scale, stage2_steps=50):
    # stage 2 for the remaining 7 images, generate 7*4=28 views
    if 90-elev <= 75:
        zero123_infer(model, exp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        zero123_infer(model, exp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)

def reconstruct(exp_dir, output_format=".ply", device_idx=0, resolution=256):
    exp_dir = os.path.abspath(exp_dir)
    main_dir_path = os.path.abspath(os.path.dirname("./"))
    os.chdir('reconstruction/')

    bash_script = f'CUDA_VISIBLE_DEVICES={device_idx} python exp_runner_generic_blender_val.py \
                    --specific_dataset_name {exp_dir} \
                    --mode export_mesh \
                    --conf confs/one2345_lod0_val_demo.conf \
                    --resolution {resolution}'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(exp_dir, f"mesh.ply")
    if output_format == ".ply":
        return ply_path
    if output_format not in [".obj", ".glb"]:
        print("Invalid output format, must be one of .ply, .obj, .glb")
        return ply_path
    return convert_mesh_format(exp_dir, output_format=output_format)

def get_clip_featrue(output_ims, ori_img):
    with torch.no_grad():
        clip_similarity = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
        ori_img = clip_preprocess(ori_img).unsqueeze(0).to(device)
        target_img_feature = clip_model.encode_image(ori_img)
        target_img_feature /= target_img_feature.norm(dim=-1, keepdim=True)
        for img in output_ims:
            img = clip_preprocess(img).unsqueeze(0).to(device)
            img_feature = clip_model.encode_image(img)
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
            similarity = img_feature @ target_img_feature.T
            clip_similarity.append(similarity.item())
    return sum(clip_similarity)/len(clip_similarity)



def predict_multiview(shape_dir, args, img_path):
    device = f"cuda:{args.gpu_idx}"
    global is_init
    if is_init == False:
    # initialize the zero123 model
        global models
        global model_zero123
        global predictor
        models = init_model(device, '/mnt/petrelfs/share_data/zhangbeichen/one2345/zero123-xl.ckpt', half_precision=args.half_precision)
        model_zero123 = models["turncam"]
        # initialize the Segment Anything model
        predictor = sam_init(args.gpu_idx)
        is_init = True

    input_raw = Image.open(img_path)

    # preprocess the input image
    input_256 = preprocess(predictor, input_raw)

    # generate multi-view images in two stages with Zero123.
    # first stage: generate N=8 views cover 360 degree of the input shape.
    clip_similarity = stage1_run(model_zero123, device, shape_dir, input_256, scale=3, ddim_steps=75)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return clip_similarity
    #elev, stage1_imgs = stage1_run(model_zero123, device, shape_dir, input_256, scale=3, ddim_steps=75)
    # second stage: 4 local views for each of the first-stage view, resulting in N*4=32 source view images.
    stage2_run(model_zero123, device, shape_dir, elev, scale=3, stage2_steps=50)

if __name__ == "__main__":
    img_root = '/mnt/petrelfs/zhangbeichen/One-2-3-45/data/pc+monitor/'
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_path', type=str, default="/mnt/petrelfs/zhangbeichen/One-2-3-45/image/calculator_2.jpeg", help='Path to the input image')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision')
    parser.add_argument('--mesh_resolution', type=int, default=256, help='Mesh resolution')
    parser.add_argument('--output_format', type=str, default=".ply", help='Output format: .ply, .obj, .glb')

    args = parser.parse_args()


    assert(torch.cuda.is_available())
    total_image = os.listdir(img_root)
    print(len(total_image))
    print(total_image[0])
    total_clip = []
    count = 0
    for img_path in total_image:
        count = count + 1
        shape_id = img_path.split('/')[-1].split('.')[0]
        shape_dir = f"./exp/{shape_id}"
        #print(shape_dir)
        os.makedirs(shape_dir, exist_ok=True)

        sim = predict_multiview(shape_dir, args, img_root+img_path)
        print(sim)
        total_clip.append(sim)
        if count == 10:
            break
    print(img_root)
    print("average clip score:")
    print(sum(total_clip)/len(total_clip))

    # utilize cost volume-based 3D reconstruction to generate textured 3D mesh
    #mesh_path = reconstruct(shape_dir, output_format=args.output_format, device_idx=args.gpu_idx, resolution=args.mesh_resolution)
    #print("Mesh saved to:", mesh_path)
