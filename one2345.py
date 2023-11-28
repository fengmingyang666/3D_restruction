#使用前，请关闭任何中华人民共和国不允许的软件（如：梯子）（旺柴）

from gradio_client import Client
import shutil
import os
client = Client("https://one-2-3-45-one-2-3-45.hf.space/")

def mkdir(path):
 
        folder = os.path.exists(path)

        if not folder:                   
                os.makedirs(path)            
        return

model = 'smartphone'
mash_pre_path = './mash_pre' + os.sep + model
data_path = './data' + os.sep + model

mkdir(data_path)
mkdir(mash_pre_path)


def one_2_3_45(input_img_path):

        ### Elevation estimation 
        # DON'T TO ASK USERS TO ESTIMATE ELEVATION! This OFF-THE-SHELF algorithm is ALL YOU NEED!
        elevation_angle_deg = client.predict(
                input_img_path,
                True,           # image preprocessing
                api_name="/estimate_elevation"
        )
        #print('elevation_angle_deg: ', elevation_angle_deg)

        ### Image preprocessing: segment, rescale, and recenter
        segmented_img_filepath = client.predict(
                input_img_path, 
                api_name="/preprocess"
        )
        #print('segmented_img_filepath: ', segmented_img_filepath)

        ### Single image to 3D mesh
        generated_mesh_filepath = client.predict(
                input_img_path, 
                True,           # image preprocessing
                api_name="/generate_mesh"
        )
        print('generated_mesh_filepath: ', generated_mesh_filepath)

        print("finish.")
        return generated_mesh_filepath


filelist = os.listdir(data_path)

for file in filelist:
        if file.endswith(".jpeg"):
                #run model
                model_idx = file.replace('.jpeg','')
                file_path = data_path + os.sep + file
                mash_path = one_2_3_45(file_path)
                #break

                #move the file to now directory
                p = mash_path.split("gradio")[0]
                print(p)
                p = p + 'gradio'
                new_mash_path = p + os.sep + model
                mkdir(new_mash_path)
                new_mash_file = new_mash_path + os.sep + model_idx + '.ply'
                os.rename(mash_path,new_mash_file)
                shutil.move(new_mash_file,mash_pre_path)
                