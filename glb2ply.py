import os
import trimesh
import time

def mkdir(path):
 
        folder = os.path.exists(path)

        if not folder:                   
                os.makedirs(path)            
        return
def glb2ply(type:str):
    model = type
    mash_gt_path = './mash_gt' + os.sep + model
    data_path = './data' + os.sep + model

    mkdir(data_path)
    mkdir(mash_gt_path)

    filelist = os.listdir(data_path)
    print(model)
    for file in filelist:
            if file.endswith(".glb"):
                    model_idx = file.replace('.glb','')
                    file_path = data_path + os.sep + file
                    try:
                        m = trimesh.load_mesh(file_path)
                        me = m.copy()
                        #me.apply_scale(1.1)
                        file_gt = mash_gt_path + os.sep + model_idx + '.ply'
                        me.export(file_gt)
                        time.sleep(2)
                    except:
                        print(file)
                        time.sleep(2)
    return

types_all = ['calculator','computer+mouse','digital+camera','game+boy',
         'iPad','keyboard','laptop','pc monitor','polaroid','smartphone']
types = ['keyboard','laptop','pc monitor','polaroid','smartphone']

for type in types:
      glb2ply(type=type)
                