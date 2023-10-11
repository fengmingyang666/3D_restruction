
# 需要解压的文件夹
ZIP_PATH = "D:\Download"
# 解压到的根目录
EXTRACT_BASE_PATH = "D:\Download"

import os
import zipfile

# "F:\A"的长度是2
lenth_zip_path = len(ZIP_PATH.split('\\'))

for dir_index, (root, ds, fs) in enumerate(os.walk(ZIP_PATH)):
    for file_index, f in enumerate(fs):
        # 判断是不是.zip文件, 后缀名是zip, 或者压根没有后缀名pass
        if (f.split('.')[-1] != 'zip') or (len(f.split('.')) == 1):
            continue 
        print("==========================================")
        print("Found zip file...")
        # 确定相对路径
        # 如"F:\A" ---> "F:\A\B\C"
        # 需要提取"B\C" 
        # len(ZIP_PATH.split('\\'))在前面避免多次调用
        root_spilt = root.split('\\')
        relative_path = root_spilt[lenth_zip_path:len(root_spilt)]        
        # 拼接解压目录
        extract_path = EXTRACT_BASE_PATH
        for item in relative_path:
            extract_path = os.path.join(extract_path, item)
        print("extract_path = {}".format(extract_path))
        print("root = {}".format(root))
        print("file = {}".format(f))
        zipfilename = os.path.join(root, f)
        print("zipfilename = {}".format(zipfilename))
        # 解压文件
        print("尝试打开压缩文件中...", end='')
        zip = zipfile.ZipFile(zipfilename,'r')
        print("打开成功,开始解压……")
        zip.extractall(path=extract_path)
        zip.close()
print('ok')