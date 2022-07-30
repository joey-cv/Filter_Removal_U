import os
import shutil
file_root ='/root/data/AIM_RESULT/1'
all_dirs = os.listdir(file_root)
class_names = ['_Amaro', '_Clarendon', '_Gingham', '_He-Fe',
         '_Hudson', '_Mayfair', '_Nashville', '_Perpetua',
         '_Valencia', '_X-ProII']
src_name = '_Lo-Fi'
if os.path.exists(file_root):
    for per_dir in all_dirs:
        for class_name in class_names:
            src_file_name = os.path.join(file_root, per_dir, per_dir + src_name+'.png')
            out_file_name = os.path.join(file_root, per_dir, per_dir + class_name+'.png')
            print(src_file_name)
            print(out_file_name)
            # shutil.copy(src_file_name, out_file_name)
            # print('1')



else:
    print('no exists src')