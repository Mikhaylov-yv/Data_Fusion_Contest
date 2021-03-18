from shutil import copy
from shutil import copyfile
import os.path
import zipfile

def warp_data(data_name, path = 'Model_to_send'):
    arch_name = 'out_mode.zip'
    path = f"data/output/{path}"
    pac_list = ['script.py',
                'main.py',
                'model.py',
                'model',
                'model_param',
                'tokenizer',
                'vocab',
                'cat_dict'
                ]
    zipf = zipfile.ZipFile(arch_name, 'w', zipfile.ZIP_DEFLATED)
    # Копирование модулей
    if os.path.exists(path):
        print('Папка есть')
    else:
        print('Паки нет')
        os.makedirs(path)
    for fil_name in pac_list:
        copy(fil_name, path)
        zipf.write(os.path.join(path, fil_name))
    data_path = f"{path}/data"
    # Копирование данных для тестирования
    if os.path.exists(data_path):
        print('Папка есть')
    else:
        print('Паки нет')
    copyfile(f"data/input/{data_name}", f"{data_path}/task1_test_for_user.parquet")
    # Создание архива

    zipf.close()

if __name__ == '__main__':
    warp_data()
