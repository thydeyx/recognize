# -*- coding: utf-8 -*-

import os
import struct
import sys
import re
import shutil
import PIL.Image

import numpy as np


def readFromGnt(data_file, total_bytes, code_map):
    '''
    从文件对象中读取数据并返回

    param data_file， 文件对象
    param total_bytes: 文件总byte数
    param code_map： 由于汉字编码不连续，作为分类label并不合适，该dict保存汉字码和label的映射关系
    '''
    decoded_bytes = 0;
    image_list = [];
    label_list = [];
    new_label = len(code_map) + 1;

    while decoded_bytes != total_bytes:
        try:
            data_length = struct.unpack("<I", data_file.read(4))[0];
            tag_code = struct.unpack(">H", data_file.read(2))[0];
            image_width = struct.unpack("<H", data_file.read(2))[0];
            image_height = struct.unpack("<H", data_file.read(2))[0];
            result_image = np.zeros((image_height, image_width));

            for row in range(image_height):
                for col in range(image_width):
                    result_image[row][col] = struct.unpack('B', data_file.read(1))[0];

            decoded_bytes += data_length;
        except struct.error:
            break;

        try:
            tag_code_str = struct.pack(">H", tag_code).decode("GB2312");
        except UnicodeDecodeError:
            continue;

        # print(hex(tag_code), tag_code_str);

        if tag_code_str not in code_map:
            code_map[tag_code_str] = new_label;
            new_label += 1;

        image_list.append(result_image);
        label_list.append(code_map[tag_code_str]);
 
    return image_list, label_list, code_map;


def gnt2npy(src_dir, dst_dir, map_file):
    '''
    将gnt文件存为npy格式

    param src_file: 源文件名，gnt文件
    param dst_file: 目标文件名， 若此参数设置为'xxx'，则会生成xxx_images.npy 和 xxx_labels.npy
    param map_file: 由于汉字编码不连续，作为分类label并不合适，该文件保存汉字码和label的映射关系
    '''

    code_map = {};
    if os.path.exists(map_file):
        with open(map_file, "r", encoding="UTF-8") as fp:
            for line in fp:
                line = line.strip();
                if len(line) == 0:
                    continue;

                code, label = line.split("\t");
                code_map[code] = int(label);

    shutil.rmtree(dst_dir, ignore_errors=True);
    os.mkdir(dst_dir);

    if os.path.isdir(src_dir): #包含gnt文件的文件夹
        file_list = os.listdir(src_dir);

        for file_name in file_list:
            file_path = os.path.join(src_dir, file_name);
            image_list = [];
            label_list = [];

            if os.path.isfile(file_path) and re.match(".*.gnt", file_name):
                print("processing {} ...".format(file_path));
                data_file = open(file_path, "rb");
                total_bytes = os.path.getsize(file_path);
                image_list, label_list, code_map = readFromGnt(data_file, total_bytes, code_map);
                for i in range(len(image_list)):
                    res_file_name = str(label_list[i]) + "_" + os.path.splitext(file_name)[0] + "_" + str(i) + ".npy";
                    print(res_file_name);
                    np.save(os.path.join(dst_dir, res_file_name), image_list[i]);
                    # im = PIL.Image.fromarray(image_list[i]);
                    # im.convert("RGB").save(os.path.join(dst_dir, str(label_list[i]) + "_" + res_file_name + ".png"));

            del image_list;
            del label_list;

    else:
        sys.stderr.write("Please check your input again.\n");
        return None;

    with open(map_file, "w", encoding="UTF-8") as fp:
        for code in code_map:
            fp.write(code + "\t" + str(code_map[code]) + "\n");


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 " + sys.argv[0] + "src_dir dst_dir map_file\n");
        sys.exit();

    src_file = sys.argv[1];
    dst_file = sys.argv[2];
    map_file = sys.argv[3];
    gnt2npy(src_file, dst_file, map_file);


if __name__ == "__main__":
    main();