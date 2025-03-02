#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# Author: Yue Gaofeng
# Date: 2025.2.17
# This subprogram mainly merges the six small graphs from the main program run into one line.

from PIL import Image

def round_figs(iter):
    # 打开多个 PNG 图像
    image1 = Image.open(f'Results/fig9_{iter}_Client 1.png')
    image2 = Image.open(f'Results/fig9_{iter}_Client 2.png')
    image3 = Image.open(f'Results/fig9_{iter}_Client 3.png')
    image4 = Image.open(f'Results/fig9_{iter}_Client 4.png')
    image5 = Image.open(f'Results/fig9_{iter}_Client 5.png')
    image6 = Image.open(f'Results/fig9_{iter}_Global.png')

    # 获取图像的宽度和高度
    total_width = image1.width + image2.width + image3.width + image4.width + image5.width + image6.width
    max_height = max(image1.height, image2.height, image3.height, image4.height, image5.height, image6.height)

    # 创建一个新的空白图像，宽度为六个图像宽度之和，高度为最高的图像高度
    new_image = Image.new('RGB', (total_width, max_height))

    # 将每个图像粘贴到新图像中
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    new_image.paste(image3, (image1.width + image2.width, 0))
    new_image.paste(image4, (image1.width + image2.width + image3.width, 0))
    new_image.paste(image5, (image1.width + image2.width + image3.width + image4.width, 0))
    new_image.paste(image6, (image1.width + image2.width + image3.width + image4.width + image5.width, 0))

    new_image.save(f'Results/fig9-1-{iter+1}.pdf', format='pdf')


if __name__ == "__main__":
    iter = 6
    for i in range(0, iter):
        round_figs(iter=i)
