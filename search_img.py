import matplotlib.pyplot as plt
import numpy as np
import cv2

# どんな画像があるか確認
name = '0000047'                                                        # ファイル名を指定
img = cv2.imread(f'./iccv09Data/images/{name}.jpg')                     # jpg画像
label_regions = np.loadtxt(f'./iccv09Data/labels/{name}.regions.txt')   # 意味クラス（空, 木, 道, 草, 水, 建物, 山, 前景のオブジェクト）を示すマスク
label_surfaces = np.loadtxt(f'./iccv09Data/labels/{name}.surfaces.txt') # 幾何学的なクラス (空, 水平, 垂直) を示すマスク
label_layers = np.loadtxt(f'./iccv09Data/labels/{name}.layers.txt')     # 別々の画像領域を示すマスク

# 画像表示
display_list = [img, label_regions, label_surfaces, label_layers]
title = ['jpg', 'regions', 'surfaces', 'layers']
plt.figure(figsize=(15, 15))

for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
plt.show()
