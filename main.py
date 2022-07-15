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

from segmentation_models import Unet
from segmentation_models import get_preprocessing

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)


import glob

# データセットの準備(今回は3種類のマスクの内、surfacesを選択)

# 画像の一覧取得
images = sorted(glob.glob(f'./iccv09Data/images/*.jpg'))
labels = sorted(glob.glob(f'./iccv09Data/labels/*.surfaces.txt'))

x = []
y = []
classes = 3                 # クラス数
ratio = 0.8                 # 学習データの割合
input_shape = (224, 224)    # 32の倍数でないといけない https://github.com/qubvel/segmentation_models/issues/1

# 入力画像
for img_path in images:
    img = cv2.imread(img_path)
    img = cv2.resize(img, input_shape)      # 入力サイズに変換
    img = np.array(img, dtype=np.float32)   # float形に変換
    img *= 1./255                           # 0～1に正規化
    x.append(img)

# 正解ラベル
for label_path in labels:
    label = np.loadtxt(label_path)
    label = cv2.resize(label, input_shape)  # 入力サイズに変換
    img = []
    for label_index in range(classes):      # 各クラスごとに画像を作成（クラス0であれば、元のマスク0の部分が1、それ以外は0の画像となる）
        img.append(label == label_index)
    img = np.array(img, np.float32)         # float形に変換
    img = img.transpose(1, 2, 0)            # (クラス数, 224, 224) => (224, 224, クラス数)
    y.append(img)

x = np.array(x)
y = np.array(y)
x = preprocess_input(x)

# データを分割
p = int(ratio * len(x))
x_train = x[:p]
y_train = y[:p]
x_val = x[p:]
y_val = y[p:]

# チュートリアル=>シンプルなトレーニングパイプラインを参考

# モデルを定義
model = Unet(BACKBONE, classes=classes, encoder_weights=None)
model.compile('Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 学習
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=20,
    validation_data=(x_val, y_val)
)

# 学習曲線のグラフ
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))

# [左側] metricsについてのグラフ
axL.plot(history.history['accuracy'])
axL.plot(history.history['val_accuracy'])
axL.grid(True)
axL.set_title('Accuracy_vs_Epoch')
axL.set_ylabel('accuracy')
axL.set_xlabel('epoch')
axL.legend(['train', 'val'], loc='upper left')

# [右側] lossについてのグラフ
axR.plot(history.history['loss'])
axR.plot(history.history['val_loss'])
axR.grid(True)
axR.set_title("Loss_vs_Epoch")
axR.set_ylabel('loss')
axR.set_xlabel('epoch')
axR.legend(['train', 'val'], loc='upper left')

# グラフを表示
plt.show()

# 結果を確認
num = 0                                                     # 確認したい画像を指定
input_img = x_train[num]                                    # 入力画像
true_img = cv2.resize(np.loadtxt(labels[num]), input_shape) # 正解マスク
preds = model.predict(x_train[num][np.newaxis, ...])        # 予測（長さ１の配列で渡す）
pred_img = np.argmax(preds[0], axis=2)                      # 予測マスク (224, 224, クラス数) => (224, 224)

# 結果表示
display_list = [input_img, true_img, pred_img]
title = ['Input Image', 'True Mask', 'Predicted Mask']

plt.figure(figsize=(15, 15))
for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
plt.show()