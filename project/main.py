import os
import numpy as np
import screenshot_multi
from final_resut_GUI import show_predict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#放擷取+二質化之後的截圖
dir = 'final_result/'
#路徑的head部分(不包含檔案名稱)
img_path = os.listdir(dir.split('/')[0])
#從檔案名稱中拆出字體名稱
for i in range(len(img_path)):
    img_path[i] = dir + img_path[i]
predict_all = []
predict_total = np.zeros(87, dtype=float)
#截圖後放入模型內運行
for img_name in img_path:
    #載入圖片並放入模型中預測
    network = load_model('all.h5')
    img = image.load_img(img_name, target_size=(100, 100), color_mode="grayscale")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    pred = network.predict(img)[0]
    #有超過60%以上的準確度後列出相似度最高前幾名
    if max(pred) >= 0.6:
        network = load_model('letter_all.h5')
        img = image.load_img(img_name, target_size=(100, 100), color_mode="grayscale")
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255
        pred = network.predict(img)[0]
        predict_total = predict_total + pred
        pred = pred.tolist()
        predict_all.append(pred)

top_idx = predict_total.argsort()[-1:-4:-1]
predict_total = predict_total/len(predict_all)
TopBER = predict_total[top_idx]
#顯示結果
show_predict(top_idx, TopBER)
#將對應字體檔案放入暫存資料夾中
if os.path.exists('fonts'):
    for fileName in os.listdir('fonts'):
        os.remove('fonts/' + fileName)
    os.rmdir('fonts')

screenshot_multi.clean_file()
print()
