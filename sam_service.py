# sam_service.py

# 导入所需的库
from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from PIL import Image
import numpy as np

# from segment_anything import SamPredictor, sam_model_registry

# 创建Flask应用
app = Flask(__name__)


from PIL import Image
from lang_sam import LangSAM


print('Loading LangSAM...')
model = LangSAM()
print('LangSAM loaded.')

def expand_mask_rectangle(mask, expand_pixels=5):
    """
    基于给定的二值掩码，找到目标的外接矩形，向外扩展指定像素数，生成新的掩码。

    参数：
    - mask: np.array，二值掩码，包含0和1
    - expand_pixels: int，向外扩展的像素数

    返回：
    - new_mask: np.array，扩展后的掩码，类型与输入mask相同
    """
    # 找到mask中值为255的像素的行和列索引
    indices = np.argwhere(mask == int(255))
    if indices.size == 0:
        # 如果没有找到值为1的像素，返回全零掩码
        return np.zeros_like(mask)

    # 获取行和列的最小和最大值
    min_row, min_col = indices.min(axis=0)
    max_row, max_col = indices.max(axis=0)

    # 向外扩展
    min_row = max(min_row - expand_pixels, 0)
    min_col = max(min_col - expand_pixels, 0)
    max_row = min(max_row + expand_pixels, mask.shape[0] - 1)
    max_col = min(max_col + expand_pixels, mask.shape[1] - 1)

    # 创建新的掩码
    new_mask = np.zeros_like(mask)
    new_mask[min_row:max_row+1, min_col:max_col+1] = int(255)

    return new_mask


# 定义分割接口
@app.route('/segment', methods=['POST'])
def segment():
    data = request.get_json()
    image_path = data['image_path']
    text_prompt = data['text_prompt']
    
    # 从路径加载图像并转换为 PIL 格式
    image_pil = Image.open(image_path).convert("RGB")

    # 使用模型进行预测
    results = model.predict([image_pil], [text_prompt])
    masks=results[0]['masks']
    
    #只选择一个物体,后面再进一步丰富
    mask=masks[0]
    # mask=masks[0].transpose(1,0)
    mask[mask != 0] = int(255)
    mask_ori=mask
    
    mask=expand_mask_rectangle(mask,80)
    
    

    # 创建一个新的 numpy 数组，将255变为0，0变为255
    # inverted_mask = 255 - mask

    # 将反转后的 mask 数组转换为 Image 对象
    # image = Image.fromarray(mask.astype(np.uint8))

    # 保存为 PNG 图片
    # image.save('mask_image.png')
    cv2.imwrite('images/mask_ori.png',mask_ori)
    
    cv2.imwrite('images/mask.png',mask)
    
    print('mask image saved')
    return "ok"

if __name__ == '__main__':
    # 运行服务
    app.run(host='127.0.0.1', port=5000)
