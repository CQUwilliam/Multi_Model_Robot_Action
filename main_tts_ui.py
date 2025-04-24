# main.py
# 原错误导入方式
from pyaudio import PyAudio, paInt16
import tkinter as tk
from tkinter import ttk
import threading
import time
# 修正为
import pyaudio
import wave
import speech_recognition as sr
# 导入所需的库
import openai
import requests
import cv2
import pyrealsense2 as rs
import numpy as np
import base64
import custom_urx as urx
import json
from PIL import Image
from scipy.spatial.transform import Rotation as R
from custom_urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

global text_record

# 设置OpenAI的API密钥
openai.api_key = 'YOUR_OPENAI_API_KEY'  # 请替换为您的OpenAI API密钥
class RobotUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("智能机械臂控制系统")
        self.geometry("800x600")
        self.configure(bg="#2E3440")
        
        # 录音控制变量
        self.is_recording = False
        self.audio_frames = []
        
        # 初始化UI组件
        self.create_widgets()
        
    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=6)
        style.configure('TLabel', background='#2E3440', foreground='#D8DEE9')
        
        # 主容器
        main_frame = ttk.Frame(self)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # 状态显示区
        self.status_label = ttk.Label(main_frame, text="就绪", font=('Arial', 14))
        self.status_label.pack(pady=10)
        
        # 波形可视化画布
        self.canvas = tk.Canvas(main_frame, bg="#3B4252", height=100)
        self.canvas.pack(fill=tk.X, pady=10)
        
        # 控制按钮区
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        self.record_btn = ttk.Button(
            btn_frame, 
            text="🎤 开始录音", 
            command=self.toggle_recording,
            style='TButton'
        )
        self.record_btn.pack(side=tk.LEFT, padx=10)
        
        self.task_btn = ttk.Button(
            btn_frame,
            text="🤖 执行任务",
            command=self.start_task,
            state=tk.DISABLED
        )
        self.task_btn.pack(side=tk.LEFT, padx=10)
        
        # 识别结果显示
        self.result_text = tk.Text(
            main_frame, 
            height=8, 
            bg="#434C5E", 
            fg="#D8DEE9",
            font=('Arial', 12)
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
    
    def toggle_recording(self):
        """切换录音状态"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """开始录音"""
        self.is_recording = True
        self.record_btn.config(text="⏹ 停止录音")
        self.status_label.config(text="录音中...")
        self.audio_frames = []
        
        # 启动录音线程
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()
        
        # 启动波形动画
        self.animate_wave()
    
    def stop_recording(self):
        global text_record
        """停止录音"""
        self.is_recording = False
        self.record_btn.config(text="🎤 开始录音")
        self.status_label.config(text="录音结束")
        
        # 保存录音文件
        self.save_recording()
        
        # 语音识别
        text_record = speech_recognition()
        self.show_result(f"识别结果：{text_record}")
        self.task_btn.config(state=tk.NORMAL)

    
    def record_audio(self):
        """录音线程函数"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        while self.is_recording:
            data = stream.read(1024)
            self.audio_frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def save_recording(self):
        """保存录音文件"""
        with wave.open('temp.wav', 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.audio_frames))
    
    def animate_wave(self):
        """波形动画"""
        if self.is_recording:
            self.canvas.delete("wave")
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            
            for i in range(0, width, 10):
                amplitude = np.random.randint(10, 40)
                self.canvas.create_line(
                    i, height/2 - amplitude,
                    i+8, height/2 + amplitude,
                    fill="#88C0D0",
                    tags="wave",
                    width=2
                )
            self.after(100, self.animate_wave)
    
    def show_result(self, text):
        """显示识别结果"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
    
    def start_task(self):
        """启动任务执行"""
        self.task_btn.config(state=tk.DISABLED)
        task_thread = threading.Thread(target=self.run_robot_task)
        task_thread.start()
    
    def run_robot_task(self):
        """机械臂任务线程"""
        # 在这里调用原有的main()函数逻辑
        self.status_label.config(text="任务执行中...")
        try:
            # 调用原有的main函数逻辑
            main()
        except Exception as e:
            self.show_result(f"任务执行出错：{str(e)}")
        finally:
            self.status_label.config(text="任务完成")
def record(DURATION=5):
    """录音函数"""
    CHUNK = 1024
    FORMAT = paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("开始录音...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("录音结束")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 保存为临时文件
    with wave.open('temp.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def speech_recognition():
    """语音识别函数"""
    r = sr.Recognizer()
    with sr.AudioFile('temp.wav') as source:
        audio = r.record(source)
    
    try:
        text = r.recognize_google(audio, language='zh-CN')
        print("识别结果：" + text)
        return text
    except sr.UnknownValueError:
        print("无法识别音频")
        return ""
    except sr.RequestError as e:
        print(f"服务错误：{e}")
        return ""

# 定义GPT-4任务规划函数
def task_planning(task_description):
    """
    使用GPT-4将任务描述转换为可执行的步骤。
    """
    prompt = f"""
将以下任务分解为机器人可执行的步骤，并使用JSON格式输出。每个步骤应包括：
- action：要执行的动作（如detect、grasp、place）
- object：动作涉及的对象
- purpose：如果是检测动作，说明检测的目的（如grasp或place）
- target：如果是放置动作，指明放置的目标

任务描述：
“{task_description}”

请输出JSON格式的步骤列表。
"""
    response = openai.Completion.create(
        engine="gpt-4",  # 如果有GPT-4权限，可以使用"gpt-4"
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0
    )
    steps = response.choices[0].text.strip()
    return steps


# 从Realsense相机获取图像
def get_camera_images(pipeline,align):
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # 将图像转换为numpy数组
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    cv2.imwrite('/home/cavalier/lgx/llmrobot/images/image.png',color_image)
    cv2.imwrite('/home/cavalier/lgx/llmrobot/images/depth.png',depth_image)

# 像素坐标和深度值转换为相机坐标系下的3D坐标
def pixel_to_camera_coords(centroid, depth_value, intrinsics):
    x = (centroid[1] - intrinsics[0][2]) / intrinsics[0][0] * depth_value
    y = (centroid[0] - intrinsics[1][2]) / intrinsics[1][1] * depth_value
    z = depth_value
    return np.array([x, y, z,1]).reshape(4,1)

# 坐标转换
def transform_coordinates(camera_coords):
    # 使用已知的外参矩阵进行转换
    transformation_matrix = np.eye(4)  # 请替换为实际的外参矩阵
    camera_coords_homogeneous = np.append(camera_coords, 1)
    base_coords = transformation_matrix.dot(camera_coords_homogeneous)
    return base_coords[:3]

# 转换抓取姿态
def transform_grasp_pose(grasp_pose):
    position = np.array(grasp_pose['position'])
    rotation_matrix = np.array(grasp_pose['rotation'])
    # 转换位置
    base_position = transform_coordinates(position)
    # 转换方向
    # 需要将旋转矩阵也转换到机械臂基座坐标系
    base_rotation_matrix = rotation_matrix  # 简化处理，需根据实际情况调整
    return {'position': base_position, 'rotation': base_rotation_matrix}

def homogeneous_matrix(rotation, translation):
    """
    根据旋转矩阵和平移向量构造齐次变换矩阵
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T
# 控制UR5机械臂抓取
def execute_grasp(robot,gripper,translation_cam_obj, rotation_cam_obj,width):
    print('\n')
    print('\n')
    print("======正在执行抓取======")
    
    T=np.array([[0,0,1],[0,1,0],[-1,0,0]])  # best
    # T=np.array([[0,1,0],[0,0,1],[1,0,0]])
    # T=np.array([[0,0,1],[0,1,0],[1,0,0]])

    # # 计算右手系旋转矩阵 R_rh
    rotation_cam_obj = rotation_cam_obj @ T


    # 相机到末端的变换矩阵M（4x4）
    M_cam2ee = np.load('/home/cavalier/lgx/llmrobot/config/hand_eye_transform.npy') # 请替换为实际的数值

    # 计算末端到相机的变换矩阵（M_cam2ee的逆）
    # T_ee_cam = np.linalg.inv(M_cam2ee)
    T_ee_cam=M_cam2ee

    # 构造物体在相机坐标系下的齐次变换矩阵
    T_cam_obj = homogeneous_matrix(rotation_cam_obj, translation_cam_obj)

    # 计算物体在末端坐标系下的位姿
    T_ee_obj = np.dot(T_ee_cam, T_cam_obj)



    # 获取当前末端在基座坐标系下的位姿矩阵（4x4）
    T_base_ee = robot.get_pose().get_matrix()

    # 计算物体在基座坐标系下的位姿
    T_base_obj = np.dot(T_base_ee, T_ee_obj)

    # 从T_base_obj中提取旋转矩阵和平移向量
    rotation_base_obj = T_base_obj[:3, :3]
    translation_base_obj = T_base_obj[:3, 3]

    # 将旋转矩阵转换为旋转向量（轴角表示）

    r = R.from_matrix(rotation_base_obj)
    rot_vec = r.as_rotvec()
    
    # tmp=translation_base_obj[0]
    # tmp1=translation_base_obj[0,0]
    # tmp3=translation_base_obj[0][0][0]
    # tmp2=float(translation_base_obj[0][0][0])
    # tmp3=tmp2.item()

    pose = [
    float(translation_base_obj[0,0]),
    float(translation_base_obj[1,0]),
    float(translation_base_obj[2,0])+0.01,
    2.709342051317487, 1.463335081708574, -0.08384753609461765
    # rot_vec[0],
    # rot_vec[1],
    # rot_vec[2]
    # 0
    ]

    # aa = [np.rad2deg(rot_vec[0]), np.rad2deg(rot_vec[1]),np.rad2deg(rot_vec[2])]
    # print(pose)

    # 移动机器人到目标位姿
    robot.movel(pose, acc=0.01, vel=0.08)
    
    # robot.movel(pose, acc=0.1, vel=0.05)
    # 等夹爪好了再写
    gripper.close_gripper()
    

# 控制UR5机械臂放置
def execute_place(robot,gripper, place_pose):
    print('\n')
    print('\n')
    print("======正在执行放置======")
    
    # approach_pose = place_pose + np.array([0, 0, 0.1,0,0,0])  # 上方10cm
    # robot.movel(approach_pose.tolist(), wait=True)
    robot.movel(place_pose, acc=0.01,vel=0.08)
    # 控制夹爪打开
    gripper.open_gripper()
    

# 主程序
def main():
    # # -------------------------------tts-------------------------------------
    # # 语音/指令输入模块
    # while True:
    #     start_record_ok = input('是否开启录音？\n'
    #                             '输入数字录音指定时长（秒）\n'
    #                             '按 k 使用键盘输入\n'
    #                             '按 c 使用默认指令\n'
    #                             '>>> ')
        
    #     task_description = ""
    #     if str.isnumeric(start_record_ok):
    #         DURATION = int(start_record_ok)
    #         record(DURATION=DURATION)
    #         task_description = speech_recognition()
    #     elif start_record_ok.lower() == 'k':
    #         task_description = input("请输入任务指令：")
    #     elif start_record_ok.lower() == 'c':
    #         task_description = "将绿色苹果放到红色杯子上"
    #     else:
    #         print("无效输入，请重新选择")
    #         continue
        
    #     if task_description.strip():
    #         break
    # # -------------------------------tts-------------------------------------
    # print('\n')
    print('\n')
    print("======开始执行任务======")
    hand_eye_transform = np.load('/home/cavalier/lgx/llmrobot/config/hand_eye_transform.npy')
    # 初始化UR5机械臂
    robot = urx.Robot("192.168.1.40")  # 请替换为UR5的实际IP地址
    gripper=Robotiq_Two_Finger_Gripper(robot)
    init_pose=[-0.07408505689348382, 0.48661989071612584, 0.23827063269992171, 2.264987695687536, 2.147798631721059, 0.10441536241257253]
    robot.movel(init_pose,acc=0.1,vel=0.08)
    gripper.open_gripper()
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)

    # 对齐深度帧到彩色帧
    align_to = rs.stream.color
    align = rs.align(align_to)
    # 捕获一帧RGB和深度图像
    for _ in range(30):  # 跳过前面的几帧，确保图像稳定
        frames = pipeline.wait_for_frames()
        
    intrinsics = [[906.6353759765625, 0, 654.9713745117188], [0, 906.5818481445312, 358.79400634765625], [0, 0, 1]]

    # # 获取任务描述
    # task_description = "将红色方块放到绿色方块上"
    #
    # # 使用GPT-4进行任务规划
    # steps = task_planning(task_description)
    # print("任务规划结果：", steps)
    
    ## try
    steps=[
  {
    "action": "detect",
    "object": "green apple",
    "purpose": "grasp"
  },
  {
    "action": "grasp",
    "object": "green apple"
  },
  {
    "action": "detect",
    "object": "red cup",
    "purpose": "place"
  },
  {
    "action": "place",
    "object": "green apple",
    "target": "red cup"
  }]
  

  
    steps_2=[
  {
    "action": "detect",
    "object": "red apple",
    "purpose": "grasp"
  },
  {
    "action": "grasp",
    "object": "red apple"
  },
  {
    "action": "detect",
    "object": "green cup",
    "purpose": "place"
  },
  {
    "action": "place",
    "object": "red apple",
    "target": "red cup"
  }
]


    # 解析步骤
    # steps = json.loads(steps)

    # 初始化变量
    mask = None
    place_position = None

    # 在此处判断一下用steps还是steps2(红放绿还是绿放红)


    for step in steps:
        action = step['action']
        if action == 'detect':
            print('\n')
            print('\n')
            print("======正在执行检测程序======")
            
            robot.movel(init_pose,vel=0.08,acc=0.1)
            object_name = step['object']
            purpose = step.get('purpose', '')
            # 获取相机图像
            get_camera_images(pipeline,align)
            # 调用SAM服务
            sam_payload = {
                'image_path': '/home/cavalier/lgx/llmrobot/images/image.png',
                'text_prompt': object_name
            }
            sam_response = requests.post('http://localhost:5000/segment', json=sam_payload)
            
            if sam_response.status_code == 200:
                mask_path='/home/cavalier/lgx/llmrobot/images/mask.png'
                if purpose == 'grasp':
                    pass
                elif purpose == 'place':
                    mask = np.array(Image.open(mask_path))
                    # 计算放置位置
                    indices = np.argwhere(mask)
                    centroid = indices.mean(axis=0)
                    depth_image= np.array(Image.open('/home/cavalier/lgx/llmrobot/images/depth.png'))
                    depth_value = depth_image[int(centroid[0]), int(centroid[1])]/1000.0 # x0.001?
                    # 将像素坐标和深度值转换为相机坐标系下的3D坐标
                    point_in_camera = pixel_to_camera_coords(centroid, depth_value, intrinsics)
                    # 转换到机械臂基座坐标系
                    # print(object_name)
                    # 获取当前机械臂末端位姿
                    # tcp_pose_list = list(robot.get_pos())  # 返回[x, y, z, rx, ry, rz]
                    tcp_pose_list = robot.getl()
                    # 提取位置和方向
                    position = np.array(tcp_pose_list[:3])  # [x, y, z]
                    orientation = np.array(tcp_pose_list[3:])  # [rx, ry, rz]，旋转向量
                    # orientation[:2] = 0

                    # 将旋转向量转换为旋转矩阵
                    rotation_vector = orientation  # 旋转向量
                    rotation = R.from_rotvec(rotation_vector)
                    rotation_matrix = rotation.as_matrix()  # 3x3旋转矩阵

                    # 构建4x4齐次变换矩阵
                    # tcp_pose_matrix = np.eye(4)
                    # tcp_pose_matrix[:3, :3] = rotation_matrix
                    # tcp_pose_matrix[:3, 3] = position

                    T_base_tcp = np.eye(4)
                    T_base_tcp[:3, :3] = rotation_matrix
                    T_base_tcp[:3, 3] = position
                    # print(f'T_base_tcp{T_base_tcp}')

                    # 计算目标点在机械臂基座坐标系下的坐标
                    # T_base_tcp * T_tcp_cam * point_in_camera = point_in_base
                    point_in_base = T_base_tcp @ hand_eye_transform @ point_in_camera
                    
                    place_pose=robot.getl()
                    place_pose[:3]=point_in_base[:3].flatten()
                    # 在上面3cm放
                    place_pose[2]+=0.04
                    # print(f'放置高度{place_pose}')
                    
                    # place_position = point_in_base[:3].flatten()
            else:
                print(f"SAM服务调用失败：{sam_response.status_code}")
        elif action == 'grasp':
            # 编码RGB图像、深度图像和掩码
            # 调用GraspNet服务
            graspnet_payload = {
                'color_image_path': '/home/cavalier/lgx/llmrobot/images/image.png',
                'depth_image_path': '/home/cavalier/lgx/llmrobot/images/depth.png',
                'mask_path':'/home/cavalier/lgx/llmrobot/images/mask.png'
            }
            graspnet_response = requests.post('http://localhost:6001/grasp', json=graspnet_payload)
            if graspnet_response.status_code == 200:
                translation=np.load('/home/cavalier/lgx/llmrobot/gripper_info/translation.npy')
                rotation=np.load('/home/cavalier/lgx/llmrobot/gripper_info/rotation.npy')
                width=np.load('/home/cavalier/lgx/llmrobot/gripper_info/width.npy')
                if translation is not None:
                    # 执行抓取
                    execute_grasp(robot,gripper,translation, rotation, width)
                else:
                    print("未能获取抓取姿态")
            else:
                print(f"GraspNet服务调用失败：{graspnet_response.status_code}")

        elif action == 'place':
            if place_pose is not None:
                # 执行放置
                execute_place(robot,gripper, place_pose)
            else:
                print("未找到放置位置，无法执行放置")
        else:
            print(f"未知的动作：{action}")
    
    # time.sleep(30)
    # get_camera_images(pipeline,align)

    # 停止相机
    pipeline.stop()

    # 关闭机械臂连接
    robot.close()

if __name__ == "__main__":
    # main()
    ui = RobotUI()
    ui.mainloop()
