import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 創建 RealSense 管道
    pipeline = rs.pipeline()

    # 配置流
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 開始流
    pipeline.start(config)

    try:
        while True:
            # 等待一組幀
            frames = pipeline.wait_for_frames()
            
            # 獲取彩色和深度幀
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 將幀轉換為 numpy 數組
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 將深度圖像標準化為 0-255
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # 顯示彩色和深度圖像
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', images)

            # 按 'q' 鍵退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 停止流並釋放資源
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
