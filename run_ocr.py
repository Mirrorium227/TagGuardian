import os
import sys
import argparse

def main():
    # 默认参数设置
    parser = argparse.ArgumentParser(description='运行PaddleOCR识别')
    parser.add_argument('--image_path', type=str, default='4.png', help='要识别的图片路径')
    parser.add_argument('--det_model_dir', type=str, default='ch_PP-OCRv4_det_infer', help='检测模型目录')
    parser.add_argument('--rec_model_dir', type=str, default='ch_PP-OCRv4_rec_infer', help='识别模型目录')
    parser.add_argument('--use_angle_cls', action='store_true', help='是否使用方向分类器')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU')
    
    args = parser.parse_args()
    
    # 构建命令
    cmd = f'python tools/infer/predict_system.py --image_dir="{args.image_path}" --det_model_dir="{args.det_model_dir}" --rec_model_dir="{args.rec_model_dir}"'
    
    # 添加可选参数
    if args.use_angle_cls:
        cmd += ' --use_angle_cls=true'
    
    if args.use_gpu:
        cmd += ' --use_gpu=true'
    
    # 执行命令
    print(f"执行命令: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()