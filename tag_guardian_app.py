import sys
import os
import cv2
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QTextEdit, QSplitter, QProgressBar, QMessageBox,
                            QRadioButton, QButtonGroup, QGroupBox, QLineEdit,
                            QCheckBox, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
import subprocess
from openai import OpenAI
import openpyxl # 新增：导入openpyxl库
from openpyxl.utils.exceptions import InvalidFileException # 新增：处理无效文件异常
import re # 新增：导入re库

# ==============================================================================
# OCR处理线程
# 把OCR这种耗时的操作放到单独的线程里，防止主界面卡死，提升用户体验。
# ==============================================================================
class OCRThread(QThread):
    """
    OCR处理线程，专门负责调用外部的PaddleOCR脚本进行图像文字识别。
    通过信号(pyqtSignal)与主线程通信，更新进度、状态和返回识别结果。
    """
    progress_update = pyqtSignal(int) # 进度更新信号，传递整数表示百分比
    ocr_complete = pyqtSignal(str)    # OCR完成信号，传递字符串表示识别结果或错误信息
    status_update = pyqtSignal(str)   # 状态更新信号，传递字符串描述当前操作

    def __init__(self, image_path, det_model_dir, rec_model_dir, use_angle_cls, use_gpu):
        super().__init__()
        self.image_path = image_path            # 待识别图片路径
        self.det_model_dir = det_model_dir      # OCR检测模型路径
        self.rec_model_dir = rec_model_dir      # OCR识别模型路径
        self.use_angle_cls = use_angle_cls      # 是否使用角度分类器
        self.use_gpu = use_gpu                  # 是否使用GPU进行推理

    def run(self):
        """线程执行的核心逻辑"""
        self.status_update.emit("OCR正在识别...") # 更新状态：开始识别
        self.progress_update.emit(30) # 更新进度

        # 核心步骤：构建调用PaddleOCR脚本的命令
        # 这里用f-string动态构建命令，包含了图片路径和模型路径等参数
        cmd = f'python tools/infer/predict_system.py --image_dir="{self.image_path}" --det_model_dir="{self.det_model_dir}" --rec_model_dir="{self.rec_model_dir}"'
        
        if self.use_angle_cls:
            cmd += ' --use_angle_cls=true' # 如果启用了角度分类
        
        if self.use_gpu:
            cmd += ' --use_gpu=true' # 如果启用了GPU
        
        self.progress_update.emit(50) # 更新进度

        # 执行命令并捕获标准输出和标准错误
        # 使用subprocess.Popen来异步执行，并通过communicate()获取结果
        # text=True 和 encoding='utf-8' 确保输出是文本格式且编码正确
        try:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            stdout, stderr = process.communicate() # 等待命令执行完成

            # 检查OCR脚本的返回码，非0通常表示执行出错
            if process.returncode != 0:
                error_message = f"OCR脚本执行失败。\n错误信息：\n{stderr}" # 优先使用stderr中的错误信息
                self.ocr_complete.emit(error_message)
                self.status_update.emit("OCR识别失败")
                self.progress_update.emit(100) # 确保进度条走完
                return # 提前退出

        except FileNotFoundError: # 特定异常：如果python或脚本路径找不到
            self.ocr_complete.emit("OCR处理失败：未找到 predict_system.py 脚本。请检查路径。")
            self.status_update.emit("OCR脚本未找到")
            self.progress_update.emit(100)
            return
        except Exception as e: # 其他未知异常
            self.ocr_complete.emit(f"OCR处理发生未知错误：{str(e)}")
            self.status_update.emit("OCR未知错误")
            self.progress_update.emit(100)
            return
            
        self.progress_update.emit(80) # 更新进度

        # OCR脚本执行成功后，读取它生成的结果文件
        # 结果文件路径是固定的，在 "inference_results/system_results.txt"
        result_file = os.path.join("inference_results", "system_results.txt")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.ocr_complete.emit(content) # 发送OCR结果
            except Exception as e: # 文件读取也可能发生异常
                self.ocr_complete.emit(f"读取OCR结果文件失败: {str(e)}")
        else:
            # 如果结果文件不存在，需要通知用户
            # 同时，如果stderr有内容，也一并显示，可能有助于排查问题
            if stderr: # stderr可能包含脚本内部的警告或非致命错误
                 self.ocr_complete.emit(f"OCR处理完成，但未找到结果文件。\n脚本错误信息：\n{stderr.strip()}")
            else:
                 self.ocr_complete.emit("OCR处理完成，但未找到结果文件。")
            
        self.progress_update.emit(100) # 最终进度
        self.status_update.emit("OCR识别完成") # 最终状态

# ==============================================================================
# 大模型处理线程
# 同样是为了避免UI卡顿，将调用大模型API的操作放到独立线程
# ==============================================================================
class AIModelThread(QThread):
    """
    大模型处理线程，用于调用AI模型API（如硅基流动或Ollama）对OCR结果进行校正和信息提取。
    通过信号与主线程通信。
    """
    progress_update = pyqtSignal(int) # 进度更新
    ai_complete = pyqtSignal(str)     # AI处理完成，返回处理结果或错误
    status_update = pyqtSignal(str)   # 状态更新

    def __init__(self, ocr_text, model_type, api_key=None, model_name=None, ollama_url=None):
        super().__init__()
        self.ocr_text = ocr_text        # OCR识别出的原始文本
        self.model_type = model_type    # 大模型类型："siliconflow" 或 "ollama"
        self.api_key = api_key          # API密钥 (主要用于siliconflow)
        self.model_name = model_name    # 使用的具体模型名称
        self.ollama_url = ollama_url    # Ollama服务的URL (如果使用Ollama)

    def run(self):
        """线程执行的核心逻辑：调用大模型API"""
        self.status_update.emit("大模型正在修正...")
        self.progress_update.emit(30)

        try:
            # 系统提示词System Prompt
            system_prompt = (
                "你是一个专业的商超价签信息提取助手。"
                "请从以下OCR识别结果中，提取并整理出商品的基本信息。"
                "要求：1. 提取商品名称、规格、单位和价格四项关键信息"
                "2. 按照固定格式，每行输出一个内容输出：\n"
                "- 商品名称：[名称]\n"
                "- 规格：[规格]\n"
                "- 单位：[单位]\n"
                "- 价格：[价格]元。\n"
                "OCR识别结果：[OCR结果内容]" # 这个占位符会在后面被实际OCR文本替换，或者在构建messages时处理
            )
            
            # 准备用户输入内容，将OCR文本嵌入
            full_user_content = f"OCR识别结果：\n{self.ocr_text}"

            if self.model_type == "siliconflow":
                # --------------------------------------------------------------
                # 调用硅基流动 (SiliconFlow) API
                # --------------------------------------------------------------
                client = OpenAI(
                    api_key=self.api_key, 
                    base_url="https://api.siliconflow.cn/v1" # 硅基流动的API基地址
                )
                
                self.progress_update.emit(50)
                
                # 调用chat completions接口
                response = client.chat.completions.create(
                    model=self.model_name, # 使用指定的模型
                    messages=[
                        # 这里的system_prompt处理，如果OCR内容较长，直接替换占位符可能导致prompt过长
                        # 更稳妥的方式是将OCR内容作为用户消息的一部分，或者如果API支持，作为单独的上下文
                        # 当前实现是将OCR内容放在用户消息中，system_prompt中的占位符可以移除
                        {'role': 'system', 'content': system_prompt.replace("[OCR结果内容]", "")}, 
                        {'role': 'user', 'content': full_user_content}
                    ],
                    stream=False # 非流式输出，一次性返回结果
                )
                
                self.progress_update.emit(80)
                
                # 解析返回结果
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    result = response.choices[0].message.content
                    self.ai_complete.emit(result)
                else:
                    self.ai_complete.emit("大模型未返回有效结果") # API调用成功但内容为空或格式不对
                    
            elif self.model_type == "ollama":
                # --------------------------------------------------------------
                # 调用本地 Ollama API
                # --------------------------------------------------------------
                import requests # requests库只在Ollama模式下使用，所以在这里导入
                
                self.progress_update.emit(50)
                
                url = f"{self.ollama_url}/api/chat" # Ollama的聊天API端点
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt.replace("[OCR结果内容]", "")},
                        {"role": "user", "content": full_user_content}
                    ],
                    "stream": False # 非流式
                }
                
                # 发送POST请求
                response = requests.post(url, json=payload)
                response.raise_for_status() # 如果HTTP状态码是4xx或5xx，则抛出异常
                
                self.progress_update.emit(80)
                
                result_json = response.json() # 解析JSON响应
                if "message" in result_json and "content" in result_json["message"]:
                    self.ai_complete.emit(result_json["message"]["content"])
                else:
                    self.ai_complete.emit("Ollama返回格式异常") # JSON结构不符合预期
            else:
                self.ai_complete.emit("未知的模型类型") # 配置错误
                
        except requests.exceptions.RequestException as e: # 处理requests库可能抛出的网络相关异常
            self.ai_complete.emit(f"Ollama请求失败: {str(e)}")
        except Exception as e: # 捕获其他所有可能的异常
            self.ai_complete.emit(f"大模型处理出错: {str(e)}")
            
        self.progress_update.emit(100)
        self.status_update.emit("大模型修正完成")

# ==============================================================================
# 主应用程序类
# 继承自QMainWindow，是整个GUI的骨架
# ==============================================================================
class TagGuardianApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- 实例变量初始化 ---
        self.image_path = None          # 当前选择的图片路径
        self.ocr_result = None          # OCR线程返回的原始结果
        self.ai_corrected_text = None   # AI线程返回的修正后文本
        self.conversations = {}         # (未使用)原意可能是存储对话历史
        self.current_image_name = None  # 当前处理的图片文件名，用于显示或保存
        self.is_one_click_active = False # 新增：用于标记是否正在执行一键操作流程

        # 使用QSettings持久化用户配置，比如API密钥、模型选择等
        self.settings = QSettings("TagGuardian", "OCRApp") # "公司名", "应用名"
        
        self.excel_file_path = "tag_guardian_data.xlsx" # 定义Excel数据文件的固定名称

        # --- 初始化顺序很重要 ---
        self.initUI() # 1. 先初始化UI控件，这样status_text等控件才存在
        self.init_excel_file() # 2. 然后初始化Excel文件，这里会用到status_text更新状态
        self.loadSettings() # 3. 最后加载保存的设置并应用到UI
        
    def init_excel_file(self):
        """
        初始化Excel文件。
        如果文件不存在，则创建一个新的，并写入表头。
        如果文件已存在，则尝试加载它。
        包含基本的错误处理，如权限问题。
        """
        try:
            if not os.path.exists(self.excel_file_path):
                # 文件不存在，创建新的工作簿和工作表
                workbook = openpyxl.Workbook()
                sheet = workbook.active # 获取活动工作表
                headers = ["品名", "规格", "单位", "价格"] # 定义表头
                sheet.append(headers) # 写入表头行
                workbook.save(self.excel_file_path) # 保存文件
                self.update_status(f"已创建新的Excel文件: {self.excel_file_path}")
            else:
                # 文件已存在，可以进行一些检查，比如表头是否完整等（当前未实现）
                self.update_status(f"已加载Excel文件: {self.excel_file_path}")
        except PermissionError: # 文件权限不足
            QMessageBox.critical(self, "Excel错误", f"无法访问Excel文件 {self.excel_file_path}。\n请确保文件未被其他程序占用且具有写入权限。")
            self.update_status(f"Excel文件访问权限错误")
        except Exception as e: # 其他创建/加载Excel时的错误
            QMessageBox.critical(self, "Excel错误", f"初始化Excel文件时发生错误: {str(e)}")
            self.update_status(f"Excel文件初始化错误")

    def initUI(self):
        # --- 基本窗口设置 ---
        self.setWindowTitle('TagGuardian - OCR与文本校正工具')
        self.setGeometry(100, 100, 1200, 800) # 初始位置和大小
        
        # 主布局，垂直排列
        main_layout = QVBoxLayout()
        
        # 后面很多控件都会用到这个字体，统一设置黑体
        bold_font = QFont("SimHei") 
        bold_font.setBold(False)
        bold_font.setPointSize(10)
        
        config_layout = QHBoxLayout() # 水平排列OCR配置和大模型配置
        
        # OCR配置区 (左侧)
        ocr_group = QGroupBox("OCR配置")
        ocr_group.setFont(bold_font) # 组标题也用粗体
        ocr_layout = QVBoxLayout()
        
        # OCR设备选择：GPU还是CPU
        self.gpu_radio = QRadioButton("使用GPU")
        self.gpu_radio.setFont(bold_font)
        self.cpu_radio = QRadioButton("使用CPU")
        self.cpu_radio.setFont(bold_font)
        self.cpu_radio.setChecked(True)  # 默认选CPU
        
        ocr_device_group = QButtonGroup(self) # 确保单选
        ocr_device_group.addButton(self.gpu_radio)
        ocr_device_group.addButton(self.cpu_radio)
        
        ocr_layout.addWidget(self.gpu_radio)
        ocr_layout.addWidget(self.cpu_radio)
        ocr_layout.addStretch() # 占位符，让控件靠上
        ocr_group.setLayout(ocr_layout)
        
        # 大模型配置区 (右侧)
        model_group = QGroupBox("大模型配置")
        model_group.setFont(bold_font)
        model_layout = QVBoxLayout()
        
        # 大模型类型选择：硅基流动还是Ollama
        model_type_layout = QHBoxLayout()
        self.siliconflow_radio = QRadioButton("硅基流动")
        self.siliconflow_radio.setFont(bold_font)
        self.ollama_radio = QRadioButton("Ollama")
        self.ollama_radio.setFont(bold_font)
        
        model_type_group = QButtonGroup(self) # 确保单选
        model_type_group.addButton(self.siliconflow_radio)
        model_type_group.addButton(self.ollama_radio)
        
        model_type_layout.addWidget(self.siliconflow_radio)
        model_type_layout.addWidget(self.ollama_radio)
        model_layout.addLayout(model_type_layout)
        
        # 硅基流动 API 的具体配置项
        self.siliconflow_config = QWidget() # 用一个QWidget包起来，方便整体显示/隐藏
        siliconflow_layout = QGridLayout() # 网格布局，适合标签+输入框
        
        siliconflow_layout.addWidget(QLabel("API密钥:"), 0, 0)
        self.siliconflow_api_key = QLineEdit()
        self.siliconflow_api_key.setEchoMode(QLineEdit.Password) # 密钥嘛，输的时候隐藏掉
        self.siliconflow_api_key.setFont(bold_font)
        siliconflow_layout.addWidget(self.siliconflow_api_key, 0, 1)
        
        siliconflow_layout.addWidget(QLabel("模型名称:"), 1, 0)
        self.siliconflow_model_name = QLineEdit("Qwen/Qwen2.5-7B-Instruct") # 给个默认模型
        self.siliconflow_model_name.setFont(bold_font)
        siliconflow_layout.addWidget(self.siliconflow_model_name, 1, 1)
        
        self.siliconflow_config.setLayout(siliconflow_layout)
        
        # Ollama 的具体配置项
        self.ollama_config = QWidget() # 同样用QWidget包起来
        ollama_layout = QGridLayout()
        
        ollama_layout.addWidget(QLabel("Ollama URL:"), 0, 0)
        self.ollama_url = QLineEdit("http://localhost:11434") # Ollama默认地址
        self.ollama_url.setFont(bold_font)
        ollama_layout.addWidget(self.ollama_url, 0, 1)
        
        ollama_layout.addWidget(QLabel("模型名称:"), 1, 0)
        self.ollama_model_name = QLineEdit("llama3") # Ollama常用的模型
        self.ollama_model_name.setFont(bold_font)
        ollama_layout.addWidget(self.ollama_model_name, 1, 1)
        
        self.ollama_config.setLayout(ollama_layout)
        
        # 默认情况下，具体的API配置先藏起来，根据用户选择再显示
        self.siliconflow_config.hide()
        self.ollama_config.hide()
        
        # 联动：当选择不同的大模型类型时，显示/隐藏对应的配置项
        self.siliconflow_radio.toggled.connect(self.toggle_model_config)
        self.ollama_radio.toggled.connect(self.toggle_model_config)
        
        model_layout.addWidget(self.siliconflow_config)
        model_layout.addWidget(self.ollama_config)
        model_layout.addStretch() # 占位符
        model_group.setLayout(model_layout)
        
        # 把OCR配置组和大模型配置组添加到顶部的config_layout
        config_layout.addWidget(ocr_group, 1) # 参数1代表拉伸因子
        config_layout.addWidget(model_group, 2) # 参数2代表拉伸因子，让大模型配置区更宽些
        
        top_layout = QHBoxLayout() # 水平排列图片区和按钮区
        
        # 左侧图片显示区域
        self.image_label = QLabel("请选择图片") # 初始提示
        self.image_label.setAlignment(Qt.AlignCenter) # 居中显示
        self.image_label.setStyleSheet("border: 2px dashed #cccccc; background-color: #f5f5f5;") # 给点样式，像个占位框
        self.image_label.setMinimumSize(400, 300) # 最小尺寸
        self.image_label.setFont(bold_font)
        
        # 右侧按钮区域
        button_layout = QVBoxLayout() # 按钮垂直排列
        
        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.select_image) # 点击就调用select_image方法
        self.select_image_btn.setMinimumHeight(40) # 按钮大一点好看
        self.select_image_btn.setFont(bold_font)
        
        self.run_ocr_btn = QPushButton("运行OCR识别")
        self.run_ocr_btn.clicked.connect(self.run_ocr)
        self.run_ocr_btn.setMinimumHeight(40)
        self.run_ocr_btn.setEnabled(False) # 初始不可用，选了图片才能点
        self.run_ocr_btn.setFont(bold_font)

        self.run_ai_btn = QPushButton("大模型校正")
        self.run_ai_btn.clicked.connect(self.run_ai_model)
        self.run_ai_btn.setMinimumHeight(40)
        self.run_ai_btn.setEnabled(False) # 初始不可用，OCR完了才能点
        self.run_ai_btn.setFont(bold_font)
        
        self.save_result_btn = QPushButton("保存结果") # 保存到文本文件
        self.save_result_btn.clicked.connect(self.save_result)
        self.save_result_btn.setMinimumHeight(40)
        self.save_result_btn.setEnabled(False) # 初始不可用，AI校正完了才能点
        self.save_result_btn.setFont(bold_font)
        
        self.save_to_excel_btn = QPushButton("存储入库") # 保存到Excel
        self.save_to_excel_btn.clicked.connect(self.save_to_excel)
        self.save_to_excel_btn.setMinimumHeight(40)
        self.save_to_excel_btn.setEnabled(False) # 初始不可用，AI校正完了才能点
        self.save_to_excel_btn.setFont(bold_font)
        
        # 状态显示 (例如："正在识别...", "已完成")
        status_layout = QHBoxLayout()
        status_label = QLabel("处理状态:")
        status_label.setFont(bold_font)
        self.status_text = QLabel("就绪") # 初始状态
        self.status_text.setFont(bold_font)
        self.status_text.setStyleSheet("color: blue;") # 状态用蓝色字
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.status_text)
        
        # 进度条
        progress_layout = QHBoxLayout()
        progress_label = QLabel("处理进度:")
        progress_label.setFont(bold_font)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0) # 初始进度0
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
        
        # 把按钮、状态、进度条都加到右侧的button_layout
        button_layout.addWidget(self.select_image_btn)
        button_layout.addWidget(self.run_ocr_btn)
        button_layout.addWidget(self.run_ai_btn)
        button_layout.addWidget(self.save_result_btn)
        button_layout.addWidget(self.save_to_excel_btn)
        button_layout.addStretch() # 占位符，让按钮靠上
        button_layout.addLayout(status_layout) # 状态显示在按钮下方
        button_layout.addLayout(progress_layout) # 进度条在状态下方
        
        # 把图片区和按钮区添加到中部的top_layout
        top_layout.addWidget(self.image_label, 7) # 图片区占7份宽度
        top_layout.addLayout(button_layout, 3)    # 按钮区占3份宽度
        
        # --- 底部区域 (OCR结果 和 AI校正结果) ---
        splitter = QSplitter(Qt.Horizontal) # 可拖动的分割器，左右两边
        
        self.ocr_text = QTextEdit() # 左边显示OCR原始结果
        self.ocr_text.setReadOnly(True) # 只读
        self.ocr_text.setPlaceholderText("OCR识别结果将显示在这里") # 提示文字
        self.ocr_text.setFont(bold_font)
        
        self.ai_text = QTextEdit() # 右边显示AI校正后的结果
        self.ai_text.setReadOnly(True) # 只读
        self.ai_text.setPlaceholderText("大模型校正结果将显示在这里") # 提示文字
        self.ai_text.setFont(bold_font)
        
        splitter.addWidget(self.ocr_text)
        splitter.addWidget(self.ai_text)
        splitter.setSizes([600, 600]) # 初始时左右两边一样宽
        
        main_layout.addLayout(config_layout, 2) # 顶部配置区，占2份高度
        main_layout.addLayout(top_layout, 5)    # 中部图片和按钮区，占5份高度
        main_layout.addWidget(splitter, 5)      # 底部文本区，占5份高度
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout) # 把主布局设置给一个QWidget
        self.setCentralWidget(central_widget) # 再把这个QWidget设为中央部件
        
    def toggle_model_config(self):
        """根据选择的模型类型显示对应的配置"""
        # 这个逻辑很简单，哪个radio被选中，就显示哪个的config，隐藏另一个
        if self.siliconflow_radio.isChecked():
            self.siliconflow_config.show()
            self.ollama_config.hide()
        elif self.ollama_radio.isChecked():
            self.siliconflow_config.hide()
            self.ollama_config.show()
    
    def loadSettings(self):
        """加载保存的设置 - 程序启动时恢复上次的配置"""
        # OCR设置 - GPU/CPU选择
        use_gpu = self.settings.value("ocr/use_gpu", False, type=bool) # 读"ocr/use_gpu"键，默认False
        if use_gpu:
            self.gpu_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)
            
        # 大模型设置 - 类型选择 (siliconflow/ollama)
        model_type = self.settings.value("model/type", "siliconflow", type=str) # 默认"siliconflow"
        if model_type == "siliconflow":
            self.siliconflow_radio.setChecked(True)
        else:
            self.ollama_radio.setChecked(True)
            
        # 硅基流动具体设置 - API密钥和模型名
        self.siliconflow_api_key.setText(self.settings.value("siliconflow/api_key", "", type=str))
        self.siliconflow_model_name.setText(self.settings.value("siliconflow/model_name", "Qwen/Qwen2.5-7B-Instruct", type=str))
        
        # Ollama具体设置 - URL和模型名
        self.ollama_url.setText(self.settings.value("ollama/url", "http://localhost:11434", type=str))
        self.ollama_model_name.setText(self.settings.value("ollama/model_name", "llama3", type=str))
        
        # 根据加载的设置，触发一次UI更新，显示对应的配置项
        self.toggle_model_config()
        
    def saveSettings(self):
        """保存当前设置 - 程序关闭前或手动保存时调用"""
        # OCR设置
        self.settings.setValue("ocr/use_gpu", self.gpu_radio.isChecked())
        
        # 大模型类型
        if self.siliconflow_radio.isChecked():
            self.settings.setValue("model/type", "siliconflow")
        else:
            self.settings.setValue("model/type", "ollama")
            
        # 硅基流动配置
        self.settings.setValue("siliconflow/api_key", self.siliconflow_api_key.text())
        self.settings.setValue("siliconflow/model_name", self.siliconflow_model_name.text())
        
        # Ollama配置
        self.settings.setValue("ollama/url", self.ollama_url.text())
        self.settings.setValue("ollama/model_name", self.ollama_model_name.text())
        
    def closeEvent(self, event):
        """重写窗口关闭事件，确保在关闭前保存设置"""
        self.saveSettings() # 调用保存设置的函数
        event.accept()      # 接受关闭事件，程序正常退出
        
    def select_image(self):
        """弹出文件对话框让用户选择图片"""
        options = QFileDialog.Options()
        # 打开文件对话框，限定图片格式
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", options=options
        )
        
        if file_name: # 如果用户选择了文件 (file_name非空)
            self.image_path = file_name # 保存完整路径
            self.current_image_name = os.path.basename(file_name) # 提取文件名
            self.display_image(file_name) # 在UI上显示图片
            
            # 重置UI状态和按钮可用性
            self.run_ocr_btn.setEnabled(True) # 选了图，OCR按钮可用
            self.ocr_text.clear()
            self.ai_text.clear()
            self.run_ai_btn.setEnabled(False)
            self.save_result_btn.setEnabled(False)
            self.save_to_excel_btn.setEnabled(False) # Excel按钮也禁用
            self.status_text.setText("已选择图片，等待处理")
            self.progress_bar.setValue(0)
            
    def display_image(self, file_path):
        """在UI的image_label上显示指定路径的图片，并自动缩放"""
        image = cv2.imread(file_path) # 用OpenCV读取图片
        if image is None: # 读取失败就返回
            self.update_status(f"无法加载图片: {file_path}")
            QMessageBox.warning(self, "图片加载失败", f"无法加载图片：{file_path}")
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV默认BGR，转为RGB给Qt用
        h, w, c = image.shape # 图片原始高、宽、通道数
        
        # 获取用于显示图片的QLabel的尺寸
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        
        # 计算缩放比例，确保图片完整显示在QLabel内且保持宽高比
        scale = min(label_width / w, label_height / h)
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        image_resized = cv2.resize(image, (new_width, new_height)) # 缩放图片
        
        # 将OpenCV的图像数据转换为Qt的QImage，然后是QPixmap
        bytes_per_line = 3 * new_width # RGB三通道
        q_image = QImage(image_resized.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.image_label.setPixmap(pixmap) # 设置到QLabel上
        
    def run_ocr(self):
        """执行OCR识别流程"""
        if not self.image_path: # 防御：没选图片不执行
            QMessageBox.warning(self, "警告", "请先选择图片")
            return
            
        # 准备开始OCR，重置UI状态
        self.progress_bar.setValue(10) # 进度条初始值
        self.ocr_text.clear()
        self.ai_text.clear()
        self.run_ocr_btn.setEnabled(False) # OCR执行中，按钮禁用
        self.run_ai_btn.setEnabled(False)
        self.save_result_btn.setEnabled(False)
        self.save_to_excel_btn.setEnabled(False)
        
        # 创建OCR线程实例
        # 模型路径是写死的，可以考虑做成可配置
        self.ocr_thread = OCRThread(
            image_path=self.image_path,
            det_model_dir='ch_PP-OCRv4_det_infer', # 检测模型
            rec_model_dir='ch_PP-OCRv4_rec_infer', # 识别模型
            use_angle_cls=False, # 通常不需要角度分类，除非图片方向很多变
            use_gpu=self.gpu_radio.isChecked()  # 根据UI选择使用GPU与否
        )
        
        # 连接线程的信号到主线程的槽函数
        self.ocr_thread.progress_update.connect(self.update_progress)
        self.ocr_thread.ocr_complete.connect(self.handle_ocr_result)
        self.ocr_thread.status_update.connect(self.update_status)
        self.ocr_thread.start() # 启动线程
        
    def handle_ocr_result(self, result):
        """处理OCR线程返回的结果"""
        self.ocr_result = result # 保存原始结果
        
        # 尝试解析OCR脚本输出的特定格式 (通常是 "文件名\tJSON数据")
        try:
            # 简单判断下是不是错误信息，如果是，直接显示
            if "失败" in result or "错误" in result or not result.strip().startswith(os.path.basename(self.image_path if self.image_path else "")):
                self.ocr_text.setText(result)
                self.run_ocr_btn.setEnabled(True) # OCR失败，允许重试
                self.run_ai_btn.setEnabled(False)
                self.save_to_excel_btn.setEnabled(False)
                return

            lines = result.strip().split('\n') # PaddleOCR可能一行一个结果
            formatted_text = ""
            
            for line in lines:
                parts = line.split('\t', 1) # 按第一个制表符分割
                if len(parts) >= 2:
                    # file_name_part = parts[0] # 文件名部分，这里用self.current_image_name代替
                    json_data_str = parts[1]  # JSON字符串部分
                    
                    json_data = json.loads(json_data_str) # 解析JSON
                    
                    formatted_text += f"文件: {self.current_image_name or parts[0]}\n" # 显示当前文件名
                    for item in json_data: # 遍历JSON中的识别项
                        formatted_text += f"- 文本: {item['transcription']}\n" # 'transcription'是PaddleOCR结果中的文本内容
                    formatted_text += "\n"
            
            self.ocr_text.setText(formatted_text if formatted_text else result) # 显示格式化文本，如果失败则显示原始结果
            self.run_ai_btn.setEnabled(True) # OCR成功，启用AI校正按钮
        except json.JSONDecodeError:
            # JSON解析失败，说明结果格式不对或者不是预期的JSON
            self.ocr_text.setText(f"OCR结果JSON解析失败。\n原始结果:\n{result}")
            self.run_ai_btn.setEnabled(False)
        except Exception as e:
            # 其他未知错误
            self.ocr_text.setText(f"处理OCR结果时发生错误: {str(e)}\n原始结果:\n{result}")
            self.run_ai_btn.setEnabled(False)
        
        self.run_ocr_btn.setEnabled(True) # OCR流程结束（无论成功失败），允许重新运行OCR
        self.save_to_excel_btn.setEnabled(False) # AI结果还没出来，Excel按钮保持禁用
        
    def run_ai_model(self):
        """执行大模型校正流程"""
        # 检查OCR结果是否有效
        if not self.ocr_result or not self.ocr_text.toPlainText() or "失败" in self.ocr_text.toPlainText() or "错误" in self.ocr_text.toPlainText():
            QMessageBox.warning(self, "警告", "请先成功运行OCR识别，或确保OCR结果有效。")
            return
            
        # 准备开始AI校正，重置UI
        self.progress_bar.setValue(10)
        self.ai_text.clear()
        self.run_ai_btn.setEnabled(False) # AI执行中，按钮禁用
        self.save_result_btn.setEnabled(False)
        self.save_to_excel_btn.setEnabled(False)
        
        ocr_content_for_ai = self.ocr_text.toPlainText() # 获取OCR文本框的全部内容给大模型

        # 创建AI模型线程实例
        self.ai_thread = AIModelThread(
            ocr_text=ocr_content_for_ai,
            model_type="siliconflow" if self.siliconflow_radio.isChecked() else "ollama", # 根据UI选择
            api_key=self.siliconflow_api_key.text() if self.siliconflow_radio.isChecked() else None,
            model_name=self.siliconflow_model_name.text() if self.siliconflow_radio.isChecked() else self.ollama_model_name.text(),
            ollama_url=self.ollama_url.text() if self.ollama_radio.isChecked() else None
        )
        
        # 连接信号槽
        self.ai_thread.progress_update.connect(self.update_progress)
        self.ai_thread.ai_complete.connect(self.handle_ai_model_result)
        self.ai_thread.status_update.connect(self.update_status)
        self.ai_thread.start() # 启动线程
        
    def handle_ai_model_result(self, result):
        """处理AI模型线程返回的结果"""
        self.ai_corrected_text = result # 保存AI校正后的文本
        self.ai_text.setText(result)    # 显示在UI上
        self.run_ai_btn.setEnabled(True) # AI流程结束，允许重新运行
        
        # 根据AI结果是否有效，决定是否启用保存按钮
        if "失败" not in result and "错误" not in result and "未返回有效结果" not in result and "异常" not in result:
            self.save_result_btn.setEnabled(True)
            self.save_to_excel_btn.setEnabled(True) # AI成功，Excel按钮可用
        else:
            self.save_result_btn.setEnabled(False)
            self.save_to_excel_btn.setEnabled(False) # AI失败，Excel按钮禁用

    def save_result(self):
        """将AI校正后的结果保存到文本文件"""
        if not self.ai_corrected_text: # 防御：没结果不保存
            QMessageBox.warning(self, "警告", "没有可保存的AI校正结果。")
            return

        options = QFileDialog.Options()
        # 默认文件名包含原图片名
        default_filename = f"{self.current_image_name}_corrected.txt" if self.current_image_name else "corrected_result.txt"
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存结果", default_filename, 
            "文本文件 (*.txt);;所有文件 (*)", options=options
        )

        if file_name: # 如果用户选择了保存路径
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(self.ai_corrected_text) # 写入AI结果
                self.update_status(f"结果已保存到: {file_name}")
                QMessageBox.information(self, "成功", "结果已成功保存！")
            except Exception as e: # 保存文件可能发生的异常
                QMessageBox.critical(self, "错误", f"保存文件失败: {str(e)}")
                self.update_status("保存文件失败")

    def save_to_excel(self):
        """将AI校正结果解析并存储到Excel文件中"""
        if not self.ai_corrected_text or "失败" in self.ai_corrected_text or "错误" in self.ai_corrected_text:
            QMessageBox.warning(self, "警告", "没有有效的大模型修正结果可供存储。")
            return

        raw_text = self.ai_corrected_text # AI输出的文本
        
        # 初始化提取信息的字典，给个默认值
        extracted_info = {
            "品名": "无",
            "规格": "无",
            "单位": "无",
            "价格": 0.00  # 价格用浮点数
        }

        # 解析AI输出的固定格式文本
        lines = raw_text.split('\n')
        for line in lines:
            line = line.strip() # 去掉首尾空格
            if line.startswith("- 商品名称："):
                extracted_info["品名"] = line.replace("- 商品名称：", "").strip()
            elif line.startswith("- 规格："):
                extracted_info["规格"] = line.replace("- 规格：", "").strip()
            elif line.startswith("- 单位："):
                extracted_info["单位"] = line.replace("- 单位：", "").strip()
            elif line.startswith("- 价格："):
                price_text = line.replace("- 价格：", "").strip()
                price_text = price_text.replace("元。", "").replace("元", "").strip() # 去掉"元"和句号
                if price_text.lower() == "无": # AI可能输出"无"
                    extracted_info["价格"] = 0.00
                else:
                    # 用正则提取价格中的数字部分
                    match = re.search(r"(\d+(\.\d+)?)", price_text)
                    if match:
                        extracted_info["价格"] = float(match.group(1))
                    else:
                        extracted_info["价格"] = 0.00 # 没找到数字也给个默认
        
        # 准备写入Excel的一行数据
        excel_row = [
            extracted_info["品名"],
            extracted_info["规格"],
            extracted_info["单位"],
            extracted_info["价格"]
        ]

        try:
            workbook = openpyxl.load_workbook(self.excel_file_path) # 加载已存在的Excel
            sheet = workbook.active # 获取活动工作表
            sheet.append(excel_row) # 追加新行
            workbook.save(self.excel_file_path) # 保存Excel
            self.update_status(f"数据已成功存入: {self.excel_file_path}")
            QMessageBox.information(self, "成功", f"商品信息 '{extracted_info['品名']}' 已成功存入Excel！")
        except FileNotFoundError: # Excel文件如果被删了
             QMessageBox.critical(self, "Excel错误", f"Excel文件 {self.excel_file_path} 未找到。\n请尝试重新启动程序或检查文件路径。")
             self.update_status(f"Excel文件未找到")
        except InvalidFileException: # Excel文件损坏
            QMessageBox.critical(self, "Excel错误", f"Excel文件 {self.excel_file_path} 格式无效或已损坏。")
            self.update_status(f"Excel文件损坏")
        except PermissionError: # Excel文件被其他程序占用，或没有写入权限
            QMessageBox.critical(self, "Excel错误", f"无法写入Excel文件 {self.excel_file_path}。\n请确保文件未被其他程序占用且具有写入权限。")
            self.update_status(f"Excel文件写入权限错误")
        except Exception as e: # 其他写入Excel的错误
            QMessageBox.critical(self, "错误", f"存入Excel失败: {str(e)}")
            self.update_status("存入Excel失败")

    def update_progress(self, value):
        """更新进度条的值"""
        self.progress_bar.setValue(value)
        
    def update_status(self, status):
        """更新状态栏文本"""
        self.status_text.setText(status)
        
    # 注意：这里有一个重复定义的 save_result 方法，保留上面那个更完善的，下面这个可以考虑删除或合并
    # def save_result(self):
    #     options = QFileDialog.Options()
    #     file_name, _ = QFileDialog.getSaveFileName(
    #         self, "保存结果", "", "文本文件 (*.txt);;所有文件 (*)", options=options
    #     )
        
    #     if file_name:
    #         with open(file_name, 'w', encoding='utf-8') as f:
    #             f.write("=== OCR识别结果 ===\n\n")
    #             f.write(self.ocr_text.toPlainText())
    #             f.write("\n\n=== 大模型校正结果 ===\n\n")
    #             f.write(self.ai_text.toPlainText())
            
    #         self.status_text.setText("结果已保存")
    #         QMessageBox.information(self, "成功", f"结果已保存到 {file_name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，看起来更现代
    window = TagGuardianApp()
    window.show()
    sys.exit(app.exec_())