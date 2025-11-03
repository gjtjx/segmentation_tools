#!/usr/bin/env python3
"""
图像语义标注工具
支持功能：
- 批量导入 JPG 图像与 PNG 标签
- Alpha 叠加显示标签 (默认 0.5，可调节)
- 画笔/橡皮擦（矩形，支持旋转），支持可调节半宽/半高（默认 5/5）
- 图像导航 (上一张/下一张/跳转)
- 自动保存到 labels 文件夹
- 撤销/重做功能
- 鼠标滚轮以鼠标为中心缩放；右键拖拽平移；R 重置视图
"""

import sys
import os
import glob
import shutil
import datetime
import base64
import requests
import io
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
                            QFileDialog, QMessageBox, QToolBar, QAction, QStatusBar,
                            QButtonGroup, QRadioButton, QGroupBox, QLineEdit,
                            QSplitter, QFrame, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRect, QSize
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QBrush, QColor, QFont, 
                        QIcon, QKeySequence, QCursor, QPalette, QImage)


class ImageCanvas(QLabel):
    """自定义图像画布，支持绘制操作"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid gray;")
        self.setFocusPolicy(Qt.ClickFocus)
        
        # 图像相关
        self.original_image = None
        self.mask_image = None
        self.display_image = None
        self.scaled_pixmap = None  # 缓存缩放后的图像
        self.alpha = 0.5

        # 绘制相关
        self.drawing = False
        self.tool_mode = "brush"  # "brush" or "eraser"
        # 画笔半宽/半高（图像像素），默认为 5/5
        self.brush_width = 5
        self.brush_height = 5
        self.last_point = QPoint()
        
        # 缩放相关
        self.scale_factor = 1.0
        self.image_offset = QPoint(0, 0)
        self.zoom_factor = 1.0  # 用户手动缩放倍数
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        
        # 拖拽相关
        self.dragging = False
        self.drag_start_pos = QPoint()
        self.image_pos = QPoint(0, 0)  # 图像的位置偏移
        
        # 光标旋转角度
        self.cursor_rotation = 45  # 默认旋转45度
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        
        # 光标相关
        self.cursor_visible = False
        self.cursor_pos = QPoint()
        # 显示模式: 'overlay' (默认), 'original', 'mask'
        self.display_mode = 'overlay'
        
        # 多边形绘制相关（以图像坐标存储点）
        self.polygon_points = []  # type: list
        self.polygon_active = False
        
        # AI分割相关
        self.ai_points = []  # AI点采集列表 [[x, y, label], ...]
        self.ai_mode_active = False
        self.previous_tool_mode = "brush"  # 用于P键恢复
        
    def set_images(self, image_path: str, mask_path: str):
        """设置要显示的图像和mask"""
        try:
            # 加载原始图像
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # 加载或创建mask - 允许mask不存在，不报错
            if os.path.exists(mask_path):
                try:
                    self.mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if self.mask_image is None:
                        # 读取失败，创建空mask
                        self.mask_image = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
                except Exception as e:
                    # 读取异常，创建空mask，不中断
                    print(f"Warning: Failed to load mask {mask_path}: {e}")
                    self.mask_image = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            else:
                # mask不存在，创建空mask，允许用户从头标注
                self.mask_image = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            
            # 确保mask尺寸与图像匹配
            if self.mask_image.shape != self.original_image.shape[:2]:
                self.mask_image = cv2.resize(self.mask_image, 
                                           (self.original_image.shape[1], self.original_image.shape[0]))
            
            self.update_display()
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
            return False
    
    def update_display(self):
        """更新显示的图像"""
        if self.original_image is None:
            return
        # 创建彩色mask (绿色)
        mask_colored = np.zeros_like(self.original_image)
        # 使用绿色通道显示mask
        mask_colored[self.mask_image > 0] = [0, 255, 0]

        # 根据display_mode选择输出
        if self.display_mode == 'original':
            self.display_image = self.original_image.copy()
        elif self.display_mode == 'mask':
            # 纯mask视图（黑底绿前景）
            self.display_image = mask_colored.copy()
        else:
            # overlay 混合
            self.display_image = cv2.addWeighted(
                self.original_image, 1.0,
                mask_colored, self.alpha, 0
            )
        
        # 转换为QPixmap
        height, width, channel = self.display_image.shape
        bytes_per_line = 3 * width
        
        # 使用QImage直接从numpy数组创建
        q_image = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        
        # 计算显示参数（考虑用户缩放）
        label_size = self.size()
        auto_scale_factor = min(
            label_size.width() / width,
            label_size.height() / height
        )
        
        # 应用用户缩放
        self.scale_factor = auto_scale_factor * self.zoom_factor
        
        # 缩放图像
        scaled_size = q_pixmap.size() * self.scale_factor
        scaled_pixmap = q_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        # 存储缩放后的图像，交由 paintEvent 绘制
        self.scaled_pixmap = scaled_pixmap
        
        # 触发重绘
        self.update()
    
    def set_alpha(self, alpha: float):
        """设置alpha值"""
        self.alpha = alpha
        self.update_display()
    
    def set_tool_mode(self, mode: str):
        """设置工具模式"""
        # 如果从多边形模式切换出去，取消未完成的多边形
        if getattr(self, 'tool_mode', None) in ("polygon_add", "polygon_erase") and mode not in ("polygon_add", "polygon_erase"):
            self.cancel_polygon()
        # 如果从AI模式切换出去，清除AI点
        if getattr(self, 'tool_mode', None) == "ai_segment" and mode != "ai_segment":
            self.ai_points = []
            self.ai_mode_active = False
        # 保存切换前的模式（用于P键恢复）
        current = getattr(self, 'tool_mode', 'brush')
        if mode == "ai_segment" and current != "ai_segment":
            self.previous_tool_mode = current
        self.tool_mode = mode
        # 隐藏系统光标，使用自定义绘制
        self.setCursor(QCursor(Qt.BlankCursor))
        self.update()

    def set_display_mode(self, mode: str):
        """设置显示模式：'overlay'|'original'|'mask'"""
        if mode in ('overlay', 'original', 'mask'):
            self.display_mode = mode
            self.update_display()
    
    def set_brush_width(self, width: int):
        """设置画笔半宽（像素）"""
        self.brush_width = max(1, int(width))
        self.update()

    def set_brush_height(self, height: int):
        """设置画笔半高（像素）"""
        self.brush_height = max(1, int(height))
        self.update()
    
    def set_cursor_rotation(self, rotation: int):
        """设置光标旋转角度"""
        self.cursor_rotation = rotation
        self.update()
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton and self.original_image is not None:
            # 左键：开始绘制（先保存撤销快照，合并整段笔划为一次撤销）
            if self.tool_mode in ("brush", "eraser"):
                win = self.window()
                if win is not None and hasattr(win, 'save_undo_state'):
                    win.save_undo_state()
                self.drawing = True
                self.last_point = event.pos()
                self.draw_at_position(event.pos())
            elif self.tool_mode in ("polygon_add", "polygon_erase"):
                # 多边形模式：添加一个点（以图像坐标）
                img_pos = self.screen_to_image_coords(event.pos())
                if img_pos is not None:
                    if not self.polygon_active:
                        # 开始新的多边形并保存撤销快照
                        win = self.window()
                        if win is not None and hasattr(win, 'save_undo_state'):
                            win.save_undo_state()
                        self.polygon_points = []
                        self.polygon_active = True
                    # 如果点击接近首点且点数>=3，则闭合并提交
                    if len(self.polygon_points) >= 3 and self._near_first_point(img_pos):
                        self.commit_polygon()
                    else:
                        self.polygon_points.append(img_pos)
                        self.update()
            elif self.tool_mode == "ai_segment":
                # AI分割模式：收集点
                img_pos = self.screen_to_image_coords(event.pos())
                if img_pos is not None:
                    # 左键添加正样本点（label=1）
                    self.ai_points.append([img_pos[0], img_pos[1], 1])
                    self.ai_mode_active = True
                    self.update()
                    print(f"AI点添加: {img_pos}, label=1 (正样本)")
        elif event.button() == Qt.RightButton and self.original_image is not None:
            if self.tool_mode == "ai_segment":
                # AI模式下右键添加负样本点（label=0）
                img_pos = self.screen_to_image_coords(event.pos())
                if img_pos is not None:
                    self.ai_points.append([img_pos[0], img_pos[1], 0])
                    self.ai_mode_active = True
                    self.update()
                    print(f"AI点添加: {img_pos}, label=0 (负样本)")
            else:
                # 右键：开始拖拽平移
                self.dragging = True
                self.drag_start_pos = event.pos()
                self.drag_start_offset = QPoint(self.image_pos)
                self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseDoubleClickEvent(self, event):
        """双击结束多边形"""
        if self.original_image is None:
            return
        if event.button() == Qt.LeftButton and self.tool_mode in ("polygon_add", "polygon_erase") and self.polygon_active:
            if len(self.polygon_points) >= 3:
                self.commit_polygon()
            else:
                # 点数不足，取消
                self.cancel_polygon()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        # 更新光标位置
        self.cursor_pos = event.pos()
        self.cursor_visible = True

        if self.dragging and (event.buttons() & Qt.RightButton):
            # 拖拽模式 - 平移图像
            delta = event.pos() - self.drag_start_pos
            self.image_pos = QPoint(
                self.drag_start_offset.x() + delta.x(),
                self.drag_start_offset.y() + delta.y()
            )
            self.update_display()
        elif self.drawing and (event.buttons() & Qt.LeftButton):
            # 绘制模式 - 左键绘制
            self.draw_line(self.last_point, event.pos())
            self.last_point = event.pos()

        # 更新显示以绘制光标
        self.update()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.RightButton and self.dragging:
            # 结束拖拽
            self.dragging = False
            self.setCursor(QCursor(Qt.BlankCursor))
        elif event.button() == Qt.LeftButton and self.drawing:
            # 结束绘制
            self.drawing = False

    def keyPressEvent(self, event):
        """键盘事件（处理多边形：回车提交、退格撤点、ESC取消）"""
        handled = False
        # 快捷键 F/G 支持：在多边形模式下快速切换为增加/删除模式
        if event.key() == Qt.Key_F:
            # 切换到多边形增加模式
            self.window().poly_add_radio.setChecked(True) if self.window() is not None else None
            self.set_tool_mode('polygon_add')
            handled = True
        elif event.key() == Qt.Key_G:
            # 切换到多边形删除模式
            self.window().poly_erase_radio.setChecked(True) if self.window() is not None else None
            self.set_tool_mode('polygon_erase')
            handled = True
        if self.tool_mode in ("polygon_add", "polygon_erase") and self.polygon_active:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if len(self.polygon_points) >= 3:
                    self.commit_polygon()
                else:
                    self.cancel_polygon()
                handled = True
            elif event.key() == Qt.Key_Escape:
                self.cancel_polygon()
                handled = True
            elif event.key() == Qt.Key_Backspace:
                if self.polygon_points:
                    self.polygon_points.pop()
                    self.update()
                else:
                    self.cancel_polygon()
                handled = True
        if not handled:
            super().keyPressEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开事件"""
        self.cursor_visible = False
        self.update()
    
    def enterEvent(self, event):
        """鼠标进入事件"""
        self.cursor_visible = True
        self.update()
    
    def wheelEvent(self, event):
        """鼠标滚轮事件 - 实现鼠标中心缩放"""
        if event.modifiers() == Qt.ControlModifier and self.original_image is not None:
            # 获取鼠标在widget中的位置
            mouse_widget_pos = event.pos()
            
            # 计算缩放因子
            zoom_in = event.angleDelta().y() > 0
            zoom_factor = 1.1 if zoom_in else 1.0 / 1.1

            # 记录缩放前的总缩放与平移（总缩放=自适应缩放*用户缩放）
            widget_size = self.size()
            img_w = self.original_image.shape[1]
            img_h = self.original_image.shape[0]

            # 自适应缩放（与 update_display 保持一致）
            auto_scale = min(
                widget_size.width() / img_w,
                widget_size.height() / img_h
            )

            old_total_scale = self.scale_factor  # 等同于 auto_scale * self.zoom_factor（上次 update_display 的结果）
            old_pan = QPoint(self.image_pos)     # 平移偏移（屏幕像素）

            # 计算缩放前鼠标对应的图像坐标（保持鼠标锚点）
            left_old = (widget_size.width() - int(img_w * old_total_scale)) / 2 + old_pan.x()
            top_old = (widget_size.height() - int(img_h * old_total_scale)) / 2 + old_pan.y()
            x_img = (mouse_widget_pos.x() - left_old) / old_total_scale
            y_img = (mouse_widget_pos.y() - top_old) / old_total_scale

            # 更新用户缩放并裁剪
            self.zoom_factor *= zoom_factor
            self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor))

            # 新的总缩放
            new_total_scale = auto_scale * self.zoom_factor

            # 计算新的平移，使鼠标锚点位置不变
            left_center_new = (widget_size.width() - img_w * new_total_scale) / 2
            top_center_new = (widget_size.height() - img_h * new_total_scale) / 2
            new_pan_x = mouse_widget_pos.x() - left_center_new - new_total_scale * x_img
            new_pan_y = mouse_widget_pos.y() - top_center_new - new_total_scale * y_img

            self.image_pos = QPoint(int(new_pan_x), int(new_pan_y))

            self.update_display()
        else:
            super().wheelEvent(event)
    
    def draw_at_position(self, pos: QPoint):
        """在指定位置绘制（考虑旋转和缩放）"""
        if self.original_image is None:
            return
        
        # 转换屏幕坐标到图像坐标
        image_pos = self.screen_to_image_coords(pos)
        if image_pos is None:
            return
        
        center_x, center_y = image_pos

        # 计算在图像坐标系中的实际画笔半宽/半高
        half_w = self.brush_width
        half_h = self.brush_height

        # 获取旋转角度（转换为弧度）
        import math
        rotation_rad = math.radians(self.cursor_rotation)

        # 如果旋转角度接近于0、90、180、270度，使用轴对齐矩形
        angle_mod_90 = abs(self.cursor_rotation % 90)
        if angle_mod_90 <= 5 or angle_mod_90 >= 85:
            # 使用简单的轴对齐矩形
            x1 = max(0, int(center_x - half_w))
            y1 = max(0, int(center_y - half_h))
            x2 = min(self.mask_image.shape[1] - 1, int(center_x + half_w))
            y2 = min(self.mask_image.shape[0] - 1, int(center_y + half_h))

            fill_val = 255 if self.tool_mode == "brush" else 0
            cv2.rectangle(self.mask_image, (x1, y1), (x2, y2), fill_val, -1)
        else:
            # 使用旋转矩形
            self.draw_rotated_rectangle(center_x, center_y, half_w, half_h, rotation_rad)
        
        self.update_display()
        # 通知主窗口有变更
        win = self.window()
        if win is not None and hasattr(win, 'trigger_auto_save'):
            win.unsaved_changes = True
            win.trigger_auto_save()
    
    def draw_line(self, start_pos: QPoint, end_pos: QPoint):
        """绘制线条（使用矩形模式）"""
        if self.original_image is None:
            return
        
        # 转换屏幕坐标到图像坐标
        start_image = self.screen_to_image_coords(start_pos)
        end_image = self.screen_to_image_coords(end_pos)
        
        if start_image is None or end_image is None:
            return
        
        # 计算线条上的点，在每个点上绘制矩形
        x1, y1 = start_image
        x2, y2 = end_image
        
        # 计算线条长度
        import math
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if distance < 1:
            # 距离太小，只绘制一个矩形
            self.draw_rectangle_at_point(x1, y1)
        else:
            # 沿着线条绘制多个矩形
            steps = max(int(distance), 1)
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                self.draw_rectangle_at_point(x, y)
        
        self.update_display()
        # 通知主窗口有变更
        win = self.window()
        if win is not None and hasattr(win, 'trigger_auto_save'):
            win.unsaved_changes = True
            win.trigger_auto_save()
    
    def draw_rotated_rectangle(self, center_x: float, center_y: float, half_w: float, half_h: float, rotation_rad: float):
        """绘制旋转矩形（支持不同半宽/半高）"""
        import math
        
        # 计算旋转矩形的四个顶点
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)
        
        # 使用传入的半宽/半高
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]
        
        # 旋转并转换为整数坐标
        rotated_points = []
        for dx, dy in corners:
            new_x = int(center_x + dx * cos_r - dy * sin_r)
            new_y = int(center_y + dx * sin_r + dy * cos_r)
            rotated_points.append([new_x, new_y])
        
        # 使用OpenCV的fillPoly填充旋转矩形
        import numpy as np
        points = np.array(rotated_points, dtype=np.int32)
        
        if self.tool_mode == "brush":
            cv2.fillPoly(self.mask_image, [points], 255)
        elif self.tool_mode == "eraser":
            cv2.fillPoly(self.mask_image, [points], 0)
    
    def draw_rectangle_at_point(self, x: int, y: int):
        """在指定点绘制矩形（考虑旋转）"""
    # 计算在图像坐标系中的实际画笔半宽/半高
        half_w = self.brush_width
        half_h = self.brush_height
        
        # 获取旋转角度
        import math
        rotation_rad = math.radians(self.cursor_rotation)
        
        # 如果旋转角度接近于0、90、180、270度，使用简单矩形
        angle_mod_90 = self.cursor_rotation % 90
        if angle_mod_90 < 5 or angle_mod_90 > 85:
            # 使用简单矩形
            x1 = max(0, x - half_w)
            y1 = max(0, y - half_h)
            x2 = min(self.mask_image.shape[1] - 1, x + half_w)
            y2 = min(self.mask_image.shape[0] - 1, y + half_h)
            
            if self.tool_mode == "brush":
                cv2.rectangle(self.mask_image, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
            elif self.tool_mode == "eraser":
                cv2.rectangle(self.mask_image, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
        else:
            # 使用旋转矩形
            self.draw_rotated_rectangle(x, y, half_w, half_h, rotation_rad)
    
    def screen_to_image_coords(self, screen_pos: QPoint) -> Optional[Tuple[int, int]]:
        """将屏幕坐标转换为图像坐标（使用总缩放scale_factor与平移image_pos）"""
        if self.original_image is None:
            return None

        img_h, img_w = self.original_image.shape[:2]
        widget_size = self.size()

        # 使用总缩放（自适应缩放*用户缩放），与 update_display 保持一致
        total_scale = self.scale_factor
        scaled_w = int(img_w * total_scale)
        scaled_h = int(img_h * total_scale)

        # 图像左上角位置 = 居中位置 + 平移
        left = (widget_size.width() - scaled_w) // 2 + self.image_pos.x()
        top = (widget_size.height() - scaled_h) // 2 + self.image_pos.y()

        # 鼠标相对位置
        rel_x = screen_pos.x() - left
        rel_y = screen_pos.y() - top

        # 不在图像区域内
        if rel_x < 0 or rel_x >= scaled_w or rel_y < 0 or rel_y >= scaled_h:
            return None

        # 映射到原始图像坐标
        img_x = int(rel_x / total_scale)
        img_y = int(rel_y / total_scale)

        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))

        return (img_x, img_y)
    
    def get_current_mask(self) -> np.ndarray:
        """获取当前mask"""
        return self.mask_image.copy() if self.mask_image is not None else None
    
    def set_mask(self, mask: np.ndarray):
        """设置mask"""
        if mask is not None:
            self.mask_image = mask.copy()
            self.update_display()
    
    def paintEvent(self, event):
        """绘制事件 - 绘制图像与自定义光标"""
        painter = QPainter(self)
        
        # 先绘制图像
        if self.scaled_pixmap is not None and not self.scaled_pixmap.isNull():
            widget_size = self.size()
            scaled_w = self.scaled_pixmap.width()
            scaled_h = self.scaled_pixmap.height()
            # 图像左上角位置 = 居中位置 + 平移
            left = (widget_size.width() - scaled_w) // 2 + self.image_pos.x()
            top = (widget_size.height() - scaled_h) // 2 + self.image_pos.y()
            painter.drawPixmap(left, top, self.scaled_pixmap)
        
        # 再绘制自定义光标
        if self.cursor_visible and self.original_image is not None:
            self.draw_cursor(painter)

        # 绘制多边形预览
        if self.original_image is not None and self.polygon_active and len(self.polygon_points) >= 1:
            self.draw_polygon_preview(painter)
        
        # 绘制AI点
        if self.original_image is not None and self.ai_mode_active and len(self.ai_points) > 0:
            self.draw_ai_points(painter)
        
        painter.end()
    
    def draw_cursor(self, painter: QPainter):
        """绘制光标"""
    # 计算光标矩形大小（以屏幕像素绘制）
    # 光标宽/高 =（图像像素中的直径）*（总缩放比例）
        cursor_w = int(self.brush_width * 2 * self.scale_factor)
        cursor_h = int(self.brush_height * 2 * self.scale_factor)
        
        # 设置画笔颜色为绿色
        pen = QPen(QColor(0, 255, 0), 2)
        if self.tool_mode == "brush":
            pen.setStyle(Qt.SolidLine)
        else:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.NoBrush))  # 不填充
        
        # 保存当前变换状态
        painter.save()
        painter.translate(self.cursor_pos.x(), self.cursor_pos.y())
        painter.rotate(self.cursor_rotation)
        rotated_rect = QRect(
            int(-cursor_w / 2),
            int(-cursor_h / 2),
            int(cursor_w),
            int(cursor_h)
        )
        painter.drawRect(rotated_rect)
        # 绘制中心小红点（便于精确对齐）
        dot_radius = max(2, int(min(cursor_w, cursor_h) * 0.06))
        painter.setPen(QPen(QColor(255, 0, 0)))
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.drawEllipse(QRect(int(-dot_radius), int(-dot_radius), int(dot_radius * 2), int(dot_radius * 2)))
        painter.restore()

    def image_to_screen_coords(self, img_x: int, img_y: int) -> Optional[QPoint]:
        """将图像坐标转换为屏幕坐标"""
        if self.original_image is None or self.scaled_pixmap is None:
            return None
        img_h, img_w = self.original_image.shape[:2]
        widget_size = self.size()
        total_scale = self.scale_factor
        scaled_w = int(img_w * total_scale)
        scaled_h = int(img_h * total_scale)
        left = (widget_size.width() - scaled_w) // 2 + self.image_pos.x()
        top = (widget_size.height() - scaled_h) // 2 + self.image_pos.y()
        sx = int(left + img_x * total_scale)
        sy = int(top + img_y * total_scale)
        return QPoint(sx, sy)

    def draw_polygon_preview(self, painter: QPainter):
        """绘制多边形预览（线与锚点）"""
        if not self.polygon_points:
            return
        # 转为屏幕坐标
        screen_points: List[QPoint] = []
        for (ix, iy) in self.polygon_points:
            sp = self.image_to_screen_coords(ix, iy)
            if sp is not None:
                screen_points.append(sp)
        if not screen_points:
            return
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        # 画线段
        for i in range(1, len(screen_points)):
            painter.drawLine(screen_points[i-1], screen_points[i])
        # 画到当前鼠标位置的预览线
        if self.cursor_visible and len(screen_points) >= 1:
            painter.setPen(QPen(QColor(0, 255, 0), 1, Qt.DashLine))
            painter.drawLine(screen_points[-1], self.cursor_pos)
        # 画锚点
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        for p in screen_points:
            painter.drawEllipse(p, 3, 3)

    def draw_ai_points(self, painter: QPainter):
        """绘制AI标注点（正样本绿色，负样本红色）"""
        if not self.ai_points:
            return
        for point_data in self.ai_points:
            ix, iy, label = point_data[0], point_data[1], point_data[2]
            sp = self.image_to_screen_coords(int(ix), int(iy))
            if sp is not None:
                # 正样本绿色，负样本红色
                color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
                painter.setPen(QPen(color, 3))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(sp, 5, 5)
                # 画十字标记
                painter.drawLine(sp.x() - 8, sp.y(), sp.x() + 8, sp.y())
                painter.drawLine(sp.x(), sp.y() - 8, sp.x(), sp.y() + 8)


    def _near_first_point(self, img_pos: Tuple[int, int], threshold: int = 5) -> bool:
        """判断是否接近首点（图像像素距离）"""
        if not self.polygon_points:
            return False
        x0, y0 = self.polygon_points[0]
        x, y = img_pos
        dx = x - x0
        dy = y - y0
        return (dx*dx + dy*dy) <= threshold*threshold

    def commit_polygon(self):
        """提交多边形到mask（根据模式增加或删除）"""
        if len(self.polygon_points) < 3 or self.mask_image is None:
            self.cancel_polygon()
            return
        import numpy as np
        pts = np.array(self.polygon_points, dtype=np.int32)
        val = 255 if self.tool_mode == "polygon_add" else 0
        cv2.fillPoly(self.mask_image, [pts], val)
        self.polygon_points = []
        self.polygon_active = False
        self.update_display()
        # 通知主窗口有变更
        win = self.window()
        if win is not None and hasattr(win, 'trigger_auto_save'):
            win.unsaved_changes = True
            win.trigger_auto_save()

    def cancel_polygon(self):
        """取消当前多边形编辑"""
        self.polygon_points = []
        self.polygon_active = False
        self.update()


class AnnotationTool(QMainWindow):
    """主标注工具窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像语义标注工具")
        self.setGeometry(100, 100, 1400, 900)
        
        # 数据相关
        self.image_files = []
        self.current_index = 0
        self.images_dir = ""
        self.labels_dir = ""
        
        # 撤销/重做
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50
        
        # AI模式切换状态
        self.previous_tool_radio = None
        
        # 初始化UI
        self.init_ui()
        self.init_shortcuts()
        
        # 状态
        self.unsaved_changes = False
        
        # 自动保存定时器
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_timer.setSingleShot(True)  # 单次触发
        self.auto_save_interval = 2000  # 2秒后自动保存
    
    def keyPressEvent(self, event):
        """键盘按下事件"""
        key = event.key()
        modifiers = event.modifiers()
        
        # 旋转角度调节快捷键
        if key == Qt.Key_Z and modifiers == Qt.NoModifier:
            # Z键：减小1度
            current_value = self.rotation_slider.value()
            new_value = max(0, current_value - 1)
            self.rotation_slider.setValue(new_value)
            return
        elif key == Qt.Key_V and modifiers == Qt.NoModifier:
            # V键：增加1度（原先使用 C，已改为 V 以保留 C 用作视图切换）
            current_value = self.rotation_slider.value()
            new_value = min(360, current_value + 1)
            self.rotation_slider.setValue(new_value)
            return
        elif key == Qt.Key_R and modifiers == Qt.NoModifier:
            # R键：重置图像缩放与位置
            self.reset_view()
            return

        # 一键清除当前图像（先备份）：T
        if key == Qt.Key_T and modifiers == Qt.NoModifier:
            self.clear_current_annotation_with_backup()
            return

        # 快捷键 F/G：切换多边形增加/删除模式（主窗口级别，避免焦点问题）
        if key == Qt.Key_F and modifiers == Qt.NoModifier:
            try:
                self.poly_add_radio.setChecked(True)
            except Exception:
                pass
            self.canvas.set_tool_mode('polygon_add')
            return
        if key == Qt.Key_G and modifiers == Qt.NoModifier:
            try:
                self.poly_erase_radio.setChecked(True)
            except Exception:
                pass
            self.canvas.set_tool_mode('polygon_erase')
            return
        
        # H键：切换到AI分割模式
        if key == Qt.Key_H and modifiers == Qt.NoModifier:
            try:
                self.ai_radio.setChecked(True)
            except Exception:
                pass
            self.canvas.set_tool_mode('ai_segment')
            return
        
        # P键：切换绘制点模式/恢复原模式
        if key == Qt.Key_P and modifiers == Qt.NoModifier:
            if self.canvas.tool_mode == "ai_segment":
                # 当前是AI模式，恢复到之前的工具
                prev_mode = getattr(self.canvas, 'previous_tool_mode', 'brush')
                if prev_mode == "brush":
                    self.brush_radio.setChecked(True)
                elif prev_mode == "eraser":
                    self.eraser_radio.setChecked(True)
                elif prev_mode == "polygon_add":
                    self.poly_add_radio.setChecked(True)
                elif prev_mode == "polygon_erase":
                    self.poly_erase_radio.setChecked(True)
                self.canvas.set_tool_mode(prev_mode)
            else:
                # 切换到AI绘制点模式
                self.ai_radio.setChecked(True)
                self.canvas.set_tool_mode('ai_segment')
            return

        # 切换显示模式：C，在 overlay->original->mask 三模式间循环
        if key == Qt.Key_C and modifiers == Qt.NoModifier:
            modes = ['overlay', 'original', 'mask']
            cur = getattr(self.canvas, 'display_mode', 'overlay')
            try:
                idx = modes.index(cur)
            except ValueError:
                idx = 0
            new = modes[(idx + 1) % len(modes)]
            self.canvas.set_display_mode(new)
            try:
                self.display_toggle_btn.setText(f"C键切换显示: {new.capitalize()}")
            except Exception:
                pass
            return
        
        # 原有的快捷键处理
        if key == Qt.Key_A:
            # A键：上一张
            self.previous_image()
        elif key == Qt.Key_D:
            # D键：下一张
            self.next_image()
        elif key == Qt.Key_S:
            # S键：切换画笔模式
            self.toggle_tool()
        elif modifiers == Qt.ControlModifier:
            if key == Qt.Key_Z:
                # Ctrl+Z：撤销
                self.undo()
            elif key == Qt.Key_Y:
                # Ctrl+Y：重做
                self.redo()
        else:
            super().keyPressEvent(event)

    def toggle_tool(self):
        """在画笔与橡皮擦之间切换"""
        if self.brush_radio.isChecked():
            self.eraser_radio.setChecked(True)
            self.canvas.set_tool_mode("eraser")
        else:
            self.brush_radio.setChecked(True)
            self.canvas.set_tool_mode("brush")

    def brush_width_changed(self, val: int):
        """画笔半宽变化"""
        self.canvas.set_brush_width(val)

    def brush_height_changed(self, val: int):
        """画笔半高变化"""
        self.canvas.set_brush_height(val)
        
    def init_ui(self):
        """初始化用户界面"""
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧图像显示区域
        self.canvas = ImageCanvas()
        splitter.addWidget(self.canvas)
        
        # 设置分割器比例
        splitter.setSizes([300, 1100])
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 创建工具栏
        self.create_toolbar()
    
    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        
        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        load_btn = QPushButton("加载图像文件夹")
        load_btn.clicked.connect(self.load_images)
        file_layout.addWidget(load_btn)
        
        save_btn = QPushButton("保存当前标注")
        save_btn.clicked.connect(self.save_current_annotation)
        file_layout.addWidget(save_btn)
        
        layout.addWidget(file_group)
        
        # 导航组
        nav_group = QGroupBox("图像导航")
        nav_layout = QVBoxLayout(nav_group)
        
        nav_buttons_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张 (A键)")
        self.prev_btn.clicked.connect(self.previous_image)
        self.next_btn = QPushButton("下一张 (D键)")
        self.next_btn.clicked.connect(self.next_image)
        nav_buttons_layout.addWidget(self.prev_btn)
        nav_buttons_layout.addWidget(self.next_btn)
        nav_layout.addLayout(nav_buttons_layout)
        
        # 跳转输入
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("跳转到:"))
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("图像编号")
        self.jump_input.returnPressed.connect(self.jump_to_image)
        jump_layout.addWidget(self.jump_input)
        jump_btn = QPushButton("跳转")
        jump_btn.clicked.connect(self.jump_to_image)
        jump_layout.addWidget(jump_btn)
        nav_layout.addLayout(jump_layout)
        
        # 当前图像信息
        self.image_info_label = QLabel("未加载图像")
        nav_layout.addWidget(self.image_info_label)
        
        layout.addWidget(nav_group)
        
        # 工具组
        tools_group = QGroupBox("绘制工具")
        tools_layout = QVBoxLayout(tools_group)
        
        # 工具选择
        self.tool_group = QButtonGroup()
        self.brush_radio = QRadioButton("画笔 (增加) - S键")
        self.eraser_radio = QRadioButton("橡皮擦 (删除) - E键")
        self.poly_add_radio = QRadioButton("多边形 (增加) -F键")
        self.poly_erase_radio = QRadioButton("多边形 (删除) -G键")
        self.ai_radio = QRadioButton("绘制点 (AI) - P键")
        self.brush_radio.setChecked(True)

        self.tool_group.addButton(self.brush_radio, 0)
        self.tool_group.addButton(self.eraser_radio, 1)
        self.tool_group.addButton(self.poly_add_radio, 2)
        self.tool_group.addButton(self.poly_erase_radio, 3)
        self.tool_group.addButton(self.ai_radio, 4)
        self.tool_group.buttonClicked.connect(self.tool_changed)

        tools_layout.addWidget(self.brush_radio)
        tools_layout.addWidget(self.eraser_radio)
        tools_layout.addWidget(self.poly_add_radio)
        tools_layout.addWidget(self.poly_erase_radio)
        tools_layout.addWidget(self.ai_radio)

        # 画笔半宽/半高
        brush_w_layout = QHBoxLayout()
        brush_w_layout.addWidget(QLabel("画笔半宽:"))
        self.brush_width_spin = QSpinBox()
        self.brush_width_spin.setRange(1, 100)
        self.brush_width_spin.setValue(5)
        self.brush_width_spin.valueChanged.connect(self.brush_width_changed)
        brush_w_layout.addWidget(self.brush_width_spin)
        tools_layout.addLayout(brush_w_layout)

        brush_h_layout = QHBoxLayout()
        brush_h_layout.addWidget(QLabel("画笔半高:"))
        self.brush_height_spin = QSpinBox()
        self.brush_height_spin.setRange(1, 100)
        self.brush_height_spin.setValue(5)
        self.brush_height_spin.valueChanged.connect(self.brush_height_changed)
        brush_h_layout.addWidget(self.brush_height_spin)
        tools_layout.addLayout(brush_h_layout)
        
        # 旋转角度
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("旋转角度:"))
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(0, 360)
        self.rotation_slider.setValue(45)
        self.rotation_slider.valueChanged.connect(self.rotation_changed)
        rotation_layout.addWidget(self.rotation_slider)
        self.rotation_label = QLabel("45°")
        rotation_layout.addWidget(self.rotation_label)
        tools_layout.addLayout(rotation_layout)
        
        layout.addWidget(tools_group)
        
        # AI分割设置组
        ai_group = QGroupBox("AI分割设置")
        ai_layout = QVBoxLayout(ai_group)
        
        # SAM服务器配置
        sam_server_layout = QHBoxLayout()
        sam_server_layout.addWidget(QLabel("SAM服务器:"))
        self.sam_server_input = QLineEdit()
        self.sam_server_input.setText("http://segmentation.ensightful.xyz")
        self.sam_server_input.setPlaceholderText("http://IP:端口")
        sam_server_layout.addWidget(self.sam_server_input)
        ai_layout.addLayout(sam_server_layout)
        
        # AI操作按钮
        ai_buttons_layout = QHBoxLayout()
        self.ai_segment_btn = QPushButton("执行分割")
        self.ai_segment_btn.clicked.connect(self.run_ai_segmentation)
        self.ai_segment_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        ai_buttons_layout.addWidget(self.ai_segment_btn)
        
        self.ai_clear_btn = QPushButton("清除点")
        self.ai_clear_btn.clicked.connect(self.clear_ai_points)
        ai_buttons_layout.addWidget(self.ai_clear_btn)
        ai_layout.addLayout(ai_buttons_layout)
        
        # AI提示信息
        ai_tip = QLabel("左键:正样本点 右键:负样本点\n点击\"执行分割\"调用SAM")
        ai_tip.setStyleSheet("color: gray; font-size: 10px;")
        ai_tip.setWordWrap(True)
        ai_layout.addWidget(ai_tip)
        
        layout.addWidget(ai_group)
        
        # 显示设置组
        display_group = QGroupBox("显示设置")
        display_layout = QVBoxLayout(display_group)
        
        # Alpha设置
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("透明度:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self.alpha_changed)
        alpha_layout.addWidget(self.alpha_slider)
        self.alpha_label = QLabel("0.5")
        alpha_layout.addWidget(self.alpha_label)
        display_layout.addLayout(alpha_layout)
        
        # 缩放提示
        zoom_tip = QLabel("提示: Ctrl+滚轮缩放，右键拖拽平移")
        zoom_tip.setStyleSheet("color: gray; font-size: 10px;")
        zoom_tip.setWordWrap(True)
        display_layout.addWidget(zoom_tip)
        # 显示模式切换按钮
        self.display_toggle_btn = QPushButton("C键切换显示: Overlay")
        self.display_toggle_btn.clicked.connect(self.cycle_display_mode)
        display_layout.addWidget(self.display_toggle_btn)
        
        layout.addWidget(display_group)
        
        # 编辑操作组
        edit_group = QGroupBox("编辑操作")
        edit_layout = QVBoxLayout(edit_group)
        
        edit_buttons_layout = QHBoxLayout()
        self.undo_btn = QPushButton("撤销Ctrl+Z")
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn = QPushButton("重做Ctrl+Y")
        self.redo_btn.clicked.connect(self.redo)
        edit_buttons_layout.addWidget(self.undo_btn)
        edit_buttons_layout.addWidget(self.redo_btn)
        edit_layout.addLayout(edit_buttons_layout)

        # 清除当前图像标注（已有）
        clear_current_btn = QPushButton("清除当前标注 T键")
        clear_current_btn.clicked.connect(self.clear_all_annotations)
        edit_layout.addWidget(clear_current_btn)

        # 清除所有图像的标注（批量）
        #clear_all_btn = QPushButton("清除所有图像标注")
        #clear_all_btn.clicked.connect(self.clear_all_annotations_all_images)
        #edit_layout.addWidget(clear_all_btn)
        
        layout.addWidget(edit_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        return panel
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = self.addToolBar("主工具栏")
        
        # 加载动作
        load_action = QAction("加载图像", self)
        load_action.setShortcut(QKeySequence.Open)
        load_action.triggered.connect(self.load_images)
        toolbar.addAction(load_action)
        
        # 保存动作
        save_action = QAction("保存", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_current_annotation)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # 重置图像动作
        reset_action = QAction("重置图像", self)
        reset_action.setToolTip("重置缩放和位置到初始状态")
        reset_action.triggered.connect(self.reset_view)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()
        
        # 撤销/重做动作
        undo_action = QAction("撤销Ctrl+Z", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)
        
        redo_action = QAction("重做Ctrl+Y", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)
    
    def init_shortcuts(self):
        """初始化快捷键"""
        # 导航快捷键
        self.prev_btn.setShortcut(QKeySequence(Qt.Key_A))  # 改为A键
        self.next_btn.setShortcut(QKeySequence(Qt.Key_D))  # 改为D键
        
        # 工具切换快捷键
        self.brush_radio.setShortcut(QKeySequence(Qt.Key_S))  # 改为S键
        self.eraser_radio.setShortcut(QKeySequence(Qt.Key_E))
        self.ai_radio.setShortcut(QKeySequence(Qt.Key_P))  # 绘制点模式
        # 多边形快捷键 F/G（快速切换多边形增加/删除模式）
        # 直接在主窗口或canvas捕获 F/G，通过 focusPolicy 已允许 canvas 接收
        # 另外为显示切换添加快捷键 Shift+D
        toggle_seq = QKeySequence(Qt.SHIFT + Qt.Key_D)
        try:
            self.display_toggle_btn.setShortcut(toggle_seq)
        except Exception:
            pass
    
    def load_images(self):
        """加载图像文件夹"""
        if self.image_files and not self.maybe_save_pending_changes():
            return

        base_dir = QFileDialog.getExistingDirectory(
            self, "选择包含images和labels子文件夹的根目录", "", QFileDialog.ShowDirsOnly
        )
        
        if not base_dir:
            return
        
        # 检查是否存在images子文件夹
        images_subdir = os.path.join(base_dir, "images")
        if not os.path.exists(images_subdir):
            QMessageBox.warning(self, "警告", "选定目录下未找到'images'子文件夹")
            return
        
        # 查找图像文件
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_files = []
        
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(images_subdir, pattern)))
            #image_files.extend(glob.glob(os.path.join(images_subdir, pattern.upper())))
        
        if not image_files:
            QMessageBox.warning(self, "警告", "在images子文件夹中未找到图像文件")
            return
        
        self.image_files = sorted(image_files)
        self.images_dir = images_subdir  # 实际的images子文件夹
        
        # 设置标签文件夹 - labels是base_dir的子文件夹，与images平级
        self.labels_dir = os.path.join(base_dir, "labels")
        
        # 如果labels文件夹不存在，创建它
        if not os.path.exists(self.labels_dir):
            os.makedirs(self.labels_dir)
        
        self.current_index = 0
        self.load_current_image()
        
        # 显示正确的图像总数
        self.status_bar.showMessage(f"已加载 {len(self.image_files)} 张图像 (来自images文件夹)")
    
    def load_current_image(self):
        """加载当前图像"""
        if not self.image_files:
            return
        
        # 保存当前图像的撤销状态
        self.save_undo_state()
        
        image_path = self.image_files[self.current_index]
        image_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(image_name)[0]
        mask_path = os.path.join(self.labels_dir, f"{name_without_ext}.png")
        
        # 加载图像
        if self.canvas.set_images(image_path, mask_path):
            self.image_info_label.setText(
                f"图像 {self.current_index + 1}/{len(self.image_files)}\n{image_name}"
            )
            self.jump_input.setText(str(self.current_index + 1))
            
            # 更新按钮状态
            self.prev_btn.setEnabled(self.current_index > 0)
            self.next_btn.setEnabled(self.current_index < len(self.image_files) - 1)
            
            # 清空撤销/重做栈
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.update_undo_redo_buttons()
            
            self.unsaved_changes = False
    
    def previous_image(self):
        """上一张图像"""
        if self.current_index > 0:
            if not self.maybe_save_pending_changes():
                return
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """下一张图像"""
        if self.current_index < len(self.image_files) - 1:
            if not self.maybe_save_pending_changes():
                return
            self.current_index += 1
            self.load_current_image()
    
    def jump_to_image(self):
        """跳转到指定图像"""
        try:
            index = int(self.jump_input.text()) - 1
            if 0 <= index < len(self.image_files):
                if index == self.current_index:
                    return
                if not self.maybe_save_pending_changes():
                    return
                self.current_index = index
                self.load_current_image()
            else:
                QMessageBox.warning(self, "警告", "无效的图像编号")
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数字")
    
    def save_current_annotation(self) -> bool:
        """保存当前标注，成功返回 True"""
        if not self.image_files or self.canvas.mask_image is None:
            return False
        
        try:
            image_path = self.image_files[self.current_index]
            image_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(image_name)[0]
            mask_path = os.path.join(self.labels_dir, f"{name_without_ext}.png")

            self.auto_save_timer.stop()

            saved = cv2.imwrite(mask_path, self.canvas.mask_image)
            if not saved:
                raise IOError("cv2.imwrite 返回 False")

            self.unsaved_changes = False
            self.status_bar.showMessage(f"已保存: {mask_path}", 2000)
            return True

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
            return False
    
    def auto_save(self):
        """自动保存"""
        if self.unsaved_changes:
            self.save_current_annotation()
    
    def reset_view(self):
        """重置图像的缩放和位置到初始状态"""
        if hasattr(self.canvas, 'zoom_factor'):
            self.canvas.zoom_factor = 1.0
        if hasattr(self.canvas, 'image_pos'):
            self.canvas.image_pos = QPoint(0, 0)
        
        # 更新显示
        self.canvas.update_display()
        
        # 显示状态信息
        self.status_bar.showMessage("已重置图像的缩放和位置", 2000)
    
    def trigger_auto_save(self):
        """触发自动保存定时器"""
        self.auto_save_timer.stop()  # 停止之前的定时器
        self.auto_save_timer.start(self.auto_save_interval)  # 启动新的定时器

    def maybe_save_pending_changes(self) -> bool:
        """在切换图像或数据集前确保保存当前更改"""
        if not self.unsaved_changes:
            self.auto_save_timer.stop()
            return True

        if self.save_current_annotation():
            return True

        self.status_bar.showMessage("保存当前标注失败，已取消切换。", 4000)
        return False
    
    def tool_changed(self):
        """工具改变"""
        if self.brush_radio.isChecked():
            self.canvas.set_tool_mode("brush")
        elif self.eraser_radio.isChecked():
            self.canvas.set_tool_mode("eraser")
        elif self.poly_add_radio.isChecked():
            self.canvas.set_tool_mode("polygon_add")
        elif self.poly_erase_radio.isChecked():
            self.canvas.set_tool_mode("polygon_erase")
        elif self.ai_radio.isChecked():
            self.canvas.set_tool_mode("ai_segment")
    
    def brush_size_changed(self, size):
        """画笔大小改变"""
        self.canvas.set_brush_size(size)
    
    def rotation_changed(self, rotation):
        """旋转角度改变"""
        self.rotation_label.setText(f"{rotation}°")
        self.canvas.set_cursor_rotation(rotation)
    
    def alpha_changed(self, value):
        """透明度改变"""
        alpha = value / 100.0
        self.alpha_label.setText(f"{alpha:.2f}")
        self.canvas.set_alpha(alpha)

    def cycle_display_mode(self):
        """在 overlay -> original -> mask 之间切换显示模式"""
        modes = ['overlay', 'original', 'mask']
        cur = self.canvas.display_mode if hasattr(self.canvas, 'display_mode') else 'overlay'
        try:
            idx = modes.index(cur)
        except ValueError:
            idx = 0
        next_mode = modes[(idx + 1) % len(modes)]
        self.canvas.set_display_mode(next_mode)
        # 更新按钮文字
        self.display_toggle_btn.setText(f"C键切换显示: {next_mode.capitalize()}")

    def clear_all_annotations_all_images(self):
        """清除已加载数据集下所有图像的标注（labels文件夹内所有png置零）"""
        if not self.image_files or not self.labels_dir:
            QMessageBox.information(self, "信息", "尚未加载任何图像文件夹")
            return

        reply = QMessageBox.question(
            self, "确认", "确定要清除当前数据集中所有图像的标注吗？此操作不可逆。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 保存当前状态以便撤销
        self.save_undo_state()

        count = 0
        for img_path in self.image_files:
            name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.labels_dir, f"{name}.png")
            try:
                # 写入全零mask，确保与原图大小匹配
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                zero_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.imwrite(mask_path, zero_mask)
                count += 1
            except Exception:
                continue

        # 重新加载当前mask并刷新显示
        self.canvas.set_images(self.image_files[self.current_index], os.path.join(self.labels_dir, f"{os.path.splitext(os.path.basename(self.image_files[self.current_index]))[0]}.png"))
        self.status_bar.showMessage(f"已清除 {count} 张图像的标注", 4000)
        self.unsaved_changes = True
    
    def save_undo_state(self):
        """保存撤销状态"""
        if self.canvas.mask_image is not None:
            # 限制撤销栈大小
            if len(self.undo_stack) >= self.max_undo_steps:
                self.undo_stack.pop(0)
            
            self.undo_stack.append(self.canvas.mask_image.copy())
            self.redo_stack.clear()  # 清空重做栈
            self.update_undo_redo_buttons()
    
    def undo(self):
        """撤销操作"""
        if self.undo_stack and self.canvas.mask_image is not None:
            # 保存当前状态到重做栈
            self.redo_stack.append(self.canvas.mask_image.copy())
            
            # 恢复上一个状态
            previous_state = self.undo_stack.pop()
            self.canvas.set_mask(previous_state)
            
            self.update_undo_redo_buttons()
            self.unsaved_changes = True
    
    def redo(self):
        """重做操作"""
        if self.redo_stack and self.canvas.mask_image is not None:
            # 保存当前状态到撤销栈
            self.undo_stack.append(self.canvas.mask_image.copy())
            
            # 恢复重做状态
            next_state = self.redo_stack.pop()
            self.canvas.set_mask(next_state)
            
            self.update_undo_redo_buttons()
            self.unsaved_changes = True
    
    def update_undo_redo_buttons(self):
        """更新撤销/重做按钮状态"""
        self.undo_btn.setEnabled(len(self.undo_stack) > 0)
        self.redo_btn.setEnabled(len(self.redo_stack) > 0)
    
    def clear_all_annotations(self):
        """清除当前图像的所有标注（带撤销栈备份）"""
        reply = QMessageBox.question(
            self, "确认", "确定要清除当前图像的所有标注吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes and self.canvas.mask_image is not None:
            self.save_undo_state()
            # 进行单图备份
            self._backup_current_mask()
            self.canvas.mask_image.fill(0)
            self.canvas.update_display()
            self.unsaved_changes = True

    def _backup_current_mask(self):
        """将当前图像的 mask 备份到 labels/backup/<timestamp>/ 下，保留原文件名"""
        if not self.image_files:
            return None
        img_path = self.image_files[self.current_index]
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.labels_dir, f"{name}.png")
        if not os.path.exists(mask_path):
            return None

        backup_dir = os.path.join(self.labels_dir, 'backup', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        try:
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(mask_path, os.path.join(backup_dir, os.path.basename(mask_path)))
            self.status_bar.showMessage(f"已备份当前mask到 {backup_dir}", 3000)
            return backup_dir
        except Exception as e:
            QMessageBox.warning(self, "备份失败", f"无法备份当前mask: {e}")
            return None

    def clear_current_annotation_with_backup(self):
        """为快捷键 T 使用：备份并清除当前图像的 mask（等同 clear_all_annotations 的单图版本）"""
        if self.canvas.mask_image is None:
            QMessageBox.information(self, "信息", "当前没有加载mask可清除")
            return

        reply = QMessageBox.question(
            self, "确认", "确定要备份并清除当前图像的标注吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.save_undo_state()
        self._backup_current_mask()
        self.canvas.mask_image.fill(0)
        self.canvas.update_display()
        self.unsaved_changes = True

    def clear_ai_points(self):
        """清除AI标注点"""
        self.canvas.ai_points = []
        self.canvas.ai_mode_active = False
        self.canvas.update()
        self.status_bar.showMessage("已清除AI标注点", 2000)

    def run_ai_segmentation(self):
        """执行AI分割"""
        if not self.canvas.ai_points:
            QMessageBox.warning(self, "警告", "请先在图像上标注点（左键:正样本，右键:负样本）")
            return
        
        if self.canvas.original_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
        
        # 获取SAM服务器配置
        server_url = self.sam_server_input.text().strip()
        model = "vit_h"  # 固定使用vit_h模型
        
        if not server_url:
            QMessageBox.warning(self, "警告", "请配置SAM服务器地址")
            return
        
        try:
            # 编码图像
            self.status_bar.showMessage("正在编码图像...", 0)
            QApplication.processEvents()
            
            # 将numpy图像转为PIL Image再转base64
            pil_image = Image.fromarray(self.canvas.original_image)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            image_data = base64.b64encode(img_buffer.read()).decode('utf-8')
            image_base64 = f"data:image/jpeg;base64,{image_data}"
            
            # 准备请求数据
            data = {
                "model": model,
                "prompt_type": "point",
                "image": image_base64,
                "use_tensorrt": True,
                "alpha": 0.5,
                "points": self.canvas.ai_points,  # [[x, y, label], ...]
            }
            
            self.status_bar.showMessage(f"正在调用SAM服务 ({len(self.canvas.ai_points)}个点)...", 0)
            QApplication.processEvents()
            
            # 发送请求
            response = requests.post(f"{server_url}/api/segment", json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # 检查API返回格式
                if not result.get('success', False):
                    error_msg = result.get('error', 'Unknown error')
                    QMessageBox.critical(self, "错误", f"SAM服务返回错误: {error_msg}")
                    self.status_bar.showMessage(f"分割失败: {error_msg}", 3000)
                    return
                
                self.status_bar.showMessage(
                    f"分割成功! 耗时: {result['performance']['total_time_ms']:.0f}ms", 
                    3000
                )
                
                # 解析mask - 检查新的返回格式
                result_data = result.get('result', {})
                if 'masks' in result_data:
                    masks_data = result_data['masks']
                    if masks_data:
                        # 取第一个mask（最佳结果）
                        mask_item = masks_data[0]
                        
                        # 处理不同的返回格式
                        if isinstance(mask_item, dict):
                            # 如果是字典格式，可能包含data字段
                            mask_base64 = mask_item.get('data', mask_item.get('mask', ''))
                        elif isinstance(mask_item, str):
                            # 如果是字符串格式（base64）
                            mask_base64 = mask_item
                        else:
                            QMessageBox.warning(self, "警告", f"未知的mask格式: {type(mask_item)}")
                            return
                        
                        # 解码base64
                        if isinstance(mask_base64, str):
                            if ',' in mask_base64:
                                mask_base64 = mask_base64.split(',')[1]
                            mask_bytes = base64.b64decode(mask_base64)
                        else:
                            QMessageBox.warning(self, "警告", f"mask_base64不是字符串: {type(mask_base64)}")
                            return
                        
                        # 转为numpy数组
                        mask_pil = Image.open(io.BytesIO(mask_bytes))
                        new_mask = np.array(mask_pil)
                        
                        # 如果是RGB，转为灰度
                        if len(new_mask.shape) == 3:
                            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_RGB2GRAY)
                        
                        # 二值化处理
                        new_mask = (new_mask > 0).astype(np.uint8) * 255
                        
                        # 调整到原图尺寸
                        if new_mask.shape != self.canvas.mask_image.shape:
                            new_mask = cv2.resize(
                                new_mask, 
                                (self.canvas.mask_image.shape[1], self.canvas.mask_image.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        
                        # 保存撤销状态
                        self.save_undo_state()
                        
                        # 叠加到现有mask（或运算）
                        self.canvas.mask_image = np.maximum(self.canvas.mask_image, new_mask)
                        self.canvas.update_display()
                        self.unsaved_changes = True
                        
                        # 清除AI点
                        self.canvas.ai_points = []
                        self.canvas.ai_mode_active = False
                        self.canvas.update()
                        
                        num_masks = result_data.get('info', {}).get('num_masks', len(masks_data))
                        QMessageBox.information(
                            self, 
                            "成功", 
                            f"AI分割完成并已叠加到现有mask\n生成了 {num_masks} 个mask"
                        )
                    else:
                        QMessageBox.warning(self, "警告", "服务器返回的mask为空")
                else:
                    QMessageBox.warning(self, "警告", "服务器响应中没有masks数据")
            else:
                QMessageBox.critical(
                    self, 
                    "错误", 
                    f"SAM服务调用失败 ({response.status_code})\n{response.text[:200]}"
                )
                self.status_bar.showMessage(f"分割失败: {response.status_code}", 3000)
                
        except requests.exceptions.Timeout:
            QMessageBox.critical(self, "错误", "请求超时，请检查服务器是否正常运行")
            self.status_bar.showMessage("分割请求超时", 3000)
        except requests.exceptions.ConnectionError:
            QMessageBox.critical(self, "错误", f"无法连接到SAM服务器: {server_url}")
            self.status_bar.showMessage("连接失败", 3000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"AI分割失败: {str(e)}")
            self.status_bar.showMessage(f"分割失败: {str(e)}", 3000)
            import traceback
            traceback.print_exc()
        self.canvas.update_display()
        self.unsaved_changes = True
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self, "确认退出", "有未保存的更改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                self.save_current_annotation()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("图像语义标注工具 V0.2")
    app.setOrganizationName("Image Annotation Tools")
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = AnnotationTool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()