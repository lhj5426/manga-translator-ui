"""
区域文本图形项 — Qt Graphics Item 层。

核心设计：
- RegionTextItem 的 pos() = 源区域中心（world），rotation() = angle
- 局部坐标系：以源区域中心为原点，线段/白框均用局部坐标表示
- 白框用于定义文字渲染边界，是用户可手动调整的矩形
- 文字 pixmap 以 render_center（白框中心的世界坐标）为锚点定位

旋转 bug 修复说明：
  旧实现中 update_text_pixmap 用 mapFromScene(pos) 定位 pixmap 左上角。
  但 pos 是世界坐标轴对齐矩形的左上角，经 mapFromScene（含逆旋转）后，
  pixmap 中心在局部坐标中会偏离白框中心。
  修复：以 render_center → mapFromScene → 局部中心 → 减半尺寸，
  让 pixmap 始终以白框中心为定位基准。
"""

import copy
import logging
import traceback
from typing import List

import numpy as np
from editor.desktop_ui_geometry import (
    calculate_new_edge_on_drag,
    calculate_new_vertices_on_drag,
    rotate_point,
)
from editor.geometry_commit_pipeline import (
    build_rotate_region_data,
    build_white_frame_region_data,
)
from editor.region_geometry_state import RegionGeometryState
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QCursor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QStyle,
)

logger = logging.getLogger("manga_translator")


# ======================================================================
# 辅助类
# ======================================================================

class TransparentPixmapItem(QGraphicsPixmapItem):
    """对鼠标事件完全透明的 Pixmap item，不阻挡父 item 的选择。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

    def shape(self) -> QPainterPath:
        return QPainterPath()


# ======================================================================
# 主图形项
# ======================================================================

class RegionTextItem(QGraphicsItemGroup):

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self, region_data, region_index, geometry_callback, parent=None):
        super().__init__(parent)
        self.region_data = copy.deepcopy(region_data)
        self.region_index = region_index
        self.geometry_callback = geometry_callback

        self._image_item = None          # 图像项引用，用于坐标转换
        self._in_callback = False        # 防止回调重入

        # 单一几何状态源：geo
        self.geo = RegionGeometryState.from_region_data(self.region_data)

        # Qt 坐标：pos = 源区域中心，rotation = angle
        self.rotation_angle = float(self.geo.angle)
        self.visual_center = QPointF(
            float(self.geo.center[0]),
            float(self.geo.center[1]),
        )
        self.setPos(self.visual_center)
        self.setRotation(self.rotation_angle)
        self.setTransformOriginPoint(QPointF(0, 0))

        # 构建局部坐标多边形
        self.polygons: List[QPolygonF] = []
        self._rebuild_qt_polygons()

        # 文字 pixmap 子项
        self.text_item = TransparentPixmapItem(self)
        self.text_item.setZValue(-1)

        # 交互状态
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self._interaction_mode = "none"
        self._is_dragging = False

        # 拖拽状态变量
        self._drag_handle_indices = None
        self._drag_start_pos = QPointF()
        self._drag_start_polygons: List[QPolygonF] = []
        self._drag_start_rotation = 0.0
        self._drag_start_visual_center = QPointF()
        self._drag_start_white_frame_local = None
        self._drag_start_scene_pos = QPointF()
        self._drag_start_pivot_scene = QPointF()
        self._drag_start_text_item_pos = QPointF()
        self._drag_start_white_handle_world = None

        # 显示
        self._polygons_visible = True
        self._show_white_box = True
        self._shape_path = None

        self._setup_pens()

    # ------------------------------------------------------------------
    # 内部构建
    # ------------------------------------------------------------------

    def _rebuild_qt_polygons(self):
        """从 geo.polygons_local 重建 Qt QPolygonF 列表。"""
        self.polygons = []
        for local_poly_data in self.geo.polygons_local:
            poly = QPolygonF()
            for x, y in local_poly_data:
                poly.append(QPointF(x, y))
            self.polygons.append(poly)

    def _setup_pens(self):
        self.white_pen = QPen(QColor("white"), 3)
        self.white_pen.setStyle(Qt.PenStyle.DashLine)
        self.white_brush = QBrush(Qt.BrushStyle.NoBrush)

    # ------------------------------------------------------------------
    # 坐标转换（图像↔场景）
    # ------------------------------------------------------------------

    def set_image_item(self, item):
        self._image_item = item

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def _apply_region_state(self, region_data: dict):
        """将完整 region_data 同步到 item 本地状态。"""
        self.region_data = copy.deepcopy(region_data)
        self.geo = RegionGeometryState.from_region_data(self.region_data)

        self.rotation_angle = float(self.geo.angle)
        self.visual_center = QPointF(
            float(self.geo.center[0]),
            float(self.geo.center[1]),
        )

        self.prepareGeometryChange()
        self._shape_path = None

        self.setPos(self.visual_center)
        self.setRotation(self.rotation_angle)
        self.setTransformOriginPoint(QPointF(0, 0))
        self._rebuild_qt_polygons()

    def update_from_data(self, region_data: dict):
        """从新的 region_data 更新整个 item 状态。"""
        try:
            if self._is_dragging or self._in_callback:
                return
            if not self.scene() or not hasattr(self, "region_data"):
                return

            old_rect = self.sceneBoundingRect() if self.scene() else None
            was_selected = self.isSelected()

            # 模型数据是唯一事实来源，避免旧 item 残留状态污染
            self._apply_region_state(region_data)

            if was_selected != self.isSelected():
                self.setSelected(was_selected)

            self.update()
            self._invalidate_scene_rect(old_rect)

        except (RuntimeError, AttributeError) as e:
            logger.debug(f"[RegionTextItem] update_from_data: {e}")
        except Exception as e:
            logger.error(f"[RegionTextItem] update_from_data: {e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # 核心修复: 文字 pixmap 定位
    # ------------------------------------------------------------------

    def update_text_pixmap(self, pixmap, pos, rotation=0.0, pivot_point=None, render_center=None):
        """更新文字 pixmap 位置。

        当 render_center 可用时（白框中心世界坐标），以此为锚点居中放置 pixmap，
        避免旋转变换导致 mapFromScene(pos) 引起的中心偏移。
        """
        self.text_item.setPixmap(pixmap)

        if render_center is not None and pixmap and not pixmap.isNull():
            # 以白框中心为锚点定位
            local_center = self.mapFromScene(
                QPointF(float(render_center[0]), float(render_center[1]))
            )
            local_pos = QPointF(
                local_center.x() - pixmap.width() / 2.0,
                local_center.y() - pixmap.height() / 2.0,
            )
        else:
            # 回退：直接映射世界坐标左上角
            local_pos = self.mapFromScene(QPointF(float(pos.x()), float(pos.y())))

        self.text_item.setPos(local_pos)
        self.text_item.setTransformOriginPoint(self.text_item.boundingRect().center())
        self.text_item.setRotation(0)   # 父 item 已旋转

    def set_dst_points(self, dst_points):
        """设置渲染 dst_points，自动模式下同步到白框。"""
        self.geo.set_render_box(dst_points)
        self.prepareGeometryChange()
        self._shape_path = None
        self.update()

    # ------------------------------------------------------------------
    # ID / 序列化
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 可见性
    # ------------------------------------------------------------------

    def set_text_visible(self, visible: bool):
        self.text_item.setVisible(visible)

    def set_box_visible(self, visible: bool):
        if self._polygons_visible != visible:
            self._polygons_visible = visible
            self.update()

    def set_white_box_visible(self, visible: bool):
        self._show_white_box = visible
        self.update()

    # ------------------------------------------------------------------
    # Qt 必需重写: shape / boundingRect / paint
    # ------------------------------------------------------------------

    def shape(self) -> QPainterPath:
        try:
            if self._shape_path is not None:
                return self._shape_path

            path = QPainterPath()
            if self._show_white_box and self.geo.white_frame_local is not None:
                left, top, right, bottom = self.geo.white_frame_local
                path.addRect(QRectF(left, top, right - left, bottom - top))

                if self.isSelected():
                    metrics = self._white_handle_metrics()
                    r = metrics["hit_radius"]
                    for p in self._white_corner_points() + self._white_edge_points():
                        path.addEllipse(p, r, r)
                    ri = self._rotate_handle_info()
                    rr = (ri["handle_size"] / 2.0) + (4.0 / ri["lod"])
                    path.addEllipse(ri["rot_pos"], rr, rr)
                    path.moveTo(ri["center"])
                    path.lineTo(ri["rot_pos"])
            else:
                path = self._core_polygon_path()

            self._shape_path = path
            return path
        except Exception as e:
            logger.error(f"[RegionTextItem] shape: {e}\n{traceback.format_exc()}")
            return QPainterPath()

    def boundingRect(self) -> QRectF:
        try:
            return self.shape().boundingRect().adjusted(-10, -10, 10, 10)
        except Exception:
            return QRectF(0, 0, 100, 100)

    def paint(self, painter, option, widget=None):
        try:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            is_sel = bool(option.state & QStyle.StateFlag.State_Selected)

            if self._show_white_box and self._polygons_visible and self.geo.white_frame_local is not None:
                self._draw_white_box(painter, is_sel)

            if is_sel and self._show_white_box and self.geo.white_frame_local is not None and self._polygons_visible:
                ri = self._rotate_handle_info()
                hs = ri["handle_size"]
                pw = ri["pen_width"]

                painter.setPen(QPen(QColor("red"), pw * 1.5))
                painter.drawLine(ri["center"], ri["rot_pos"])
                painter.setBrush(QBrush(QColor("red")))
                painter.setPen(QPen(QColor("white"), pw))
                painter.drawEllipse(
                    int(ri["rot_pos"].x() - hs / 2),
                    int(ri["rot_pos"].y() - hs / 2),
                    int(hs), int(hs),
                )
                self._draw_white_handles(painter)

            painter.restore()
        except Exception as e:
            logger.error(f"[RegionTextItem] paint: {e}\n{traceback.format_exc()}")
            try:
                painter.restore()
            except Exception:
                pass

    def itemChange(self, change, value):
        try:
            if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
                self.prepareGeometryChange()
                self._shape_path = None
        except Exception as e:
            logger.error(f"[RegionTextItem] itemChange: {e}")
        return super().itemChange(change, value)

    # ------------------------------------------------------------------
    # 白框绘制辅助
    # ------------------------------------------------------------------

    def _white_corner_points(self) -> list:
        wf = self.geo.white_frame_local
        if wf is None:
            return []
        l, t, r, b = wf
        return [QPointF(l, t), QPointF(r, t), QPointF(r, b), QPointF(l, b)]

    def _white_edge_points(self) -> list:
        wf = self.geo.white_frame_local
        if wf is None:
            return []
        l, t, r, b = wf
        return [
            QPointF((l + r) / 2, t), QPointF(r, (t + b) / 2),
            QPointF((l + r) / 2, b), QPointF(l, (t + b) / 2),
        ]

    def _draw_white_box(self, painter, is_selected: bool):
        wf = self.geo.white_frame_local
        if wf is None:
            return
        l, t, r, b = wf
        poly = QPolygonF([QPointF(l, t), QPointF(r, t), QPointF(r, b), QPointF(l, b)])

        if is_selected:
            painter.setPen(QPen(QColor(0, 255, 255), 4))
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawPolygon(poly)
            painter.setPen(QPen(QColor("black"), 2))
            painter.drawPolygon(poly)
        else:
            # 非选中态使用高可见颜色，避免在浅色背景上“看不见”
            pen = QPen(QColor(0, 255, 180), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawPolygon(poly)

    def _draw_white_handles(self, painter):
        mtx = self._white_handle_metrics()
        hs = mtx["visual_size"]
        pw = mtx["pen_width"]
        half = hs / 2.0

        painter.setBrush(QBrush(QColor(255, 255, 100)))
        painter.setPen(QPen(QColor("black"), pw))
        for p in self._white_corner_points():
            painter.drawRect(int(p.x() - half), int(p.y() - half), int(hs), int(hs))

        painter.setBrush(QBrush(QColor(255, 165, 0)))
        painter.setPen(QPen(QColor("black"), pw))
        for p in self._white_edge_points():
            painter.drawEllipse(int(p.x() - half), int(p.y() - half), int(hs), int(hs))

    # ------------------------------------------------------------------
    # 几何 / 手柄参数
    # ------------------------------------------------------------------

    def _lod(self) -> float:
        if self.scene() and self.scene().views():
            return self.scene().views()[0].transform().m11()
        return 1.0

    def _rotate_handle_info(self) -> dict:
        lod = self._lod()
        hs = 10.0 / lod
        if self.geo.white_frame_local is not None:
            l, t, r, b = self.geo.white_frame_local
            cx = (l + r) / 2
            cy = (t + b) / 2
            center = QPointF(cx, cy)
            rot_pos = QPointF(cx, t - 40.0 / lod)
        else:
            center = QPointF(0, 0)
            rot_pos = QPointF(0, -40.0 / lod)
        return {
            "lod": lod, "handle_size": hs,
            "pen_width": 1.5 / lod,
            "center": center, "rot_pos": rot_pos,
        }

    def _white_handle_metrics(self) -> dict:
        lod = self._lod()
        vs = 12.0 / lod
        return {
            "lod": lod, "visual_size": vs,
            "hit_radius": (vs / 2.0) + (4.0 / lod),
            "pen_width": max(1.0 / lod, 1.0),
        }

    def _rotation_pivot_local(self) -> QPointF:
        """旋转支点：白框中心（回退到局部原点）。"""
        if self.geo.white_frame_local is not None:
            l, t, r, b = self.geo.white_frame_local
            return QPointF((l + r) / 2, (t + b) / 2)
        return QPointF(0, 0)

    def _core_polygon_path(self) -> QPainterPath:
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        seen = set()
        for poly in self.polygons:
            if not poly.isEmpty():
                key = tuple((p.x(), p.y()) for p in poly)
                if key not in seen:
                    path.addPolygon(poly)
                    path.closeSubpath()
                    seen.add(key)
        return path

    # ------------------------------------------------------------------
    # 手柄命中检测
    # ------------------------------------------------------------------

    def _get_handle_at(self, pos: QPointF):
        ri = self._rotate_handle_info()
        rot_hit_r = (ri["handle_size"] / 2.0) + (4.0 / ri["lod"])
        dx = ri["rot_pos"].x() - pos.x()
        dy = ri["rot_pos"].y() - pos.y()
        if dx * dx + dy * dy <= rot_hit_r * rot_hit_r:
            return "rotate", (-1, -1)

        if self._show_white_box and self.geo.white_frame_local is not None:
            result = self._white_handle_at(pos)
            if result[0] is not None:
                return result
            if self._point_in_white_frame(pos):
                return "white_move", (0, 0)

        return None, (-1, -1)

    def _white_handle_at(self, pos: QPointF):
        mtx = self._white_handle_metrics()
        hr_sq = mtx["hit_radius"] ** 2

        for i, p in enumerate(self._white_corner_points()):
            if (p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2 <= hr_sq:
                return "white_corner", i

        for i, p in enumerate(self._white_edge_points()):
            if (p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2 <= hr_sq:
                return "white_edge", i

        return None, (-1, -1)

    def _point_in_white_frame(self, pos: QPointF) -> bool:
        wf = self.geo.white_frame_local
        if wf is None:
            return False
        return wf[0] <= pos.x() <= wf[2] and wf[1] <= pos.y() <= wf[3]

    # ------------------------------------------------------------------
    # 提交入口
    # ------------------------------------------------------------------

    def _commit_region_data(self, new_data: dict):
        self.region_data = copy.deepcopy(new_data)

    def _emit_region_update(self, event, new_data: dict):
        cb = self.geometry_callback
        idx = self.region_index
        self._commit_region_data(new_data)
        super().mouseReleaseEvent(event)
        self._in_callback = True
        try:
            cb(idx, new_data)
        finally:
            self._in_callback = False

    # ------------------------------------------------------------------
    # 鼠标交互
    # ------------------------------------------------------------------

    def hoverMoveEvent(self, event: QGraphicsSceneMouseEvent):
        try:
            if not self.scene():
                super().hoverMoveEvent(event)
                return
            views = self.scene().views()
            view = views[0] if views else None
            if view and hasattr(view, "_active_tool") and view._active_tool in ("pen", "brush", "eraser"):
                super().hoverMoveEvent(event)
                return

            if self.isSelected():
                handle, idx = self._get_handle_at(event.pos())
                cursor_map = {
                    "rotate": Qt.CursorShape.SizeAllCursor,
                    "white_move": Qt.CursorShape.SizeAllCursor,
                }
                if handle == "white_corner":
                    cursor_map["white_corner"] = (
                        Qt.CursorShape.SizeFDiagCursor if idx in (0, 2)
                        else Qt.CursorShape.SizeBDiagCursor
                    )
                elif handle == "white_edge":
                    cursor_map["white_edge"] = (
                        Qt.CursorShape.SizeVerCursor if idx in (0, 2)
                        else Qt.CursorShape.SizeHorCursor
                    )
                cursor = cursor_map.get(handle)
                if cursor:
                    self._apply_hover_cursor(cursor)
                else:
                    self._clear_hover_cursor()
            else:
                self._clear_hover_cursor()
            super().hoverMoveEvent(event)
        except Exception as e:
            logger.debug(f"[RegionTextItem] hoverMoveEvent: {e}")

    def hoverLeaveEvent(self, event):
        self._clear_hover_cursor()
        super().hoverLeaveEvent(event)

    def _apply_hover_cursor(self, shape):
        self.setCursor(shape)
        if self.scene() and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, "_active_tool") and view._active_tool == "select":
                view.viewport().setCursor(QCursor(shape))

    def _clear_hover_cursor(self):
        self.unsetCursor()
        if self.scene() and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, "_active_tool") and view._active_tool == "select":
                view.viewport().unsetCursor()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        try:
            if not self.scene():
                super().mousePressEvent(event)
                return

            local_pos = event.pos()

            if event.button() == Qt.MouseButton.LeftButton:
                views = self.scene().views()
                view = views[0] if views else None
                if not view or not hasattr(view, "model"):
                    super().mousePressEvent(event)
                    return

                was_selected = self.isSelected()
                handle, indices = (None, (-1, -1))
                if was_selected:
                    handle, indices = self._get_handle_at(local_pos)
                super().mousePressEvent(event)

                if was_selected and self.isSelected():
                    self._save_drag_start_state(event, local_pos, handle, indices)
                else:
                    self._interaction_mode = "none"
                    self._is_dragging = False
                    self._drag_handle_indices = None
                event.accept()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                if self.isSelected():
                    event.accept()
                    return
                else:
                    super().mousePressEvent(event)
                    return
            else:
                super().mousePressEvent(event)
        except Exception as e:
            logger.error(f"[RegionTextItem] mousePressEvent: {e}\n{traceback.format_exc()}")

    def _save_drag_start_state(self, event, local_pos, handle, indices):
        """保存拖动开始时的所有状态。"""
        self._is_dragging = bool(handle)
        self._drag_start_pos = local_pos
        self._drag_start_scene_pos = event.scenePos()
        self._drag_start_polygons = [QPolygonF(p) for p in self.polygons]
        self._drag_start_rotation = self.rotation()
        self._drag_start_angle = self.rotation_angle
        self._drag_start_center = self._rotation_pivot_local()
        self._drag_start_scene_rect = self.sceneBoundingRect() if self.scene() else None
        self._drag_start_visual_center = QPointF(self.visual_center)

        if handle:
            self._interaction_mode = handle
            self._drag_handle_indices = indices

            if handle in ("white_corner", "white_edge"):
                self._drag_start_white_frame_local = (
                    list(self.geo.white_frame_local) if self.geo.white_frame_local else None
                )
                self._drag_start_white_handle_world = self._white_handle_world_at_start()
            elif handle == "white_move":
                self._drag_start_white_frame_local = (
                    list(self.geo.white_frame_local) if self.geo.white_frame_local else None
                )
                self._drag_start_text_item_pos = QPointF(self.text_item.pos())
            elif handle == "rotate":
                self.setTransformOriginPoint(QPointF(0, 0))
                self._drag_start_pivot_scene = self.mapToScene(self._drag_start_center)
                vec = event.scenePos() - self._drag_start_pivot_scene
                self._drag_start_angle_rad = np.arctan2(vec.y(), vec.x())
            else:
                self._interaction_mode = "none"
                self._is_dragging = False
        else:
            self._interaction_mode = "none"
            self._is_dragging = False

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        try:
            if self._interaction_mode == "none":
                super().mouseMoveEvent(event)
                return

            if self._interaction_mode == "rotate":
                self._handle_rotate_drag(event)
            elif self._interaction_mode in ("white_corner", "white_edge"):
                self._handle_white_frame_edit(event)
            elif self._interaction_mode == "white_move":
                self._handle_white_frame_move(event)
            else:
                super().mouseMoveEvent(event)
            event.accept()
        except Exception as e:
            logger.error(f"[RegionTextItem] mouseMoveEvent: {e}\n{traceback.format_exc()}")

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        try:
            if self._interaction_mode == "none":
                self._is_dragging = False
                super().mouseReleaseEvent(event)
                return

            mode = self._interaction_mode
            self._interaction_mode = "none"

            if mode == "rotate":
                self._commit_rotation(event)
            elif mode in ("white_corner", "white_edge", "white_move"):
                self._commit_white_frame(event, mode)
            else:
                super().mouseReleaseEvent(event)
            self._is_dragging = False
        except Exception as e:
            self._interaction_mode = "none"
            self._is_dragging = False
            logger.error(f"[RegionTextItem] mouseReleaseEvent: {e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # 旋转拖拽
    # ------------------------------------------------------------------

    def _handle_rotate_drag(self, event):
        center_scene = self._drag_start_pivot_scene
        vec = event.scenePos() - center_scene
        new_angle_rad = np.arctan2(vec.y(), vec.x())
        delta_deg = np.degrees(new_angle_rad - self._drag_start_angle_rad)
        new_rot = self._drag_start_rotation + delta_deg
        self.setRotation(new_rot)

        # 保持白框中心（局部点）在场景中不动
        theta = np.radians(new_rot)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        px, py = self._drag_start_center.x(), self._drag_start_center.y()
        self.setPos(QPointF(
            center_scene.x() - (px * cos_t - py * sin_t),
            center_scene.y() - (px * sin_t + py * cos_t),
        ))
        self.visual_center = QPointF(self.pos())

    def _commit_rotation(self, event):
        new_angle = self.rotation()
        self.rotation_angle = float(new_angle)

        old_center = self.geo.center
        if not (isinstance(old_center, (list, tuple)) and len(old_center) >= 2):
            old_center = [self.visual_center.x(), self.visual_center.y()]
        delta_x = float(self.pos().x()) - float(old_center[0])
        delta_y = float(self.pos().y()) - float(old_center[1])

        new_lines = []
        for poly in self.region_data.get("lines", []):
            new_poly = []
            for p in poly:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    new_poly.append([float(p[0]) + delta_x, float(p[1]) + delta_y])
            if new_poly:
                new_lines.append(new_poly)

        new_cx, new_cy = float(self.pos().x()), float(self.pos().y())
        self.visual_center = QPointF(new_cx, new_cy)

        # 同步 geo 状态：更新 center、lines、angle，并重建 polygons_local 和白框
        self.geo.center = [new_cx, new_cy]
        self.geo.angle = float(new_angle)
        if new_lines:
            self.geo.lines = new_lines
        self.geo._rebuild_polygons_local()
        if not self.geo.has_custom_white_frame:
            self.geo._auto_update_white_frame()

        new_data = build_rotate_region_data(
            self.region_data, new_angle,
            new_center=[new_cx, new_cy],
            new_lines=new_lines or None,
        )
        self._emit_region_update(event, new_data)

    # ------------------------------------------------------------------
    # 白框编辑
    # ------------------------------------------------------------------

    def _handle_white_frame_edit(self, event: QGraphicsSceneMouseEvent):
        try:
            if self._drag_handle_indices is None or self._drag_start_white_frame_local is None:
                return

            l0, t0, r0, b0 = self._drag_start_white_frame_local
            cx, cy = self.geo.center
            start_verts = [
                [l0 + cx, t0 + cy], [r0 + cx, t0 + cy],
                [r0 + cx, b0 + cy], [l0 + cx, b0 + cy],
            ]

            delta = event.scenePos() - self._drag_start_scene_pos
            if self._drag_start_white_handle_world is not None:
                bx, by = self._drag_start_white_handle_world
                mouse = (bx + delta.x(), by + delta.y())
            else:
                mouse = (event.scenePos().x(), event.scenePos().y())

            if self._interaction_mode == "white_corner":
                new_verts = calculate_new_vertices_on_drag(
                    start_verts, self._drag_handle_indices,
                    mouse, self.rotation_angle, (cx, cy),
                )
            elif self._interaction_mode == "white_edge":
                new_verts = calculate_new_edge_on_drag(
                    start_verts, self._drag_handle_indices,
                    mouse, self.rotation_angle, (cx, cy),
                )
            else:
                return

            if not new_verts or len(new_verts) != 4:
                return

            nl = min(p[0] for p in new_verts) - cx
            nr = max(p[0] for p in new_verts) - cx
            nt = min(p[1] for p in new_verts) - cy
            nb = max(p[1] for p in new_verts) - cy
            min_s = 8.0
            if nr - nl < min_s:
                e = (min_s - (nr - nl)) / 2.0
                nl -= e
                nr += e
            if nb - nt < min_s:
                e = (min_s - (nb - nt)) / 2.0
                nt -= e
                nb += e

            old_rect = self.sceneBoundingRect() if self.scene() else None
            self.prepareGeometryChange()
            self._shape_path = None
            self.geo.set_custom_white_frame_local([nl, nt, nr, nb])
            self.update()
            self._invalidate_scene_rect(old_rect)

        except Exception as e:
            logger.error(f"[RegionTextItem] _handle_white_frame_edit: {e}\n{traceback.format_exc()}")

    def _handle_white_frame_move(self, event: QGraphicsSceneMouseEvent):
        try:
            if self._drag_start_white_frame_local is None:
                return

            old_rect = self.sceneBoundingRect() if self.scene() else None
            scene_delta = event.scenePos() - self._drag_start_scene_pos

            # 场景位移 → 局部位移（逆旋转）
            angle_rad = np.radians(self.rotation())
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            dx = scene_delta.x() * cos_a + scene_delta.y() * sin_a
            dy = -scene_delta.x() * sin_a + scene_delta.y() * cos_a

            l, t, r, b = self._drag_start_white_frame_local
            moved = [l + dx, t + dy, r + dx, b + dy]

            self.prepareGeometryChange()
            self._shape_path = None
            self.geo.set_custom_white_frame_local(moved)
            # 轻量预览：仅平移文字层
            self.text_item.setPos(self._drag_start_text_item_pos + QPointF(dx, dy))
            self.update()
            self._invalidate_scene_rect(old_rect)

        except Exception as e:
            logger.error(f"[RegionTextItem] _handle_white_frame_move: {e}\n{traceback.format_exc()}")

    def _commit_white_frame(self, event, edit_mode=None):
        scene = self.scene()
        if scene and hasattr(self, "_drag_start_scene_rect") and self._drag_start_scene_rect is not None:
            update_rect = self._drag_start_scene_rect.united(self.sceneBoundingRect())
            scene.invalidate(update_rect, QGraphicsScene.SceneLayer.ItemLayer)
            scene.update(update_rect)

        patch = self.geo.to_region_data_patch()
        new_data = build_white_frame_region_data(
            self.region_data, patch, self.geo.white_frame_local,
            old_white_frame_local=self._drag_start_white_frame_local,
            edit_mode=edit_mode,
        )
        self._emit_region_update(event, new_data)

    def _white_handle_world_at_start(self):
        if self._drag_start_white_frame_local is None:
            return None
        l, t, r, b = self._drag_start_white_frame_local
        cx, cy = self.geo.center
        angle = self.rotation_angle

        if self._interaction_mode == "white_corner":
            idx = self._drag_handle_indices
            corners = [(l + cx, t + cy), (r + cx, t + cy), (r + cx, b + cy), (l + cx, b + cy)]
            if not (0 <= idx < 4):
                return None
            x, y = corners[idx]
            return rotate_point(x, y, angle, cx, cy) if angle != 0 else (x, y)

        if self._interaction_mode == "white_edge":
            idx = self._drag_handle_indices
            verts = [(l + cx, t + cy), (r + cx, t + cy), (r + cx, b + cy), (l + cx, b + cy)]
            x, y = verts[idx]
            return rotate_point(x, y, angle, cx, cy) if angle != 0 else (x, y)

        return None

    # ------------------------------------------------------------------
    # 场景刷新
    # ------------------------------------------------------------------

    def _invalidate_scene_rect(self, old_rect):
        if self.scene() and old_rect is not None:
            new_rect = self.sceneBoundingRect()
            update_rect = old_rect.united(new_rect)
            self.scene().invalidate(update_rect, QGraphicsScene.SceneLayer.ItemLayer)
            self.scene().update(update_rect)

    # ------------------------------------------------------------------
    # WYSIWYG 占位（实际渲染由 GraphicsView 驱动）
    # ------------------------------------------------------------------
