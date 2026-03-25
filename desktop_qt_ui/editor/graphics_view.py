
import cv2
import numpy as np

# --- 新增Imports for Refactoring ---
from editor import text_renderer_backend
from editor.editor_model import EditorModel
from editor.graphics_items import RegionTextItem, TransparentPixmapItem
from editor.region_render_snapshot import RegionRenderSnapshot
from editor.render_layout_pipeline import (
    build_region_specific_params,
    calculate_region_dst_points,
    prepare_layout_context,
)
from editor.selection_manager import SelectionManager
from editor.text_render_pipeline import (
    build_region_render_params as pipeline_build_region_render_params,
)
from editor.text_render_pipeline import (
    build_text_block_from_region as pipeline_build_text_block_from_region,
)
from editor.text_render_pipeline import (
    clear_region_text,
    make_text_render_cache_key,
    render_region_text,
)
from main_view_parts.theme import get_current_theme, get_theme_colors
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QMenu
from services import get_render_parameter_service

from manga_translator.utils import TextBlock

# --- 结束新增 ---

class GraphicsView(QGraphicsView):
    """
    自定义画布视图，继承自 QGraphicsView。
    负责显示和交互式操作图形项（如图片、文本框）。
    """
    region_geometry_changed = pyqtSignal(int, dict)
    _layout_result_ready = pyqtSignal(list)  # 布局计算结果信号
    view_state_changed = pyqtSignal(object, object)

    def __init__(self, model: EditorModel, parent=None):
        super().__init__(parent)
        self.model = model

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self._image_item: QGraphicsPixmapItem = None
        self._raw_mask_item: QGraphicsPixmapItem = None
        self._refined_mask_item: QGraphicsPixmapItem = None
        self._inpainted_image_item: QGraphicsPixmapItem = None
        self._q_image_ref = None # Keep a reference to the QImage to prevent crashes
        self._inpainted_q_image_ref = None
        self._preview_item: QGraphicsPixmapItem = None # Dedicated item for drawing previews

        self._region_items = []
        self._image_np = None
        self._pending_geometry_edit_kinds: dict[int, str] = {}

        # --- Mask Editing State ---
        self._active_tool = 'select'
        self._brush_size = 30
        self._is_drawing = False
        self._current_draw_scene_points: list[QPointF] = []
        self._current_draw_mask_points: list[tuple[int, int]] = []
        self._current_draw_mask_shape: tuple[int, int] | None = None

        # 拖动相关状态
        self._potential_drag = False  # 是否可能开始拖动
        self._drag_start_pos = None  # 拖动起始位置
        self._drag_threshold = 5  # 拖动阈值（像素）

        # --- Text Box Drawing State ---
        self._is_drawing_textbox = False
        self._textbox_start_pos = None
        self._textbox_preview_item = None

        # --- Text Rendering Cache ---
        self._text_render_cache = {}

        # --- Debounce timer for rendering ---
        self.render_debounce_timer = QTimer(self)
        self.render_debounce_timer.setSingleShot(True)
        self.render_debounce_timer.setInterval(150) # 150ms debounce time
        self.render_debounce_timer.timeout.connect(self._perform_render_update)


        # --- 新增渲染器和缓存 ---
        self._text_blocks_cache: list[TextBlock] = []
        self._dst_points_cache: list[np.ndarray] = []
        self._render_snapshot_cache: list[RegionRenderSnapshot | None] = []
        # --- 结束新增 ---

        self._setup_view()
        self._connect_model_signals()

        # 连接布局结果信号
        self._layout_result_ready.connect(self._apply_layout_result)

    def _setup_view(self):
        """配置视图属性"""
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 性能优化配置
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # 启用滚动条
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.apply_theme()

        # 选择管理器：集中管理选择逻辑和双向同步
        self.selection_manager = SelectionManager(
            self.model, self.scene, lambda: self._region_items
        )

    def apply_theme(self, theme: str | None = None):
        colors = get_theme_colors(theme or get_current_theme())
        canvas_color = QColor(colors["bg_canvas"])
        self.scene.setBackgroundBrush(canvas_color)
        self.setBackgroundBrush(canvas_color)
        self.scene.update()
        self.viewport().update()

    def _connect_model_signals(self):
        """连接模型信号到视图的槽"""
        self.model.image_changed.connect(self.on_image_changed)
        self.model.regions_changed.connect(self.on_regions_changed)
        self.model.raw_mask_changed.connect(lambda mask: self.on_mask_data_changed('raw', mask))
        self.model.refined_mask_changed.connect(lambda mask: self.on_mask_data_changed('refined', mask))
        self.model.display_mask_type_changed.connect(self.on_display_mask_type_changed)
        self.model.inpainted_image_changed.connect(self.on_inpainted_image_changed)
        self.model.region_display_mode_changed.connect(self.on_region_display_mode_changed)
        self.model.original_image_alpha_changed.connect(self.on_original_image_alpha_changed)
        self.model.region_style_updated.connect(self.on_region_style_updated) # NEW: For targeted style updates
        # selection_changed 由 SelectionManager 处理，无需在此连接

        # --- Mask Editing Connections ---
        self.model.active_tool_changed.connect(self._on_active_tool_changed)
        self.model.brush_size_changed.connect(self._on_brush_size_changed)

    def get_view_state(self):
        if self._image_item is None:
            return None, None
        center_scene = self.mapToScene(self.viewport().rect().center())
        return QTransform(self.transform()), QPointF(center_scene)

    def _emit_view_state_changed(self):
        transform, center_scene = self.get_view_state()
        if transform is None or center_scene is None:
            return
        self.view_state_changed.emit(transform, center_scene)

    def _scale_mask_item(self, mask_item: QGraphicsPixmapItem):
        """Helper function to scale a mask item to match the base image item."""
        if not self._image_item or not mask_item:
            return

        img_rect = self._image_item.boundingRect()
        mask_rect = mask_item.boundingRect()

        if mask_rect.width() > 0 and mask_rect.height() > 0:
            scale_x = img_rect.width() / mask_rect.width()
            scale_y = img_rect.height() / mask_rect.height()
            transform = QTransform()
            transform.scale(scale_x, scale_y)
            mask_item.setTransform(transform)

    def clear_all_state(self):
        """清空所有状态,包括items、缓存、计时器"""
        self.selection_manager.suppress_forward_sync(True)  # 防止 removeItem 触发选择同步
        try:
            self._reset_drawing_state()
            # 停止防抖计时器
            if hasattr(self, 'render_debounce_timer') and self.render_debounce_timer.isActive():
                self.render_debounce_timer.stop()

            # 清空region items - 使用副本遍历避免修改时出错
            items_to_remove = list(self._region_items) if hasattr(self, '_region_items') else []
            for item in items_to_remove:
                try:
                    if item and hasattr(item, 'scene') and item.scene():
                        self.scene.removeItem(item)
                except (RuntimeError, AttributeError):
                    pass
            if hasattr(self, '_region_items'):
                self._region_items.clear()

            # 清空image items
            if self._image_item and self._image_item.scene():
                self.scene.removeItem(self._image_item)
                self._image_item = None

            if self._inpainted_image_item and self._inpainted_image_item.scene():
                self.scene.removeItem(self._inpainted_image_item)
                self._inpainted_image_item = None

            # 清空mask items
            if self._raw_mask_item and self._raw_mask_item.scene():
                self.scene.removeItem(self._raw_mask_item)
                self._raw_mask_item = None

            if self._refined_mask_item and self._refined_mask_item.scene():
                self.scene.removeItem(self._refined_mask_item)
                self._refined_mask_item = None


            # 清空预览项（文本框、几何编辑、蒙版绘制等）
            if self._textbox_preview_item and self._textbox_preview_item.scene():
                self.scene.removeItem(self._textbox_preview_item)
                self._textbox_preview_item = None

            if self._preview_item and self._preview_item.scene():
                self.scene.removeItem(self._preview_item)
                self._preview_item = None

            # 清空框选状态（由 SelectionManager 管理）
            self.selection_manager.clear_state()

            # 清空缓存
            if hasattr(self, '_text_render_cache'):
                self._text_render_cache.clear()
            self._text_blocks_cache = []
            self._dst_points_cache = []
            self._render_snapshot_cache = []

            # 重置所有绘制状态
            self._is_drawing = False
            self._is_drawing_textbox = False
            self._clear_pending_geometry_edits()

            # 关闭线程池（如果存在）
            if hasattr(self, '_render_executor'):
                try:
                    self._render_executor.shutdown(wait=False)
                    del self._render_executor
                except Exception:
                    pass
        except (RuntimeError, AttributeError) as e:
            # 清理过程中可能遇到已删除的对象
            print(f"[View] Warning: Error during clear_all_state: {e}")
        finally:
            self.selection_manager.suppress_forward_sync(False)

    def on_image_changed(self, image):
        """槽：当模型中的图像变化时更新背景"""

        # 先清空所有状态
        self.clear_all_state()

        # 清空场景（防止触发选择同步）
        self.selection_manager.suppress_forward_sync(True)
        self.scene.clear()
        self.selection_manager.suppress_forward_sync(False)
        self.selection_manager.on_scene_cleared()  # scene.clear() 会删除所有 item
        self._image_item = None
        self._raw_mask_item = None
        self._refined_mask_item = None
        self._inpainted_image_item = None
        self._preview_item = None
        self._image_np = None

        if image is None:
            return

        try:
            self._image_np = np.array(image.convert("RGB"))
        except Exception as convert_error:
            print(f"[VIEW WARN] Failed to convert image to numpy array: {convert_error}")
            self._image_np = None

        # 修复：ImageQt 不支持 'LA' 模式，需要转换为支持的模式
        # 支持的模式：'1', 'L', 'P', 'RGB', 'RGBA'
        if image.mode not in ('1', 'L', 'P', 'RGB', 'RGBA'):
            if image.mode == 'LA':
                # LA (灰度+Alpha) -> RGBA
                image = image.convert('RGBA')
            elif 'A' in image.mode:
                # 其他带Alpha通道的模式 -> RGBA
                image = image.convert('RGBA')
            else:
                # 其他模式 -> RGB
                image = image.convert('RGB')
        
        self._q_image_ref = ImageQt(image)
        pixmap = QPixmap.fromImage(self._q_image_ref)

        self._image_item = self.scene.addPixmap(pixmap)
        self._image_item.setZValue(2) # Place it above the inpainted image (Z=1)
        # 设置原图透明度（从模型获取）
        alpha = self.model.get_original_image_alpha()
        self._image_item.setOpacity(alpha)
        # 不设置场景矩形，让它自动管理
        self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_view_state_changed()

        # 竞态条件修复：图片加载后，强制触发一次区域重绘和渲染数据计算
        # 以确保在文本区域先于图片加载的情况下，文本也能被正确渲染
        self.on_regions_changed(self.model.get_regions())

    def on_mask_data_changed(self, mask_type: str, mask_array: np.ndarray):
        """当任一蒙版数据在模型中更新时，更新对应的PixmapItem"""
        target_item = self._raw_mask_item if mask_type == 'raw' else self._refined_mask_item

        if mask_array is None or mask_array.size == 0:
            if target_item:
                target_item.setPixmap(QPixmap())
            return

        h, w = mask_array.shape[:2]
        color_mask = np.zeros((h, w, 4), dtype=np.uint8)
        color_mask[mask_array > 0] = [255, 0, 0, 128]
        # 【关键修复】使用.copy()确保QImage拥有自己的内存，防止numpy数组被回收后崩溃
        q_image = QImage(color_mask.data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
        pixmap = QPixmap.fromImage(q_image)

        if target_item is None or target_item.scene() is None:
            # Create or recreate the mask item if it doesn't exist or was removed from scene
            if mask_type == 'raw':
                if self._raw_mask_item and self._raw_mask_item.scene():
                    self.scene.removeItem(self._raw_mask_item)
                self._raw_mask_item = TransparentPixmapItem()
                self._raw_mask_item.setPixmap(pixmap)
                self._raw_mask_item.setZValue(10)
                self.scene.addItem(self._raw_mask_item)
                self._scale_mask_item(self._raw_mask_item)
                self._raw_mask_item.setVisible(self.model.get_display_mask_type() == 'raw')
                target_item = self._raw_mask_item
            else:
                if self._refined_mask_item and self._refined_mask_item.scene():
                    self.scene.removeItem(self._refined_mask_item)
                self._refined_mask_item = TransparentPixmapItem()
                self._refined_mask_item.setPixmap(pixmap)
                self._refined_mask_item.setZValue(11)
                self.scene.addItem(self._refined_mask_item)
                self._scale_mask_item(self._refined_mask_item)
                self._refined_mask_item.setVisible(self.model.get_display_mask_type() == 'refined')
                target_item = self._refined_mask_item
        else:
            target_item.setPixmap(pixmap)
            self._scale_mask_item(target_item) # Rescale on update too

        # Force update after mask change
        self.scene.update()
        self.viewport().update()  # Also update the viewport
        self.update()

        # Additional refresh for visibility if this mask type should be visible
        current_display_type = self.model.get_display_mask_type()
        if mask_type == current_display_type and target_item:
            target_item.setVisible(True)

    def on_display_mask_type_changed(self, mask_type: str):
        """根据模型状态，切换哪个蒙版图层可见"""
        # --- Defensive Fix: Ensure mask items exist if their data exists ---
        if mask_type == 'raw' and self._raw_mask_item is None and self.model.get_raw_mask() is not None:
            self.on_mask_data_changed('raw', self.model.get_raw_mask())

        if mask_type == 'refined' and self._refined_mask_item is None and self.model.get_refined_mask() is not None:
            self.on_mask_data_changed('refined', self.model.get_refined_mask())
        # --- End Defensive Fix ---

        if self._raw_mask_item:
            is_visible = (mask_type == 'raw')
            self._raw_mask_item.setVisible(is_visible)
        if self._refined_mask_item:
            is_visible = (mask_type == 'refined')
            self._refined_mask_item.setVisible(is_visible)

        # Force scene refresh
        self.scene.update()
        self.viewport().update()  # Also update the viewport
        self.update()

        # Additional repaint call for immediate visual update
        self.repaint()

    def on_inpainted_image_changed(self, image):
        """槽：当模型中的修复后图像变化时更新"""
        if self._image_item is None:
            return # Cannot display inpainted if there is no base image

        if image is None:
            if self._inpainted_image_item:
                self._inpainted_image_item.setVisible(False)
            self._inpainted_q_image_ref = None
            return

        # 修复：ImageQt 不支持 'LA' 模式，需要转换为支持的模式
        if image.mode not in ('1', 'L', 'P', 'RGB', 'RGBA'):
            if image.mode == 'LA':
                image = image.convert('RGBA')
            elif 'A' in image.mode:
                image = image.convert('RGBA')
            else:
                image = image.convert('RGB')
        
        self._inpainted_q_image_ref = ImageQt(image)
        pixmap = QPixmap.fromImage(self._inpainted_q_image_ref)

        if self._inpainted_image_item is None:
            self._inpainted_image_item = TransparentPixmapItem()
            self._inpainted_image_item.setPixmap(pixmap)
            self._inpainted_image_item.setZValue(1) # Below masks and text
            self._inpainted_image_item.setOpacity(1.0) # 始终完全不透明
            self.scene.addItem(self._inpainted_image_item)
        else:
            self._inpainted_image_item.setPixmap(pixmap)
            self._inpainted_image_item.setOpacity(1.0) # 始终完全不透明

        self._inpainted_image_item.setVisible(True)

    @pyqtSlot(int)
    def on_region_style_updated(self, region_index: int):
        """Slot for targeted update when only style changes."""
        self._perform_single_item_update(region_index)


    @pyqtSlot(float)
    def on_original_image_alpha_changed(self, alpha: float):
        """槽：当原图透明度变化时更新"""
        if self._image_item:
            self._image_item.setOpacity(alpha)

    def on_region_display_mode_changed(self, mode: str):
        """槽：当区域显示模式变化时更新所有文本项的可见性"""
        for item in self.scene.items():
            if isinstance(item, RegionTextItem):
                if mode == "full":
                    item.setVisible(True)
                    item.set_text_visible(True)
                    item.set_box_visible(True)
                    item.set_white_box_visible(True)
                elif mode == "text_only":
                    item.setVisible(True)
                    item.set_text_visible(True)
                    item.set_box_visible(False)
                    item.set_white_box_visible(False)
                elif mode == "box_only":
                    item.setVisible(True)
                    item.set_text_visible(False)
                    item.set_box_visible(True)
                    item.set_white_box_visible(True)
                elif mode == "none":
                    item.setVisible(False)
                    item.set_white_box_visible(False)



    def on_regions_changed(self, regions):
        """槽：当模型中的区域数据变化时，根据情况选择性更新或完全更新。"""
        # 优先处理由交互编辑触发的几何 targeted 更新（按 region 独立消费上下文）
        # 仅在 item 数量未变化时使用 targeted 路径；增删区域必须走完整重建。
        same_item_count = len(regions) == len(self._region_items)
        pending_indices = list(self._pending_geometry_edit_kinds.keys())
        handled = False
        if same_item_count:
            for region_index in pending_indices:
                edit_kind = self._consume_pending_geometry_edit(region_index)
                if edit_kind is None:
                    continue
                if 0 <= region_index < len(self._region_items):
                    self._perform_single_item_update(region_index, edit_kind=edit_kind)
                    handled = True

        if handled:
            return

        # 走完整更新时清空交互上下文，避免旧状态污染下一次刷新。
        self._clear_pending_geometry_edits()

        # A general change occurred (e.g., new translation). Perform full debounced update.
        self._text_render_cache.clear()
        self._render_snapshot_cache = []
        self.render_debounce_timer.start()

    def _values_equal(self, left, right) -> bool:
        try:
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                return np.array_equal(np.asarray(left), np.asarray(right))
            return left == right
        except Exception:
            return False

    def _infer_geometry_edit_kind(self, region_index: int, new_region_data: dict) -> str:
        """根据旧/新 region_data 差异判断几何编辑类型。"""
        old_region_data = self.model.get_region_by_index(region_index)
        if not isinstance(old_region_data, dict) or not isinstance(new_region_data, dict):
            return "unknown"

        if not self._values_equal(old_region_data.get("angle"), new_region_data.get("angle")):
            return "rotate"
        if not self._values_equal(old_region_data.get("center"), new_region_data.get("center")):
            return "move"
        if not self._values_equal(old_region_data.get("lines"), new_region_data.get("lines")):
            return "shape"

        white_changed = (
            not self._values_equal(
                old_region_data.get("white_frame_rect_local"),
                new_region_data.get("white_frame_rect_local"),
            )
            or not self._values_equal(
                old_region_data.get("has_custom_white_frame", False),
                new_region_data.get("has_custom_white_frame", False),
            )
        )
        if white_changed:
            return "white_frame"
        return "other"

    def _perform_single_item_update(self, index, edit_kind: str | None = None):
        """对单个区域执行targeted更新。"""
        try:
            if not (0 <= index < len(self._region_items)):
                return

            region_data = self.model.get_region_by_index(index)
            item = self._region_items[index]

            if not region_data or item is None or item.scene() is None:
                return

            if edit_kind is None:
                edit_kind = self._consume_pending_geometry_edit(index)

            if edit_kind == "white_frame":
                # 白框编辑：不调 update_from_data（避免覆盖当前白框）
                # 从 item 的白框构建 override_dst_points
                override = self._build_dst_points_from_item(item)
                self._recalculate_single_region_render_data(index, override_dst_points=override)
            else:
                # 风格/文本类 targeted 更新尽量不回滚当前 item 几何。
                # 但当模型几何与 item 几何不一致（如 undo/redo 白框）时，
                # 必须以模型几何为准，否则会出现“尺寸回滚但位置不回滚”。
                region_for_item = region_data.copy()
                if (
                    hasattr(item, "geo")
                    and item.geo is not None
                    and self._region_geometry_matches_item(region_data, item)
                ):
                    try:
                        region_for_item.update(item.geo.to_region_data_patch())
                        region_for_item["center"] = list(item.geo.center)
                    except Exception:
                        pass
                item.update_from_data(region_for_item)
                self._recalculate_single_region_render_data(index)

            self._update_single_region_text_visual(index)

            if item.scene() is not None:
                item.update()
        except (RuntimeError, AttributeError) as e:
            print(f"[View] Warning: Item update failed for index {index}: {e}")

    def _set_pending_geometry_edit(self, region_index: int, edit_kind: str):
        self._pending_geometry_edit_kinds[int(region_index)] = str(edit_kind)

    def _consume_pending_geometry_edit(self, region_index: int) -> str | None:
        return self._pending_geometry_edit_kinds.pop(int(region_index), None)

    def _clear_pending_geometry_edits(self):
        self._pending_geometry_edit_kinds.clear()

    def _region_geometry_matches_item(self, region_data: dict, item: RegionTextItem) -> bool:
        """判断 model 中几何是否与当前 item 几何一致。"""
        try:
            if not hasattr(item, "geo") or item.geo is None:
                return False

            if not self._values_equal(region_data.get("center"), list(item.geo.center)):
                return False
            if not self._values_equal(region_data.get("angle"), float(item.geo.angle)):
                return False
            if not self._values_equal(region_data.get("lines"), item.geo.lines):
                return False
            if not self._values_equal(
                region_data.get("has_custom_white_frame", False),
                bool(item.geo.has_custom_white_frame),
            ):
                return False

            model_wf = region_data.get("white_frame_rect_local")
            item_wf = item.geo.white_frame_local
            if model_wf is None and item_wf is None:
                return True
            return self._values_equal(model_wf, item_wf)
        except Exception:
            return False

    def _build_dst_points_from_item(self, item):
        """从 item 的白框构建渲染 dst_points（世界坐标轴对齐矩形）。"""
        wf = item.geo.white_frame_local
        if wf is None or len(wf) != 4:
            return None
        left, top, right, bottom = wf
        box_w = float(right - left)
        box_h = float(bottom - top)
        if box_w <= 0.0 or box_h <= 0.0:
            return None

        cx, cy = item.geo.center
        angle = float(getattr(item.geo, "angle", 0.0) or 0.0)
        theta = np.deg2rad(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        local_cx = (left + right) / 2.0
        local_cy = (top + bottom) / 2.0
        render_cx = float(cx + local_cx * cos_t - local_cy * sin_t)
        render_cy = float(cy + local_cx * sin_t + local_cy * cos_t)
        half_w = box_w / 2.0
        half_h = box_h / 2.0

        return np.array(
            [[
                [render_cx - half_w, render_cy - half_h],
                [render_cx + half_w, render_cy - half_h],
                [render_cx + half_w, render_cy + half_h],
                [render_cx - half_w, render_cy + half_h],
            ]],
            dtype=np.float32,
        )

    def _ensure_render_cache_size(self, index: int):
        while len(self._text_blocks_cache) <= index:
            self._text_blocks_cache.append(None)
        while len(self._dst_points_cache) <= index:
            self._dst_points_cache.append(None)
        while len(self._render_snapshot_cache) <= index:
            self._render_snapshot_cache.append(None)

    def _build_render_snapshot(self, index: int, region_data: dict, item: RegionTextItem | None) -> RegionRenderSnapshot:
        geo_state = item.geo if (item is not None and hasattr(item, "geo")) else None
        return RegionRenderSnapshot.from_sources(
            region_index=index,
            region_data=region_data,
            geo_state=geo_state,
        )

    def _render_region_text_visual(self, index: int, use_cache: bool):
        """统一渲染并应用单个区域文本显示。"""
        if not (0 <= index < len(self._region_items)):
            return
        item = self._region_items[index]
        if item is None or not hasattr(item, 'scene') or item.scene() is None:
            return
        if not hasattr(item, 'text_item') or item.text_item is None:
            return

        if self.model.get_region_display_mode() in ["box_only", "none"]:
            item.text_item.setVisible(False)
            return
        item.text_item.setVisible(True)

        if index >= len(self._text_blocks_cache) or index >= len(self._dst_points_cache):
            return
        text_block = self._text_blocks_cache[index]
        dst_points = self._dst_points_cache[index]
        if text_block is None or dst_points is None:
            clear_region_text(item)
            return

        snapshot = self._render_snapshot_cache[index] if index < len(self._render_snapshot_cache) else None
        if snapshot is None:
            region_data = self.model.get_region_by_index(index)
            if not region_data:
                return
            snapshot = self._build_render_snapshot(index, region_data, item)
            self._ensure_render_cache_size(index)
            self._render_snapshot_cache[index] = snapshot

        region_data_for_render = snapshot.text_block_input()

        unrotated_text_block = pipeline_build_text_block_from_region(
            region_data_for_render,
            font_size_override=getattr(text_block, 'font_size', None),
            log_tag=f" for region {index}"
        )
        if unrotated_text_block is None:
            clear_region_text(item)
            return

        render_parameter_service = get_render_parameter_service()
        render_params = pipeline_build_region_render_params(
            render_parameter_service,
            text_renderer_backend,
            index,
            snapshot.style_input(),
            unrotated_text_block,
        )

        cache_key = None
        if use_cache:
            cache_key = make_text_render_cache_key(unrotated_text_block, dst_points, render_params)

        cached_result = self._text_render_cache.get(cache_key) if cache_key is not None else None
        if cached_result is None:
            render_result = render_region_text(
                text_renderer_backend,
                unrotated_text_block,
                dst_points,
                render_params,
                len(self._text_blocks_cache),
            )
            if render_result and cache_key is not None:
                self._text_render_cache[cache_key] = render_result
            cached_result = render_result

        if cached_result:
            pixmap, pos = cached_result
            # 先应用 dst_points（可能更新白框与旋转原点），再映射文字位置，
            # 避免在变换更新后出现文字与白框中心错位。
            item.set_dst_points(dst_points)
            item.update_text_pixmap(
                pixmap,
                pos,
                0,
                None,
                render_center=snapshot.render_center,
            )
        else:
            clear_region_text(item)



    def _perform_render_update(self):
        """执行实际的渲染更新，由防抖计时器调用。"""
        self.selection_manager.suppress_forward_sync(True)  # 防止增删 items 触发选择同步
        try:
            # Reuse existing items to improve performance and stability
            regions = self.model.get_regions()
            current_items = self._region_items

            # 1. Remove excess items
            while len(current_items) > len(regions):
                item = current_items.pop()
                try:
                    if item and hasattr(item, 'scene') and item.scene():
                        self.scene.removeItem(item)
                except (RuntimeError, AttributeError):
                    pass

            # 2. Create new items if needed
            while len(current_items) < len(regions):
                i = len(current_items)
                region_data = regions[i]
                if not region_data.get('lines'):
                    pass

                item = RegionTextItem(
                    region_data,
                    i,
                    geometry_callback=self._on_region_geometry_changed,
                )
                item.set_image_item(self._image_item)
                item.setZValue(100)
                self.scene.addItem(item)
                current_items.append(item)

            # 3. Update all items with new data
            for i, region_data in enumerate(regions):
                if i < len(current_items):
                    item = current_items[i]
                    item.set_image_item(self._image_item)
                    item.region_index = i
                    item.update_from_data(region_data)

            # After updating items, recalculate all rendering data
            self.recalculate_render_data()
        except Exception as e:
            print(f"[View] Warning: Render update failed: {e}")
        finally:
            self.selection_manager.suppress_forward_sync(False)
            # 恢复 Qt items 的选择状态，与 model 同步
            self.selection_manager.restore_selection_after_rebuild()

    def _update_text_visuals(self):
        try:
            if self.model.get_region_display_mode() in ["box_only", "none"]:
                for item in self._region_items:
                    if item and hasattr(item, 'text_item') and item.text_item and item.scene():
                        item.text_item.setVisible(False)
                return

            for i in range(min(len(self._region_items), len(self._text_blocks_cache), len(self._dst_points_cache))):
                self._render_region_text_visual(i, use_cache=True)

        except (RuntimeError, AttributeError) as e:
            # 处理item在更新过程中被删除的情况
            print(f"[View] Warning: Text visuals update failed: {e}")

    def _recalculate_single_region_render_data(self, index, override_dst_points=None):
        """重新计算单个区域的渲染数据。

        override_dst_points: 如果提供，直接使用此值作为 dst_points（白框编辑场景），
                            跳过默认布局计算。
        """
        regions = self.model.get_regions()
        if self._image_np is None or not regions or not (0 <= index < len(regions)):
            return

        self._ensure_render_cache_size(index)

        item = self._region_items[index] if 0 <= index < len(self._region_items) else None
        snapshot = self._build_render_snapshot(index, regions[index], item)
        self._render_snapshot_cache[index] = snapshot

        region_dict = snapshot.text_block_input()
        text_block = pipeline_build_text_block_from_region(region_dict, log_tag=f" for region {index}")
        self._text_blocks_cache[index] = text_block
        if text_block is None:
            self._dst_points_cache[index] = None
            return

        render_parameter_service = get_render_parameter_service()
        global_params_dict, config_obj = prepare_layout_context(
            render_parameter_service,
            text_renderer_backend,
        )
        region_specific_params = build_region_specific_params(global_params_dict, text_block)
        if region_dict.get("line_spacing") is not None:
            region_specific_params["line_spacing"] = region_dict.get("line_spacing")
        if region_dict.get("letter_spacing") is not None:
            region_specific_params["letter_spacing"] = region_dict.get("letter_spacing")

        # 3. 计算渲染框 dst_points
        try:
            self._dst_points_cache[index] = calculate_region_dst_points(
                text_block,
                region_specific_params,
                config_obj,
                override_dst_points=override_dst_points,
            )
        except Exception as e:
            print(f"[View] Failed to calculate dst_points for region {index}: {e}")
            print(f"  font_size={text_block.font_size}, horizontal={text_block.horizontal}, "
                  f"center={text_block.center}, xyxy={text_block.xyxy}, "
                  f"line_spacing={region_specific_params.get('line_spacing')}, "
                  f"letter_spacing={region_specific_params.get('letter_spacing')}, "
                  f"angle={region_dict.get('angle')}")
            import traceback
            traceback.print_exc()
            self._dst_points_cache[index] = None

    def _update_single_region_text_visual(self, index, use_cache=False):
        """重新渲染单个区域的文字"""
        try:
            self._render_region_text_visual(index, use_cache=use_cache)
        except (RuntimeError, AttributeError) as e:
            # Item可能在渲染过程中被删除
            print(f"[View] Warning: Text visual update failed for index {index}: {e}")
        except Exception as e:
            print(f"[View] Error in _update_single_region_text_visual for index {index}: {e}")

    def recalculate_render_data(self):
        """
        执行昂贵的布局计算并将结果缓存。
        这个方法应该在 regions 数据变化后被调用。
        使用线程池异步执行以避免阻塞UI。
        """
        regions = self.model.get_regions()
        if self._image_np is None or not regions:
            self._text_blocks_cache = []
            self._dst_points_cache = []
            self._render_snapshot_cache = []
            return

        # 1. 构建统一渲染快照（避免 model/item 混用造成位置跳变）
        snapshots: list[RegionRenderSnapshot] = []
        for i, region_dict in enumerate(regions):
            item = self._region_items[i] if i < len(self._region_items) else None
            snapshots.append(self._build_render_snapshot(i, region_dict, item))
        self._render_snapshot_cache = snapshots

        # 2. 将快照数据转换为 TextBlock
        self._text_blocks_cache = [
            pipeline_build_text_block_from_region(snapshot.text_block_input(), log_tag=f" for region {i}")
            for i, snapshot in enumerate(snapshots)
        ]

        render_parameter_service = get_render_parameter_service()
        global_params_dict, config_obj = prepare_layout_context(
            render_parameter_service,
            text_renderer_backend,
        )

        # 3. 计算每个区域的渲染框
        dst_points_list = []
        for i, text_block in enumerate(self._text_blocks_cache):
            if text_block is None:
                dst_points_list.append(None)
                continue

            try:
                snapshot = snapshots[i] if i < len(snapshots) else None
                region_dict = snapshot.region_data if snapshot is not None else {}
                region_params = build_region_specific_params(global_params_dict, text_block)
                if region_dict.get("line_spacing") is not None:
                    region_params["line_spacing"] = region_dict.get("line_spacing")
                if region_dict.get("letter_spacing") is not None:
                    region_params["letter_spacing"] = region_dict.get("letter_spacing")
                dst_points_list.append(
                    calculate_region_dst_points(
                        text_block,
                        region_params,
                        config_obj,
                    )
                )
            except Exception as e:
                print(f"[View] Failed to calculate dst_points for region {i}: {e}")
                print(f"  font_size={text_block.font_size}, horizontal={text_block.horizontal}, "
                      f"center={text_block.center}, xyxy={text_block.xyxy}, "
                      f"line_spacing={global_params_dict.get('line_spacing')}, "
                      f"letter_spacing={global_params_dict.get('letter_spacing')}, "
                      f"angle={regions[i].get('angle') if i < len(regions) else 'N/A'}")
                dst_points_list.append(None)

        self._dst_points_cache = dst_points_list
        self._update_text_visuals()
        self.scene.update()
    
    @pyqtSlot(list)
    def _apply_layout_result(self, dst_points_cache):
        """在主线程应用布局计算结果"""
        try:
            self._dst_points_cache = dst_points_cache
            self._update_text_visuals()
            self.scene.update()
        except Exception as e:
            print(f"[View] Error applying layout result: {e}")



    def _on_region_geometry_changed(self, region_index, new_region_data):
        self._set_pending_geometry_edit(
            region_index,
            self._infer_geometry_edit_kind(region_index, new_region_data),
        )
        self.region_geometry_changed.emit(region_index, new_region_data)



    def resizeEvent(self, event):
        """处理视图大小改变事件"""
        # 不自动适应视图，保持用户的缩放级别
        super().resizeEvent(event)
        self._emit_view_state_changed()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self._emit_view_state_changed()

    def wheelEvent(self, event):
        """处理鼠标滚轮事件以实现缩放"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

        self._update_cursor() # Update cursor size on zoom
        self._emit_view_state_changed()

    def mousePressEvent(self, event):
        """处理鼠标按下事件以实现平移、选择和开始绘图"""
        # 确保点击画布时，保存文本框的编辑内容
        parent_view = self.parent()
        if hasattr(parent_view, 'force_save_property_panel_edits'):
            parent_view.force_save_property_panel_edits()
        
        # 让画布获取焦点
        self.setFocus()
        
        if self._active_tool == 'draw_textbox' and event.button() == Qt.MouseButton.LeftButton:
            self._start_drawing_textbox(event.pos())
            event.accept()
            return

        if self._active_tool in ['pen', 'eraser', 'brush'] and event.button() == Qt.MouseButton.LeftButton:
            self._start_drawing(event.pos())
            event.accept()
            return

        # --- Original Selection Logic ---
        if event.button() == Qt.MouseButton.LeftButton and self._active_tool == 'select':
            # 不再依赖 scene.items()，直接让事件传播
            # Qt 的事件分发系统会自动找到正确的 item（通过 shape().contains()）
            # 如果有 item 处理了事件，事件会被 accept
            # 如果没有 item 处理，事件会返回到这里，我们再处理空白点击
            pass  # 让事件继续传播

        # --- Panning and Item Interaction Logic ---
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Create a new event with LeftButton instead of modifying the original
            from PyQt6.QtGui import QMouseEvent
            dummy_event = QMouseEvent(
                event.type(),
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(dummy_event)
        elif event.button() == Qt.MouseButton.LeftButton:
            # 检查是否点击在空白区域
            item_at_pos = self.itemAt(event.pos())
            
            # 检查是否点击在 RegionTextItem 上（包括其子元素）
            clicked_region_item = False
            if item_at_pos:
                # 向上查找父元素，看是否是 RegionTextItem
                check_item = item_at_pos
                while check_item:
                    if isinstance(check_item, RegionTextItem):
                        clicked_region_item = True
                        break
                    check_item = check_item.parentItem()

            # 如果点击在空白区域（没有 item 或只有图片），按下 Alt 键时框选，否则拖拽画布
            if item_at_pos is None or item_at_pos == self._image_item:
                alt_pressed = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                if alt_pressed:
                    self.selection_manager.start_box_select(self.mapToScene(event.pos()))
                    event.accept()
                else:
                    self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                    # 直接传递原始事件即可，因为当前已经是左键按下了
                    super().mousePressEvent(event)
                return

            # 如果点击在 RegionTextItem 上，让 item 处理事件
            if clicked_region_item:
                super().mousePressEvent(event)
                # 选择同步由 scene.selectionChanged 统一处理
            else:
                super().mousePressEvent(event)
                # 空白区域点击：手动清除选择（Qt不一定自动清除item选择）
                alt_pressed = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                if not alt_pressed:
                    # 清除所有 Qt item 选择，scene.selectionChanged 会自动同步到 model
                    self.scene.clearSelection()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing."""
        # 处理框选
        if self.selection_manager.is_box_selecting:
            current_pos = self.mapToScene(event.pos())
            if self.selection_manager.update_box_select(current_pos):
                event.accept()
                return
            else:
                return

        if self._is_drawing_textbox:
            self._update_textbox_drawing(event.pos())
            event.accept()
            return

        if self._is_drawing:
            self._update_preview_drawing(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        # 处理框选完成
        if self.selection_manager.is_box_selecting and event.button() == Qt.MouseButton.LeftButton:
            alt_pressed = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            self.selection_manager.finish_box_select(alt_pressed)
            event.accept()
            return

        if self._is_drawing_textbox and event.button() == Qt.MouseButton.LeftButton:
            self._finish_textbox_drawing()
            event.accept()
            return

        if self._is_drawing and event.button() == Qt.MouseButton.LeftButton:
            self._finish_drawing()

        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            # 清除拖动状态
            if self._drag_start_pos:
                self._potential_drag = False
                self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _start_drawing(self, pos):
        if self._image_item is None:
            return
        mask_shape = self._get_edit_mask_shape()
        if mask_shape is None:
            return
        self._is_drawing = True
        self._current_draw_scene_points = []
        self._current_draw_mask_points = []
        self._current_draw_mask_shape = mask_shape

        scene_point = self.mapToScene(pos)
        self._append_draw_point(scene_point)

        # Ensure the preview item exists and is ready
        if self._preview_item is None:
            pixmap = QPixmap(self._image_item.pixmap().size())
            pixmap.fill(Qt.GlobalColor.transparent)
            self._preview_item = self.scene.addPixmap(pixmap)
            self._preview_item.setZValue(150) # Ensure preview is on top of everything
            self._scale_mask_item(self._preview_item)
        self._preview_item.setVisible(True)
        self._redraw_preview_drawing()

    def _update_preview_drawing(self, pos):
        if not self._is_drawing:
            return
        self._append_draw_point(self.mapToScene(pos))
        self._redraw_preview_drawing()

    def _redraw_preview_drawing(self):
        if self._preview_item is None:
            return

        pixmap = self._preview_item.pixmap()
        pixmap.fill(Qt.GlobalColor.transparent) # Clear previous preview
        if self._current_draw_scene_points:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            if self._active_tool in ['pen', 'brush']:
                preview_pen = QPen(QColor(255, 0, 0, 128), self._brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            else:
                preview_pen = QPen(QColor(0, 150, 255, 100), self._brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(preview_pen)

            draw_points = [self._scene_to_image_point(point) for point in self._current_draw_scene_points]
            if len(draw_points) == 1:
                radius = max(1.0, self._brush_size / 2.0)
                painter.drawEllipse(draw_points[0], radius, radius)
            else:
                for idx in range(1, len(draw_points)):
                    painter.drawLine(draw_points[idx - 1], draw_points[idx])

            painter.end()

        self._preview_item.setPixmap(pixmap)
        # Force immediate scene update for real-time feedback
        self.scene.update()
        self.viewport().update()

    def _get_edit_mask_shape(self):
        current_mask = self.model.get_refined_mask()
        if current_mask is not None:
            mask = np.array(current_mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.ndim == 2 and mask.size > 0:
                return mask.shape[0], mask.shape[1]
        if self._image_item is None:
            return None
        return self._image_item.pixmap().height(), self._image_item.pixmap().width()

    def _scene_to_image_point(self, scene_point: QPointF) -> QPointF:
        if self._image_item is None:
            return scene_point
        image_point = self._image_item.mapFromScene(scene_point)
        image_w = self._image_item.pixmap().width()
        image_h = self._image_item.pixmap().height()
        if image_w <= 0 or image_h <= 0:
            return QPointF(0.0, 0.0)
        x = min(max(float(image_point.x()), 0.0), float(image_w - 1))
        y = min(max(float(image_point.y()), 0.0), float(image_h - 1))
        return QPointF(x, y)

    def _scene_to_mask_point(self, scene_point: QPointF, mask_shape: tuple[int, int]):
        if self._image_item is None:
            return None
        mask_h, mask_w = mask_shape
        if mask_h <= 0 or mask_w <= 0:
            return None

        image_point = self._scene_to_image_point(scene_point)
        image_w = self._image_item.pixmap().width()
        image_h = self._image_item.pixmap().height()
        if image_w <= 0 or image_h <= 0:
            return None

        x_ratio = float(image_point.x()) / float(max(image_w - 1, 1))
        y_ratio = float(image_point.y()) / float(max(image_h - 1, 1))
        x_mask = int(round(x_ratio * float(max(mask_w - 1, 0))))
        y_mask = int(round(y_ratio * float(max(mask_h - 1, 0))))
        x_mask = min(max(x_mask, 0), mask_w - 1)
        y_mask = min(max(y_mask, 0), mask_h - 1)
        return x_mask, y_mask

    def _append_draw_point(self, scene_point: QPointF):
        if not self._is_drawing or self._current_draw_mask_shape is None:
            return
        self._current_draw_scene_points.append(scene_point)
        mask_point = self._scene_to_mask_point(scene_point, self._current_draw_mask_shape)
        if mask_point is None:
            return
        if not self._current_draw_mask_points or self._current_draw_mask_points[-1] != mask_point:
            self._current_draw_mask_points.append(mask_point)

    def _build_stroke_mask(self, points: list[tuple[int, int]], mask_shape: tuple[int, int]) -> np.ndarray:
        mask_h, mask_w = mask_shape
        stroke_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        if not points:
            return stroke_mask

        image_w = self._image_item.pixmap().width() if self._image_item else mask_w
        image_h = self._image_item.pixmap().height() if self._image_item else mask_h
        scale_x = float(mask_w) / float(max(image_w, 1))
        scale_y = float(mask_h) / float(max(image_h, 1))
        stroke_size = max(1, int(round(self._brush_size * (scale_x + scale_y) * 0.5)))
        radius = max(1, stroke_size // 2)

        if len(points) == 1:
            cv2.circle(stroke_mask, points[0], radius, 255, thickness=-1, lineType=cv2.LINE_8)
            return stroke_mask

        for idx in range(1, len(points)):
            cv2.line(
                stroke_mask,
                points[idx - 1],
                points[idx],
                255,
                thickness=stroke_size,
                lineType=cv2.LINE_8,
            )
        for point in points:
            cv2.circle(stroke_mask, point, radius, 255, thickness=-1, lineType=cv2.LINE_8)
        return stroke_mask

    def _normalize_binary_mask_array(self, mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if mask is None:
            return np.zeros(target_shape, dtype=np.uint8)
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
        if mask_np.shape[:2] != target_shape:
            mask_np = cv2.resize(mask_np, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
        return mask_np

    def _finish_drawing(self):
        if not self._is_drawing or not self._current_draw_mask_points or self._current_draw_mask_shape is None:
            self._reset_drawing_state()
            # Clear preview even if no valid path was drawn
            self._clear_preview()
            return

        current_mask = self.model.get_refined_mask()
        old_mask_np = self._normalize_binary_mask_array(current_mask, self._current_draw_mask_shape) if current_mask is not None else None
        base_mask = self._normalize_binary_mask_array(current_mask, self._current_draw_mask_shape)
        stroke_mask = self._build_stroke_mask(self._current_draw_mask_points, self._current_draw_mask_shape)

        new_mask_np = base_mask.copy()
        if self._active_tool in ['pen', 'brush']:
            new_mask_np[stroke_mask > 0] = 255
        elif self._active_tool == 'eraser':
            new_mask_np[stroke_mask > 0] = 0
        else:
            self._reset_drawing_state()
            self._clear_preview()
            return

        # Clear the preview immediately
        self._clear_preview()

        # 没有实际变化时不写入历史
        if old_mask_np is not None and np.array_equal(old_mask_np, new_mask_np):
            self._reset_drawing_state()
            return
        if old_mask_np is None and not np.any(new_mask_np):
            self._reset_drawing_state()
            return

        # --- Refactored to Command Pattern ---
        from .commands import MaskEditCommand
        command = MaskEditCommand(
            model=self.model,
            old_mask=old_mask_np,
            new_mask=new_mask_np.copy()
        )
        
        # Access controller through model's controller reference
        controller = getattr(self.model, 'controller', None)
        if controller and hasattr(controller, 'execute_command'):
            controller.execute_command(command)
        else:
            # Fallback: try to find controller through parent hierarchy
            # GraphicsView -> QSplitter -> EditorView
            editor_view = self.parent()
            while editor_view and not hasattr(editor_view, 'controller'):
                editor_view = editor_view.parent()
            
            if editor_view and hasattr(editor_view, 'controller'):
                editor_view.controller.execute_command(command)
            else:
                # Last resort: directly update the model (no undo support)
                self.model.set_refined_mask(new_mask_np.copy())
                import logging
                logging.warning("Could not find controller, updated mask directly without undo support")

        self._reset_drawing_state()

    def _reset_drawing_state(self):
        self._is_drawing = False
        self._current_draw_scene_points = []
        self._current_draw_mask_points = []
        self._current_draw_mask_shape = None

    def _clear_preview(self):
        """Clear the preview item immediately."""
        if self._preview_item:
            self._preview_item.pixmap().fill(Qt.GlobalColor.transparent)
            self._preview_item.setVisible(False)
            # Force immediate update
            self.scene.update()
            self.viewport().update()




    # --- Mask Editing Slots & Methods ---

    @pyqtSlot(str)
    def _on_active_tool_changed(self, tool: str):
        """Slot to handle tool changes from the model."""
        self._active_tool = tool
        self._update_cursor()
        # Force immediate cursor update
        self.viewport().update()

    @pyqtSlot(int)
    def _on_brush_size_changed(self, size: int):
        """Slot to handle brush size changes from the model."""
        self._brush_size = size
        self._update_cursor()

    def _update_cursor(self):
        """Updates the cursor to match the selected tool and brush size."""
        if self._active_tool in ['pen', 'eraser', 'brush']:
            size = max(10, int(self._brush_size * self.transform().m11()))

            # 创建自定义光标
            cursor_size = size + 6
            pixmap = QPixmap(cursor_size, cursor_size)
            pixmap.fill(Qt.GlobalColor.transparent)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            center = cursor_size // 2
            radius = size // 2

            if self._active_tool in ['pen', 'brush']:
                # 画笔：红色圆圈
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
                painter.setBrush(Qt.GlobalColor.transparent)
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
                painter.setPen(QPen(Qt.GlobalColor.red, 1))
                painter.drawEllipse(center - radius + 1, center - radius + 1, (radius - 1) * 2, (radius - 1) * 2)
            else:  # eraser
                # 橡皮擦：蓝色圆圈
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
                painter.setBrush(Qt.GlobalColor.transparent)
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
                painter.setPen(QPen(Qt.GlobalColor.blue, 1))
                painter.drawEllipse(center - radius + 1, center - radius + 1, (radius - 1) * 2, (radius - 1) * 2)

            painter.end()

            cursor = QCursor(pixmap, center, center)
            # 在视图和viewport上都设置光标
            self.setCursor(cursor)
            self.viewport().setCursor(cursor)
        elif self._active_tool == 'draw_textbox':
            # 在视图和viewport上都设置光标
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.viewport().setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            # 选择工具下不要强制覆盖光标，让 item 手柄光标生效
            self.unsetCursor()
            self.viewport().unsetCursor()

    def enterEvent(self, event):
        """Handle mouse enter event."""
        self._update_cursor()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave event."""
        # Reset cursor when leaving the view
        self.unsetCursor()
        self.viewport().unsetCursor()
        super().leaveEvent(event)


    # --- Public Slots for Toolbar ---
    @pyqtSlot()
    def zoom_in(self):
        self.scale(1.15, 1.15)
        self._emit_view_state_changed()

    @pyqtSlot()
    def zoom_out(self):
        self.scale(1 / 1.15, 1 / 1.15)
        self._emit_view_state_changed()

    @pyqtSlot()
    def fit_to_window(self):
        if self._image_item:
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._emit_view_state_changed()

    def contextMenuEvent(self, event):
        """处理右键菜单事件"""
        # 检查是否有选中的区域
        selected_regions = self.model.get_selection()
        selection_count = len(selected_regions)
        
        # 创建右键菜单
        menu = QMenu(self)
        
        if selection_count > 0:
            # 有选中区域时的菜单项
            menu.addAction("🔍 OCR识别选中项", self._ocr_selected_regions)
            menu.addAction("🌐 翻译选中项", self._translate_selected_regions)
            menu.addSeparator()
            
            if selection_count == 1:
                # 单选时的额外选项
                menu.addAction("📋 复制区域", self._copy_selected_region)
                menu.addAction("🎨 粘贴样式", self._paste_region_style)
                menu.addSeparator()
            
            # 删除选项
            menu.addAction(f"🗑️ 删除选中的 {selection_count} 个区域", self._delete_selected_regions)
        else:
            # 无选中区域时的菜单项
            menu.addAction("➕ 添加文本框", self._add_text_box)
            menu.addAction("📋 粘贴区域", self._paste_region)
            menu.addSeparator()
            menu.addAction("🔄 刷新视图", self._refresh_view)
        
        # 显示菜单
        menu.exec(event.globalPos())

    def _get_controller(self):
        """获取控制器实例"""
        # GraphicsView -> QSplitter -> EditorView
        # 需要向上查找两级才能找到 EditorView
        parent = self.parent()
        # 向上遍历查找 EditorView
        current = parent
        while current:
            if hasattr(current, 'controller'):
                return current.controller
            current = current.parent() if hasattr(current, 'parent') and callable(current.parent) else None
        return None

    def _ocr_selected_regions(self):
        """OCR识别选中的区域"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if controller and selected_regions:
            controller.ocr_regions(selected_regions)

    def _translate_selected_regions(self):
        """翻译选中的区域"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if controller and selected_regions:
            controller.translate_regions(selected_regions)

    def _copy_selected_region(self):
        """复制选中的区域"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if len(selected_regions) == 1 and controller:
            controller.copy_region(selected_regions[0])

    def _paste_region_style(self):
        """粘贴区域样式"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if len(selected_regions) == 1 and controller:
            controller.paste_region_style(selected_regions[0])

    def _delete_selected_regions(self):
        """删除选中的区域"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if controller:
            controller.delete_regions(selected_regions)

    def _add_text_box(self):
        """添加新的文本框"""
        controller = self._get_controller()
        if controller:
            controller.enter_drawing_mode()

    def _paste_region(self):
        """粘贴区域到鼠标位置"""
        controller = self._get_controller()
        if controller and self._image_item:
            # 获取鼠标在场景中的位置
            mouse_pos_scene = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
            # 转换为图像坐标
            mouse_pos_image = self._image_item.mapFromScene(mouse_pos_scene)
            controller.paste_region(mouse_pos_image)

    def _refresh_view(self):
        """刷新视图"""
        self.scene.update()
        self.update()

    def _start_drawing_textbox(self, pos):
        """开始绘制新文本框"""
        if self._image_item is None:
            return

        self._is_drawing_textbox = True
        self._textbox_start_pos = self.mapToScene(pos)

        # 创建预览矩形
        if self._textbox_preview_item is None:
            pen = QPen(QColor(255, 0, 0, 200))  # 红色，半透明
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            brush = QColor(255, 0, 0, 50)  # 红色填充，更透明
            self._textbox_preview_item = self.scene.addRect(0, 0, 0, 0, pen, brush)
            self._textbox_preview_item.setZValue(200)

        # 重置预览矩形
        self._textbox_preview_item.setRect(0, 0, 0, 0)
        self._textbox_preview_item.setVisible(True)

    def _update_textbox_drawing(self, pos):
        """更新文本框绘制预览"""
        if not self._is_drawing_textbox or self._textbox_start_pos is None:
            return

        current_pos = self.mapToScene(pos)

        # 计算矩形
        x1, y1 = self._textbox_start_pos.x(), self._textbox_start_pos.y()
        x2, y2 = current_pos.x(), current_pos.y()

        # 确保矩形有正确的方向
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 更新预览矩形
        if self._textbox_preview_item:
            self._textbox_preview_item.setRect(left, top, width, height)

    def _finish_textbox_drawing(self):
        """完成文本框绘制"""
        if not self._is_drawing_textbox or self._textbox_start_pos is None:
            return

        # 获取最终矩形
        rect = self._textbox_preview_item.rect()

        # 隐藏并重置预览
        if self._textbox_preview_item:
            self._textbox_preview_item.setVisible(False)
            self._textbox_preview_item.setRect(0, 0, 0, 0)

        # 检查矩形大小是否足够大
        min_size = 20
        if rect.width() < min_size or rect.height() < min_size:
            self._is_drawing_textbox = False
            self._textbox_start_pos = None
            return

        # 创建新的文本区域
        self._create_new_text_region(rect)

        # 重置状态
        self._is_drawing_textbox = False
        self._textbox_start_pos = None

        # 退出绘制模式
        self.model.set_active_tool('select')

    def _create_new_text_region(self, rect):
        """创建新的文本区域"""
        # 将场景坐标转换为图像坐标
        if not self._image_item:
            return
        
        # 获取图像的变换矩阵
        image_transform = self._image_item.transform()
        try:
            # 获取逆变换
            inverse_transform = image_transform.inverted()[0]
            
            # 获取模板角度（如果有选中区域的话）
            template_angle = 0
            controller = self._get_controller()
            if controller:
                selected_regions = self.model.get_selection()
                if selected_regions:
                    template_region = self.model.get_region_by_index(selected_regions[-1])
                    if template_region:
                        template_angle = template_region.get('angle', 0)
            
            # 创建矩形的四个角点（考虑模板角度）
            rect_center_x = (rect.left() + rect.right()) / 2
            rect_center_y = (rect.top() + rect.bottom()) / 2
            rect_width = rect.right() - rect.left()
            rect_height = rect.bottom() - rect.top()
            
            # 创建相对于中心的矩形点
            half_width = rect_width / 2
            half_height = rect_height / 2
            
            # 未旋转的矩形角点（相对于中心）
            relative_points = [
                (-half_width, -half_height),  # 左上
                (half_width, -half_height),   # 右上
                (half_width, half_height),    # 右下
                (-half_width, half_height)    # 左下
            ]
            
            # 如果有模板角度，旋转这些点
            if template_angle != 0:
                import math
                cos_a = math.cos(math.radians(template_angle))
                sin_a = math.sin(math.radians(template_angle))
                
                rotated_points = []
                for x, y in relative_points:
                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a
                    rotated_points.append((new_x, new_y))
                relative_points = rotated_points
            
            # 转换为场景坐标
            scene_points = []
            for x, y in relative_points:
                scene_points.append(QPointF(rect_center_x + x, rect_center_y + y))
            
            # 转换为图像坐标
            image_points = []
            for point in scene_points:
                image_point = inverse_transform.map(point)
                image_points.append([image_point.x(), image_point.y()])

            # 计算中心点（在图像坐标系中）— 必须先于 white_frame_rect_local
            center_scene = QPointF(rect_center_x, rect_center_y)
            center_image = inverse_transform.map(center_scene)
            cx, cy = center_image.x(), center_image.y()

            # 白框数据：局部坐标（相对于 center 的偏移）
            xs = [p[0] for p in image_points]
            ys = [p[1] for p in image_points]
            white_frame_rect_local = [min(xs) - cx, min(ys) - cy, max(xs) - cx, max(ys) - cy]
            box_w = max(xs) - min(xs)
            box_h = max(ys) - min(ys)
            inferred_direction = 'vertical' if box_h > box_w else 'horizontal'
            
            # 使用之前获取的模板区域数据
            template_data = {}
            if controller:
                selected_regions = self.model.get_selection()
                if selected_regions:
                    template_region = self.model.get_region_by_index(selected_regions[-1])
                    if template_region:
                        template_data = {
                            'font_family': template_region.get('font_family', 'Arial'),
                            'font_size': template_region.get('font_size', 24),
                            'font_color': template_region.get('font_color', '#000000'),
                            'bg_colors': template_region.get('bg_colors', template_region.get('bg_color', [255, 255, 255])),
                            'alignment': template_region.get('alignment', 'center'),
                            'direction': template_region.get('direction', 'auto'),
                            'angle': template_region.get('angle', 0),
                            'letter_spacing': template_region.get('letter_spacing', 1.0),
                        }
            
            # 从配置服务获取默认渲染参数
            from services import get_config_service
            config_service = get_config_service()
            config = config_service.get_config()
            default_line_spacing = config.render.line_spacing if hasattr(config.render, 'line_spacing') else 1.0
            default_letter_spacing = config.render.letter_spacing if hasattr(config.render, 'letter_spacing') else 1.0
            default_stroke_width = config.render.stroke_width if hasattr(config.render, 'stroke_width') else 0.07
            # 确保不为 None
            if default_line_spacing is None:
                default_line_spacing = 1.0
            if default_letter_spacing is None:
                default_letter_spacing = 1.0
            if default_stroke_width is None:
                default_stroke_width = 0.07
            
            # 创建新区域数据，使用模板样式或默认值
            new_region_data = {
                'text': '',
                'texts': [''],  # 添加 texts 字段,避免 TextBlock 创建失败
                'translation': '',
                'polygons': [image_points],
                'lines': [image_points],
                'white_frame_rect_local': white_frame_rect_local,
                'has_custom_white_frame': True,
                'center': [cx, cy],
                'font_family': template_data.get('font_family', 'Arial'),
                'font_size': template_data.get('font_size', 24),
                'font_color': template_data.get('font_color', '#000000'),
                'bg_colors': template_data.get('bg_colors', [255, 255, 255]),
                'alignment': template_data.get('alignment', 'center'),
                'direction': inferred_direction,
                'angle': template_data.get('angle', 0),
                'line_spacing': default_line_spacing,
                'letter_spacing': template_data.get('letter_spacing', default_letter_spacing),
                'stroke_width': default_stroke_width,
                'font_path': ''  # 空字符串，渲染时会自动使用主页配置的默认字体
            }
            
            # 通知控制器添加新区域 - 使用命令模式以支持撤销
            controller = self._get_controller()
            if controller:
                from editor.commands import AddRegionCommand

                # 新建文本框前清理几何交互上下文，避免污染后续刷新路径
                self._clear_pending_geometry_edits()

                # 使用命令模式添加新区域
                command = AddRegionCommand(
                    model=self.model,
                    region_data=new_region_data,
                    description="Add New Text Box"
                )
                controller.execute_command(command)

                # 选中新创建的区域
                new_index = len(self.model.get_regions()) - 1
                self.model.set_selection([new_index])

                # 强制更新 view
                self.viewport().update()
                self.scene.update()

        except Exception as e:
            import traceback
            print(f"创建文本区域失败: {e}")
            traceback.print_exc()

