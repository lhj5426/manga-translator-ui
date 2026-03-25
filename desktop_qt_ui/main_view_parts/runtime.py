from PyQt6.QtCore import QTimer

from main_view_parts.theme import repolish_widget


def _set_progress_state(self, state: str):
    if hasattr(self, "progress_bar"):
        self.progress_bar.setProperty("progressState", state)
        repolish_widget(self.progress_bar)


def _set_start_button_state(self, state: str):
    if hasattr(self, "start_button"):
        self.start_button.setProperty("translationState", state)
        repolish_widget(self.start_button)


def update_workflow_mode_description(self, index: int | None = None):
    """根据翻译流程模式更新翻译页标题下方的介绍文字。"""
    if not hasattr(self, "translation_page_subtitle"):
        return

    if index is None:
        if hasattr(self, "workflow_mode_combo"):
            index = self.workflow_mode_combo.currentIndex()
        else:
            index = 0

    mode_keys = {
        0: "Normal Translation",
        2: "Import YOLO Label Data",
        3: "OCR Only",
        4: "Translate JSON Only",
        5: "Import Translation and Generate Mask",
        6: "Render Only (Use Existing Mask)",
        8: "Export Translation",
        9: "Export Original Text",
        10: "Import Translation and Render",
        11: "Colorize Only",
        12: "Upscale Only",
        13: "Inpaint Only",
        14: "Replace Translation",
    }
    tip_keys = {
        0: "Tip: Standard translation pipeline with detection, OCR, translation and rendering",
        2: "Tip: Import YOLO labels for all images first and save reusable detection boxes to JSON without running OCR",
        3: "Tip: Load existing detection boxes from JSON and run OCR only, then export original text template",
        4: "Tip: Requires existing JSON data. The app reads original text from JSON, translates it, writes results back to JSON, and deletes imagename_original.txt after success",
        5: "Tip: Import translation JSON first and only generate/refine mask PNG files, without rendering output text",
        6: "Tip: Use existing mask PNG from JSON (mask_file) to inpaint and render. No automatic mask regeneration",
        8: "Tip: After exporting, check manga_translator_work/translations/ for imagename_translated.txt files",
        9: "Tip: After exporting, manually translate imagename_original.txt in manga_translator_work/originals/, then use 'Import Translation and Render' mode",
        10: "Tip: Will read TXT files from manga_translator_work/originals/ or translations/ and render (prioritize _original.txt)",
        11: "Tip: Only colorize images, no detection, OCR, translation or rendering",
        12: "Tip: Only upscale images, no detection, OCR, translation or rendering",
        13: "Tip: Detect text regions and inpaint to output clean images, no translation or rendering",
        14: "Tip: Place translated images in manga_translator_work/translated_images with matching filenames. The app extracts translated text, matches regions on raw images, inpaints originals, and renders translated text.",
    }
    mode_key = mode_keys.get(index, mode_keys[0])
    tip_key = tip_keys.get(index, tip_keys[0])
    if hasattr(self, "translation_page_title"):
        self.translation_page_title.setText(self._t(mode_key))
    self.translation_page_subtitle.setText(self._t(tip_key))







def update_progress(self, current: int, total: int, message: str = ""):
    """更新进度条。"""
    if total > 0:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        percentage = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setFormat(f"{current}/{total} ({percentage}%)")
        if hasattr(self, "progress_info_label"):
            self.progress_info_label.setText(message or f"已完成 {current}/{total}")

        if not getattr(self, "_progress_active", False):
            self._progress_active = True
            _set_progress_state(self, "active")
    else:
        self._progress_active = False
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0/0 (0%)")
        if hasattr(self, "progress_info_label"):
            self.progress_info_label.setText("")
        _set_progress_state(self, "idle")


def reset_progress(self):
    """重置进度条为初始状态（灰色）。"""
    self._progress_active = False
    self.progress_bar.setMaximum(100)
    self.progress_bar.setValue(0)
    self.progress_bar.setFormat("0/0 (0%)")
    if hasattr(self, "progress_info_label"):
        self.progress_info_label.setText("")
    _set_progress_state(self, "idle")


def on_translation_state_changed(self, is_translating: bool):
    """根据翻译状态更新开始/停止按钮。"""
    if is_translating:
        self.start_button.setEnabled(False)
        self.start_button.setText(self._t("Starting..."))
        QTimer.singleShot(2000, self._enable_stop_button)
    else:
        self.start_button.setEnabled(True)
        _set_start_button_state(self, "ready")

        try:
            self.start_button.clicked.disconnect()
        except TypeError:
            pass
        self.start_button.clicked.connect(self.controller.start_backend_task)
        self.update_start_button_text()


def enable_stop_button(self):
    """启用停止按钮（延迟调用）。"""
    if self.controller.state_manager.is_translating():
        self.start_button.setEnabled(True)
        mode_key = _get_current_workflow_mode_key(self)
        if mode_key:
            self.start_button.setText(f"停止{self._t(mode_key)}")
        else:
            self.start_button.setText("停止当前任务")
        _set_start_button_state(self, "stop")
        try:
            self.start_button.clicked.disconnect()
        except TypeError:
            pass
        self.start_button.clicked.connect(self.controller.stop_task)


def set_stopping_state(self):
    """设置按钮为“停止中...”状态，避免重复点击。"""
    self.start_button.setEnabled(False)
    self.start_button.setText(self._t("Stopping..."))
    _set_start_button_state(self, "stopping")
    try:
        self.start_button.clicked.disconnect()
    except TypeError:
        pass


def _get_current_workflow_mode_key(self):
    try:
        config = self.config_service.get_config()
        if config.cli.replace_translation:
            return "Replace Translation"
        if getattr(config.cli, "ocr_only", False):
            return "OCR Only"
        if getattr(config.cli, "import_yolo_only", False):
            return "Import YOLO Label Data"
        if getattr(config.cli, "load_text_generate_mask_only", False):
            return "Import Translation and Generate Mask"
        if getattr(config.cli, "load_text_render_only", False):
            return "Render Only (Use Existing Mask)"
        if config.cli.inpaint_only:
            return "Inpaint Only"
        if config.cli.upscale_only:
            return "Upscale Only"
        if config.cli.colorize_only:
            return "Colorize Only"
        if config.cli.translate_json_only:
            return "Translate JSON Only"
        if config.cli.load_text:
            return "Import Translation and Render"
        if config.cli.template:
            return "Export Original Text"
        if config.cli.generate_and_export:
            return "Export Translation"
        return "Normal Translation"
    except Exception:
        return None


def sync_workflow_mode_from_config(self):
    """从配置同步下拉框的选择。"""
    try:
        config = self.config_service.get_config()
        self.workflow_mode_combo.blockSignals(True)

        if config.cli.replace_translation:
            self.workflow_mode_combo.setCurrentIndex(14)
        elif getattr(config.cli, "ocr_only", False):
            self.workflow_mode_combo.setCurrentIndex(3)
        elif getattr(config.cli, "import_yolo_only", False):
            self.workflow_mode_combo.setCurrentIndex(2)
        elif getattr(config.cli, "load_text_generate_mask_only", False):
            self.workflow_mode_combo.setCurrentIndex(5)
        elif getattr(config.cli, "load_text_render_only", False):
            self.workflow_mode_combo.setCurrentIndex(6)
        elif config.cli.load_text:
            self.workflow_mode_combo.setCurrentIndex(10)
        elif config.cli.inpaint_only:
            self.workflow_mode_combo.setCurrentIndex(13)
        elif config.cli.upscale_only:
            self.workflow_mode_combo.setCurrentIndex(12)
        elif config.cli.colorize_only:
            self.workflow_mode_combo.setCurrentIndex(11)
        elif config.cli.translate_json_only:
            self.workflow_mode_combo.setCurrentIndex(4)
        elif config.cli.template:
            self.workflow_mode_combo.setCurrentIndex(9)
        elif config.cli.generate_and_export:
            self.workflow_mode_combo.setCurrentIndex(8)
        else:
            self.workflow_mode_combo.setCurrentIndex(0)

        self.workflow_mode_combo.blockSignals(False)
        self._last_workflow_mode_index = self.workflow_mode_combo.currentIndex()
        update_workflow_mode_description(self, self.workflow_mode_combo.currentIndex())
    except Exception as e:
        print(f"Error syncing workflow mode: {e}")


def on_workflow_mode_changed(self, index: int):
    """处理翻译流程模式改变并持久化。"""
    if index in (1, 7):
        fallback_index = getattr(self, "_last_workflow_mode_index", 0)
        if fallback_index in (1, 7):
            fallback_index = 0
        self.workflow_mode_combo.blockSignals(True)
        self.workflow_mode_combo.setCurrentIndex(fallback_index)
        self.workflow_mode_combo.blockSignals(False)
        return

    config = self.config_service.get_config()

    config.cli.load_text = False
    config.cli.load_text_generate_mask_only = False
    config.cli.load_text_render_only = False
    config.cli.translate_json_only = False
    config.cli.template = False
    config.cli.generate_and_export = False
    config.cli.colorize_only = False
    config.cli.upscale_only = False
    config.cli.inpaint_only = False
    config.cli.replace_translation = False
    config.cli.import_yolo_only = False
    config.cli.ocr_only = False

    if index == 2:
        config.cli.import_yolo_only = True
    elif index == 3:
        config.cli.ocr_only = True
    elif index == 4:
        config.cli.translate_json_only = True
    elif index == 5:
        config.cli.load_text = True
        config.cli.load_text_generate_mask_only = True
    elif index == 6:
        config.cli.load_text = True
        config.cli.load_text_render_only = True
    elif index == 8:
        config.cli.generate_and_export = True
    elif index == 9:
        config.cli.template = True
    elif index == 10:
        config.cli.load_text = True
    elif index == 11:
        config.cli.colorize_only = True
    elif index == 12:
        config.cli.upscale_only = True
    elif index == 13:
        config.cli.inpaint_only = True
    elif index == 14:
        config.cli.replace_translation = True

    self.config_service.set_config(config)
    self.config_service.save_config_file()
    self._last_workflow_mode_index = index
    self.update_start_button_text()
    update_workflow_mode_description(self, index)


def update_start_button_text(self):
    """根据当前模式更新开始按钮文案。"""
    if self.controller.state_manager.is_translating():
        return

    try:
        config = self.config_service.get_config()
        if config.cli.replace_translation:
            self.start_button.setText(self._t("Start Replace Translation"))
        elif getattr(config.cli, "ocr_only", False):
            self.start_button.setText(self._t("Start OCR Only"))
        elif getattr(config.cli, "import_yolo_only", False):
            self.start_button.setText(self._t("Start Importing YOLO Labels"))
        elif getattr(config.cli, "load_text_generate_mask_only", False):
            self.start_button.setText(self._t("Start Generating Mask"))
        elif getattr(config.cli, "load_text_render_only", False):
            self.start_button.setText(self._t("Start Render Only"))
        elif config.cli.inpaint_only:
            self.start_button.setText(self._t("Start Inpainting"))
        elif config.cli.upscale_only:
            self.start_button.setText(self._t("Start Upscaling"))
        elif config.cli.colorize_only:
            self.start_button.setText(self._t("Start Colorizing"))
        elif config.cli.translate_json_only:
            self.start_button.setText(self._t("Start JSON Translation"))
        elif config.cli.load_text:
            self.start_button.setText(self._t("Import Translation and Render"))
        elif config.cli.template:
            self.start_button.setText(self._t("Generate Original Text Template"))
        elif config.cli.generate_and_export:
            self.start_button.setText(self._t("Export Translation"))
        else:
            self.start_button.setText(self._t("Start Translation"))
    except Exception as e:
        self.start_button.setText(self._t("Start Translation"))
        print(f"Could not update button text: {e}")
