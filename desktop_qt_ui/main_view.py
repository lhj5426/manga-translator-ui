
from main_view_parts import dynamic_settings as main_view_dynamic
from main_view_parts import env_management as main_view_env
from main_view_parts import layout as layout_parts
from main_view_parts import runtime as main_view_runtime
from main_view_parts.style_generator import generate_main_view_style
from main_view_parts.theme import apply_widget_stylesheet, get_current_theme
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QSplitter,
    QWidget,
)
from services import get_config_service, get_i18n_manager


class MainView(QWidget):
    """
    主翻译视图，对应旧UI的 MainView。
    包含文件列表、设置和日志。
    """
    setting_changed = pyqtSignal(str, object)
    env_var_changed = pyqtSignal(str, str)
    editor_view_requested = pyqtSignal()
    theme_change_requested = pyqtSignal(str)
    language_change_requested = pyqtSignal(str)

    _open_filter_list = main_view_dynamic._open_filter_list
    _open_ai_ocr_prompt_editor = main_view_dynamic._open_ai_ocr_prompt_editor
    _open_ai_colorizer_prompt_editor = main_view_dynamic._open_ai_colorizer_prompt_editor
    _open_ai_renderer_prompt_editor = main_view_dynamic._open_ai_renderer_prompt_editor
    _process_next_setting_chunk = main_view_dynamic._process_next_setting_chunk
    _finalize_settings_ui = main_view_dynamic._finalize_settings_ui
    _create_dynamic_settings = main_view_dynamic._create_dynamic_settings
    _on_setting_changed = main_view_dynamic._on_setting_changed
    _on_upscale_ratio_changed = main_view_dynamic._on_upscale_ratio_changed
    _on_numeric_input_changed = main_view_dynamic._on_numeric_input_changed
    _update_upscale_ratio_options = main_view_dynamic._update_upscale_ratio_options
    _create_param_widgets = main_view_dynamic._create_param_widgets

    _create_left_sidebar = layout_parts.create_left_sidebar
    _create_translation_page = layout_parts.create_translation_page
    _create_settings_page = layout_parts.create_settings_page
    _create_env_page = layout_parts.create_env_page
    _create_prompt_page = layout_parts.create_prompt_page
    _create_font_page = layout_parts.create_font_page
    _create_right_panel = layout_parts.create_right_panel
    _switch_content_page = layout_parts.switch_content_page
    _on_nav_prompt_clicked = layout_parts.on_nav_prompt_clicked
    _on_nav_editor_clicked = layout_parts.on_nav_editor_clicked
    _on_nav_font_clicked = layout_parts.on_nav_font_clicked
    _populate_theme_combo = layout_parts.populate_theme_combo
    _populate_language_combo = layout_parts.populate_language_combo
    _on_theme_combo_changed = layout_parts.on_theme_combo_changed
    _on_language_combo_changed = layout_parts.on_language_combo_changed
    _refresh_prompt_manager = layout_parts.refresh_prompt_manager
    _apply_selected_prompt = layout_parts.apply_selected_prompt
    _on_prompt_selection_changed = layout_parts.on_prompt_selection_changed
    _open_prompt_editor = layout_parts.open_prompt_editor
    _create_new_prompt = layout_parts.create_new_prompt
    _copy_selected_prompt = layout_parts.copy_selected_prompt
    _rename_selected_prompt = layout_parts.rename_selected_prompt
    _delete_selected_prompt = layout_parts.delete_selected_prompt
    _refresh_font_manager = layout_parts.refresh_font_manager
    _import_fonts = layout_parts.import_fonts
    _delete_selected_font = layout_parts.delete_selected_font
    _apply_selected_font = layout_parts.apply_selected_font
    _on_font_selection_changed = layout_parts._on_font_selection_changed
    _refresh_font_preview_styles = layout_parts.refresh_font_preview_styles

    update_progress = main_view_runtime.update_progress
    reset_progress = main_view_runtime.reset_progress

    _create_env_widgets = main_view_env.create_env_widgets
    _get_env_default_placeholder = main_view_env.get_env_default_placeholder
    _debounced_save_env_var = main_view_env.debounced_save_env_var
    _on_open_custom_api_params_file = main_view_env.on_open_custom_api_params_file
    _on_test_api_clicked = main_view_env.on_test_api_clicked
    _on_get_models_clicked = main_view_env.on_get_models_clicked
    _refresh_preset_list = main_view_env.refresh_preset_list
    _on_add_preset_clicked = main_view_env.on_add_preset_clicked
    _on_delete_preset_clicked = main_view_env.on_delete_preset_clicked
    _on_preset_changed = main_view_env.on_preset_changed
    update_output_path_display = main_view_env.update_output_path_display
    _trigger_add_files = main_view_env.trigger_add_files

    _enable_stop_button = main_view_runtime.enable_stop_button
    set_stopping_state = main_view_runtime.set_stopping_state
    _sync_workflow_mode_from_config = main_view_runtime.sync_workflow_mode_from_config
    _on_workflow_mode_changed = main_view_runtime.on_workflow_mode_changed
    _update_workflow_mode_description = main_view_runtime.update_workflow_mode_description
    update_start_button_text = main_view_runtime.update_start_button_text

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.config_service = get_config_service()
        self.i18n = get_i18n_manager()
        self.env_widgets = {}
        self._env_debounce_timer = QTimer(self)
        self._env_debounce_timer.setSingleShot(True)
        self._env_debounce_timer.setInterval(500) # 500ms debounce delay

        self.layout = QHBoxLayout(self)
        self.env_var_changed.connect(self.controller.save_env_var)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setObjectName("main_view_root")
        
        # --- 创建主分割器 (左右) ---
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.setObjectName("main_view_splitter")
        self.layout.addWidget(main_splitter)

        # --- 左侧侧边栏 ---
        left_panel = self._create_left_sidebar()

        # --- 右侧面板 ---
        right_panel = self._create_right_panel()

        # --- 组合布局 ---
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 0) # 侧边栏固定为主
        main_splitter.setStretchFactor(1, 1) # 内容区可拉伸
        main_splitter.setCollapsible(0, False) # 侧边栏不折叠
        main_splitter.setCollapsible(1, True) # 内容区可以折叠
        main_splitter.setSizes([220, 1060]) # 设置初始比例
        main_splitter.setHandleWidth(6) # 设置分隔条宽度

        # 不在这里调用 _create_dynamic_settings，等待 app_logic.initialize 发送 config_loaded 信号
        # self._create_dynamic_settings()  # 删除这行，避免重复创建

        # Connect signals for button state management
        self.controller.state_manager.is_translating_changed.connect(self.on_translation_state_changed, type=Qt.ConnectionType.QueuedConnection)
        self.controller.state_manager.current_config_changed.connect(self.update_start_button_text)
        QTimer.singleShot(100, self.update_start_button_text) # Set initial text
        QTimer.singleShot(100, self._sync_workflow_mode_from_config) # Sync workflow mode dropdown
        self._apply_reference_ui_style()
    
    def _t(self, key: str, **kwargs) -> str:
        """翻译辅助方法"""
        if self.i18n:
            return self.i18n.translate(key, **kwargs)
        return key

    def _apply_reference_ui_style(self, theme: str | None = None):
        theme = theme or get_current_theme()
        apply_widget_stylesheet(self, generate_main_view_style(theme))
        if hasattr(self, "prompt_preview_panel") and self.prompt_preview_panel:
            self.prompt_preview_panel.apply_theme()
        if hasattr(self, "_refresh_font_preview_styles"):
            self._refresh_font_preview_styles()

    @pyqtSlot(dict)
    def set_parameters(self, config: dict):
        main_view_dynamic.set_parameters(self, config)

    def _show_setting_description(self, key: str, name: str, description: str):
        """更新右侧描述面板"""
        if hasattr(self, 'settings_desc_name'):
            self.settings_desc_name.setText(name)
        if hasattr(self, 'settings_desc_key'):
            self.settings_desc_key.setText(self._t("Settings Desc Key", config_key=key))
        if hasattr(self, 'settings_desc_text'):
            self.settings_desc_text.setText(description or self._t("Settings Desc No Description"))

    def refresh_tab_titles(self):
        """刷新标签页标题（用于语言切换）。"""
        tab_titles = getattr(self, "settings_tab_title_keys", None)
        if not tab_titles:
            tab_titles = ["Application Settings", "Basic Settings", "Advanced Settings", "Options"]
        for i, title_key in enumerate(tab_titles):
            if i < self.settings_tabs.count():
                self.settings_tabs.setTabText(i, self._t(title_key))

    def refresh_ui_texts(self):
        """刷新所有UI文本（用于语言切换）。"""
        self.refresh_tab_titles()

        if hasattr(self, "sidebar_start_label"):
            self.sidebar_start_label.setText(self._t("Start Translation"))
        if hasattr(self, "sidebar_settings_label"):
            self.sidebar_settings_label.setText(self._t("Settings"))
        if hasattr(self, "sidebar_tools_label"):
            self.sidebar_tools_label.setText(self._t("Data Management"))
        if hasattr(self, "sidebar_editor_label"):
            self.sidebar_editor_label.setText(self._t("Editor"))
        if hasattr(self, "nav_translation_button"):
            self.nav_translation_button.setText(self._t("Translation Interface"))
        if hasattr(self, "nav_editor_button"):
            self.nav_editor_button.setText(self._t("Editor View"))
        if hasattr(self, "nav_settings_button"):
            self.nav_settings_button.setText(self._t("Settings"))
        if hasattr(self, "nav_env_button"):
            self.nav_env_button.setText(self._t("API Management"))
        if hasattr(self, "nav_prompt_button"):
            self.nav_prompt_button.setText(self._t("Prompt Management"))
        if hasattr(self, "nav_font_button"):
            self.nav_font_button.setText(self._t("Font Management"))

        if hasattr(self, "theme_label"):
            self.theme_label.setText(self._t("Theme:"))
        if hasattr(self, "language_label"):
            self.language_label.setText(self._t("Language:"))
        self._populate_theme_combo()
        self._populate_language_combo()

        if hasattr(self, "translation_page_title"):
            self.translation_page_title.setText(self._t("Translation Interface"))
        if hasattr(self, "translation_input_card"):
            self.translation_input_card.setTitle(self._t("Input Files"))
        if hasattr(self, "translation_task_card"):
            self.translation_task_card.setTitle(self._t("Translation Task"))
        if hasattr(self, "add_files_button"):
            self.add_files_button.setText(self._t("Add Files"))
        if hasattr(self, "add_folder_button"):
            self.add_folder_button.setText(self._t("Add Folder"))
        if hasattr(self, "clear_list_button"):
            self.clear_list_button.setText(self._t("Clear List"))

        if hasattr(self, "output_folder_label"):
            self.output_folder_label.setText(self._t("Output Directory:"))
        if hasattr(self, "output_folder_input"):
            self.output_folder_input.setPlaceholderText(self._t("Select or drag output folder..."))
        if hasattr(self, "browse_button"):
            self.browse_button.setText(self._t("Browse..."))
        if hasattr(self, "open_button"):
            self.open_button.setText(self._t("Open"))

        if hasattr(self, "workflow_mode_hint_label"):
            self.workflow_mode_hint_label.setText(
                self._t("Choose translation workflow mode before starting the task.")
            )
        if hasattr(self, "workflow_mode_label"):
            self.workflow_mode_label.setText(self._t("Translation Workflow Mode:"))
        current_index = 0
        if hasattr(self, "workflow_mode_combo"):
            current_index = self.workflow_mode_combo.currentIndex()
            self.workflow_mode_combo.blockSignals(True)
            self.workflow_mode_combo.clear()
            self.workflow_mode_combo.addItems(
                [
                    self._t("Normal Translation"),
                    self._t("------ Split Execution ------"),
                    self._t("Import YOLO Label Data"),
                    self._t("OCR Only"),
                    self._t("Translate JSON Only"),
                    self._t("Import Translation and Generate Mask"),
                    self._t("Render Only (Use Existing Mask)"),
                    self._t("------ Other Optional Modes ------"),
                    self._t("Export Translation"),
                    self._t("Export Original Text"),
                    self._t("Import Translation and Render"),
                    self._t("Colorize Only"),
                    self._t("Upscale Only"),
                    self._t("Inpaint Only"),
                    self._t("Replace Translation"),
                ]
            )
            model = self.workflow_mode_combo.model()
            if model is not None:
                split_item = model.item(1)
                optional_item = model.item(7)
                if split_item is not None:
                    split_item.setEnabled(False)
                if optional_item is not None:
                    optional_item.setEnabled(False)
            self.workflow_mode_combo.setCurrentIndex(current_index)
            self.workflow_mode_combo.blockSignals(False)
        self._update_workflow_mode_description(current_index)

        self.update_start_button_text()

        if hasattr(self, "export_config_button"):
            self.export_config_button.setText(self._t("Export Config"))
        if hasattr(self, "import_config_button"):
            self.import_config_button.setText(self._t("Import Config"))

        if hasattr(self, "settings_page_title"):
            self.settings_page_title.setText(self._t("Settings Page Title"))
        if hasattr(self, "settings_page_subtitle"):
            self.settings_page_subtitle.setText(self._t("Settings Page Subtitle"))
        if hasattr(self, "settings_desc_header_label"):
            self.settings_desc_header_label.setText(self._t("Settings Desc Header"))
        if hasattr(self, "settings_desc_name"):
            self.settings_desc_name.setText("")
        if hasattr(self, "settings_desc_key"):
            self.settings_desc_key.setText("")
        if hasattr(self, "settings_desc_text"):
            self.settings_desc_text.setText(self._t("Settings Desc Placeholder"))

        if hasattr(self, "env_page_title_label"):
            self.env_page_title_label.setText(self._t("API Management"))
        if hasattr(self, "env_page_subtitle_label"):
            self.env_page_subtitle_label.setText(
                self._t("Manage API keys and environment variables for each translator")
            )
        if hasattr(self, "env_tab_widget"):
            self.env_tab_widget.setTabText(0, self._t("Translation"))
            self.env_tab_widget.setTabText(1, self._t("OCR"))
            self.env_tab_widget.setTabText(2, self._t("Colorization"))
            self.env_tab_widget.setTabText(3, self._t("Render"))

        if hasattr(self, "file_list") and hasattr(self.file_list, "refresh_ui_texts"):
            self.file_list.refresh_ui_texts()

        if hasattr(self, "prompt_page_title_label"):
            self.prompt_page_title_label.setText(self._t("Prompt Management"))
        if hasattr(self, "prompt_page_subtitle_label"):
            self.prompt_page_subtitle_label.setText(
                self._t("Manage and apply prompt files for translation")
            )
        if hasattr(self, "prompt_card"):
            self.prompt_card.setTitle(self._t("Prompt List"))
        if hasattr(self, "prompt_refresh_button"):
            self.prompt_refresh_button.setText(self._t("Refresh"))
        if hasattr(self, "prompt_open_dir_button"):
            self.prompt_open_dir_button.setText(self._t("Open Directory"))
        if hasattr(self, "prompt_apply_button"):
            self.prompt_apply_button.setText(self._t("Apply Selected Prompt"))
        if hasattr(self, "prompt_new_button"):
            self.prompt_new_button.setText(self._t("New"))
        if hasattr(self, "prompt_copy_button"):
            self.prompt_copy_button.setText(self._t("Copy"))
        if hasattr(self, "prompt_rename_button"):
            self.prompt_rename_button.setText(self._t("Rename"))
        if hasattr(self, "prompt_delete_button"):
            self.prompt_delete_button.setText(self._t("Delete"))
        if hasattr(self, "prompt_preview_panel") and hasattr(self.prompt_preview_panel, "refresh_ui_texts"):
            self.prompt_preview_panel.refresh_ui_texts()

        if hasattr(self, "font_page_title_label"):
            self.font_page_title_label.setText(self._t("Font Management"))
        if hasattr(self, "font_page_subtitle_label"):
            self.font_page_subtitle_label.setText(
                self._t("Manage and preview fonts for text rendering")
            )
        if hasattr(self, "font_card"):
            self.font_card.setTitle(self._t("Font List"))
        if hasattr(self, "font_import_button"):
            self.font_import_button.setText(self._t("Import"))
        if hasattr(self, "font_delete_button"):
            self.font_delete_button.setText(self._t("Delete"))
        if hasattr(self, "font_refresh_button"):
            self.font_refresh_button.setText(self._t("Refresh"))
        if hasattr(self, "font_open_dir_button"):
            self.font_open_dir_button.setText(self._t("Open Directory"))
        if hasattr(self, "font_apply_button"):
            self.font_apply_button.setText(self._t("Apply Selected Font"))
        if hasattr(self, "font_preview_card"):
            self.font_preview_card.setTitle(self._t("Font Preview"))

        self._clear_dynamic_settings()
        self._create_dynamic_settings()
        if hasattr(self, "prompt_list_widget"):
            self._refresh_prompt_manager()
        if hasattr(self, "font_list_widget"):
            self._refresh_font_manager()

    def _clear_dynamic_settings(self):
        """清理所有动态创建的设置控件。"""
        if hasattr(self, "env_group_container_layout"):
            while self.env_group_container_layout.count():
                item = self.env_group_container_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        for panel in getattr(self, "tab_frames", {}).values():
            if panel and panel.layout():
                layout = panel.layout()
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()

    @pyqtSlot(bool)
    def on_translation_state_changed(self, is_translating: bool):
        main_view_runtime.on_translation_state_changed(self, is_translating)
