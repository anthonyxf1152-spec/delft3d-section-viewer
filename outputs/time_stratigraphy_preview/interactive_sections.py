"""Desktop controls for selected-time, selected-location Delft3D sections."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import sys
import traceback

os.environ.setdefault('QT_API', 'pyside6')
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np

from stratigraphy_model import StratigraphyModel
from section_figures import draw_sections, export_sections
from cigvis_time_viewer import create_canvas, save_canvas
from colorbar_settings import color_limits
from colorbar_editor import SandColorEditor, sand_color_state

HERE = Path(__file__).resolve().parent
def reference_positions(model):
    """Grid reference points for arbitrary projects; not inferred fan zones."""
    indices = np.argwhere(model.valid)
    positions = {}
    for title, fraction in [('上部参考', .8), ('中部参考', .5), ('下部参考', .2)]:
        target = np.array([np.median(indices[:, 0]), np.quantile(indices[:, 1], fraction)])
        i, j = indices[np.argmin(np.sum((indices - target)**2, axis=1))]
        positions[title] = (float(model.x[i]), float(model.y[j]))
    return positions


class CigvisWindow(QtWidgets.QMainWindow):
    def __init__(self, canvas, metadata, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.metadata = metadata
        self.setWindowTitle(f"Cigvis 时间层序｜记录 {metadata['record']}｜左键旋转 · 滚轮缩放")
        self.setCentralWidget(canvas.native)
        self.resize(1660, 900)
        self.statusBar().showMessage('左：所选时刻层面选集；右：当前指定位置剖面。左键旋转，滚轮缩放，Shift＋左键平移。')

    def closeEvent(self, event):
        self.canvas.close()
        super().closeEvent(event)


class SectionWindow(QtWidgets.QMainWindow):
    def __init__(self, model, record=-1, x_m=None, y_m=None,
                 direction='both', half_width_m=6000., export_root=None):
        super().__init__()
        self.model = model
        self.sections = []
        self.sand_values = {}
        self.map_ax = None
        self.three_d_window = None
        self.seismic_window = None
        self.export_root = Path(export_root or HERE / 'exports')
        self._ready = False
        presets = reference_positions(model)
        default_x, default_y = presets['中部参考']
        if x_m is None and y_m is None:
            for axis, centre in ((model.x, default_x), (model.y, default_y)):
                distances = np.sort(abs(axis - centre))
                half_width_m = max(half_width_m, float(distances[1]) * 1.001)
        x_m = default_x if x_m is None else x_m
        y_m = default_y if y_m is None else y_m
        self.setWindowTitle('Delft3D｜任意时刻与位置的层序及含砂率剖面')
        self.resize(1800, 960)
        central = QtWidgets.QWidget(); layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12); layout.setSpacing(14)
        controls = QtWidgets.QWidget(); controls.setFixedWidth(304)
        panel = QtWidgets.QVBoxLayout(controls); panel.setContentsMargins(0, 0, 0, 0)
        heading = QtWidgets.QLabel('Delft3D 时间层序')
        heading.setStyleSheet('font-size:20px;font-weight:600;color:#19354c;')
        panel.addWidget(heading)
        subtitle = QtWidgets.QLabel('选择时刻与位置，查看最终保留到该时刻的层段。')
        subtitle.setWordWrap(True); panel.addWidget(subtitle)
        self.open_source_button = QtWidgets.QPushButton('打开 / 合并 DAT / DEF 结果…')
        self.open_source_button.clicked.connect(self.open_project); panel.addWidget(self.open_source_button)
        source = getattr(model, 'source_info', {})
        source_path = Path(source.get('dat_file', ''))
        source_name = f'{source_path.parent.name} / {source_path.name}' if source.get('dat_file') else model.volume_dir.name
        if source.get('kind') == 'continuation_series':
            source_name = f"多期合并 · {len(source['sources'])} 期 · {model.record_count} 个时刻"
        self.source_label = QtWidgets.QLabel(f'当前结果：{source_name}')
        self.source_label.setTextFormat(QtCore.Qt.PlainText)
        self.source_label.setWordWrap(True)
        details = str(source.get('dat_file', model.volume_dir))
        if source.get('kind') == 'continuation_series':
            details = '\n'.join(Path(s['dat_file']).name for s in source['sources'])
            gaps = [s['unobserved_interval'] for s in source['seams'] if 'unobserved_interval' in s]
            if gaps:
                self.source_label.setText(self.source_label.text() + '\n含未保存时段（悬停查看）')
                details += '\n' + '\n'.join(f"未保存：水动力 {g['previous_hydro_days']:g} → {g['next_hydro_days']:g} 天；未插值" for g in gaps)
        self.source_label.setToolTip(details)
        panel.addWidget(self.source_label)
        time_group = QtWidgets.QGroupBox('1  选择保存时刻')
        time_layout = QtWidgets.QVBoxLayout(time_group)
        self.time_combo = QtWidgets.QComboBox()
        for i in range(model.record_count):
            self.time_combo.addItem(f'记录 {i:03d} · 形态 {model.morph_days[i]:.2f} 天', i)
        self.time_combo.setMaxVisibleItems(15)
        self.time_combo.setCurrentIndex(model.record_count-1 if record == -1 else record)
        time_layout.addWidget(self.time_combo)
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setRange(0, model.record_count-1)
        self.time_slider.setValue(self.time_combo.currentIndex()); time_layout.addWidget(self.time_slider)
        nav = QtWidgets.QHBoxLayout()
        for label, delta in [('上一时刻', -1), ('下一时刻', 1)]:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _=False, step=delta: self.time_combo.setCurrentIndex(
                max(0, min(model.record_count-1, self.time_combo.currentIndex()+step))))
            nav.addWidget(button)
        time_layout.addLayout(nav)
        self.time_info = QtWidgets.QLabel(); self.time_info.setWordWrap(True); time_layout.addWidget(self.time_info)
        panel.addWidget(time_group)
        location_group = QtWidgets.QGroupBox('2  指定剖面位置')
        form = QtWidgets.QFormLayout(location_group)
        self.x_spin = self.coordinate_spin(model.x, x_m)
        self.y_spin = self.coordinate_spin(model.y, y_m)
        form.addRow('原工程 X / m', self.x_spin); form.addRow('原工程 Y / m', self.y_spin)
        preset_row = QtWidgets.QHBoxLayout()
        for label, xy in presets.items():
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(lambda _=False, pos=xy: self.set_position(*pos))
            preset_row.addWidget(button)
        form.addRow('参考位置', preset_row)
        self.direction = QtWidgets.QComboBox()
        for label, key in [('XZ与YZ两者', 'both'), ('XZ：固定Y，看X–Z', 'x'), ('YZ：固定X，看Y–Z', 'y')]:
            self.direction.addItem(label, key)
        self.direction.setCurrentIndex(self.direction.findData(direction)); form.addRow('剖面方向', self.direction)
        self.width_spin = QtWidgets.QDoubleSpinBox()
        max_width_km = max(80., float(max(np.ptp(model.x), np.ptp(model.y))) / 1000.)
        self.width_spin.setRange(.001, max_width_km)
        self.width_spin.setDecimals(2); self.width_spin.setSingleStep(1); self.width_spin.setValue(half_width_m/1000)
        form.addRow('剖面半宽 / km', self.width_spin)
        hint = QtWidgets.QLabel('也可点击右侧平面图选点。位置采用最近原始网格，实际坐标显示在剖面标题中。')
        hint.setWordWrap(True); form.addRow(hint)
        panel.addWidget(location_group)
        colors = QtWidgets.QGroupBox('3  调整Colorbar与含砂率')
        colors_form = QtWidgets.QFormLayout(colors)
        self.stage_mode = QtWidgets.QComboBox()
        for label, key in [('全时段统一', 'global'), ('随当前时刻', 'current'), ('手动范围', 'manual')]:
            self.stage_mode.addItem(label, key)
        self.bed_mode = QtWidgets.QComboBox()
        for label, key in [('随当前时刻', 'current'), ('全时段统一', 'global'), ('手动范围', 'manual')]:
            self.bed_mode.addItem(label, key)
        self.stage_min = self.color_spin(0, 0, 1e6)
        self.stage_max = self.color_spin(float(model.morph_days[-1]), 0, 1e6)
        self.bed_min = self.color_spin(-10, -1e6, 1e6)
        self.bed_max = self.color_spin(30, -1e6, 1e6)
        colors_form.addRow('沉积阶段 / 天', self.stage_mode)
        colors_form.addRow('阶段下限 / 上限', self.color_pair(self.stage_min, self.stage_max))
        colors_form.addRow('床面高程 / m', self.bed_mode)
        colors_form.addRow('高程下限 / 上限', self.color_pair(self.bed_min, self.bed_max))
        self.sand_mode = QtWidgets.QComboBox()
        self.sand_mode.addItem('分期估算＋缺测补全', 'filled')
        self.sand_mode.addItem('分期净增量估算', 'increment')
        self.sand_mode.addItem('当前混合床平均', 'mixed')
        colors_form.addRow('二维含砂率', self.sand_mode)
        panel.addWidget(colors)
        self.sand_color_editor = SandColorEditor()
        self.sand_color_state = sand_color_state()
        panel.addWidget(self.sand_color_editor)
        self.update_button = QtWidgets.QPushButton('更新剖面')
        self.update_button.setMinimumHeight(38); self.update_button.clicked.connect(self.update_plot)
        self.update_button.setStyleSheet('background:#245a7d;color:white;font-weight:600;border-radius:5px;')
        panel.addWidget(self.update_button)
        self.seismic_button = QtWidgets.QPushButton('当前切片地震正演…')
        self.seismic_button.setMinimumHeight(36)
        self.seismic_button.setToolTip('对当前时刻和 XZ / YZ 位置计算声阻抗反射系数与 30 Hz PSF 褶积响应。')
        self.seismic_button.clicked.connect(self.show_seismic)
        panel.addWidget(self.seismic_button)
        three_group = QtWidgets.QGroupBox('4  Cigvis 三维显示')
        three_form = QtWidgets.QFormLayout(three_group)
        self.ve_spin = QtWidgets.QDoubleSpinBox(); self.ve_spin.setRange(.1, 2000)
        self.ve_spin.setDecimals(1); self.ve_spin.setSingleStep(50); self.ve_spin.setValue(500)
        self.surface_spin = QtWidgets.QSpinBox(); self.surface_spin.setRange(2, 24); self.surface_spin.setValue(10)
        self.opacity_spin = QtWidgets.QDoubleSpinBox(); self.opacity_spin.setRange(.05, 1)
        self.opacity_spin.setDecimals(2); self.opacity_spin.setSingleStep(.05); self.opacity_spin.setValue(.38)
        three_form.addRow('垂向显示夸张', self.ve_spin)
        three_form.addRow('抽选层面数量', self.surface_spin)
        three_form.addRow('层面不透明度', self.opacity_spin)
        self.curtain_field = QtWidgets.QComboBox()
        for label, key in [('保存阶段（形态时间）', 'stage'),
                           ('当前混合床平均含砂率', 'mixed'),
                           ('分期净增量估算含砂率', 'increment'),
                           ('分期含砂率＋缺测补全', 'filled')]:
            self.curtain_field.addItem(label, key)
        self.curtain_field.setCurrentIndex(3)
        three_form.addRow('三维剖面着色', self.curtain_field)
        self.three_button = QtWidgets.QPushButton('Cigvis三维展示')
        self.three_button.clicked.connect(self.show_3d); three_form.addRow(self.three_button)
        panel.addWidget(three_group)
        self.export_button = QtWidgets.QPushButton('导出当前结果')
        self.export_button.setMinimumHeight(34); self.export_button.clicked.connect(lambda: self.export_current())
        panel.addWidget(self.export_button)
        help_button = QtWidgets.QPushButton('使用方法与结果含义')
        help_button.clicked.connect(self.show_help); panel.addWidget(help_button)
        note = QtWidgets.QLabel('二维图同时显示层序与含砂率；三维剖面也可选含砂率。'
                                '默认补全缺测：同阶段邻域插值，缺少邻域时用当期混合床均值。'
                                '补全属于估算，可切回分期净增量查看灰色缺测。')
        note.setStyleSheet('color:#52657a;'); note.setWordWrap(True); panel.addWidget(note)
        panel.addStretch()
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(controls); scroll.setFrameShape(QtWidgets.QFrame.NoFrame); scroll.setFixedWidth(322)
        layout.addWidget(scroll)
        right = QtWidgets.QWidget(); right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(12.6, 8.8), layout='constrained')
        self.figure_canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.figure_canvas, self)
        right_layout.addWidget(self.toolbar); right_layout.addWidget(self.figure_canvas, 1)
        self.location_info = QtWidgets.QLabel(); self.location_info.setWordWrap(True)
        self.location_info.setStyleSheet('padding:6px;background:#eef4f7;color:#25465f;border-radius:4px;')
        right_layout.addWidget(self.location_info)
        layout.addWidget(right, 1); self.setCentralWidget(central)
        self.figure_canvas.mpl_connect('button_press_event', self.map_click)
        self.time_combo.currentIndexChanged.connect(self.time_changed)
        self.time_slider.valueChanged.connect(self.time_combo.setCurrentIndex)
        for control in (self.x_spin, self.y_spin, self.width_spin):
            control.valueChanged.connect(self.mark_dirty)
        self.direction.currentIndexChanged.connect(self.mark_dirty)
        self.sand_mode.currentIndexChanged.connect(self.mark_dirty)
        self.curtain_field.currentIndexChanged.connect(self.mark_dirty)
        for mode in (self.stage_mode, self.bed_mode):
            mode.currentIndexChanged.connect(self.sync_color_controls)
        for control in (self.stage_min, self.stage_max, self.bed_min, self.bed_max):
            control.valueChanged.connect(self.mark_dirty)
        self._ready = True
        self.sand_color_state.changed.connect(self.sand_colors_changed)
        self.sync_color_controls()
        self.time_changed(self.time_combo.currentIndex())
        self._initial_plot_ok = self.update_plot()

    @staticmethod
    def coordinate_spin(axis, value):
        spin = QtWidgets.QDoubleSpinBox(); spin.setDecimals(6)
        spin.setRange(float(axis[0]), float(axis[-1])); spin.setSingleStep(125); spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def open_project(self):
        from source_dialog import open_source
        model = open_source(self)
        if model is None:
            return
        # Keep the current view alive until loading AND first drawing succeed.
        try:
            replacement = SectionWindow(model, export_root=self.export_root)
            if not replacement._initial_plot_ok:
                replacement.close()
                return
            replacement.show()
            QtWidgets.QApplication.instance()._section_window = replacement
            self.close()
        except Exception as exc:
            self.show_error(exc)

    @staticmethod
    def color_spin(value, low, high):
        spin = QtWidgets.QDoubleSpinBox(); spin.setDecimals(3)
        spin.setRange(low, high); spin.setSingleStep(10); spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def color_pair(left, right):
        widget = QtWidgets.QWidget(); row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
        row.addWidget(left); row.addWidget(QtWidgets.QLabel('—')); row.addWidget(right)
        return widget

    def sync_color_controls(self, *_):
        record = self.time_combo.currentIndex()
        for field, mode, low, high in [('stage', self.stage_mode, self.stage_min, self.stage_max),
                                       ('bed', self.bed_mode, self.bed_min, self.bed_max)]:
            manual = mode.currentData() == 'manual'
            low.setEnabled(manual); high.setEnabled(manual)
            if not manual:
                bounds = color_limits(self.model, record, field, mode.currentData())
                for control, value in zip((low, high), bounds):
                    control.blockSignals(True); control.setValue(value); control.blockSignals(False)
        self.mark_dirty()

    def selection(self):
        record = self.time_combo.currentIndex()
        stage_mode, bed_mode = self.stage_mode.currentData(), self.bed_mode.currentData()
        stage_limits = color_limits(self.model, record, 'stage', stage_mode,
                                    (self.stage_min.value(), self.stage_max.value()))
        bed_limits = color_limits(self.model, record, 'bed', bed_mode,
                                  (self.bed_min.value(), self.bed_max.value()))
        return {'record': record, 'x_m': self.x_spin.value(), 'y_m': self.y_spin.value(),
                'direction': self.direction.currentData(), 'half_width_m': self.width_spin.value()*1000,
                'display_ve': self.ve_spin.value(), 'max_surfaces': self.surface_spin.value(),
                'surface_opacity': self.opacity_spin.value(),
                'sand_mode_2d': self.sand_mode.currentData(),
                'curtain_field': self.curtain_field.currentData(),
                'stage_color_mode': stage_mode, 'stage_color_limits_morph_days': stage_limits,
                'bed_color_mode': bed_mode, 'bed_color_limits_elevation_m': bed_limits,
                'sand_color_settings': dict(self.sand_color_state.config)}

    def sand_colors_changed(self, _config):
        if self._ready and self.update_plot():
            if self.three_d_window is not None and self.three_d_window.isVisible():
                self.show_3d()

    def mark_dirty(self, *_):
        if self._ready:
            self.update_button.setText('更新剖面（参数已修改）')
            self.statusBar().showMessage('参数已修改，点击“更新剖面”或“Cigvis三维展示”应用。')

    def time_changed(self, index):
        self.time_slider.blockSignals(True); self.time_slider.setValue(index); self.time_slider.blockSignals(False)
        self.time_info.setText(f'水动力时间：{self.model.hydro_days[index]:.2f} 天\n'
                               f'形态时间：{self.model.morph_days[index]:.2f} 天（记录从0开始）')
        source = self.model.source_info
        if source.get('kind') == 'continuation_series':
            origin = source['record_sources'][index]
            name = Path(source['sources'][origin['source_index']]['dat_file']).name
            self.time_info.setText(self.time_info.text() + f"\n来源：{name} · 原记录 {origin['source_record']}")
        self.sync_color_controls()
        self.mark_dirty()

    def set_position(self, x_m, y_m):
        self.x_spin.setValue(x_m); self.y_spin.setValue(y_m); self.update_plot()

    def map_click(self, event):
        if event.inaxes is self.map_ax and event.button == 1 and not self.toolbar.mode:
            if event.xdata is not None and event.ydata is not None:
                self.set_position(event.xdata*1000, event.ydata*1000)

    def update_plot(self, *_):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            selected = self.selection()
            if selected['sand_mode_2d'] == 'filled' and self.model._filled_record != selected['record']:
                self.statusBar().showMessage('正在按所选时刻补全含砂率，首次计算约需10–20秒；同一时刻再次选点会复用结果…')
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
            self.sections = self.model.extract_sections(selected['record'], selected['x_m'], selected['y_m'],
                                    selected['direction'], selected['half_width_m'])
            if any(len(section['horizontal_m']) < 2 for section in self.sections):
                raise ValueError('剖面窗口内不足两个原始网格点，请增加半宽。')
            self.sand_values = {sec['axis']: self.model.sand_fraction_for_section(
                sec, selected['sand_mode_2d'])[0] for sec in self.sections}
            _, self.map_ax = draw_sections(self.model, self.sections, selected['record'],
                                           selected['x_m'], selected['y_m'], self.figure,
                                           selected['stage_color_limits_morph_days'],
                                           selected['bed_color_limits_elevation_m'],
                                           self.sand_values, selected['sand_mode_2d'], selected['sand_color_settings'])
            self.figure_canvas.draw()
            self.update_button.setText('更新剖面')
            descriptions = []
            for sec in self.sections:
                fixed_axis = 'Y' if sec['axis'] == 'x' else 'X'
                error = sec['actual_fixed_m'] - sec['requested_fixed_m']
                count = sec['stage_count'][sec['valid']]
                descriptions.append(f"{sec['axis'].upper()}Z：实际{fixed_axis}={sec['actual_fixed_m']:.3f} m"
                                    f"（偏差{error:+.3f} m），≥1mm保存阶段 {int(count.min())}–{int(count.max())}")
                if selected['sand_mode_2d'] == 'filled':
                    _, source = self.model.completed_sand_for_section(sec)
                    thickness = np.diff(np.vstack((sec['base_elevation_m'], sec['interfaces'])), axis=0)
                    visible = sec['valid'][None] & (thickness >= .001)
                    filled = visible & (source >= 2)
                    descriptions[-1] += f"；含砂率补全 {int(filled.sum())}/{int(visible.sum())} 个层段采样"
            self.location_info.setText('；\n'.join(descriptions))
            self.statusBar().showMessage('剖面已更新。平面图点选可直接更新；三维显示与导出均采用当前参数。')
            return True
        except Exception as exc:
            self.show_error(exc); return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _make_canvas(self, selected=None, sections=None):
        selected = self.selection() if selected is None else selected
        sections = self.sections if sections is None else sections
        return create_canvas(self.model, sections, selected['record'], selected['display_ve'],
                             selected['max_surfaces'], selected['surface_opacity'],
                             color_limits=selected['stage_color_limits_morph_days'],
                             curtain_field=selected['curtain_field'], sand_colors=selected['sand_color_settings'])

    def show_seismic(self):
        if self.seismic_window is not None and self.seismic_window.is_running:
            self.seismic_window.raise_(); self.seismic_window.activateWindow(); return
        if not self.update_plot():
            return
        from seismic_window import SeismicWindow
        if self.seismic_window is not None:
            self.seismic_window.close()
            self.seismic_window.deleteLater()
        self.seismic_window = SeismicWindow(self.model, self.sections, self.export_root, self)
        self.seismic_window.busyChanged.connect(lambda busy: self.open_source_button.setEnabled(not busy))
        if self.testAttribute(QtCore.Qt.WA_DontShowOnScreen):
            self.seismic_window.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
        self.seismic_window.show()
        self.seismic_window.compute()

    def show_3d(self):
        if not self.update_plot():
            return
        selected, sections = self.selection(), self.sections
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.statusBar().showMessage('正在创建Cigvis层面与剖面…')
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
            canvas, metadata = self._make_canvas(selected, sections)
            if self.three_d_window is not None:
                self.three_d_window.close()
            self.three_d_window = CigvisWindow(canvas, metadata)
            if self.testAttribute(QtCore.Qt.WA_DontShowOnScreen):
                self.three_d_window.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
            self.three_d_window.show()
            self.statusBar().showMessage('Cigvis三维窗口已打开。修改参数后再次点击可更新。')
        except Exception as exc:
            self.show_error(exc)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def export_current(self, output=None, notify=True):
        if not self.update_plot():
            return None
        selected = self.selection()
        sections = self.sections
        output = Path(output) if output is not None else self.export_root / (
            f"record_{selected['record']:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.statusBar().showMessage('正在导出二维、三维和剖面数据…')
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
            export_sections(output, self.model, sections, selected['record'], selected['x_m'],
                            selected['y_m'], self.figure, selected)
            for sec in sections:
                values, supported = self.model.sand_fraction_for_section(sec, selected['sand_mode_2d'])
                extra = {}
                if selected['sand_mode_2d'] == 'filled':
                    from sand_completion import FILL_SETTINGS
                    raw_values, _ = self.model.sand_fraction_for_section(sec, 'increment')
                    _, source = self.model.completed_sand_for_section(sec)
                    extra = {'raw_fraction_0_1': raw_values, 'fill_source': source,
                             'filled_mask': source >= 2,
                             'fill_settings_json': json.dumps(FILL_SETTINGS, ensure_ascii=False)}
                np.savez_compressed(output / f"{sec['axis']}z_sand_fraction.npz",
                    fraction_0_1=values, supported=supported,
                    horizontal_m=sec['horizontal_m'], morph_days=sec['morph_days'],
                    mode=selected['sand_mode_2d'], **extra)
            canvas, metadata = self._make_canvas(selected, sections)
            canvas.native.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
            try:
                canvas.show(); save_canvas(canvas, metadata, output / 'cigvis_3d.png')
            finally:
                canvas.close()
            self.statusBar().showMessage(f'结果已保存：{output}')
            if notify:
                QtWidgets.QMessageBox.information(self, '导出完成', f'已保存PNG、NPZ、CSV与参数JSON：\n{output}')
            return output
        except Exception as exc:
            if not notify:
                raise
            self.show_error(exc); return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def show_help(self):
        dialog = QtWidgets.QDialog(self); dialog.setWindowTitle('使用方法'); dialog.resize(850, 720)
        layout = QtWidgets.QVBoxLayout(dialog); text = QtWidgets.QTextBrowser()
        text.setMarkdown((HERE / 'README_交互程序.md').read_text(encoding='utf-8'))
        layout.addWidget(text); close = QtWidgets.QPushButton('关闭'); close.clicked.connect(dialog.accept)
        layout.addWidget(close); dialog.exec()

    def show_error(self, exc):
        traceback.print_exc()
        self.statusBar().showMessage(str(exc))
        QtWidgets.QMessageBox.warning(self, '无法绘制', str(exc))

    def closeEvent(self, event):
        if self.seismic_window is not None:
            if self.seismic_window.is_running:
                self.statusBar().showMessage('地震正演仍在运行，请等待完成后关闭。')
                event.ignore(); return
            self.seismic_window.close()
        if self.three_d_window is not None:
            self.three_d_window.close()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=int, default=-1)
    parser.add_argument('--x', type=float)
    parser.add_argument('--y', type=float)
    parser.add_argument('--dat', type=Path, help='Optional NEFIS data file; opens the source settings dialog.')
    parser.add_argument('--def', dest='def_file', type=Path, help='Matching NEFIS definition file.')
    parser.add_argument('--direction', choices=['x', 'y', 'both'], default='both')
    parser.add_argument('--half-width-m', type=float, default=6000)
    parser.add_argument('--volume', type=Path, default=HERE / 'volume')
    parser.add_argument('--export-root', type=Path)
    parser.add_argument('--open-3d', action='store_true', help='Open the CIGVis window alongside the control window.')
    parser.add_argument('--render-demo', action='store_true', help='Render real UI and two snapshot exports, then exit.')
    args = parser.parse_args(argv)
    if args.def_file is not None and args.dat is None:
        parser.error('--def requires --dat.')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('Delft3DSectionViewer')
    app.setOrganizationName('Delft3DTools')
    app.setFont(QtGui.QFont('Microsoft YaHei', 9))
    if args.dat is not None or not (args.volume / 'native_geometry.npz').is_file():
        from source_dialog import open_source
        config = None
        if args.dat is not None:
            config = {'dat_file': str(args.dat), 'def_file': str(args.def_file or args.dat.with_suffix('.def'))}
        model = open_source(config=config)
        if model is None:
            return 0
    else:
        model = StratigraphyModel(args.volume)
    record = model._record(args.record)
    window = SectionWindow(model, record, args.x, args.y, args.direction, args.half_width_m, args.export_root)
    app._section_window = window
    if args.render_demo:
        window.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
        window.show(); app.processEvents()
        demo = HERE / 'interactive_demo'
        window.sand_mode.setCurrentIndex(window.sand_mode.findData('mixed'))
        window.curtain_field.setCurrentIndex(window.curtain_field.findData('mixed'))
        window.export_current(demo / f'record_{record:03d}', notify=False)
        window.sand_mode.setCurrentIndex(window.sand_mode.findData('filled'))
        window.curtain_field.setCurrentIndex(window.curtain_field.findData('filled'))
        # Exercise actual Qt controls with an earlier snapshot and an off-grid position.
        earlier = min(120, model.record_count-1)
        window.time_slider.setValue(earlier)
        window.x_spin.setValue(37980.0); window.y_spin.setValue(35590.0)
        window.stage_mode.setCurrentIndex(window.stage_mode.findData('manual'))
        window.stage_min.setValue(100); window.stage_max.setValue(600)
        window.bed_mode.setCurrentIndex(window.bed_mode.findData('manual'))
        window.bed_min.setValue(-10); window.bed_max.setValue(20)
        assert window.time_combo.currentIndex() == earlier
        assert window.update_plot()
        assert all(len(sec['interfaces']) == earlier+1 for sec in window.sections)
        window.export_current(demo / f'record_{earlier:03d}_custom', notify=False)
        window.show_3d()
        assert window.three_d_window is not None
        assert window.three_d_window.metadata['record'] == earlier
        assert window.three_d_window.metadata['color_limits_morph_days'] == [100., 600.]
        assert window.three_d_window.metadata['curtain_field'] == 'filled'
        app.processEvents(); window.grab().save(str(demo / 'interactive_program.png'))
        report = {'passed': True, 'qt_time_slider_combobox_synchronized': True,
                  'qt_cigvis_window_opened_with_matching_record': True,
                  'manual_stage_color_limits_morph_days': [100., 600.],
                  'manual_bed_color_limits_elevation_m': [-10., 20.],
                  'rendered_curtain_fields': ['mixed', 'filled'],
                  'rendered_2d_sand_modes': ['mixed', 'filled'],
                  'rendered_records': [record, earlier], 'off_grid_requested_xy_m': [37980., 35590.],
                  'actual_fixed_coordinates_m': {sec['axis']: sec['actual_fixed_m'] for sec in window.sections},
                  'exported': ['sections.png', 'cigvis_3d.png', 'selection.json', 'cigvis_3d.json'] +
                              [name for sec in window.sections for name in
                               (f"{sec['axis']}z_section.npz", f"{sec['axis']}z_interfaces.csv")],
                  'screenshot': 'interactive_program.png'}
        demo.mkdir(exist_ok=True)
        (demo / 'gui_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        window.close(); print('Interactive GUI, selected-time exports and CIGVis rendering passed.', flush=True)
        return 0
    window.show()
    if args.open_3d:
        QtCore.QTimer.singleShot(300, window.show_3d)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
