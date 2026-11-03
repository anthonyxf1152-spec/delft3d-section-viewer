"""Background current-cut forward modelling and scientific comparison display."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import matplotlib
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from seismic_forward import SeismicParameters, forward_current, export_seismic
from stratigraphy_model import StratigraphyModel
from series_loader import close_model
from colorbar_settings import sand_color_style
from colorbar_editor import SandColorEditor, sand_color_state

REFERENCE_CLIP = .07247725129127502
matplotlib.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                            'axes.unicode_minus': False, 'font.size': 9})


def draw_seismic(result, figure=None, clip=REFERENCE_CLIP, overlay=False, sand_colors=None):
    figure = figure or Figure(figsize=(16, 8), layout='constrained')
    figure.clear()
    sections, metadata = result['sections'], result['metadata']
    params = metadata['parameters']
    axes = figure.subplots(len(sections), 3, squeeze=False)
    model_axes, seismic_axes = [], []
    sand_cmap, sand_norm, sand_ticks = sand_color_style(sand_colors, percent=True)
    for row, section in enumerate(sections):
        h, z = section['horizontal_m'] / 1000, section['depth_m']
        dh, dz = float(np.median(np.diff(h))), float(z[1] - z[0])
        extent = [h[0] - dh / 2, h[-1] + dh / 2, z[-1] + dz / 2, z[0] - dz / 2]
        q = np.ma.masked_where((section['occupancy'] <= 0) | ~section['valid'][:, None], section['sand_fraction'] * 100)
        titles = ['当前切片含砂率', f"原岩石参数 · {params['frequency_hz']:g} Hz PSF",
                  f"内部非均质 · {params['frequency_hz']:g} Hz PSF"]
        for column, (data, cmap, lower, upper) in enumerate([
                (q, sand_cmap, 0, 100), (section['seismic_baseline'], 'seismic', -clip, clip),
                (section['seismic'], 'seismic', -clip, clip)]):
            ax = axes[row, column]
            if column:
                data = np.ma.masked_where(np.broadcast_to(~section['valid'][:, None], data.shape), data)
            ax.set_facecolor('#eeeeee')
            color_args = {'norm': sand_norm} if column == 0 else {'vmin': lower, 'vmax': upper}
            ax.imshow(data.T, origin='upper', aspect='auto', extent=extent, cmap=cmap,
                      **color_args, interpolation='nearest' if not column else 'bilinear')
            ax.set(xlim=(float(h[0]), float(h[-1])), ylim=(float(z[-1]), float(z[0])),
                   title=titles[column], xlabel=f"原工程 {section['axis'].upper()} / km")
            if column == 0:
                fixed_axis = 'Y' if section['axis'] == 'x' else 'X'
                ax.set_ylabel(f"{fixed_axis} = {section['actual_fixed_m'] / 1000:.3f} km\n情景深度 / m")
                model_axes.append(ax)
            else:
                seismic_axes.append(ax)
                if overlay:
                    for name in ('fan_top_depth_m', 'fan_base_depth_m'):
                        ax.plot(h, np.where(section['valid'], section[name], np.nan), color='#202020', lw=.7)
    figure.colorbar(ScalarMappable(norm=sand_norm, cmap=sand_cmap), ax=model_axes,
                   label='含砂率 / %', ticks=sand_ticks, shrink=.82, pad=.02)
    figure.colorbar(ScalarMappable(norm=Normalize(-clip, clip), cmap='seismic'), ax=seismic_axes,
                   label='相对振幅（两列共用色限，无逐道 AGC）', shrink=.82, pad=.015)
    mode = '分期估算＋缺测补全' if params['sand_mode'] == 'filled' else '当前混合床平均'
    figure.suptitle(f"当前切片地震褶积正演｜记录 {metadata['selected_record']} · 形态 {metadata['selected_morph_days']:.2f} 天\n"
                   f"输入：{mode}｜物理层厚 ×{params['vertical_scale']:g} · 埋置参考深度 {params['burial_depth_m']:g} m｜内部孔隙度研究情景，非实测岩石参数",
                   fontsize=11)
    return figure


class SeismicWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    loaded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, volume_dir, sections, params, hint):
        super().__init__()
        self.volume_dir, self.sections, self.params, self.hint = volume_dir, sections, params, hint

    @QtCore.Slot()
    def run(self):
        model = None
        try:
            model = StratigraphyModel(self.volume_dir)
            result = forward_current(model, self.sections, self.params, self.progress.emit,
                                     filled_hint=self.hint if self.params.sand_mode == 'filled' else None)
            self.loaded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.hint = None
            if model is not None:
                close_model(model)
            self.finished.emit()


class SeismicWindow(QtWidgets.QMainWindow):
    busyChanged = QtCore.Signal(bool)

    def __init__(self, model, sections, export_root, parent=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowTitle('Delft3D｜当前切片地震褶积正演')
        self.resize(1720, 960)
        self.volume_dir = model.volume_dir
        self.sections = sections
        self.export_root = Path(export_root)
        self.result = None
        self._parameters_dirty = True
        self._thread = self._worker = None
        record = sections[0]['record']
        self.filled_hint = ((model._filled_values, model._filled_sources)
                            if model._filled_record == record else None)
        self.settings = QtCore.QSettings('Delft3DTools', 'SectionViewer')
        try:
            previous = json.loads(self.settings.value('seismic_parameters', '{}'))
        except (ValueError, TypeError):
            previous = {}
        defaults = asdict(SeismicParameters())
        defaults.update({k: v for k, v in previous.items() if k in defaults})
        central = QtWidgets.QWidget(); row = QtWidgets.QHBoxLayout(central)
        controls = QtWidgets.QWidget(); controls.setFixedWidth(310)
        left = QtWidgets.QVBoxLayout(controls)
        label = QtWidgets.QLabel('当前切片 · 地震正演')
        label.setStyleSheet('font-size:19px;font-weight:600;color:#19354c;'); left.addWidget(label)
        snapshot = QtWidgets.QLabel(f'记录 {record} · 形态 {model.morph_days[record]:.2f} 天\n' + '\n'.join(
            f"{s['axis'].upper()}Z：固定{'Y' if s['axis'] == 'x' else 'X'}={s['actual_fixed_m']:.3f} m" for s in sections))
        snapshot.setWordWrap(True); left.addWidget(snapshot)
        note = QtWidgets.QLabel('默认沿用参考图第三列的 30 Hz、16 倍物理层厚和内部非均质情景。'
                                '此窗口固定本次切片；主窗口换时刻或位置后，再点“当前切片地震正演”读取新切片。')
        note.setWordWrap(True); left.addWidget(note)
        self.parameter_box = QtWidgets.QGroupBox('正演情景参数'); form = QtWidgets.QFormLayout(self.parameter_box)
        self.frequency = self.spin(defaults['frequency_hz'], 5, 120, 1)
        self.scale = self.spin(defaults['vertical_scale'], .1, 64, 1)
        self.burial = self.spin(defaults['burial_depth_m'], 50, 20000, 0)
        self.depth = self.spin(defaults['depth_max_m'], 100, 30000, 0)
        self.strength = self.spin(defaults['porosity_strength'], 0, 2, 2)
        self.sand_mode = QtWidgets.QComboBox()
        self.sand_mode.addItem('分期估算＋缺测补全', 'filled'); self.sand_mode.addItem('当前混合床平均', 'mixed')
        self.sand_mode.setCurrentIndex(max(0, self.sand_mode.findData(defaults['sand_mode'])))
        for name, widget in [('主频 / Hz', self.frequency), ('物理层厚倍率', self.scale),
                             ('埋置参考深度 / m', self.burial), ('最大深度 / m', self.depth),
                             ('内部非均质强度', self.strength), ('含砂率输入', self.sand_mode)]:
            form.addRow(name, widget)
        left.addWidget(self.parameter_box)
        self.sampling_box = QtWidgets.QGroupBox('采样与复现'); form = QtWidgets.QFormLayout(self.sampling_box)
        self.xy = self.spin(defaults['spacing_xy_m'], 25, 2000, 1)
        self.dz = self.spin(defaults['spacing_z_m'], .5, 50, 2)
        self.velocity = self.spin(defaults['reference_velocity_m_s'], 1000, 6000, 0)
        self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 2**31 - 1); self.seed.setValue(defaults['seed'])
        for name, widget in [('XY 采样 / m', self.xy), ('Z 采样 / m', self.dz),
                             ('PSF 参考速度 / m·s⁻¹', self.velocity), ('随机种子', self.seed)]:
            form.addRow(name, widget)
        left.addWidget(self.sampling_box)
        self.run_button = QtWidgets.QPushButton('计算当前切片地震')
        self.run_button.setMinimumHeight(38)
        self.run_button.setStyleSheet('background:#245a7d;color:white;font-weight:600;')
        self.run_button.clicked.connect(self.compute); left.addWidget(self.run_button)
        display_box = QtWidgets.QGroupBox('显示'); form = QtWidgets.QFormLayout(display_box)
        self.clip = self.spin(REFERENCE_CLIP, .0001, 2, 5)
        self.clip.setSingleStep(.01); form.addRow('共同振幅色限 ±', self.clip)
        self.overlay = QtWidgets.QCheckBox('叠加输入顶底边界'); form.addRow(self.overlay)
        self.clip.valueChanged.connect(self.redraw); self.overlay.toggled.connect(self.redraw)
        left.addWidget(display_box)
        self.sand_color_editor = SandColorEditor()
        self.sand_color_state = sand_color_state()
        left.addWidget(self.sand_color_editor)
        self.export_button = QtWidgets.QPushButton('导出地震剖面与参数')
        self.export_button.setEnabled(False); self.export_button.clicked.connect(lambda: self.export_current())
        left.addWidget(self.export_button)
        caption = QtWidgets.QLabel('计算顺序：含砂率 → Vp／岩石密度 → 反射系数 → 三维 PSF 褶积。'
                                   '层厚倍率会改变情景的物理厚度；初始砂床按参考图方案排除。'
                                   '孔隙度变化是研究假设，PSF 是成像域近似。')
        caption.setWordWrap(True); left.addWidget(caption); left.addStretch()
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls)
        scroll.setFixedWidth(332); scroll.setFrameShape(QtWidgets.QFrame.NoFrame); row.addWidget(scroll)
        right = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(right)
        self.figure = Figure(figsize=(15, 8), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self)); layout.addWidget(self.canvas, 1)
        self.status = QtWidgets.QLabel('点击计算；正演在后台进行。'); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 0); self.progress.hide(); layout.addWidget(self.progress)
        row.addWidget(right, 1); self.setCentralWidget(central)
        self.last_error = None
        for control in (self.frequency, self.scale, self.burial, self.depth, self.strength,
                        self.xy, self.dz, self.velocity, self.seed):
            control.valueChanged.connect(self.mark_dirty)
        self.sand_mode.currentIndexChanged.connect(self.mark_dirty)
        self.sand_color_state.changed.connect(self.redraw)

    @staticmethod
    def spin(value, lower, upper, decimals):
        control = QtWidgets.QDoubleSpinBox(); control.setRange(lower, upper)
        control.setDecimals(decimals); control.setValue(float(value)); control.setKeyboardTracking(False)
        return control

    @property
    def is_running(self):
        return self._thread is not None

    def parameters(self):
        return SeismicParameters(frequency_hz=self.frequency.value(), vertical_scale=self.scale.value(),
            burial_depth_m=self.burial.value(), depth_max_m=self.depth.value(), spacing_xy_m=self.xy.value(),
            spacing_z_m=self.dz.value(), reference_velocity_m_s=self.velocity.value(),
            porosity_strength=self.strength.value(), seed=self.seed.value(), sand_mode=self.sand_mode.currentData())

    def compute(self):
        if self.is_running:
            return
        try:
            params = self.parameters(); params.validate()
        except ValueError as exc:
            self.status.setText(str(exc)); self.last_error = str(exc); return
        self.last_error = None
        self.set_busy(True)
        self._thread = QtCore.QThread(self)
        self._worker = SeismicWorker(self.volume_dir, self.sections, params, self.filled_hint)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.loaded.connect(self.loaded)
        self._worker.failed.connect(self.failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self.thread_finished)
        self._thread.start()

    def set_busy(self, active):
        for widget in (self.parameter_box, self.sampling_box, self.run_button):
            widget.setEnabled(not active)
        self.export_button.setEnabled(not active and self.result is not None and not self._parameters_dirty)
        self.progress.setVisible(active); self.busyChanged.emit(active)
        if active:
            self.status.setText('正在准备当前切片…')

    @QtCore.Slot(object)
    def loaded(self, result):
        self.result = result
        self._parameters_dirty = False
        self.run_button.setText('重新计算当前切片地震')
        self.settings.setValue('seismic_parameters', json.dumps(result['metadata']['parameters']))
        self.settings.sync()
        self.redraw()
        self.status.setText('正演完成。右列为内部非均质情景；两列采用相同振幅色限。可调整显示或导出。')

    @QtCore.Slot(str)
    def failed(self, message):
        self.last_error = message; self.status.setText(f'正演未完成：{message}')

    @QtCore.Slot()
    def thread_finished(self):
        self._thread.deleteLater(); self._thread = self._worker = None
        self.set_busy(False)

    def redraw(self, *_):
        if self.result is not None:
            draw_seismic(self.result, self.figure, self.clip.value(), self.overlay.isChecked(), self.sand_color_state.config)
            self.canvas.draw()

    def mark_dirty(self, *_):
        self._parameters_dirty = True
        self.export_button.setEnabled(False)
        self.run_button.setText('重新计算（参数已修改）')
        self.status.setText('正演参数已修改，请重新计算；当前图仍对应上次成功计算的参数。')

    def export_current(self, output=None, notify=True):
        if self.result is None or self.is_running or self._parameters_dirty:
            return None
        output = Path(output) if output else self.export_root / (
            f"seismic_record_{self.result['metadata']['selected_record']:03d}_{datetime.now():%Y%m%d_%H%M%S_%f}")
        try:
            export_seismic(output, self.result, self.figure, {'common_clip': self.clip.value(),
                'input_boundaries_overlay': self.overlay.isChecked(), 'seismic_cmap': 'seismic', 'agc': False,
                'sand_color_settings': dict(self.sand_color_state.config)})
            self.status.setText(f'已导出：{output}')
            if notify:
                QtWidgets.QMessageBox.information(self, '地震导出完成', f'已保存 PNG、NPZ 和参数 JSON：\n{output}')
            return output
        except Exception as exc:
            self.last_error = str(exc); self.status.setText(f'导出失败：{exc}')
            if not notify:
                raise
            return None

    def closeEvent(self, event):
        if self.is_running:
            self.status.setText('正演仍在运行，请等待本次计算完成后关闭。'); event.ignore()
        else:
            self.filled_hint = None
            self.result = None
            self.figure.clear()
            super().closeEvent(event)
