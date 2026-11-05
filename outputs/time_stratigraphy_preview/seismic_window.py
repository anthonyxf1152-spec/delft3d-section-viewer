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

from seismic_forward import (SeismicParameters, forward_current, export_seismic,
                             estimate_current_grid, save_current_model)
from stratigraphy_model import StratigraphyModel
from series_loader import close_model
from colorbar_settings import sand_color_style
from colorbar_editor import SandColorEditor, sand_color_state
from seismic_properties import (draw_properties, load_saved_sections,
                                export_property_figures, export_section_arrays)

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
        dimension = '二维' if params.get('model_dimension') == '2d' else '三维'
        titles = ['当前切片含砂率', f"原岩石参数 · {params['frequency_hz']:g} Hz {dimension} PSF",
                  f"内部非均质 · {params['frequency_hz']:g} Hz {dimension} PSF"]
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
    modelSaved = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, volume_dir, sections, params, hint, model_output=None):
        super().__init__()
        self.volume_dir, self.sections, self.params, self.hint = volume_dir, sections, params, hint
        self.model_output = model_output

    @QtCore.Slot()
    def run(self):
        model = None
        try:
            model = StratigraphyModel(self.volume_dir)
            hint = self.hint if self.params.sand_mode == 'filled' else None
            if self.model_output is None:
                result = forward_current(model, self.sections, self.params, self.progress.emit, filled_hint=hint)
                self.loaded.emit(result)
            else:
                output = save_current_model(self.model_output, model, self.sections, self.params,
                                            self.progress.emit, filled_hint=hint)
                self.modelSaved.emit({'output': str(output), 'model_result': load_saved_sections(output)})
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
        self.grid_model = model
        self.sections = sections
        self.export_root = Path(export_root)
        self.result = None
        self.model_result = None
        self.grid_estimate = None
        self.last_model_output = None
        self._job_kind = None
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
        self.dimension = QtWidgets.QComboBox()
        self.dimension.addItem('二维 PSF（逐切片计算）', '2d')
        self.dimension.addItem('三维 PSF（邻域体计算）', '3d')
        self.dimension.setCurrentIndex(max(0, self.dimension.findData(previous.get('model_dimension', '2d'))))
        form.addRow('正演模式', self.dimension)
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
        for name, widget in [('水平采样 / m', self.xy), ('Z 采样 / m', self.dz),
                             ('PSF 参考速度 / m·s⁻¹', self.velocity), ('随机种子', self.seed)]:
            form.addRow(name, widget)
        left.addWidget(self.sampling_box)
        grid_box = QtWidgets.QGroupBox('正演前网格与模型'); grid_layout = QtWidgets.QVBoxLayout(grid_box)
        self.estimate_button = QtWidgets.QPushButton('估计当前参数的网格大小')
        self.estimate_button.clicked.connect(self.estimate_grid); grid_layout.addWidget(self.estimate_button)
        self.grid_info = QtWidgets.QLabel('正在估计网格…'); self.grid_info.setWordWrap(True)
        self.grid_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        grid_layout.addWidget(self.grid_info)
        self.save_model_button = QtWidgets.QPushButton('保存模型 .npy 并绘制六项物性…')
        self.save_model_button.clicked.connect(lambda: self.save_model_current())
        grid_layout.addWidget(self.save_model_button)
        self.saved_sand_button = QtWidgets.QPushButton('Cigvis 查看已保存三维含砂率…')
        self.saved_sand_button.clicked.connect(self.show_saved_sand); grid_layout.addWidget(self.saved_sand_button)
        model_note = QtWidgets.QLabel('保存含砂率、速度、密度、声阻抗、孔隙度变化和反射系数，'
                                      '并绘制六项物性图。三维采用分块保存，超过正演上限仍可导出。')
        model_note.setWordWrap(True); grid_layout.addWidget(model_note)
        left.addWidget(grid_box)
        self.run_button = QtWidgets.QPushButton('计算当前切片地震')
        self.run_button.setMinimumHeight(38)
        self.run_button.setStyleSheet('QPushButton {background:#245a7d;color:white;font-weight:600;}'
                                     'QPushButton:disabled {background:#b6bec7;color:#f4f6f8;}')
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
        caption = QtWidgets.QLabel('计算顺序：含砂率 → Vp／岩石密度 → 反射系数 → 所选维度 PSF 褶积。'
                                   '层厚倍率会改变情景的物理厚度；初始砂床按参考图方案排除。'
                                   '二维 PSF 不含面外响应，其二维孔隙度场与三维切片不同。')
        caption.setWordWrap(True); left.addWidget(caption); left.addStretch()
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls)
        scroll.setFixedWidth(332); scroll.setFrameShape(QtWidgets.QFrame.NoFrame); row.addWidget(scroll)
        right = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(right)
        self.tabs = QtWidgets.QTabWidget(); layout.addWidget(self.tabs, 1)
        seismic_tab = QtWidgets.QWidget(); seismic_layout = QtWidgets.QVBoxLayout(seismic_tab)
        self.figure = Figure(figsize=(15, 8), layout='constrained')
        self.figure.text(.5, .55, '先查看网格估算，再保存模型或计算地震', ha='center', color='#52657a', fontsize=16)
        self.figure.text(.5, .48, '修改左侧参数后，网格大小自动刷新。', ha='center', color='#52657a', fontsize=11)
        self.canvas = FigureCanvasQTAgg(self.figure)
        seismic_layout.addWidget(NavigationToolbar2QT(self.canvas, self)); seismic_layout.addWidget(self.canvas, 1)
        self.tabs.addTab(seismic_tab, '地震 PSF 正演')
        properties_tab = QtWidgets.QWidget(); properties_layout = QtWidgets.QVBoxLayout(properties_tab)
        property_selector = QtWidgets.QHBoxLayout()
        property_selector.addWidget(QtWidgets.QLabel('显示切片'))
        self.properties_direction = QtWidgets.QComboBox()
        for sec in sections:
            self.properties_direction.addItem(f"{sec['axis'].upper()}Z 六项物性", sec['axis'])
        self.properties_direction.currentIndexChanged.connect(self.redraw_properties)
        property_selector.addWidget(self.properties_direction); property_selector.addStretch()
        properties_layout.addLayout(property_selector)
        self.properties_figure = Figure(figsize=(21, 9), layout='constrained')
        self.properties_figure.text(.5, .5, '保存模型或完成正演后，显示六项二维物性', ha='center', color='#52657a', fontsize=15)
        self.properties_canvas = FigureCanvasQTAgg(self.properties_figure)
        properties_layout.addWidget(NavigationToolbar2QT(self.properties_canvas, self))
        properties_layout.addWidget(self.properties_canvas, 1)
        self.tabs.addTab(properties_tab, '二维六项物性')
        self.status = QtWidgets.QLabel('先查看网格估算，可保存模型或点击计算；任务在后台进行。'); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 0); self.progress.hide(); layout.addWidget(self.progress)
        row.addWidget(right, 1); self.setCentralWidget(central)
        self.last_error = None
        self._estimate_timer = QtCore.QTimer(self)
        self._estimate_timer.setSingleShot(True); self._estimate_timer.setInterval(200)
        self._estimate_timer.timeout.connect(self.estimate_grid)
        for control in (self.frequency, self.scale, self.burial, self.depth, self.strength,
                        self.xy, self.dz, self.velocity, self.seed):
            control.valueChanged.connect(self.mark_dirty)
        self.sand_mode.currentIndexChanged.connect(self.mark_dirty)
        self.dimension.currentIndexChanged.connect(self.mark_dirty)
        self.sand_color_state.changed.connect(self.redraw)
        self.estimate_grid()

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
            porosity_strength=self.strength.value(), seed=self.seed.value(), sand_mode=self.sand_mode.currentData(),
            model_dimension=self.dimension.currentData())

    def compute(self):
        self.start_job()

    def show_saved_sand(self):
        from saved_sand_viewer import open_saved_sand_viewer
        model_path = self.last_model_output
        if model_path is not None and not (model_path / 'sand_fraction.npy').is_file():
            model_path = None
        open_saved_sand_viewer(self,model_path)

    @staticmethod
    def byte_size(value):
        return f'{value / 1024**3:.2f} GiB' if value >= 1024**3 else f'{value / 1024**2:.1f} MiB'

    def estimate_grid(self):
        self._estimate_timer.stop()
        try:
            estimate = estimate_current_grid(self.grid_model, self.sections, self.parameters())
            self.grid_estimate = estimate
            if self.parameters().model_dimension == '2d':
                details = ['二维计算网格（水平 × 深度，含保护区）：']
                for cut in estimate['sections']:
                    nh, nz = cut['computational_shape_horizontal_depth']
                    details.append(f"{cut['axis'].upper()}Z：{nh} × {nz}")
                details.append(f"二维总样点数：{estimate['voxel_count']:,}")
            else:
                nx, ny, nz = estimate['grid_shape_xyz']
                details = [f'三维网格 X × Y × Z：{nx} × {ny} × {nz}', f"体素总数：{estimate['voxel_count']:,}"]
            details += [
                       f"单个 float32 数组：{self.byte_size(estimate['single_float32_array_bytes'])}",
                       f"建模峰值内存约：{self.byte_size(estimate['estimated_model_peak_bytes'])}",
                       f"正演峰值内存约：{self.byte_size(estimate['estimated_forward_peak_bytes'])}",
                       f"模型 .npy 文件合计约：{self.byte_size(estimate['estimated_npy_payload_bytes'])}"]
            if 'streaming_save' in estimate:
                plan = estimate['streaming_save']
                details.append(f"分块保存内存约：{self.byte_size(plan['estimated_peak_bytes'])}（{plan['tile_count']} 块）")
            for cut in estimate['sections']:
                nh, nd = cut['shape_horizontal_depth']
                details.append(f"{cut['axis'].upper()}Z 切片：{nh} 道 × {nd} 深度样点")
            details.append('内存为工作量估算，请预留余量；实际层位是否落入深度范围将在生成模型时校验。')
            if not estimate['within_grid_limit']:
                details.append('超过 2200 万正演样点上限，当前不可正演。')
                if self.parameters().model_dimension == '3d':
                    details.append('仍可点击下方按钮，分块保存完整三维 NPY 模型。')
                else:
                    details.append('请增大采样间距或缩小剖面范围。')
            self.grid_info.setText('\n'.join(details))
            self.grid_info.setStyleSheet('color:#25465f;' if estimate['within_grid_limit'] else 'color:#995a12;')
            if not self.is_running:
                self.run_button.setEnabled(estimate['within_grid_limit'])
                self.save_model_button.setEnabled(estimate['within_grid_limit'] or self.parameters().model_dimension == '3d')
            return estimate
        except (ValueError, TypeError) as exc:
            self.grid_estimate = None
            self.grid_info.setStyleSheet('color:#a52a2a;')
            self.grid_info.setText(f'无法估计：{exc}')
            if not self.is_running:
                self.run_button.setEnabled(False); self.save_model_button.setEnabled(False)
            return None

    def save_model_current(self, output=None):
        if self.is_running:
            return None
        estimate = self.estimate_grid()
        if estimate is None or (not estimate['within_grid_limit'] and self.parameters().model_dimension != '3d'):
            self.status.setText(self.grid_info.text()); return None
        if output is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, '选择模型保存位置（将新建独立文件夹）',
                str(self.export_root if self.export_root.exists() else self.export_root.parent))
            if not folder:
                return None
            output = Path(folder) / f"seismic_model_record_{self.sections[0]['record']:03d}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        output = Path(output)
        if self.start_job(output):
            return output
        return None

    def start_job(self, model_output=None):
        if self.is_running:
            return False
        try:
            params = self.parameters(); params.validate()
            estimate = self.estimate_grid()
            allow_large_save = model_output is not None and params.model_dimension == '3d'
            if estimate is None or (not estimate['within_grid_limit'] and not allow_large_save):
                raise ValueError(self.grid_info.text())
        except ValueError as exc:
            self.status.setText(str(exc)); self.last_error = str(exc); return False
        self.last_error = None
        self._job_kind = 'model' if model_output is not None else 'forward'
        self.set_busy(True)
        self._thread = QtCore.QThread(self)
        self._worker = SeismicWorker(self.volume_dir, self.sections, params, self.filled_hint, model_output)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.loaded.connect(self.loaded)
        self._worker.modelSaved.connect(self.model_saved)
        self._worker.failed.connect(self.failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self.thread_finished)
        self._thread.start()
        return True

    def set_busy(self, active):
        for widget in (self.parameter_box, self.sampling_box, self.run_button, self.estimate_button, self.save_model_button):
            widget.setEnabled(not active)
        self.export_button.setEnabled(not active and self.result is not None and not self._parameters_dirty)
        if not active and self.grid_estimate is not None:
            self.run_button.setEnabled(self.grid_estimate['within_grid_limit'])
            self.save_model_button.setEnabled(self.grid_estimate['within_grid_limit'] or self.parameters().model_dimension == '3d')
        self.progress.setVisible(active); self.busyChanged.emit(active)
        if active:
            self.status.setText('正在生成并保存模型…' if self._job_kind == 'model' else '正在准备当前切片正演…')

    @QtCore.Slot(object)
    def model_saved(self, payload):
        self.last_model_output = Path(payload['output'])
        if (self.last_model_output / 'sand_fraction.npy').is_file():
            self.settings.setValue('last_saved_model_folder',str(self.last_model_output))
        self.model_result = payload['model_result']
        self.settings.setValue('seismic_parameters', json.dumps(self.model_result['metadata']['parameters']))
        self.settings.sync()
        try:
            self.status.setText('NPY 已保存，正在绘制并保存六项物性图…')
            self.redraw_properties()
            export_property_figures(self.last_model_output, self.model_result, self.sand_color_state.config)
            self.tabs.setCurrentIndex(1)
            can_forward = self.model_result['metadata']['grid_estimate']['within_grid_limit']
            next_step = '可继续计算地震。' if can_forward else '当前网格超出正演上限，模型已完整保存。'
            self.status.setText(f'模型及六项物性图已保存：{self.last_model_output}\n包含 .npy、坐标、参数、六联图及六项单独 PNG。{next_step}')
        except Exception as exc:
            self.last_error = str(exc)
            self.status.setText(f'NPY 模型已保存，物性绘图失败：{exc}')

    @QtCore.Slot(object)
    def loaded(self, result):
        self.result = result
        self.model_result = result
        self._parameters_dirty = False
        self.run_button.setText('重新计算当前切片地震')
        self.settings.setValue('seismic_parameters', json.dumps(result['metadata']['parameters']))
        self.settings.sync()
        self.redraw()
        self.tabs.setCurrentIndex(0)
        self.status.setText('正演完成。右列为内部非均质情景；两列采用相同振幅色限。可调整显示或导出。')

    @QtCore.Slot(str)
    def failed(self, message):
        task = '模型保存' if self._job_kind == 'model' else '正演'
        self.last_error = message; self.status.setText(f'{task}未完成：{message}')

    @QtCore.Slot()
    def thread_finished(self):
        self._thread.deleteLater(); self._thread = self._worker = None
        self.set_busy(False)

    def redraw(self, *_):
        if self.result is not None:
            draw_seismic(self.result, self.figure, self.clip.value(), self.overlay.isChecked(), self.sand_color_state.config)
            self.canvas.draw()
        self.redraw_properties()

    def redraw_properties(self, *_):
        if self.model_result is not None:
            axis = self.properties_direction.currentData()
            selected = dict(self.model_result, sections=[s for s in self.model_result['sections'] if s['axis'] == axis])
            draw_properties(selected, self.properties_figure, self.sand_color_state.config)
            self.properties_canvas.draw()

    def mark_dirty(self, *_):
        self._parameters_dirty = True
        self.export_button.setEnabled(False)
        self.run_button.setText('重新计算（参数已修改）')
        self.status.setText('正演参数已修改，请重新计算；当前图仍对应上次成功计算的参数。')
        self.grid_info.setText('参数已修改，正在重新估计网格…')
        self._estimate_timer.start()

    def export_current(self, output=None, notify=True):
        if self.result is None or self.is_running or self._parameters_dirty:
            return None
        output = Path(output) if output else self.export_root / (
            f"seismic_record_{self.result['metadata']['selected_record']:03d}_{datetime.now():%Y%m%d_%H%M%S_%f}")
        try:
            export_seismic(output, self.result, self.figure, {'common_clip': self.clip.value(),
                'input_boundaries_overlay': self.overlay.isChecked(), 'seismic_cmap': 'seismic', 'agc': False,
                'sand_color_settings': dict(self.sand_color_state.config)})
            export_section_arrays(output, self.result)
            export_property_figures(output, self.result, self.sand_color_state.config)
            self.status.setText(f'已导出：{output}')
            if notify:
                QtWidgets.QMessageBox.information(self, '地震导出完成', f'已保存地震与六项物性图、NPY、NPZ 和参数 JSON：\n{output}')
            return output
        except Exception as exc:
            self.last_error = str(exc); self.status.setText(f'导出失败：{exc}')
            if not notify:
                raise
            return None

    def closeEvent(self, event):
        if self.is_running:
            self.status.setText('建模／正演任务仍在运行，请等待完成后关闭。'); event.ignore()
        else:
            self._estimate_timer.stop()
            self.grid_model = None
            self.filled_hint = None
            self.result = None
            self.model_result = None
            self.figure.clear()
            self.properties_figure.clear()
            super().closeEvent(event)
