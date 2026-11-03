"""Shared, persisted sand colour editor for the desktop and seismic windows."""
import json

from PySide6 import QtCore, QtGui, QtWidgets
from colorbar_settings import DEFAULT_SAND_COLORS, validate_sand_colors, sand_color_style


class SandColorState(QtCore.QObject):
    changed = QtCore.Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.settings = QtCore.QSettings('Delft3DTools', 'SectionViewer')
        try:
            self.config = validate_sand_colors(json.loads(self.settings.value('sand_colors', '{}')))
        except (ValueError, TypeError, KeyError):
            self.config = dict(DEFAULT_SAND_COLORS)

    def apply(self, config):
        config = validate_sand_colors(config)
        if config != self.config:
            self.config = config
            self.settings.setValue('sand_colors', json.dumps(config))
            self.settings.sync()
            self.changed.emit(dict(config))


def sand_color_state():
    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError('请先创建查看器窗口。')
    if not hasattr(app, '_sand_color_state'):
        app._sand_color_state = SandColorState(app)
    return app._sand_color_state


class ColorPreview(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = dict(DEFAULT_SAND_COLORS)
        self.setMinimumHeight(42)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        cmap, _, _ = sand_color_style(self.config)
        gradient = QtGui.QLinearGradient(4, 0, self.width() - 4, 0)
        for i in range(65):
            rgba = cmap(i / 64)
            gradient.setColorAt(i / 64, QtGui.QColor.fromRgbF(*rgba))
        painter.fillRect(4, 2, self.width() - 8, 15, QtGui.QBrush(gradient))
        painter.setPen(QtGui.QColor('#3d4955'))
        font = painter.font(); font.setPointSize(8); painter.setFont(font)
        for fraction, key in [(0, 'min_percent'), (.5, 'mid_percent'), (1, 'max_percent')]:
            label = f'{self.config[key]:g}%'
            width = painter.fontMetrics().horizontalAdvance(label)
            x = max(1, min(self.width() - width - 1, int(fraction * (self.width() - 8) + 4 - width / 2)))
            painter.drawText(x, 34, label)


class SandColorEditor(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__('含砂率色标 / %', parent)
        self.state = sand_color_state()
        self.draft = dict(self.state.config)
        self.spins, self.color_buttons = [], []
        form = QtWidgets.QFormLayout(self)
        for index, (title, key, color_key) in enumerate([
                ('最小值', 'min_percent', 'low_color'), ('中间值', 'mid_percent', 'mid_color'),
                ('最大值', 'max_percent', 'high_color')]):
            spin = QtWidgets.QDoubleSpinBox(); spin.setRange(0, 100); spin.setDecimals(2)
            spin.setSuffix(' %'); spin.setKeyboardTracking(False)
            button = QtWidgets.QPushButton('颜色…'); button.setFixedWidth(70)
            button.setToolTip('点击选择该数值对应的颜色；修改颜色后使用自定义三色渐变。')
            button.clicked.connect(lambda _=False, i=index: self.choose_color(i))
            row = QtWidgets.QHBoxLayout(); row.addWidget(spin); row.addWidget(button)
            form.addRow(title, row)
            self.spins.append(spin); self.color_buttons.append(button)
            spin.valueChanged.connect(self.preview_draft)
        self.preview = ColorPreview(); form.addRow(self.preview)
        row = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton('应用色标')
        self.reset_button = QtWidgets.QPushButton('恢复默认')
        row.addWidget(self.apply_button); row.addWidget(self.reset_button); form.addRow(row)
        self.status = QtWidgets.QLabel(); self.status.setWordWrap(True)
        self.status.setStyleSheet('color:#52657a;font-size:11px;'); form.addRow(self.status)
        self.apply_button.clicked.connect(self.apply)
        self.reset_button.clicked.connect(self.reset)
        self.state.changed.connect(self.refresh)
        self.refresh(self.state.config)

    def refresh(self, config):
        self.draft = dict(config)
        for spin, key in zip(self.spins, ('min_percent', 'mid_percent', 'max_percent')):
            spin.blockSignals(True); spin.setValue(config[key]); spin.blockSignals(False)
        self.preview_draft()

    def preview_draft(self, *_):
        for spin, key in zip(self.spins, ('min_percent', 'mid_percent', 'max_percent')):
            self.draft[key] = spin.value()
        for button, key in zip(self.color_buttons, ('low_color', 'mid_color', 'high_color')):
            color = QtGui.QColor(self.draft[key])
            foreground = '#ffffff' if color.lightnessF() < .48 else '#172536'
            button.setStyleSheet(f'background:{color.name()};color:{foreground};')
        try:
            config = validate_sand_colors(self.draft)
            self.preview.config = config; self.preview.update(); self.apply_button.setEnabled(True)
            self.status.setText('中间值对应渐变中点。应用后同步二维、三维和地震输入图。')
        except ValueError as exc:
            self.status.setText(str(exc)); self.apply_button.setEnabled(False)

    def choose_color(self, index):
        key = ('low_color', 'mid_color', 'high_color')[index]
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.draft[key]), self, '选择含砂率色标颜色')
        if color.isValid():
            self.draft[key] = color.name(); self.draft['palette'] = 'custom'; self.preview_draft()

    def apply(self):
        self.state.apply(self.draft)

    def reset(self):
        self.state.apply(DEFAULT_SAND_COLORS)
        self.refresh(DEFAULT_SAND_COLORS)
