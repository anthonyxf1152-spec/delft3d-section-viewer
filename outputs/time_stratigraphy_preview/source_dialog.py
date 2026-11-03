"""Native Windows file selection and background NEFIS project loading."""
from pathlib import Path
import json

from PySide6 import QtCore, QtWidgets
from project_loader import inspect_source, load_project


def cache_root():
    location = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppLocalDataLocation)
    return Path(location) / 'source_cache'


class LoadWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    loaded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config

    @QtCore.Slot()
    def run(self):
        try:
            model = load_project(**self.config, cache_root=cache_root(), progress=self.progress.emit)
            self.loaded.emit(model)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
        finally:
            self.finished.emit()


class SourceDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle('打开 Delft3D 结果文件')
        self.resize(760, 390)
        self.model = None
        self._thread = None
        self._worker = None
        self._inspected_pair = None
        self.browse_buttons = []
        self.settings = QtCore.QSettings('Delft3DTools', 'SectionViewer')
        try:
            previous = json.loads(self.settings.value('last_source', '{}'))
        except (ValueError, TypeError):
            previous = {}
        previous.update(config or {})
        self._initial_config = previous
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel('选择同一组结果的 DAT 数据文件和 DEF 定义文件。首次读取会建立独立缓存。')
        hint.setWordWrap(True); layout.addWidget(hint)
        form = QtWidgets.QFormLayout()
        self.dat = QtWidgets.QLineEdit(previous.get('dat_file', ''))
        self.definition = QtWidgets.QLineEdit(previous.get('def_file', ''))
        for title, control, suffix in [('DAT 文件', self.dat, 'dat'), ('DEF 文件', self.definition, 'def')]:
            row = QtWidgets.QHBoxLayout(); row.addWidget(control)
            button = QtWidgets.QPushButton('浏览…')
            self.browse_buttons.append(button)
            button.clicked.connect(lambda _=False, kind=suffix: self.browse(kind))
            row.addWidget(button); form.addRow(title, row)
        self.inspect_button = QtWidgets.QPushButton('检查文件并读取泥沙组分')
        self.inspect_button.clicked.connect(self.inspect); form.addRow(self.inspect_button)
        self.sand = QtWidgets.QComboBox(); self.mud = QtWidgets.QComboBox()
        form.addRow('砂组分', self.sand); form.addRow('泥组分', self.mud)
        self.sand_density = self.density_spin(previous.get('sand_density', 1600.))
        self.mud_density = self.density_spin(previous.get('mud_density', 500.))
        form.addRow('砂干床密度 / kg·m⁻³', self.sand_density)
        form.addRow('泥干床密度 / kg·m⁻³', self.mud_density)
        layout.addLayout(form)
        note = QtWidgets.QLabel('密度请按该模型的 .sed 参数填写；1600 / 500 仅为默认值。支持 NEFIS 5、'
                                '平直正交笛卡尔网格、砂/泥两组分及一个保存床层。')
        note.setWordWrap(True); layout.addWidget(note)
        self.status = QtWidgets.QLabel('请选择文件，然后检查组分。'); self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.PlainText); layout.addWidget(self.status)
        self.progress_bar = QtWidgets.QProgressBar(); self.progress_bar.setRange(0, 0)
        self.progress_bar.hide(); layout.addWidget(self.progress_bar)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.button(QtWidgets.QDialogButtonBox.Open).setText('读取并绘图')
        self.buttons.accepted.connect(self.start_load); self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    @staticmethod
    def density_spin(value):
        spin = QtWidgets.QDoubleSpinBox(); spin.setRange(.001, 100000.)
        spin.setDecimals(3); spin.setValue(float(value)); return spin

    def browse(self, kind):
        control = self.dat if kind == 'dat' else self.definition
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(self, f'选择 .{kind} 文件',
            control.text() or self.dat.text(), f'Delft3D 文件 (*.{kind} *.{kind.upper()});;所有文件 (*)')
        if selected:
            control.setText(selected)
            if kind == 'dat':
                paired = Path(selected).with_suffix('.def')
                if paired.is_file():
                    self.definition.setText(str(paired))
            self._inspected_pair = None
            self.status.setText('文件已选择，请检查组分。')

    def inspect(self):
        try:
            pair = (self.dat.text().strip(), self.definition.text().strip())
            if not all(pair):
                raise ValueError('请分别选择 DAT 和 DEF 文件。')
            info = inspect_source(*pair)
            for combo, key in [(self.sand, 'sand_index'), (self.mud, 'mud_index')]:
                combo.clear()
                for i, name in enumerate(info['sediment_names']):
                    combo.addItem(f'{i}: {name}', i)
                same_pair = all(Path(value).resolve() == Path(self._initial_config.get(k, '.')).resolve()
                                for value, k in zip(pair, ('dat_file', 'def_file')))
                index = self._initial_config.get(key, info.get(key)) if same_pair else info.get(key)
                combo.setCurrentIndex(int(index) if index is not None else -1)
            self._inspected_pair = pair
            self.status.setText(f"检查通过：{info['records']} 个时刻，网格 {info['grid_shape_xy']}。请确认砂/泥组分和密度。")
            return True
        except Exception as exc:
            self.status.setText(f'无法读取：{exc}'); return False

    def start_load(self):
        pair = (self.dat.text().strip(), self.definition.text().strip())
        if self._inspected_pair != pair and not self.inspect():
            return
        if self.sand.currentIndex() < 0 or self.mud.currentIndex() < 0 or self.sand.currentData() == self.mud.currentData():
            self.status.setText('请选择不同的砂组分和泥组分。'); return
        self.config = {'dat_file': pair[0], 'def_file': pair[1],
            'sand_index': self.sand.currentData(), 'mud_index': self.mud.currentData(),
            'sand_density': self.sand_density.value(), 'mud_density': self.mud_density.value()}
        self.set_loading(True)
        self._thread = QtCore.QThread(self)
        self._worker = LoadWorker(self.config); self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.loaded.connect(self.loaded)
        self._worker.failed.connect(self.failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self.thread_finished)
        self._thread.start()

    def set_loading(self, active):
        for widget in (self.dat, self.definition, self.inspect_button, self.sand, self.mud,
                       self.sand_density, self.mud_density, self.buttons, *self.browse_buttons):
            widget.setEnabled(not active)
        self.progress_bar.setVisible(active)
        self.status.setText('正在读取，请等待…' if active else self.status.text())

    @QtCore.Slot(object)
    def loaded(self, model):
        self.model = model
        self.settings.setValue('last_source', json.dumps(self.config, ensure_ascii=False))

    @QtCore.Slot(str)
    def failed(self, message):
        self.status.setText(f'读取失败：{message}')

    @QtCore.Slot()
    def thread_finished(self):
        self._thread.deleteLater(); self._thread = None; self._worker = None
        self.set_loading(False)
        if self.model is not None:
            self.accept()

    def reject(self):
        if self._thread is None:
            super().reject()

    def closeEvent(self, event):
        if self._thread is not None:
            event.ignore()
        else:
            super().closeEvent(event)


def open_source(parent=None, config=None):
    dialog = SourceDialog(parent, config)
    return dialog.model if dialog.exec() == QtWidgets.QDialog.Accepted else None
