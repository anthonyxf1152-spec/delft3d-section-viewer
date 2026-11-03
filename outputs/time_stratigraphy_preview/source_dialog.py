"""Native Windows file selection and background NEFIS project loading."""
from pathlib import Path
import json

from PySide6 import QtCore, QtWidgets
from project_loader import inspect_source, load_project
from series_loader import inspect_series, load_series


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
            loader = load_series if 'sources' in self.config else load_project
            model = loader(**self.config, cache_root=cache_root(), progress=self.progress.emit)
            self.loaded.emit(model)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
        finally:
            self.finished.emit()


class SourceDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle('打开 / 合并 Delft3D 结果文件')
        self.resize(880, 700)
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
        if config and 'dat_file' in config:
            previous.pop('sources', None)
        self._initial_config = previous
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel('可打开单期结果，或将连续模拟的多期结果合并到统一时间轴。原始 DAT / DEF 保持不变。')
        hint.setWordWrap(True); layout.addWidget(hint)
        self.multiple = QtWidgets.QCheckBox('合并多期次接续结果')
        self.multiple.setChecked(bool(previous.get('sources')))
        layout.addWidget(self.multiple)
        self.series_box = QtWidgets.QGroupBox('期次顺序（上 → 下；后一期从共享锚点替换前一期尾段）')
        series_layout = QtWidgets.QVBoxLayout(self.series_box)
        self.source_list = QtWidgets.QListWidget()
        self.source_list.setMinimumHeight(110)
        self.source_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        series_layout.addWidget(self.source_list)
        for source in previous.get('sources', []):
            self.add_source(source['dat_file'], source.get('def_file'))
        row = QtWidgets.QHBoxLayout()
        self.series_buttons = []
        for title, callback in [('添加 DAT…', self.add_files), ('扫描文件夹…', self.scan_folder),
                                ('上移', lambda: self.move_source(-1)), ('下移', lambda: self.move_source(1)),
                                ('移除', self.remove_source)]:
            button = QtWidgets.QPushButton(title); button.clicked.connect(callback)
            row.addWidget(button); self.series_buttons.append(button)
        series_layout.addLayout(row)
        self.time_mode = QtWidgets.QComboBox()
        self.time_mode.addItem('文件已有累计时间（保留原时间和缺失间隔）', 'absolute')
        self.time_mode.addItem('各期重新计时（校验首尾状态后累加时间）', 'reset')
        self.time_mode.setCurrentIndex(1 if previous.get('time_mode') == 'reset' else 0)
        series_layout.addWidget(self.time_mode)
        layout.addWidget(self.series_box)
        form = QtWidgets.QFormLayout()
        self.dat = QtWidgets.QLineEdit(previous.get('dat_file', ''))
        self.definition = QtWidgets.QLineEdit(previous.get('def_file', ''))
        for title, control, suffix in [('DAT 文件', self.dat, 'dat'), ('DEF 文件', self.definition, 'def')]:
            row = QtWidgets.QHBoxLayout(); row.addWidget(control)
            button = QtWidgets.QPushButton('浏览…')
            self.browse_buttons.append(button)
            button.clicked.connect(lambda _=False, kind=suffix: self.browse(kind))
            row.addWidget(button); form.addRow(title, row)
        self.multiple.toggled.connect(self.toggle_mode)
        self.toggle_mode(self.multiple.isChecked())
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

    def toggle_mode(self, multiple):
        self.series_box.setVisible(multiple)
        for widget in (self.dat, self.definition, *self.browse_buttons):
            widget.setEnabled(not multiple)
        self._inspected_pair = None

    def add_source(self, dat, definition=None):
        dat = str(Path(dat).resolve())
        if any(self.source_list.item(i).data(QtCore.Qt.UserRole)['dat_file'] == dat
               for i in range(self.source_list.count())):
            return
        source = {'dat_file': dat, 'def_file': str(Path(definition).resolve() if definition else Path(dat).with_suffix('.def'))}
        item = QtWidgets.QListWidgetItem(Path(dat).name)
        item.setData(QtCore.Qt.UserRole, source); item.setToolTip(dat)
        self.source_list.addItem(item)
        self._inspected_pair = None

    def add_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, '选择多期 DAT 文件',
            self.dat.text(), 'Delft3D DAT (*.dat *.DAT)')
        for dat in files:
            self.add_source(dat)
        if files:
            self.status.setText('已添加；请按实际接续顺序排列，再检查组分。')

    def scan_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, '选择保存多期结果的文件夹',
            str(Path(self.dat.text()).parent) if self.dat.text() else '')
        if not folder:
            return
        candidates, failures = [], []
        from build_3d_surfaces import NefisReader
        for dat in sorted(Path(folder).iterdir()):
            if dat.suffix.lower() != '.dat' or not dat.with_suffix('.def').is_file():
                continue
            try:
                with NefisReader(dat, dat.with_suffix('.def')) as reader:
                    time = reader.scalar('map-info-series', 'ITMAPC', 0)
                    time *= reader.scalar('map-const', 'DT') * reader.scalar('map-const', 'TUNIT')
                candidates.append((float(time), dat))
            except Exception as exc:
                failures.append(f'{dat.name}: {exc}')
        for _, dat in sorted(candidates):
            self.add_source(dat)
        self.status.setText(f'扫描到 {len(candidates)} 组配对结果，按首个水动力时间添加；请确认期次顺序。'
                            + ('\n未加入：' + '\n'.join(failures) if failures else ''))

    def move_source(self, delta):
        index = self.source_list.currentRow()
        target = index + delta
        if index >= 0 and 0 <= target < self.source_list.count():
            self.source_list.insertItem(target, self.source_list.takeItem(index))
            self.source_list.setCurrentRow(target); self._inspected_pair = None

    def remove_source(self):
        self.source_list.takeItem(self.source_list.currentRow()); self._inspected_pair = None

    def selected_sources(self):
        if self.multiple.isChecked():
            return [self.source_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.source_list.count())]
        return [{'dat_file': self.dat.text().strip(), 'def_file': self.definition.text().strip()}]

    def selection_key(self):
        return tuple((s['dat_file'], s['def_file']) for s in self.selected_sources())

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
            sources = self.selected_sources()
            if not sources or any(not all(s.values()) for s in sources):
                raise ValueError('请分别选择 DAT 和 DEF 文件。')
            infos = inspect_series(sources) if self.multiple.isChecked() else [inspect_source(**sources[0])]
            info = infos[0]
            pair = (sources[0]['dat_file'], sources[0]['def_file'])
            for combo, key in [(self.sand, 'sand_index'), (self.mud, 'mud_index')]:
                combo.clear()
                for i, name in enumerate(info['sediment_names']):
                    combo.addItem(f'{i}: {name}', i)
                same_pair = all(Path(value).resolve() == Path(self._initial_config.get(k, '.')).resolve()
                                for value, k in zip(pair, ('dat_file', 'def_file')))
                index = self._initial_config.get(key, info.get(key)) if same_pair else info.get(key)
                combo.setCurrentIndex(int(index) if index is not None else -1)
            self._inspected_pair = self.selection_key()
            self.status.setText(f"检查通过：{len(infos)} 期，共 {sum(i['records'] for i in infos)} 条源记录，网格 {info['grid_shape_xy']}。请确认砂/泥组分和密度。"
                                + ('\n合并会校验共同底界及接续锚点；重叠尾段采用后一期，缺失间隔保持原样。' if self.multiple.isChecked() else ''))
            return True
        except Exception as exc:
            self.status.setText(f'无法读取：{exc}'); return False

    def start_load(self):
        if self._inspected_pair != self.selection_key() and not self.inspect():
            return
        if self.sand.currentIndex() < 0 or self.mud.currentIndex() < 0 or self.sand.currentData() == self.mud.currentData():
            self.status.setText('请选择不同的砂组分和泥组分。'); return
        self.config = {'sand_index': self.sand.currentData(), 'mud_index': self.mud.currentData(),
            'sand_density': self.sand_density.value(), 'mud_density': self.mud_density.value()}
        if self.multiple.isChecked():
            self.config.update(sources=self.selected_sources(), time_mode=self.time_mode.currentData())
        else:
            self.config.update(self.selected_sources()[0])
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
                       self.sand_density, self.mud_density, self.buttons, self.multiple, self.series_box,
                       *self.browse_buttons):
            widget.setEnabled(not active)
        if not active:
            self.toggle_mode(self.multiple.isChecked())
        self.progress_bar.setVisible(active)
        self.status.setText('正在读取，请等待…' if active else self.status.text())

    @QtCore.Slot(object)
    def loaded(self, model):
        self.model = model
        self.settings.setValue('last_source', json.dumps(self.config, ensure_ascii=False))
        self.settings.sync()

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
