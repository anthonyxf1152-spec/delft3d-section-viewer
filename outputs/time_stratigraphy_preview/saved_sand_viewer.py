"""Standalone CIGVis window for previously saved 3-D sand-fraction NPY files."""
from pathlib import Path
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from saved_sand_data import SavedSandModel, surface_mesh
from saved_sand_scene import SavedSandScene
from colorbar_editor import SandColorEditor, sand_color_state


class SandLoadWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    loaded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self,path,budget,threshold,existing_model=None):
        super().__init__()
        self.path,self.budget,self.threshold,self.existing_model = path,budget,threshold,existing_model

    @QtCore.Slot()
    def run(self):
        model = self.existing_model
        owns = model is None
        try:
            if owns:
                model = SavedSandModel(self.path,self.progress.emit)
            body = surface_mesh(model,self.budget,self.threshold,self.progress.emit)
            self.loaded.emit({'model':model,'body':body,'owns_model':owns,
                              'indices':model.initial_indices(),'threshold':self.threshold})
            owns = False
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if owns and model is not None:
                model.close()
            self.finished.emit()


class SavedSandWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cigvis｜读取三维含砂率 NPY 模型')
        self.resize(1740,960)
        self.model = self.scene = None
        self._thread = self._worker = None
        self.last_error = None
        self.settings = QtCore.QSettings('Delft3DTools','SectionViewer')
        self.state = sand_color_state()
        central = QtWidgets.QWidget(); layout = QtWidgets.QHBoxLayout(central)
        panel = QtWidgets.QWidget(); panel.setFixedWidth(316); left = QtWidgets.QVBoxLayout(panel)
        heading = QtWidgets.QLabel('三维含砂率 · NPY')
        heading.setStyleSheet('font-size:20px;font-weight:600;color:#19354c;'); left.addWidget(heading)
        self.open_button = QtWidgets.QPushButton('选择含砂率 sand_fraction.npy…')
        self.open_button.clicked.connect(self.choose_file); left.addWidget(self.open_button)
        self.folder_button = QtWidgets.QPushButton('读取已保存模型文件夹…')
        self.folder_button.clicked.connect(self.choose_folder); left.addWidget(self.folder_button)
        self.source_label = QtWidgets.QLabel('读取模型根目录；坐标、occupancy 和 valid_xy 会自动配套读取。')
        self.source_label.setWordWrap(True); self.source_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        left.addWidget(self.source_label)
        self.control_box = QtWidgets.QGroupBox('显示设置'); form = QtWidgets.QFormLayout(self.control_box)
        self.ve = QtWidgets.QDoubleSpinBox(); self.ve.setRange(.1,100); self.ve.setValue(5); self.ve.setSingleStep(1)
        self.ve.setKeyboardTracking(False); form.addRow('垂向显示夸张',self.ve)
        self.opacity = QtWidgets.QDoubleSpinBox(); self.opacity.setRange(.02,1); self.opacity.setDecimals(2)
        self.opacity.setValue(.3); self.opacity.setSingleStep(.05); self.opacity.setKeyboardTracking(False)
        form.addRow('体表面不透明度',self.opacity)
        self.threshold = QtWidgets.QDoubleSpinBox(); self.threshold.setRange(0,100); self.threshold.setDecimals(1)
        self.threshold.setSuffix(' %'); self.threshold.setKeyboardTracking(False)
        form.addRow('体表面最低含砂率',self.threshold)
        self.quality = QtWidgets.QComboBox()
        for label,value in [('快速（约50万显示样点）',500_000),('标准（约200万显示样点）',2_000_000),
                            ('精细（约500万显示样点）',5_000_000)]:
            self.quality.addItem(label,value)
        self.quality.setCurrentIndex(1); form.addRow('体表面显示分辨率',self.quality)
        self.body_visible = QtWidgets.QCheckBox('显示三维砂体表面'); self.body_visible.setChecked(True)
        form.addRow(self.body_visible)
        self.apply_button = QtWidgets.QPushButton('更新体表面（阈值 / 分辨率）')
        self.apply_button.clicked.connect(self.rebuild); form.addRow(self.apply_button)
        self.sliders,self.spins,self.plane_checks,self.coordinate_labels = [],[],[],[]
        for axis,title in enumerate(('YZ切片：固定X','XZ切片：固定Y','XY切片：固定深度')):
            check = QtWidgets.QCheckBox(title); check.setChecked(True)
            form.addRow(check); self.plane_checks.append(check)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); slider.setRange(0,0)
            spin = QtWidgets.QSpinBox(); spin.setRange(0,0); spin.setKeyboardTracking(False)
            line = QtWidgets.QHBoxLayout(); line.addWidget(slider,1); line.addWidget(spin)
            form.addRow(line)
            coordinate = QtWidgets.QLabel('尚未读取'); form.addRow(coordinate)
            self.sliders.append(slider); self.spins.append(spin); self.coordinate_labels.append(coordinate)
            slider.valueChanged.connect(lambda value,i=axis:self.set_index(i,value))
            spin.valueChanged.connect(lambda value,i=axis:self.set_index(i,value))
            check.toggled.connect(self.queue_planes)
        left.addWidget(self.control_box)
        left.addWidget(SandColorEditor())
        self.reset_button = QtWidgets.QPushButton('复位三维视角'); self.reset_button.clicked.connect(self.reset_camera)
        self.save_button = QtWidgets.QPushButton('保存当前三维图与显示参数…'); self.save_button.clicked.connect(self.save_image)
        left.addWidget(self.reset_button); left.addWidget(self.save_button)
        note = QtWidgets.QLabel('左键旋转、滚轮缩放、Shift＋左键平移。深度向下为正。'
            '体表面是按显示分辨率重建的外形；原始三维数组不变。切片位置使用保存网格的实际索引。')
        note.setWordWrap(True); left.addWidget(note); left.addStretch()
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); scroll.setFixedWidth(338)
        layout.addWidget(scroll)
        right = QtWidgets.QWidget(); self.right_layout = QtWidgets.QVBoxLayout(right)
        self.placeholder = QtWidgets.QLabel('读取保存的三维 sand_fraction.npy，查看三维砂体和 XY / XZ / YZ 切片。')
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter); self.placeholder.setWordWrap(True)
        self.right_layout.addWidget(self.placeholder,1)
        self.status = QtWidgets.QLabel('请选择已保存的三维模型。'); self.status.setWordWrap(True)
        self.right_layout.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0,0); self.progress.hide()
        self.right_layout.addWidget(self.progress)
        layout.addWidget(right,1); self.setCentralWidget(central)
        self.timer = QtCore.QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(120)
        self.timer.timeout.connect(self.update_planes)
        self.ve.valueChanged.connect(self.change_scale)
        self.opacity.valueChanged.connect(self.change_opacity)
        self.body_visible.toggled.connect(self.change_body_visibility)
        self.state.changed.connect(self.change_colors)
        self.set_busy(False)

    @property
    def is_running(self):
        return self._thread is not None

    def choose_file(self):
        if self.is_running:
            return
        initial = self.settings.value('last_saved_model_folder','')
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self,'读取已保存三维含砂率',str(initial),
            '含砂率模型 (sand_fraction.npy);;NumPy数组 (*.npy)')
        if path:
            self.load_path(path)

    def choose_folder(self):
        if self.is_running:
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self,'选择三维模型根目录',
            str(self.settings.value('last_saved_model_folder','')))
        if path:
            self.load_path(path)

    def load_path(self,path):
        self.start_job(path)

    def rebuild(self):
        if self.model is not None:
            self.start_job(self.model.path,existing_model=self.model)

    def start_job(self,path,existing_model=None):
        if self.is_running:
            return False
        self.timer.stop(); self.last_error = None
        self._thread = QtCore.QThread(self)
        self._worker = SandLoadWorker(path,self.quality.currentData(),self.threshold.value(),existing_model)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status.setText)
        self._worker.loaded.connect(self.loaded)
        self._worker.failed.connect(self.failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self.thread_finished)
        self.set_busy(True); self._thread.start()
        return True

    def set_busy(self,active):
        self.open_button.setEnabled(not active); self.folder_button.setEnabled(not active)
        self.control_box.setEnabled(not active and self.model is not None)
        self.apply_button.setEnabled(not active and self.model is not None)
        self.save_button.setEnabled(not active and self.scene is not None)
        self.reset_button.setEnabled(not active and self.scene is not None)
        self.progress.setVisible(active)
        if active:
            self.status.setText('正在读取 / 更新三维含砂率…')

    @QtCore.Slot(object)
    def loaded(self,payload):
        model = payload['model']; candidate = None
        indices = [spin.value() for spin in self.spins] if not payload['owns_model'] else payload['indices']
        try:
            candidate = SavedSandScene(model,payload['body'],indices,self.state.config,
                self.ve.value(),self.opacity.value(),payload['threshold'])
            if self.testAttribute(QtCore.Qt.WA_DontShowOnScreen):
                candidate.canvas.native.setAttribute(QtCore.Qt.WA_DontShowOnScreen,True)
            old_scene,old_model = self.scene,self.model
            self.scene,self.model = candidate,model
            self.placeholder.hide()
            self.right_layout.insertWidget(0,candidate.canvas.native,1)
            if old_scene is not None:
                self.right_layout.removeWidget(old_scene.canvas.native); old_scene.close()
                old_scene.canvas.native.setParent(None); old_scene.canvas.native.deleteLater()
            if old_model is not None and old_model is not model:
                old_model.close()
            for i,(slider,spin,index) in enumerate(zip(self.sliders,self.spins,indices)):
                for control in (slider,spin):
                    control.blockSignals(True); control.setRange(0,model.shape[i]-1); control.setValue(index); control.blockSignals(False)
            self.update_planes()
            candidate.set_body_visible(self.body_visible.isChecked())
            nx,ny,nz = model.shape
            self.source_label.setText(f'{model.path}\n原始网格：{nx} × {ny} × {nz}\n数组顺序：X,Y,深度；含砂率0～1')
            self.source_label.setToolTip(str(model.path))
            self.settings.setValue('last_saved_model_folder',str(model.folder)); self.settings.sync()
            sampled = payload['body']['sample_shape']
            self.status.setText(f'已读取三维模型；体表面显示网格 {sampled[0]} × {sampled[1]} × {sampled[2]}，'
                f"三角面 {len(payload['body']['faces']):,}。" + ('当前阈值下体表面为空，可降低阈值或查看切片。' if payload['body']['empty'] else '可旋转并调整三个切片位置。'))
        except Exception as exc:
            if candidate is not None and candidate is not self.scene:
                candidate.close()
            if payload['owns_model'] and model is not self.model:
                model.close()
            self.failed(str(exc))

    @QtCore.Slot(str)
    def failed(self,message):
        self.last_error = message; self.status.setText(f'三维含砂率读取 / 显示失败：{message}')

    @QtCore.Slot()
    def thread_finished(self):
        self._thread.deleteLater(); self._thread = self._worker = None
        self.set_busy(False)

    def set_index(self,axis,value):
        for control in (self.sliders[axis],self.spins[axis]):
            control.blockSignals(True); control.setValue(value); control.blockSignals(False)
        self.queue_planes()

    def queue_planes(self,*_):
        if self.model is not None and not self.is_running:
            self.timer.start()

    def update_planes(self):
        self.timer.stop()
        if self.scene is None:
            return
        try:
            indices = [spin.value() for spin in self.spins]
            self.scene.update_planes(indices,[check.isChecked() for check in self.plane_checks])
            for i,index in enumerate(indices):
                axis = ('X','Y','深度')[i]
                self.coordinate_labels[i].setText(f'{axis} = {self.model.axes[i][index]:.3f} m；索引 {index} / {self.model.shape[i]-1}')
        except Exception as exc:
            self.failed(str(exc))

    def change_scale(self,value):
        if self.scene is not None:
            self.scene.set_vertical_scale(value)

    def change_opacity(self,value):
        if self.scene is not None:
            self.scene.recolor(opacity=value)

    def change_colors(self,config):
        if self.scene is not None:
            self.scene.recolor(config=config)

    def change_body_visibility(self,visible):
        if self.scene is not None:
            self.scene.set_body_visible(visible)

    def reset_camera(self):
        if self.scene is not None:
            self.scene.canvas.update_camera(-55,28,35); self.scene.fit()

    def save_image(self,path=None):
        if self.scene is None or self.is_running:
            return None
        if isinstance(path,bool):
            path = None
        if path is None:
            path,_ = QtWidgets.QFileDialog.getSaveFileName(self,'保存三维含砂率图',
                str(self.model.folder/'sand_model_3d.png'),'PNG图像 (*.png)')
            if not path:
                return None
        try:
            output = self.scene.save(Path(path).with_suffix('.png'))
            self.status.setText(f'已保存三维图与显示参数：{output}')
            return output
        except Exception as exc:
            self.failed(str(exc)); return None

    def closeEvent(self,event):
        if self.is_running:
            self.status.setText('三维模型仍在读取 / 更新，请等待完成后关闭。'); event.ignore(); return
        self.timer.stop()
        if self.scene is not None:
            self.right_layout.removeWidget(self.scene.canvas.native)
            self.scene.close(); self.scene.canvas.native.setParent(None); self.scene.canvas.native.deleteLater()
            self.scene = None
        if self.model is not None:
            self.model.close(); self.model = None
        self.placeholder.show(); self.source_label.setText('请选择已保存的三维模型。')
        self.set_busy(False)
        super().closeEvent(event)


def open_saved_sand_viewer(parent=None,model_path=None):
    app = QtWidgets.QApplication.instance()
    window = getattr(app,'_saved_sand_window',None)
    if window is None:
        window = SavedSandWindow(); app._saved_sand_window = window
    if parent is not None and parent.testAttribute(QtCore.Qt.WA_DontShowOnScreen):
        window.setAttribute(QtCore.Qt.WA_DontShowOnScreen,True)
    window.show(); window.raise_(); window.activateWindow()
    if model_path is not None and Path(model_path).exists():
        window.load_path(model_path)
    elif window.model is None and not window.is_running:
        window.choose_file()
    return window


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('Delft3DSectionViewer'); app.setOrganizationName('Delft3DTools')
    app.setFont(QtGui.QFont('Microsoft YaHei',9))
    open_saved_sand_viewer(model_path=sys.argv[1] if len(sys.argv)>1 else None)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
