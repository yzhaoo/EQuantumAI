import os
import sys
import glob
import traceback
import numpy as np

from qtpy import QtCore, QtWidgets

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setMinimumSize(300, 220)

    def clear(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)


class PlotWithToolbar(QtWidgets.QWidget):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)


class ClickableScatterCanvas(MplCanvas):
    siteClicked = QtCore.Signal(int)
    lineDefined = QtCore.Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(width=7, height=7, dpi=100)
        self.parent_viewer = parent
        self._main_scatter = None
        self._coords = None
        self._sites = None
        self._draw_line_mode = False
        self._press_point = None
        self._temp_line_artist = None

        self.mpl_connect("pick_event", self._on_pick)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def set_pick_data(self, scatter, coords, sites):
        self._main_scatter = scatter
        self._coords = coords
        self._sites = sites

    def set_draw_line_mode(self, enabled):
        self._draw_line_mode = bool(enabled)
        self._press_point = None
        if self._temp_line_artist is not None:
            try:
                self._temp_line_artist.remove()
            except Exception:
                pass
            self._temp_line_artist = None
        self.draw_idle()

    def _on_pick(self, event):
        if self._draw_line_mode:
            return
        if self._main_scatter is None or event.artist != self._main_scatter:
            return
        inds = getattr(event, "ind", [])
        if len(inds) == 0:
            return
        idx = int(inds[0])
        site = int(self._sites[idx])
        self.siteClicked.emit(site)

    def _on_press(self, event):
        if not self._draw_line_mode or event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._press_point = np.array([event.xdata, event.ydata], dtype=float)

        if self._temp_line_artist is not None:
            try:
                self._temp_line_artist.remove()
            except Exception:
                pass
            self._temp_line_artist = None

    def _on_motion(self, event):
        if not self._draw_line_mode or self._press_point is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        p0 = self._press_point
        p1 = np.array([event.xdata, event.ydata], dtype=float)

        if self._temp_line_artist is not None:
            try:
                self._temp_line_artist.remove()
            except Exception:
                pass

        self._temp_line_artist = self.ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            "w--",
            lw=2,
        )[0]
        self.draw_idle()

    def _on_release(self, event):
        if not self._draw_line_mode or self._press_point is None or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            self._press_point = None
            return

        p0 = self._press_point.copy()
        p1 = np.array([event.xdata, event.ydata], dtype=float)
        self._press_point = None

        if self._temp_line_artist is not None:
            try:
                self._temp_line_artist.remove()
            except Exception:
                pass
            self._temp_line_artist = None

        if np.allclose(p0, p1):
            return

        self.lineDefined.emit(p0, p1)


class FSCViewer(QtWidgets.QMainWindow):
    def __init__(self, log_folder="fsc_logs"):
        super().__init__()
        self.setWindowTitle("FSC Viewer (QtPy)")
        self.resize(1850, 1050)

        self.log_folder = log_folder
        self.static_data = None
        self.Qsites = None
        self.coords_q = None
        self.site_ids_all = None
        self.coords_all = None
        self.snapshot_names = []
        self.file_map = {}

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self.advance_snapshot)

        self._main_scatter_artist = None
        self._selected_artist = None
        self._max_ui_artist = None
        self._cut_line_artist = None
        self._main_colorbar = None

        self.cut_p0 = None
        self.cut_p1 = None
        self._allow_empty_folder = True

        self._build_ui()
        self.load_log_folder(self.log_folder, allow_empty=True)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_widget.setMaximumWidth(360)

        self.open_btn = QtWidgets.QPushButton("Open log folder")
        self.open_btn.clicked.connect(self.choose_folder)
        controls_layout.addWidget(self.open_btn)

        self.folder_label = QtWidgets.QLabel("Folder: -")
        self.folder_label.setWordWrap(True)
        controls_layout.addWidget(self.folder_label)

        controls_layout.addWidget(QtWidgets.QLabel("Snapshot"))
        self.snapshot_combo = QtWidgets.QComboBox()
        self.snapshot_combo.currentIndexChanged.connect(self.on_snapshot_combo_changed)
        controls_layout.addWidget(self.snapshot_combo)

        self.snapshot_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.snapshot_slider.setMinimum(0)
        self.snapshot_slider.setMaximum(0)
        self.snapshot_slider.setSingleStep(1)
        self.snapshot_slider.setPageStep(1)
        self.snapshot_slider.setTracking(True)
        self.snapshot_slider.valueChanged.connect(self.on_snapshot_slider_changed)
        controls_layout.addWidget(self.snapshot_slider)

        self.snapshot_label = QtWidgets.QLabel("Index: -")
        controls_layout.addWidget(self.snapshot_label)

        player_row = QtWidgets.QHBoxLayout()

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self.toggle_play)
        player_row.addWidget(self.play_btn)

        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.prev_btn.clicked.connect(self.prev_snapshot)
        player_row.addWidget(self.prev_btn)

        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.clicked.connect(self.advance_snapshot)
        player_row.addWidget(self.next_btn)

        controls_layout.addLayout(player_row)

        controls_layout.addWidget(QtWidgets.QLabel("Main heatmap"))
        self.prop_combo = QtWidgets.QComboBox()
        self.prop_combo.addItems(["Ui", "ni", "Ci", "Qprime_mask", "ΔUi", "LDOS@0", "LDOS@Ui"])
        self.prop_combo.currentTextChanged.connect(lambda _=None: self.update_main_heatmap())
        controls_layout.addWidget(self.prop_combo)

        controls_layout.addWidget(QtWidgets.QLabel("Surface property"))
        self.surface_prop_combo = QtWidgets.QComboBox()
        self.surface_prop_combo.addItems(["Ui", "ni", "Ci", "ΔUi"])
        self.surface_prop_combo.currentIndexChanged.connect(self.update_surface_plot)
        controls_layout.addWidget(self.surface_prop_combo)

        controls_layout.addWidget(QtWidgets.QLabel("Site"))
        self.site_spin = QtWidgets.QSpinBox()
        self.site_spin.valueChanged.connect(self.on_site_changed)
        controls_layout.addWidget(self.site_spin)

        self.draw_cut_btn = QtWidgets.QPushButton("Draw cut line")
        self.draw_cut_btn.setCheckable(True)
        self.draw_cut_btn.toggled.connect(self.on_draw_cut_toggled)
        controls_layout.addWidget(self.draw_cut_btn)

        self.reset_cut_btn = QtWidgets.QPushButton("Reset cut line")
        self.reset_cut_btn.clicked.connect(self.reset_cut_line)
        controls_layout.addWidget(self.reset_cut_btn)

        self.cut_status_label = QtWidgets.QLabel("Cut line: not set")
        self.cut_status_label.setWordWrap(True)
        controls_layout.addWidget(self.cut_status_label)

        controls_layout.addWidget(QtWidgets.QLabel("Ui overlay snapshots"))
        self.ui_overlay_list = QtWidgets.QListWidget()
        self.ui_overlay_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.ui_overlay_list.setMinimumHeight(120)
        self.ui_overlay_list.itemSelectionChanged.connect(self.update_ldos_cut_plot)
        controls_layout.addWidget(self.ui_overlay_list)

        overlay_btn_row = QtWidgets.QHBoxLayout()
        self.overlay_current_btn = QtWidgets.QPushButton("Current only")
        self.overlay_current_btn.clicked.connect(self.select_current_overlay_only)
        overlay_btn_row.addWidget(self.overlay_current_btn)

        self.overlay_first_current_btn = QtWidgets.QPushButton("First + current")
        self.overlay_first_current_btn.clicked.connect(self.select_first_and_current_overlay)
        overlay_btn_row.addWidget(self.overlay_first_current_btn)
        controls_layout.addLayout(overlay_btn_row)

        controls_layout.addWidget(QtWidgets.QLabel("Cut width"))
        self.cut_width_spin = QtWidgets.QDoubleSpinBox()
        self.cut_width_spin.setDecimals(6)
        self.cut_width_spin.setRange(0.0, 1e9)
        self.cut_width_spin.setValue(0.05)
        self.cut_width_spin.setSingleStep(0.01)
        self.cut_width_spin.valueChanged.connect(self.update_cut_dependent_plots)
        controls_layout.addWidget(self.cut_width_spin)

        self.bounds_text = QtWidgets.QTextEdit()
        self.bounds_text.setReadOnly(True)
        self.bounds_text.setMaximumHeight(90)
        controls_layout.addWidget(QtWidgets.QLabel("Energy bounds"))
        controls_layout.addWidget(self.bounds_text)

        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(180)
        controls_layout.addWidget(QtWidgets.QLabel("Selected site info"))
        controls_layout.addWidget(self.info_text)

        controls_layout.addStretch(1)

        plots_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        left_plots = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.main_canvas = ClickableScatterCanvas(parent=self)
        self.main_canvas.siteClicked.connect(self.set_site)
        self.main_canvas.lineDefined.connect(self.on_cut_line_defined)
        self.main_plot_widget = PlotWithToolbar(self.main_canvas)

        self.surface_canvas = MplCanvas()
        self.surface_plot_widget = PlotWithToolbar(self.surface_canvas)

        left_plots.addWidget(self.main_plot_widget)
        left_plots.addWidget(self.surface_plot_widget)
        left_plots.setSizes([700, 350])

        right_plots = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.local_canvas = MplCanvas()
        self.local_plot_widget = PlotWithToolbar(self.local_canvas)

        self.ldos_cut_canvas = MplCanvas()
        self.ldos_cut_plot_widget = PlotWithToolbar(self.ldos_cut_canvas)

        right_plots.addWidget(self.local_plot_widget)
        right_plots.addWidget(self.ldos_cut_plot_widget)
        right_plots.setSizes([450, 450])

        plots_splitter.addWidget(left_plots)
        plots_splitter.addWidget(right_plots)
        plots_splitter.setSizes([950, 900])

        main_layout.addWidget(controls_widget)
        main_layout.addWidget(plots_splitter, 1)

        self.statusBar().showMessage("Ready")

    def load_log_folder(self, folder, allow_empty=False):
        try:
            static_path = os.path.join(folder, "run_static.npz")
            if not os.path.exists(static_path):
                if allow_empty:
                    self.log_folder = folder
                    self.folder_label.setText(f"Folder: {folder}")
                    self.statusBar().showMessage(f"Waiting for run_static.npz in {folder}")
                    return
                raise RuntimeError(f"Missing run_static.npz in {folder}")

            static = np.load(static_path, allow_pickle=True)
            static_data = static["static_data"][0]

            self.static_data = static_data
            self.Qsites = np.array(static_data["Qsites"], dtype=int)
            self.coords_q = np.array(static_data["coords_q"], dtype=float)
            self.site_ids_all = np.array(static_data["site_ids_all"], dtype=int)
            self.coords_all = np.array(static_data["coords_all"], dtype=float)

            files = sorted(
                f for f in glob.glob(os.path.join(folder, "*.npz"))
                if "run_static" not in os.path.basename(f)
            )
            self.file_map = {os.path.basename(f): f for f in files}
            self.snapshot_names = list(self.file_map.keys())

            if not self.snapshot_names:
                self.log_folder = folder
                self.folder_label.setText(f"Folder: {folder}")
                if allow_empty:
                    self.statusBar().showMessage(f"Waiting for snapshots in {folder}")
                    return
                raise RuntimeError(f"No snapshot files found in {folder}")

            self.log_folder = folder
            self.folder_label.setText(f"Folder: {folder}")

            self.snapshot_combo.blockSignals(True)
            self.snapshot_combo.clear()
            self.snapshot_combo.addItems(self.snapshot_names)
            self.snapshot_combo.blockSignals(False)

            self.snapshot_slider.blockSignals(True)
            self.snapshot_slider.setMinimum(0)
            self.snapshot_slider.setMaximum(max(len(self.snapshot_names) - 1, 0))
            self.snapshot_slider.setValue(0)
            self.snapshot_slider.blockSignals(False)
            self.update_snapshot_label()

            self.ui_overlay_list.blockSignals(True)
            self.ui_overlay_list.clear()
            for name in self.snapshot_names:
                self.ui_overlay_list.addItem(name)
            self.ui_overlay_list.blockSignals(False)

            self.site_spin.blockSignals(True)
            self.site_spin.setRange(int(np.min(self.Qsites)), int(np.max(self.Qsites)))
            self.site_spin.setValue(int(self.Qsites[0]))
            self.site_spin.blockSignals(False)

            self.reset_cut_line(update=False)
            self.select_current_overlay_only()
            self.update_all()
            self.statusBar().showMessage(f"Loaded {len(self.snapshot_names)} snapshots from {folder}")
        except Exception as exc:
            if allow_empty:
                self.log_folder = folder
                self.folder_label.setText(f"Folder: {folder}")
                self.statusBar().showMessage(f"Waiting for log data in {folder}")
                return
            QtWidgets.QMessageBox.critical(
                self,
                "Load error",
                f"Failed to load folder:\n{folder}\n\n{exc}\n\n{traceback.format_exc()}",
            )

    def refresh_log_folder(self):
        self.load_log_folder(self.log_folder, allow_empty=True)

    def choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose FSC log folder",
            self.log_folder or os.getcwd(),
        )
        if folder:
            self.load_log_folder(folder)

    def current_snapshot_name(self):
        if not self.snapshot_names:
            return None
        return self.snapshot_combo.currentText()

    def update_snapshot_label(self):
        if not self.snapshot_names:
            self.snapshot_label.setText("Index: -")
            return
        idx = self.snapshot_combo.currentIndex()
        total = len(self.snapshot_names)
        name = self.current_snapshot_name()
        self.snapshot_label.setText(f"Index: {idx + 1}/{total}  |  {name}")

    def on_snapshot_combo_changed(self, index):
        if index < 0 or index >= len(self.snapshot_names):
            return
        self.snapshot_slider.blockSignals(True)
        self.snapshot_slider.setValue(index)
        self.snapshot_slider.blockSignals(False)
        self.update_snapshot_label()
        self.select_current_overlay_only()
        self.update_all()

    def on_snapshot_slider_changed(self, value):
        if value < 0 or value >= len(self.snapshot_names):
            return
        if self.snapshot_combo.currentIndex() != value:
            self.snapshot_combo.blockSignals(True)
            self.snapshot_combo.setCurrentIndex(value)
            self.snapshot_combo.blockSignals(False)
        self.update_snapshot_label()
        self.select_current_overlay_only()
        self.update_all()

    def load_snapshot(self, name=None):
        if name is None:
            name = self.current_snapshot_name()
        return np.load(self.file_map[name], allow_pickle=True)

    def selected_ui_overlay_snapshots(self):
        return [item.text() for item in self.ui_overlay_list.selectedItems()]

    def select_current_overlay_only(self):
        current = self.current_snapshot_name()
        self.ui_overlay_list.blockSignals(True)
        for i in range(self.ui_overlay_list.count()):
            item = self.ui_overlay_list.item(i)
            item.setSelected(item.text() == current)
        self.ui_overlay_list.blockSignals(False)
        self.update_ldos_cut_plot()

    def select_first_and_current_overlay(self):
        current = self.current_snapshot_name()
        first = self.snapshot_names[0] if self.snapshot_names else None
        keep = {name for name in (first, current) if name is not None}
        self.ui_overlay_list.blockSignals(True)
        for i in range(self.ui_overlay_list.count()):
            item = self.ui_overlay_list.item(i)
            item.setSelected(item.text() in keep)
        self.ui_overlay_list.blockSignals(False)
        self.update_ldos_cut_plot()

    def get_prev_snapshot_name(self, name):
        idx = self.snapshot_names.index(name)
        if idx == 0:
            return None
        return self.snapshot_names[idx - 1]

    def qprime_mask_from_data(self, data):
        return np.isin(self.Qsites, np.asarray(data["Qprime"], dtype=int))

    def mapper_range_from_qprime(self, vals, qmask):
        vals = np.asarray(vals, dtype=float)

        vals_q = vals[qmask]
        vals_q = vals_q[np.isfinite(vals_q)]
        if vals_q.size == 0:
            vals_q = vals[np.isfinite(vals)]

        if vals_q.size == 0:
            return 0.0, 1.0

        low = float(np.nanmin(vals_q))
        high = float(np.nanmax(vals_q))
        if np.isclose(low, high):
            pad = 1e-12 if low == 0 else 1e-6 * abs(low)
            low -= pad
            high += pad
        return low, high

    def ldos_at_energy(self, data, energy_mode="Ui"):
        if "ildos" not in data:
            return np.full(len(self.Qsites), np.nan, dtype=float)

        ildos = data["ildos"]
        Ui_all = np.asarray(data["Ui"], dtype=float)
        vals = np.full(len(self.Qsites), np.nan, dtype=float)

        for sidx, site in enumerate(self.Qsites):
            try:
                E = np.asarray(ildos[sidx][0], dtype=float)
                rho = np.asarray(ildos[sidx][1], dtype=float)
                if E.size < 2:
                    continue
                target_E = float(Ui_all[site]) if energy_mode == "Ui" else 0.0
                vals[sidx] = np.interp(target_E, E, rho, left=np.nan, right=np.nan)
            except Exception:
                vals[sidx] = np.nan

        return vals

    def scalar_values_quantum(self, data, prop_name, snapshot_name):
        qmask = self.qprime_mask_from_data(data)

        if prop_name == "Ui":
            vals = np.asarray(data["Ui"], dtype=float)[self.Qsites]
        elif prop_name == "ni":
            vals = np.asarray(data["ni"], dtype=float)[self.Qsites]
        elif prop_name == "Ci":
            vals = np.full(len(self.Qsites), np.nan, dtype=float)
            Ci = np.asarray(data["Ci"], dtype=float)
            Qp = np.asarray(data["Qprime"], dtype=int)
            for i, s in enumerate(Qp[:len(Ci)]):
                idx = np.where(self.Qsites == s)[0]
                if len(idx):
                    vals[idx[0]] = Ci[i]
        elif prop_name == "Qprime_mask":
            vals = qmask.astype(float)
        elif prop_name == "ΔUi":
            prev = self.get_prev_snapshot_name(snapshot_name)
            if prev is None:
                vals = np.zeros(len(self.Qsites), dtype=float)
            else:
                prev_data = self.load_snapshot(prev)
                vals = (
                    np.asarray(data["Ui"], dtype=float)[self.Qsites]
                    - np.asarray(prev_data["Ui"], dtype=float)[self.Qsites]
                )
        elif prop_name == "LDOS@0":
            vals = self.ldos_at_energy(data, energy_mode="0")
        elif prop_name == "LDOS@Ui":
            vals = self.ldos_at_energy(data, energy_mode="Ui")
        else:
            vals = np.zeros(len(self.Qsites), dtype=float)

        return vals

    def scalar_values_surface(self, data, prop_name, snapshot_name):
        ids = self.site_ids_all
        if prop_name == "Ui":
            return np.asarray(data["Ui"], dtype=float)[ids]
        elif prop_name == "ni":
            return np.asarray(data["ni"], dtype=float)[ids]
        elif prop_name == "ΔUi":
            prev = self.get_prev_snapshot_name(snapshot_name)
            if prev is None:
                return np.zeros(len(ids), dtype=float)
            prev_data = self.load_snapshot(prev)
            return (
                np.asarray(data["Ui"], dtype=float)[ids]
                - np.asarray(prev_data["Ui"], dtype=float)[ids]
            )
        elif prop_name == "Ci":
            vals = np.full(len(ids), np.nan, dtype=float)
            Ci = np.asarray(data["Ci"], dtype=float)
            Qp = np.asarray(data["Qprime"], dtype=int)
            id_to_all = {sid: i for i, sid in enumerate(ids)}
            for i, sid in enumerate(Qp[:len(Ci)]):
                if sid in id_to_all:
                    vals[id_to_all[sid]] = Ci[i]
            return vals
        return np.full(len(ids), np.nan, dtype=float)

    def _points_near_segment(self, points_xy, p0, p1, width):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        v = p1 - p0
        L2 = float(np.dot(v, v))
        if L2 <= 0:
            return np.zeros(len(points_xy), dtype=bool), np.zeros(len(points_xy), dtype=float)

        w = np.asarray(points_xy, dtype=float) - p0[None, :]
        t = (w @ v) / L2
        t_clip = np.clip(t, 0.0, 1.0)
        proj = p0[None, :] + t_clip[:, None] * v[None, :]
        dist = np.linalg.norm(points_xy - proj, axis=1)
        mask = dist <= (width / 2.0)
        coord_along = t_clip * np.sqrt(L2)
        return mask, coord_along

    def get_surface_cut_indices(self, cut_width):
        if self.cut_p0 is None or self.cut_p1 is None:
            return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

        points_xy = self.coords_all[:, :2]
        z = self.coords_all[:, 2]
        mask, coord_along = self._points_near_segment(points_xy, self.cut_p0, self.cut_p1, cut_width)
        idx = np.where(mask)[0]
        order = np.argsort(coord_along[mask])
        return idx[order], coord_along[mask][order], z[mask][order]

    def get_quantum_line_cut_indices(self, cut_width):
        if self.cut_p0 is None or self.cut_p1 is None:
            return np.array([], dtype=int), np.array([], dtype=float)

        points_xy = self.coords_q[:, :2]
        mask, coord_along = self._points_near_segment(points_xy, self.cut_p0, self.cut_p1, cut_width)
        idx = np.where(mask)[0]
        order = np.argsort(coord_along[mask])
        return idx[order], coord_along[mask][order]

    def update_cut_status_label(self):
        if self.cut_p0 is None or self.cut_p1 is None:
            self.cut_status_label.setText("Cut line: not set")
            return
        self.cut_status_label.setText(
            f"Cut line: ({self.cut_p0[0]:.3g}, {self.cut_p0[1]:.3g}) → ({self.cut_p1[0]:.3g}, {self.cut_p1[1]:.3g})"
        )

    def toggle_play(self, checked):
        if checked:
            self.play_btn.setText("Pause")
            self.play_timer.start(500)
        else:
            self.play_btn.setText("Play")
            self.play_timer.stop()

    def advance_snapshot(self):
        n = self.snapshot_combo.count()
        if n == 0:
            return
        idx = (self.snapshot_combo.currentIndex() + 1) % n
        self.snapshot_slider.setValue(idx)

    def prev_snapshot(self):
        n = self.snapshot_combo.count()
        if n == 0:
            return
        idx = (self.snapshot_combo.currentIndex() - 1) % n
        self.snapshot_slider.setValue(idx)

    def update_all(self):
        if not self.snapshot_names:
            return
        self.update_main_heatmap()
        self.update_bounds_panel()
        self.update_local_plot()
        self.update_surface_plot()
        self.update_ldos_cut_plot()

    def update_cut_dependent_plots(self):
        self.update_surface_plot()
        self.update_ldos_cut_plot()
        self.update_main_heatmap(selected_only=True)

    def on_site_changed(self):
        self.update_main_heatmap(selected_only=True)
        self.update_local_plot()

    def set_site(self, site):
        self.site_spin.blockSignals(True)
        self.site_spin.setValue(int(site))
        self.site_spin.blockSignals(False)
        self.on_site_changed()

    def on_draw_cut_toggled(self, checked):
        self.main_canvas.set_draw_line_mode(checked)
        self.statusBar().showMessage(
            "Draw a cut line on the main heatmap" if checked else "Draw cut line mode off"
        )

    def on_cut_line_defined(self, p0, p1):
        self.cut_p0 = np.asarray(p0, dtype=float)
        self.cut_p1 = np.asarray(p1, dtype=float)
        self.update_cut_status_label()

        self.draw_cut_btn.blockSignals(True)
        self.draw_cut_btn.setChecked(False)
        self.draw_cut_btn.blockSignals(False)
        self.main_canvas.set_draw_line_mode(False)

        self.update_main_heatmap()
        self.update_surface_plot()
        self.update_ldos_cut_plot()

    def reset_cut_line(self, update=True):
        if self.coords_q is None or len(self.coords_q) == 0:
            self.cut_p0 = None
            self.cut_p1 = None
        else:
            y0 = float(np.median(self.coords_q[:, 1]))
            self.cut_p0 = np.array([float(np.min(self.coords_q[:, 0])), y0], dtype=float)
            self.cut_p1 = np.array([float(np.max(self.coords_q[:, 0])), y0], dtype=float)

        self.update_cut_status_label()
        if update:
            self.update_main_heatmap()
            self.update_surface_plot()
            self.update_ldos_cut_plot()

    def update_main_heatmap(self, selected_only=False):
        if self.Qsites is None:
            return

        snapshot_name = self.current_snapshot_name()
        data = self.load_snapshot(snapshot_name)

        if selected_only and self._main_scatter_artist is not None:
            self._update_selected_artist()
            self._update_cut_line_artist()
            self.main_canvas.draw_idle()
            return

        qmask = self.qprime_mask_from_data(data)
        vals = self.scalar_values_quantum(data, self.prop_combo.currentText(), snapshot_name)

        self.main_canvas.clear()
        ax = self.main_canvas.ax
        low, high = self.mapper_range_from_qprime(vals, qmask)

        self._main_scatter_artist = ax.scatter(
            self.coords_q[:, 0],
            self.coords_q[:, 1],
            c=vals,
            cmap="turbo",
            s=50,
            vmin=low,
            vmax=high,
            picker=True,
            linewidths=0.0,
        )

        self._main_colorbar = self.main_canvas.fig.colorbar(
            self._main_scatter_artist,
            ax=ax,
            pad=0.02,
        )

        Ui_vals = np.asarray(data["Ui"], dtype=float)[self.Qsites]
        max_idx = int(np.argmax(Ui_vals))
        self._max_ui_artist = ax.scatter(
            [self.coords_q[max_idx, 0]],
            [self.coords_q[max_idx, 1]],
            marker="D",
            s=110,
            facecolors="none",
            edgecolors="orange",
            linewidths=2,
            label="max Ui",
        )

        self.main_canvas.set_pick_data(self._main_scatter_artist, self.coords_q, self.Qsites)
        self._update_selected_artist()
        self._update_cut_line_artist()

        ax.set_title(f"Quantum-layer heatmap: {self.prop_combo.currentText()}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.legend(loc="best")
        self.main_canvas.draw_idle()

    def _update_selected_artist(self):
        ax = self.main_canvas.ax
        if self._selected_artist is not None:
            try:
                self._selected_artist.remove()
            except Exception:
                pass
            self._selected_artist = None

        site = int(self.site_spin.value())
        idx = np.where(self.Qsites == site)[0]
        if len(idx):
            i = int(idx[0])
            self._selected_artist = ax.scatter(
                [self.coords_q[i, 0]],
                [self.coords_q[i, 1]],
                marker="x",
                s=120,
                c="red",
                linewidths=2,
                label="selected",
            )

    def _update_cut_line_artist(self):
        ax = self.main_canvas.ax
        if self._cut_line_artist is not None:
            try:
                self._cut_line_artist.remove()
            except Exception:
                pass
            self._cut_line_artist = None

        if self.cut_p0 is None or self.cut_p1 is None:
            return

        self._cut_line_artist = ax.plot(
            [self.cut_p0[0], self.cut_p1[0]],
            [self.cut_p0[1], self.cut_p1[1]],
            color="white",
            linestyle="--",
            linewidth=2,
            label="cut line",
        )[0]

    def update_bounds_panel(self):
        data = self.load_snapshot()
        if "energy_bounds" in data and len(data["energy_bounds"]) == 2:
            emin, emax = data["energy_bounds"]
            self.bounds_text.setPlainText(f"Emin = {emin:.4g}\nEmax = {emax:.4g}")
        else:
            self.bounds_text.setPlainText("N/A")

    def update_local_plot(self):
        data = self.load_snapshot()
        site = int(self.site_spin.value())

        self.local_canvas.fig.clear()

        if site not in self.Qsites:
            ax = self.local_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Invalid site", ha="center", va="center")
            ax.axis("off")
            self.local_canvas.draw_idle()
            self.info_text.setPlainText("Invalid site")
            return

        if "ildos" not in data:
            ax = self.local_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No ildos saved", ha="center", va="center")
            ax.axis("off")
            self.local_canvas.draw_idle()
            self.info_text.setPlainText("No ildos saved")
            return

        gs = self.local_canvas.fig.add_gridspec(2, 1, height_ratios=[1, 1])
        ax_local = self.local_canvas.fig.add_subplot(gs[0, 0])
        ax_ldos = self.local_canvas.fig.add_subplot(gs[1, 0])

        sidx = np.where(self.Qsites == site)[0][0]
        ildos = data["ildos"]
        E = np.asarray(ildos[sidx][0], dtype=float)
        rho = np.asarray(ildos[sidx][1], dtype=float)

        Ui = float(data["Ui"][site])
        ni = float(data["ni"][site])
        rho_ui = np.interp(Ui, E, rho, left=np.nan, right=np.nan)
        rho_0 = np.interp(0.0, E, rho, left=np.nan, right=np.nan)

        info_lines = [
            f"Site {site}",
            f"Ui = {Ui:.4g}",
            f"ni = {ni:.4g}",
            f"LDOS(0) = {rho_0:.4g}",
            f"LDOS(Ui) = {rho_ui:.4g}",
        ]

        Qp = np.asarray(data["Qprime"], dtype=int)
        if site in Qp:
            qidx = np.where(Qp == site)[0][0]
            Ci = float(data["Ci"][qidx])
            x = E
            ildos_int = np.zeros_like(rho)
            ildos_int[1:] = np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(x))
            poisson = x * Ci + ni
            diff = np.abs(poisson - ildos_int)
            imin = int(np.argmin(diff))

            info_lines.extend([
                f"Ci = {Ci:.4g}",
                f"best ΔU ≈ {x[imin]:.4g}",
            ])

            ax_local.plot(x, poisson, label="Poisson")
            ax_local.plot(x, ildos_int, label="Integrated LDOS")
            ax_local.axvline(x[imin], ls="--", color="k")
            ax_local.set_xlabel("Energy / ΔU axis")
            ax_local.set_ylabel("Density")
            ax_local.set_title(f"Local consistency (site {site})")
            ax_local.legend(loc="best")
        else:
            ax_local.text(0.5, 0.5, "Site not in Qprime", ha="center", va="center")
            ax_local.axis("off")
            info_lines.append("Not in Qprime")

        ax_ldos.plot(E, rho, label="LDOS")
        ax_ldos.axvline(Ui, ls="--", color="r", label="Ui")
        ax_ldos.axvline(0.0, ls=":", color="k", label="E=0")
        ax_ldos.set_title(f"LDOS (site {site})")
        ax_ldos.set_xlabel("Energy")
        ax_ldos.set_ylabel("LDOS")
        ax_ldos.legend(loc="best")

        self.info_text.setPlainText("\n".join(info_lines))
        self.local_canvas.fig.tight_layout()
        self.local_canvas.draw_idle()

    def update_surface_plot(self):
        data = self.load_snapshot()
        cut_width = float(self.cut_width_spin.value())
        surface_prop = self.surface_prop_combo.currentText()

        self.surface_canvas.clear()
        ax = self.surface_canvas.ax

        idx_cut, coord_along, coord_vert = self.get_surface_cut_indices(cut_width)
        if len(idx_cut) == 0:
            ax.text(0.5, 0.5, "No sites in cut", ha="center", va="center")
            ax.axis("off")
            self.surface_canvas.draw_idle()
            return

        vals_all = self.scalar_values_surface(data, surface_prop, self.current_snapshot_name())
        vals_cut = vals_all[idx_cut]

        sc = ax.scatter(coord_along, coord_vert, c=vals_cut, cmap="turbo", s=35)
        self.surface_canvas.fig.colorbar(sc, ax=ax, pad=0.02, label=surface_prop)
        ax.set_xlabel("distance along cut")
        ax.set_ylabel("z")
        ax.set_title(f"Surface cut heatmap: {surface_prop}")
        ax.set_aspect("equal")
        self.surface_canvas.fig.tight_layout()
        self.surface_canvas.draw_idle()

    def _get_static_ui_quantum(self):
        if self.static_data is None:
            return None

        # Prefer explicit initial/static Ui fields if present.
        for key in ("Ui", "Ui0", "Ui_init", "Ui_initial", "U0"):
            if key in self.static_data:
                arr = np.asarray(self.static_data[key], dtype=float)
                try:
                    return arr[self.Qsites]
                except Exception:
                    return None
        return None

    def update_ldos_cut_plot(self):
        data = self.load_snapshot()
        cut_width = float(self.cut_width_spin.value())

        self.ldos_cut_canvas.fig.clear()
        ax = self.ldos_cut_canvas.fig.add_subplot(111)
        self.ldos_cut_canvas.ax = ax

        if "ildos" not in data:
            ax.text(0.5, 0.5, "No ildos saved", ha="center", va="center")
            ax.axis("off")
            self.ldos_cut_canvas.draw_idle()
            return

        idx_cut, coord_along = self.get_quantum_line_cut_indices(cut_width)
        if len(idx_cut) == 0:
            ax.text(0.5, 0.5, "No quantum sites in cut", ha="center", va="center")
            ax.axis("off")
            self.ldos_cut_canvas.draw_idle()
            return

        ildos = data["ildos"]
        E0 = np.asarray(ildos[idx_cut[0]][0], dtype=float)

        rho_mat = []
        for i in idx_cut:
            E = np.asarray(ildos[i][0], dtype=float)
            rho = np.asarray(ildos[i][1], dtype=float)
            if len(E) != len(E0) or not np.allclose(E, E0):
                rho = np.interp(E0, E, rho, left=np.nan, right=np.nan)
            rho_mat.append(rho)

        rho_mat = np.asarray(rho_mat, dtype=float).T

        extent = [coord_along.min(), coord_along.max(), E0.min(), E0.max()]
        im = ax.imshow(
            rho_mat,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="turbo",
        )
        self.ldos_cut_canvas.fig.colorbar(im, ax=ax, pad=0.02, label="LDOS")

        overlay_names = self.selected_ui_overlay_snapshots()
        current_name = self.current_snapshot_name()

        seen = set()
        overlay_names = [name for name in overlay_names if not (name in seen or seen.add(name))]

        for snap_name in overlay_names:
            try:
                snap_data = self.load_snapshot(snap_name)
            except Exception:
                continue

            ui_vals = []
            for i in idx_cut:
                site_id = self.Qsites[i]
                ui_vals.append(float(snap_data["Ui"][site_id]))

            ui_vals = np.asarray(ui_vals, dtype=float)
            ui_vals[(ui_vals < E0.min()) | (ui_vals > E0.max())] = np.nan

            if snap_name == current_name:
                ax.plot(
                    coord_along,
                    ui_vals,
                    color="white",
                    lw=1.5,
                    ls="--",
                    label=f"Ui ({snap_name})",
                )
            else:
                ax.plot(
                    coord_along,
                    ui_vals,
                    lw=1.0,
                    label=f"Ui ({snap_name})",
                )

        ax.set_xlim(coord_along.min(), coord_along.max())
        ax.set_ylim(E0.min(), E0.max())
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("distance along cut")
        ax.set_ylabel("Energy")
        ax.set_title("LDOS heatmap along line cut")
        self.ldos_cut_canvas.fig.tight_layout()
        self.ldos_cut_canvas.draw_idle()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    log_folder = sys.argv[1] if len(sys.argv) > 1 else "fsc_logs"
    viewer = FSCViewer(log_folder=log_folder)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
