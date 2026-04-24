import argparse
import contextlib
import html
import json
import os
import sys
import traceback
import glob
from dataclasses import dataclass
from threading import Event

os.environ.setdefault("EQUANTUM_MPL_BACKEND", "Qt5Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from qtpy import QtCore, QtWidgets
from qtpy.QtGui import QFont, QPixmap

from EQuantum.agent import run_nl_query as agent
from EQuantum.agent.manual_boundary_check import generate_manual_boundary_check
from EQuantum.Equantum.EQsystem import System
from EQuantum.Equantum.fsc import FSC
from EQuantum.viewer import FSCViewer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


@dataclass(frozen=True)
class IDETheme:
    window_bg: str = "#f5f6f8"
    title_bg: str = "#ffffff"
    activity_bg: str = "#f1f3f5"
    sidebar_bg: str = "#f7f8fa"
    editor_bg: str = "#ffffff"
    panel_bg: str = "#f3f4f6"
    surface_bg: str = "#eef1f4"
    input_bg: str = "#ffffff"
    border: str = "#d6d9de"
    border_strong: str = "#c7cdd4"
    text: str = "#1f2937"
    text_muted: str = "#6b7280"
    accent: str = "#2563eb"
    accent_hover: str = "#1d4ed8"
    accent_selected: str = "#dbeafe"
    status_bg: str = "#e8f1ff"
    status_text: str = "#1e3a8a"
    danger: str = "#c2410c"
    title_font_size: int = 13
    sidebar_font_size: int = 12
    editor_font_size: int = 14
    output_font_size: int = 12
    status_font_size: int = 11
    spacing_xs: int = 4
    spacing_sm: int = 8
    spacing_md: int = 12
    spacing_lg: int = 16
    radius_sm: int = 6
    radius_md: int = 8
    activity_width: int = 56
    sidebar_width: int = 320
    bottom_panel_height: int = 230


THEME = IDETheme()


class GeometryCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumSize(320, 260)

    def render_system(self, syst, plot_mode="material"):
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")
        if plot_mode == "potential":
            syst.plot_geometry(prop="potential", ax=ax, show=False)
        elif plot_mode == "charge":
            syst.plot_geometry(prop="charge", ax=ax, show=False)
        else:
            syst.plot_geometry(ax=ax, show=False)
        self.fig.tight_layout()
        self.draw_idle()


class ResultCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumSize(320, 240)

    def show_message(self, message):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.axis("off")
        self.fig.tight_layout()
        self.draw_idle()

    def render_dos(self, data_path, ylabel="DOS", title="Density of States"):
        data = np.load(data_path, allow_pickle=True)
        energy = np.asarray(data["energy"], dtype=float)
        values = None
        if "dos" in data:
            values = np.asarray(data["dos"], dtype=float)
        elif "ldos" in data:
            values = np.asarray(data["ldos"], dtype=float)
        else:
            raise KeyError(f"Expected 'dos' or 'ldos' in {data_path}")

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(energy, values, lw=1.5)
        ax.set_xlabel("Energy")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        self.fig.tight_layout()
        self.draw_idle()


class QtLogStream:
    def __init__(self, signal):
        self.signal = signal
        self._buffer = ""

    def write(self, text):
        if not text:
            return 0
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.signal.emit(line)
        return len(text)

    def flush(self):
        if self._buffer:
            self.signal.emit(self._buffer)
            self._buffer = ""


class ManualBoundaryCheckDialog(QtWidgets.QDialog):
    def __init__(self, payload, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Boundary Check")
        self.setModal(True)
        self.resize(920, 760)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(THEME.spacing_lg, THEME.spacing_lg, THEME.spacing_lg, THEME.spacing_lg)
        layout.setSpacing(THEME.spacing_md)

        intro = QtWidgets.QLabel(
            "Please review the boundary-consistency check plot before continuing the simulation."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        details = QtWidgets.QLabel(payload.get("message", ""))
        details.setWordWrap(True)
        layout.addWidget(details)

        image_label = QtWidgets.QLabel()
        image_label.setAlignment(QtCore.Qt.AlignCenter)
        image_label.setMinimumHeight(560)
        pixmap = QPixmap(payload["plot_path"])
        if not pixmap.isNull():
            image_label.setPixmap(
                pixmap.scaled(
                    860,
                    560,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )
        else:
            image_label.setText(f"Could not load plot: {payload['plot_path']}")
        layout.addWidget(image_label, 1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        cancel_button = QtWidgets.QPushButton("Cancel")
        continue_button = QtWidgets.QPushButton("Continue")
        continue_button.setDefault(True)
        cancel_button.clicked.connect(self.reject)
        continue_button.clicked.connect(self.accept)
        button_row.addWidget(cancel_button)
        button_row.addWidget(continue_button)
        layout.addLayout(button_row)


class AgentWorker(QtCore.QObject):
    agent_response = QtCore.Signal(object)
    system_built = QtCore.Signal(object, str)
    boundary_ready = QtCore.Signal(object)
    manual_check_required = QtCore.Signal(object)
    status = QtCore.Signal(str)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, query, parser_name, model_name, strict_parser, profile_name, device_shape, session_state=None, output_dir=None):
        super().__init__()
        self.query = query
        self.parser_name = parser_name
        self.model_name = model_name
        self.strict_parser = strict_parser
        self.profile_name = profile_name
        self.device_shape = device_shape
        self.session_state = session_state
        self.output_dir = output_dir
        self._manual_check_event = None
        self._manual_check_approved = False

    @QtCore.Slot(bool)
    def resolve_manual_check(self, approved):
        self._manual_check_approved = bool(approved)
        if self._manual_check_event is not None:
            self._manual_check_event.set()

    @QtCore.Slot()
    def run(self):
        try:
            stream = QtLogStream(self.log)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                args = argparse.Namespace(
                    parser=self.parser_name,
                    openai_model=self.model_name,
                    strict_openai=self.strict_parser,
                    profile=self.profile_name,
                    device_shape=self.device_shape,
                    output_dir=self.output_dir,
                    no_scf=False,
                    ldos_method="ED",
                    # Keep the GUI on a single worker by default.
                    # On macOS, launching joblib worker processes from the Qt worker thread
                    # can abort the whole app instead of surfacing a normal Python exception.
                    ncore=1,
                    moments=256,
                    n_random=10,
                    eta=0.00015,
                    eps=0.05,
                    kernel="jackson",
                    energy_points=1024,
                    tol_poisson=1e-3,
                    tol_ildos=1e-2,
                )

                self.log.emit(f"Query: {self.query}")
                self.status.emit("Parsing request...")

                if self.session_state:
                    response = agent.continue_agent_turn(
                        self.session_state,
                        self.query,
                        args,
                        execute=False,
                    )
                else:
                    response = agent.start_agent_turn(
                        self.query,
                        args,
                        execute=False,
                    )

                self.agent_response.emit(response)
                self.log.emit("Agent response:")
                self.log.emit(json.dumps(response, indent=2))

                if response["status"] != "running":
                    self.finished.emit(response)
                    return

                spec = response["spec"]
                spec = agent.apply_defaults_to_spec(
                    spec,
                    agent.get_default_spec_values(args, spec["profile"]),
                    agent.OPTIONAL_CONFIRM_FIELDS,
                )
                profile = agent.get_runtime_profile(spec)
                agent.ensure_compatible(spec, profile)
                setup_paths = agent.ensure_profile_setup(profile, spec, require_config=True)
                setup_dir = setup_paths["setup_dir"]
                config_file = setup_paths["config_file"]
                artifact_dir = agent.make_artifact_dir(profile, self.output_dir)
                self.log.emit(f"Using setup directory: {setup_dir}")
                self.log.emit(f"Artifacts will be saved to: {artifact_dir}")

                with open(os.path.join(artifact_dir, "query_spec.json"), "w") as handle:
                    json.dump(spec, handle, indent=2)

                self.status.emit("Building geometry and system...")
                syst = System(
                    profile["geoparams"],
                    config_file=config_file,
                    ifqsystem=True,
                    quantum_builder="default",
                )
                self.log.emit("System built successfully.")
                self.log.emit(f"Number of sites: {syst.num_sites}")
                self.log.emit(f"Number of quantum sites: {len(syst.Qsites)}")
                self.system_built.emit(syst, artifact_dir)

                self.status.emit("Initializing FSC...")
                phi = agent.magnetic_field_to_phi(spec["magnetic_field_T"], syst.unit_cell_area)
                self.log.emit(f"Magnetic field B = {spec['magnetic_field_T']} T")
                self.log.emit(f"Converted flux phi = {phi}")
                qparams = {"Ufunc": lambda site: 0, "phi": phi}
                fsc = FSC(syst, ifinitial=False, qparams=qparams, approx="TF")
                fsc.update_BC(agent.build_boundary_conditions(profile, spec), ifinitial=True)
                self.log.emit("Boundary conditions updated.")
                for site_id, fsite in fsc.sites.items():
                    if site_id in syst.sites:
                        syst.sites[site_id].potential = fsite.potential
                        syst.sites[site_id].charge = fsite.charge
                self.boundary_ready.emit(syst)
                fsc.Ncore = int(spec["Ncore"])
                fsc.convergence_tol = list(spec["convergence_tol"])
                fsc.save_static_reference(artifact_dir)
                self.log.emit("Saved static FSC reference.")

                self.status.emit("Waiting for manual boundary check...")
                manual_check = generate_manual_boundary_check(fsc, artifact_dir)
                self.log.emit(manual_check["message"])
                self.log.emit(f"Manual boundary plot saved to: {manual_check['plot_path']}")
                self._manual_check_event = Event()
                self._manual_check_approved = False
                self.manual_check_required.emit(manual_check)
                self._manual_check_event.wait()
                self._manual_check_event = None
                if not self._manual_check_approved:
                    cancel_message = (
                        "Manual boundary check canceled. Tell me what to change, "
                        "or reply 'start' to reset and run again with the same parameters."
                    )
                    resumed_session = dict(response["session_state"])
                    resumed_session["active"] = True
                    resumed_session["run_confirmed"] = False
                    resumed_session["missing_fields"] = []
                    resumed_session.setdefault("history", []).append({"role": "assistant", "text": cancel_message})
                    resumed_response = {
                        "status": "needs_clarification",
                        "message": cancel_message,
                        "spec": spec,
                        "missing_fields": [],
                        "session_state": resumed_session,
                    }
                    self.agent_response.emit(resumed_response)
                    self.finished.emit(resumed_response)
                    return
                self.log.emit("Manual boundary check approved.")

                self.status.emit("Running FSC solver with step snapshots...")
                self.log.emit("Starting FSC solve with snapshot_mode='step'.")
                fsc.solve(
                    syst,
                    save=True,
                    snapshot_mode="step",
                    snapshot_every=1,
                    snapshot_folder=artifact_dir,
                    ldos_method=spec["ldos_method"],
                    save_ildos=True,
                    eta=spec["eta"],
                    M=256,
                    eps=0.05,
                    kernel="jackson",
                )
                self.log.emit("FSC solve finished.")

                self.status.emit("Exporting final DOS/LDOS artifacts...")
                energy_grid = np.linspace(-6 * syst.t, 6 * syst.t, 1024)
                result = agent.summarize_run(fsc, spec, phi, artifact_dir)
                result["setup_dir"] = setup_dir
                result["config_file"] = config_file

                if spec["task"] == "dos":
                    energy, rho = fsc.qsystem.get_dos(
                        w=energy_grid,
                        M=256,
                        n_random=10,
                        eps=0.05,
                        kernel="jackson",
                    )
                    agent.save_dos_artifacts(artifact_dir, energy, rho)
                    result["artifacts"] = {
                        "plot": os.path.join(artifact_dir, "dos.png"),
                        "data": os.path.join(artifact_dir, "dos_data.npz"),
                    }
                else:
                    site_id = agent.save_ldos_artifacts(artifact_dir, fsc, site_mode="center")
                    result["ldos_site_id"] = site_id
                    result["artifacts"] = {
                        "plot": os.path.join(artifact_dir, "ldos.png"),
                        "data": os.path.join(artifact_dir, "ldos_data.npz"),
                    }

                with open(os.path.join(artifact_dir, "run_summary.json"), "w") as handle:
                    json.dump(result, handle, indent=2)

                completed_session = dict(response["session_state"])
                completed_session["active"] = False
                completed_session["missing_fields"] = []
                completed_session.setdefault("history", []).append({"role": "assistant", "text": "Simulation completed."})

                completed_response = {
                    "status": "completed",
                    "message": "Simulation completed.",
                    "spec": spec,
                    "result": result,
                    "missing_fields": [],
                    "session_state": completed_session,
                }

                self.log.emit("Run summary:")
                self.log.emit(json.dumps(completed_response, indent=2))
                self.finished.emit(completed_response)
                stream.flush()
        except Exception:
            error_text = traceback.format_exc()
            self.log.emit(error_text)
            self.failed.emit(error_text)


class AgentInterfaceWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EQuantum AI Interface")
        self.setFixedSize(1800, 1100)

        self.worker_thread = None
        self.worker = None
        self.current_artifact_dir = None
        self.current_system = None
        self.active_session_state = None

        self._build_ui()
        self._apply_codex_style()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.title_bar = QtWidgets.QFrame()
        self.title_bar.setObjectName("titleBar")
        title_layout = QtWidgets.QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(THEME.spacing_lg, THEME.spacing_sm, THEME.spacing_lg, THEME.spacing_sm)
        title_layout.setSpacing(THEME.spacing_md)
        title_layout.addStretch(1)

        self.reset_button = QtWidgets.QPushButton("New Request")
        self.reset_button.clicked.connect(self.reset_conversation)
        title_layout.addWidget(self.reset_button)
        root_layout.addWidget(self.title_bar)

        body_widget = QtWidgets.QWidget()
        body_layout = QtWidgets.QHBoxLayout(body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        root_layout.addWidget(body_widget, 1)

        self.activity_bar = QtWidgets.QFrame()
        self.activity_bar.setObjectName("activityBar")
        self.activity_bar.setFixedWidth(THEME.activity_width)
        activity_layout = QtWidgets.QVBoxLayout(self.activity_bar)
        activity_layout.setContentsMargins(0, THEME.spacing_md, 0, THEME.spacing_md)
        activity_layout.setSpacing(THEME.spacing_sm)

        self.nav_chat_button = self._make_activity_button("C", "Conversation")
        self.nav_session_button = self._make_activity_button("S", "Session")
        self.nav_settings_button = self._make_activity_button("P", "Settings")
        self.nav_chat_button.setChecked(True)
        activity_layout.addWidget(self.nav_chat_button, 0, QtCore.Qt.AlignHCenter)
        activity_layout.addWidget(self.nav_session_button, 0, QtCore.Qt.AlignHCenter)
        activity_layout.addWidget(self.nav_settings_button, 0, QtCore.Qt.AlignHCenter)
        activity_layout.addStretch(1)
        body_layout.addWidget(self.activity_bar)

        self.body_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.body_splitter.setChildrenCollapsible(False)
        body_layout.addWidget(self.body_splitter, 1)

        self.right_column_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.right_column_splitter.setChildrenCollapsible(False)

        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setObjectName("sidebar")
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(THEME.spacing_md, THEME.spacing_md, THEME.spacing_md, THEME.spacing_md)
        sidebar_layout.setSpacing(THEME.spacing_md)

        self.sidebar_title = QtWidgets.QLabel("Conversation")
        self.sidebar_title.setObjectName("sidebarTitle")
        sidebar_layout.addWidget(self.sidebar_title)

        self.sidebar_stack = QtWidgets.QStackedWidget()
        sidebar_layout.addWidget(self.sidebar_stack, 1)

        conversation_page = QtWidgets.QWidget()
        conversation_layout = QtWidgets.QVBoxLayout(conversation_page)
        conversation_layout.setContentsMargins(0, 0, 0, 0)
        conversation_layout.setSpacing(THEME.spacing_sm)
        self.chat_output = QtWidgets.QTextBrowser()
        self.chat_output.setReadOnly(True)
        self.chat_output.setOpenLinks(False)
        self.chat_output.setOpenExternalLinks(False)
        self.chat_output.setObjectName("chatTranscript")
        self.chat_output.setPlaceholderText("Conversation with the agent will appear here.")
        conversation_layout.addWidget(self.chat_output, 1)

        self.composer_card = QtWidgets.QFrame()
        self.composer_card.setObjectName("composerCard")
        composer_layout = QtWidgets.QVBoxLayout(self.composer_card)
        composer_layout.setContentsMargins(THEME.spacing_md, THEME.spacing_md, THEME.spacing_md, THEME.spacing_md)
        composer_layout.setSpacing(THEME.spacing_sm)

        composer_layout.addWidget(QtWidgets.QLabel("Simulation Request"))
        request_row = QtWidgets.QHBoxLayout()
        request_row.setSpacing(THEME.spacing_sm)
        self.query_input = QtWidgets.QPlainTextEdit()
        self.query_input.setPlaceholderText(
            "calculate the density of states for a square lattice system with backgate voltage 0.5 and magnetic field 1T"
        )
        self.query_input.setPlainText(
            "calculate the density of states for a square lattice system with backgate voltage 0.5 and magnetic field 1T"
        )
        self.query_input.setMaximumHeight(110)
        request_row.addWidget(self.query_input, 1)

        self.run_button = QtWidgets.QPushButton(">")
        self.run_button.clicked.connect(self.submit_turn)
        self.run_button.setFixedSize(44, 44)
        self.run_button.setToolTip("Send request")
        self.run_button.setProperty("role", "send")
        request_row.addWidget(self.run_button, 0, QtCore.Qt.AlignBottom)
        composer_layout.addLayout(request_row)
        conversation_layout.addWidget(self.composer_card, 0)
        self.sidebar_stack.addWidget(conversation_page)

        session_page = QtWidgets.QWidget()
        session_layout = QtWidgets.QVBoxLayout(session_page)
        session_layout.setContentsMargins(0, 0, 0, 0)
        session_layout.setSpacing(THEME.spacing_sm)
        session_layout.addWidget(QtWidgets.QLabel("Current Spec"))
        self.json_output = QtWidgets.QPlainTextEdit()
        self.json_output.setReadOnly(True)
        session_layout.addWidget(self.json_output, 1)
        session_layout.addWidget(QtWidgets.QLabel("Run Summary"))
        self.summary_output = QtWidgets.QPlainTextEdit()
        self.summary_output.setReadOnly(True)
        session_layout.addWidget(self.summary_output, 1)
        self.sidebar_stack.addWidget(session_page)

        settings_page = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_page)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(THEME.spacing_sm)

        settings_layout.addWidget(QtWidgets.QLabel("Profile"))
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.addItems(sorted(agent.PROFILES.keys()))
        settings_layout.addWidget(self.profile_combo)

        settings_layout.addWidget(QtWidgets.QLabel("Device Shape"))
        self.device_shape_combo = QtWidgets.QComboBox()
        default_profile = agent.PROFILES[self.profile_combo.currentText()]
        self.device_shape_combo.addItems(default_profile.get("supported_device_shapes", [default_profile.get("device_shape", "dotgate")]))
        settings_layout.addWidget(self.device_shape_combo)
        self.profile_combo.currentTextChanged.connect(self.refresh_device_shape_options)

        settings_layout.addWidget(QtWidgets.QLabel("Parser"))
        self.parser_combo = QtWidgets.QComboBox()
        self.parser_combo.addItems(["langchain", "openai", "regex"])
        settings_layout.addWidget(self.parser_combo)

        settings_layout.addWidget(QtWidgets.QLabel("Model"))
        self.model_input = QtWidgets.QLineEdit("gpt-4o-mini")
        settings_layout.addWidget(self.model_input)

        self.strict_checkbox = QtWidgets.QCheckBox("Strict parser")
        settings_layout.addWidget(self.strict_checkbox)

        self.open_folder_button = QtWidgets.QPushButton("Open Current Run In Solver")
        self.open_folder_button.clicked.connect(self.open_current_folder_in_viewer)
        self.open_folder_button.setEnabled(False)
        settings_layout.addWidget(self.open_folder_button)

        settings_layout.addWidget(QtWidgets.QLabel("Run Folder"))
        self.folder_label = QtWidgets.QLabel("Log folder: -")
        self.folder_label.setWordWrap(True)
        self.folder_label.setObjectName("folderLabel")
        settings_layout.addWidget(self.folder_label)
        settings_layout.addStretch(1)
        self.sidebar_stack.addWidget(settings_page)

        self.editor_area = QtWidgets.QFrame()
        self.editor_area.setObjectName("editorArea")
        editor_layout = QtWidgets.QVBoxLayout(self.editor_area)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)

        self.tabs = QtWidgets.QTabWidget()

        geometry_tab = QtWidgets.QWidget()
        geometry_layout = QtWidgets.QVBoxLayout(geometry_tab)
        geometry_controls = QtWidgets.QHBoxLayout()
        geometry_controls.addWidget(QtWidgets.QLabel("Geometry view"))
        self.geometry_mode_combo = QtWidgets.QComboBox()
        self.geometry_mode_combo.addItems(["material", "potential", "charge"])
        self.geometry_mode_combo.currentTextChanged.connect(self.update_geometry_view)
        self.geometry_mode_combo.setEnabled(False)
        geometry_controls.addWidget(self.geometry_mode_combo)
        self.refresh_geometry_button = QtWidgets.QPushButton("Refresh Geometry")
        self.refresh_geometry_button.clicked.connect(self.refresh_geometry_from_latest_snapshot)
        self.refresh_geometry_button.setEnabled(False)
        geometry_controls.addWidget(self.refresh_geometry_button)
        geometry_controls.addStretch(1)
        geometry_layout.addLayout(geometry_controls)
        self.geometry_canvas = GeometryCanvas()
        geometry_layout.addWidget(self.geometry_canvas)
        self.tabs.addTab(geometry_tab, "Geometry")

        solver_tab = QtWidgets.QWidget()
        solver_layout = QtWidgets.QVBoxLayout(solver_tab)
        solver_controls = QtWidgets.QHBoxLayout()
        self.refresh_solver_button = QtWidgets.QPushButton("Refresh Solver")
        self.refresh_solver_button.clicked.connect(self.refresh_viewer)
        self.refresh_solver_button.setEnabled(False)
        solver_controls.addWidget(self.refresh_solver_button)
        solver_controls.addStretch(1)
        solver_layout.addLayout(solver_controls)
        self.viewer = FSCViewer("fsc_logs")
        self.viewer.setWindowFlags(QtCore.Qt.Widget)
        solver_layout.addWidget(self.viewer)
        self.tabs.addTab(solver_tab, "Solver")

        result_tab = QtWidgets.QWidget()
        result_layout = QtWidgets.QVBoxLayout(result_tab)
        self.result_canvas = ResultCanvas()
        self.result_canvas.show_message("Final DOS / LDOS plot will appear here after the run finishes.")
        result_layout.addWidget(self.result_canvas)
        self.tabs.addTab(result_tab, "Result")
        editor_layout.addWidget(self.tabs, 1)

        self.bottom_panel = QtWidgets.QFrame()
        self.bottom_panel.setObjectName("bottomPanel")
        bottom_layout = QtWidgets.QVBoxLayout(self.bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self.bottom_tabs = QtWidgets.QTabWidget()
        self.bottom_tabs.setDocumentMode(True)
        bottom_layout.addWidget(self.bottom_tabs)

        log_tab = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_tab)
        log_layout.setContentsMargins(THEME.spacing_md, THEME.spacing_md, THEME.spacing_md, THEME.spacing_md)
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(5000)
        log_layout.addWidget(self.log_output, 1)
        self.bottom_tabs.addTab(log_tab, "Run Output")

        json_tab = QtWidgets.QWidget()
        json_layout = QtWidgets.QVBoxLayout(json_tab)
        json_layout.setContentsMargins(THEME.spacing_md, THEME.spacing_md, THEME.spacing_md, THEME.spacing_md)
        self.bottom_json_output = QtWidgets.QPlainTextEdit()
        self.bottom_json_output.setReadOnly(True)
        json_layout.addWidget(self.bottom_json_output)
        self.bottom_tabs.addTab(json_tab, "Spec")

        summary_tab = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(THEME.spacing_md, THEME.spacing_md, THEME.spacing_md, THEME.spacing_md)
        self.bottom_summary_output = QtWidgets.QPlainTextEdit()
        self.bottom_summary_output.setReadOnly(True)
        summary_layout.addWidget(self.bottom_summary_output)
        self.bottom_tabs.addTab(summary_tab, "Summary")

        self.body_splitter.addWidget(self.editor_area)
        self.right_column_splitter.addWidget(self.sidebar)
        self.right_column_splitter.addWidget(self.bottom_panel)
        self.body_splitter.addWidget(self.right_column_splitter)

        self.body_splitter.setSizes([1800 - THEME.sidebar_width - THEME.activity_width, THEME.sidebar_width])
        self.right_column_splitter.setSizes([720, THEME.bottom_panel_height])

        self.nav_chat_button.clicked.connect(lambda: self.switch_sidebar_view(0, "Conversation"))
        self.nav_session_button.clicked.connect(lambda: self.switch_sidebar_view(1, "Session"))
        self.nav_settings_button.clicked.connect(lambda: self.switch_sidebar_view(2, "Settings"))

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.statusBar().addPermanentWidget(self.status_label, 1)

    def _apply_codex_style(self):
        mono = QFont("Consolas")
        if not mono.exactMatch():
            mono = QFont("Menlo")
        if not mono.exactMatch():
            mono = QFont("Monaco")
        if not mono.exactMatch():
            mono = QFont("Courier New")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(THEME.output_font_size)

        ui_font = QFont("Segoe UI")
        if not ui_font.exactMatch():
            ui_font = QFont("Inter")
        if not ui_font.exactMatch():
            ui_font = QFont()
        ui_font.setPointSize(THEME.sidebar_font_size)

        self.setFont(ui_font)

        text_widgets = [
            self.query_input,
            self.log_output,
            self.json_output,
            self.summary_output,
            self.bottom_json_output,
            self.bottom_summary_output,
        ]
        for widget in text_widgets:
            widget.setFont(mono)
            widget.setTabStopDistance(28)

        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: %(window_bg)s;
                color: %(text)s;
            }
            QFrame#titleBar {
                background: %(title_bg)s;
                border-bottom: 1px solid %(border)s;
            }
            QFrame#activityBar {
                background: %(activity_bg)s;
                border-right: 1px solid %(border)s;
            }
            QToolButton[nav="true"] {
                background: transparent;
                color: %(text_muted)s;
                border: none;
                border-left: 2px solid transparent;
                padding: 10px 0px;
                font-size: %(sidebar_font_size)dpt;
                font-weight: 600;
            }
            QToolButton[nav="true"]:hover {
                color: %(text)s;
                background: %(surface_bg)s;
            }
            QToolButton[nav="true"]:checked {
                color: %(text)s;
                background: %(surface_bg)s;
                border-left: 2px solid %(accent)s;
            }
            QFrame#sidebar {
                background: %(sidebar_bg)s;
                border-right: 1px solid %(border)s;
            }
            QLabel#sidebarTitle {
                color: %(text)s;
                font-size: %(sidebar_font_size)dpt;
                font-weight: 600;
            }
            QFrame#editorArea {
                background: %(editor_bg)s;
            }
            QLabel#folderLabel {
                color: %(text_muted)s;
                font-size: %(sidebar_font_size)dpt;
            }
            QFrame#composerCard, QFrame#bottomPanel {
                background: %(panel_bg)s;
                border-top: 1px solid %(border)s;
            }
            QTabWidget::pane {
                border: 1px solid %(border)s;
                background: %(editor_bg)s;
            }
            QTabBar::tab {
                background: %(panel_bg)s;
                color: %(text_muted)s;
                border: 1px solid %(border)s;
                padding: 8px 14px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                color: %(text)s;
                background: %(editor_bg)s;
            }
            QLabel {
                color: %(text_muted)s;
                font-size: %(sidebar_font_size)dpt;
                font-weight: 500;
            }
            QPlainTextEdit, QTextEdit {
                background: %(input_bg)s;
                color: %(text)s;
                border: 1px solid %(border_strong)s;
                border-radius: %(radius_sm)dpx;
                padding: 8px;
                selection-background-color: %(accent_selected)s;
                font-size: %(output_font_size)dpt;
            }
            QTextBrowser#chatTranscript {
                background: %(input_bg)s;
                color: %(text)s;
                border: 1px solid %(border_strong)s;
                border-radius: %(radius_sm)dpx;
                padding: 10px;
                font-size: %(sidebar_font_size)dpt;
                line-height: 1.45;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: %(input_bg)s;
                color: %(text)s;
                border: 1px solid %(border_strong)s;
                border-radius: %(radius_sm)dpx;
                padding: 6px 8px;
                font-size: %(sidebar_font_size)dpt;
            }
            QComboBox QAbstractItemView {
                background: %(surface_bg)s;
                color: %(text)s;
                border: 1px solid %(border_strong)s;
                selection-background-color: %(accent_selected)s;
            }
            QPushButton {
                background: %(surface_bg)s;
                color: %(text)s;
                border: 1px solid %(border_strong)s;
                border-radius: %(radius_sm)dpx;
                padding: 8px 12px;
                font-weight: 600;
                font-size: %(sidebar_font_size)dpt;
            }
            QPushButton:hover {
                background: %(panel_bg)s;
                border: 1px solid %(accent)s;
            }
            QPushButton:pressed {
                background: %(surface_bg)s;
            }
            QPushButton:disabled {
                color: %(text_muted)s;
                background: %(surface_bg)s;
            }
            QPushButton[role="send"] {
                background: %(accent)s;
                color: %(status_text)s;
                border: 1px solid %(accent)s;
                border-radius: 22px;
                padding: 0px;
                font-size: 18px;
                font-weight: 700;
            }
            QPushButton[role="send"]:hover {
                background: %(accent_hover)s;
                border: 1px solid %(accent_hover)s;
            }
            QPushButton[role="send"]:pressed {
                background: %(accent_selected)s;
                border: 1px solid %(accent_selected)s;
            }
            QPushButton[role="send"]:disabled {
                background: %(border_strong)s;
                color: %(text_muted)s;
                border: 1px solid %(border_strong)s;
            }
            QCheckBox {
                color: %(text)s;
                spacing: 8px;
            }
            QScrollBar:vertical {
                background: %(panel_bg)s;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: %(border_strong)s;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: %(text_muted)s;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QSplitter::handle {
                background: %(border)s;
            }
            QStatusBar {
                background: %(status_bg)s;
                color: %(status_text)s;
                border-top: 1px solid %(border)s;
                font-size: %(status_font_size)dpt;
            }
            """
            % THEME.__dict__
        )
        self.style().unpolish(self.run_button)
        self.style().polish(self.run_button)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().setContentsMargins(THEME.spacing_md, 0, THEME.spacing_md, 0)

    def _make_activity_button(self, text, tooltip):
        button = QtWidgets.QToolButton()
        button.setText(text)
        button.setCheckable(True)
        button.setAutoExclusive(True)
        button.setToolTip(tooltip)
        button.setProperty("nav", True)
        button.setFixedSize(44, 44)
        return button

    def switch_sidebar_view(self, index, title):
        self.sidebar_stack.setCurrentIndex(index)
        self.sidebar_title.setText(title)

    def refresh_device_shape_options(self, profile_name):
        profile = agent.PROFILES[profile_name]
        device_shapes = profile.get("supported_device_shapes", [profile.get("device_shape", "dotgate")])
        current = self.device_shape_combo.currentText()
        self.device_shape_combo.blockSignals(True)
        self.device_shape_combo.clear()
        self.device_shape_combo.addItems(device_shapes)
        if current in device_shapes:
            self.device_shape_combo.setCurrentText(current)
        else:
            self.device_shape_combo.setCurrentText(profile.get("device_shape", device_shapes[0]))
        self.device_shape_combo.blockSignals(False)

    def set_status(self, text):
        self.status_label.setText(text)
        self.statusBar().showMessage(text)

    def append_chat(self, speaker, text):
        is_user = speaker.strip().lower() == "user"
        align = "right" if is_user else "left"
        bubble_bg = "#dbeafe" if is_user else "#f3f4f6"
        bubble_border = "#93c5fd" if is_user else "#d1d5db"
        safe_speaker = html.escape(speaker)
        safe_text = html.escape(text).replace("\n", "<br>")
        bubble_html = f"""
        <div style="width:100%; margin: 0 0 10px 0; text-align: {align};">
          <div style="
            display:inline-block;
            max-width: 88%;
            text-align: left;
            background: {bubble_bg};
            border: 1px solid {bubble_border};
            border-radius: 12px;
            padding: 8px 10px;
          ">
            <div style="font-size: 11px; color: #6b7280; margin-bottom: 4px;">{safe_speaker}</div>
            <div style="font-size: 13px; color: #1f2937;">{safe_text}</div>
          </div>
        </div>
        """
        self.chat_output.append(bubble_html)
        scrollbar = self.chat_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def reset_conversation(self):
        self.active_session_state = None
        self.chat_output.clear()
        self.json_output.clear()
        self.summary_output.clear()
        self.bottom_json_output.clear()
        self.bottom_summary_output.clear()
        self.log_output.clear()
        self.folder_label.setText("Log folder: -")
        self.set_status("Idle")

    def _prepare_followup_session_state(self, session_state):
        if not session_state:
            return None
        prepared = dict(session_state)
        prepared["active"] = True
        prepared["run_confirmed"] = False
        prepared["setup_generation_confirmed"] = False
        prepared["pending_setup_generation"] = False
        prepared["pending_setup_dir"] = None
        prepared["missing_fields"] = list(prepared.get("missing_fields", []))
        return prepared

    def submit_turn(self):
        query = self.query_input.toPlainText().strip()
        if not query:
            QtWidgets.QMessageBox.warning(self, "Missing query", "Please enter a natural-language request.")
            return

        if self.worker_thread is not None and self.worker_thread.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "A simulation is already running.")
            return

        self.append_chat("User", query)
        self.run_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.refresh_solver_button.setEnabled(False)
        self.query_input.clear()
        self.log_output.clear()
        self.current_artifact_dir = None
        self.result_canvas.show_message("Running simulation...")
        self.set_status("Running..." if not self.active_session_state else "Waiting for clarification...")

        self.worker_thread = QtCore.QThread(self)
        self.worker = AgentWorker(
            query=query,
            parser_name=self.parser_combo.currentText(),
            model_name=self.model_input.text().strip() or "gpt-4o-mini",
            strict_parser=self.strict_checkbox.isChecked(),
            profile_name=self.profile_combo.currentText(),
            device_shape=self.device_shape_combo.currentText(),
            session_state=self.active_session_state,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.agent_response.connect(self.on_agent_response)
        self.worker.system_built.connect(self.on_system_built)
        self.worker.boundary_ready.connect(self.on_boundary_ready)
        self.worker.manual_check_required.connect(self.on_manual_check_required)
        self.worker.status.connect(self.set_status)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.on_worker_done)
        self.worker_thread.start()

    def on_agent_response(self, response):
        spec_text = json.dumps(response.get("spec", {}), indent=2)
        self.json_output.setPlainText(spec_text)
        self.bottom_json_output.setPlainText(spec_text)
        message = response.get("message", "")
        if message:
            self.append_chat("Agent", message)

        status = response.get("status")
        if status == "needs_clarification":
            self.active_session_state = response.get("session_state")
            self.run_button.setEnabled(True)
            self.query_input.setFocus()
            if self.current_artifact_dir:
                self.open_folder_button.setEnabled(True)
                self.refresh_solver_button.setEnabled(True)
            summary_text = json.dumps(response, indent=2)
            self.summary_output.setPlainText(summary_text)
            self.bottom_summary_output.setPlainText(summary_text)
            self.set_status("Waiting for clarification")
        elif status == "running":
            self.active_session_state = self._prepare_followup_session_state(response.get("session_state"))
            summary_text = json.dumps(response, indent=2)
            self.summary_output.setPlainText(summary_text)
            self.bottom_summary_output.setPlainText(summary_text)
            self.set_status("Running simulation")

    def append_log(self, text):
        self.log_output.appendPlainText(text)
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_system_built(self, syst, artifact_dir):
        self.current_artifact_dir = artifact_dir
        self.current_system = syst
        self.folder_label.setText(f"Log folder: {artifact_dir}")
        self.geometry_mode_combo.setEnabled(False)
        self.refresh_geometry_button.setEnabled(False)
        self.geometry_mode_combo.blockSignals(True)
        self.geometry_mode_combo.setCurrentText("material")
        self.geometry_mode_combo.blockSignals(False)
        self.geometry_canvas.render_system(syst, plot_mode="material")
        self.tabs.setCurrentIndex(0)
        self.viewer.log_folder = artifact_dir
        self.viewer.refresh_log_folder()
        self.open_folder_button.setEnabled(True)
        self.refresh_solver_button.setEnabled(True)

    def on_boundary_ready(self, syst):
        self.current_system = syst
        self.geometry_mode_combo.setEnabled(True)
        self.refresh_geometry_button.setEnabled(True)
        self.geometry_mode_combo.blockSignals(True)
        self.geometry_mode_combo.setCurrentText("potential")
        self.geometry_mode_combo.blockSignals(False)
        self.update_geometry_view()

    def on_manual_check_required(self, payload):
        self.append_chat("Agent", "Please review the manual boundary check plot before continuing.")
        self.set_status("Waiting for manual check")
        dialog = ManualBoundaryCheckDialog(payload, self)
        approved = dialog.exec() == QtWidgets.QDialog.Accepted
        if self.worker is not None:
            self.worker.resolve_manual_check(approved)

    def refresh_viewer(self):
        if self.current_artifact_dir is None:
            return
        self.viewer.log_folder = self.current_artifact_dir
        self.viewer.refresh_log_folder()

    def update_geometry_view(self):
        if self.current_system is None:
            return
        self.geometry_canvas.render_system(
            self.current_system,
            plot_mode=self.geometry_mode_combo.currentText(),
        )

    def refresh_geometry_from_latest_snapshot(self):
        if self.current_system is None or not self.current_artifact_dir:
            return

        pattern = os.path.join(self.current_artifact_dir, "*.npz")
        files = [
            path for path in glob.glob(pattern)
            if os.path.basename(path) not in {"run_static.npz", "dos_data.npz", "ldos_data.npz"}
        ]

        if not files:
            self.append_log("No FSC snapshot found yet for geometry refresh.")
            return

        latest_path = max(files, key=os.path.getmtime)
        data = np.load(latest_path, allow_pickle=True)

        ui = np.asarray(data["Ui"], dtype=float)
        ni = np.asarray(data["ni"], dtype=float)

        for site_id, site in self.current_system.sites.items():
            if site_id < len(ui):
                site.potential = float(ui[site_id])
            if site_id < len(ni):
                site.charge = float(ni[site_id])

        self.append_log(f"Geometry refreshed from snapshot: {os.path.basename(latest_path)}")
        self.update_geometry_view()

    def on_finished(self, result):
        summary_text = json.dumps(result, indent=2)
        self.summary_output.setPlainText(summary_text)
        self.bottom_summary_output.setPlainText(summary_text)
        if result.get("status") == "needs_clarification":
            self.active_session_state = result.get("session_state")
            self.run_button.setEnabled(True)
            self.query_input.setFocus()
            if self.current_artifact_dir:
                self.open_folder_button.setEnabled(True)
                self.refresh_solver_button.setEnabled(True)
            self.set_status("Waiting for clarification")
            return

        self.active_session_state = self._prepare_followup_session_state(result.get("session_state"))
        self.append_chat("Agent", result.get("message", "Simulation completed."))
        self.set_status("Completed")
        if result.get("status") == "completed":
            self.render_final_result(result.get("result", {}))
            self.tabs.setCurrentIndex(2)
            self.refresh_viewer()

    def on_failed(self, error_text):
        self.summary_output.setPlainText(error_text)
        self.result_canvas.show_message("Run failed. See traceback on the right.")
        self.active_session_state = None
        self.append_chat("Agent", "An error occurred while handling the request.")
        self.set_status("Error")
        QtWidgets.QMessageBox.critical(self, "Run failed", error_text)

    def on_worker_done(self):
        self.run_button.setEnabled(True)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def render_final_result(self, result):
        artifacts = result.get("artifacts", {})
        data_path = artifacts.get("data")
        if not data_path or not os.path.exists(data_path):
            self.result_canvas.show_message("No final DOS / LDOS data file was found.")
            return

        if result.get("task") == "dos":
            self.result_canvas.render_dos(
                data_path,
                ylabel="DOS",
                title="Density of States",
            )
        else:
            site_id = result.get("ldos_site_id", "?")
            self.result_canvas.render_dos(
                data_path,
                ylabel="LDOS",
                title=f"Local Density of States at site {site_id}",
            )

    def open_current_folder_in_viewer(self):
        if not self.current_artifact_dir:
            return
        self.tabs.setCurrentIndex(1)
        self.viewer.log_folder = self.current_artifact_dir
        self.viewer.refresh_log_folder()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = AgentInterfaceWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
