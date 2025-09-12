#!/usr/bin/env python3
import sys
import os
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Qt and Matplotlib backend
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib as mpl
from concurrent.futures import ThreadPoolExecutor
import threading

# ROOT reader
import uproot
import awkward as ak


class _AggregateWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(int, object, object, object)
    error = QtCore.pyqtSignal(int, str)

    def __init__(self,
                 x_hit: np.ndarray,
                 y_hit: np.ndarray,
                 is_pixel_hit: np.ndarray,
                 base_z: np.ndarray,
                 kx0: np.ndarray,
                 ky0: np.ndarray,
                 first_center: float,
                 pitch: float,
                 n: int,
                 r: int,
                 xC1: float,
                 yC1: float,
                 sel: int,
                 seq: int):
        super().__init__()
        self.x_hit = x_hit
        self.y_hit = y_hit
        self.is_pixel_hit = is_pixel_hit
        self.base_z = base_z
        self.kx0 = kx0
        self.ky0 = ky0
        self.first_center = float(first_center)
        self.pitch = float(pitch)
        self.n = int(n)
        self.r = int(r)
        self.xC1 = float(xC1)
        self.yC1 = float(yC1)
        self.sel = int(sel)
        self.seq = int(seq)
        self._cancelled = False
        self._cancel_lock = threading.Lock()

    def cancel(self):
        with self._cancel_lock:
            self._cancelled = True

    def _is_cancelled(self) -> bool:
        with self._cancel_lock:
            return self._cancelled

    def run(self):
        try:
            x_hit = self.x_hit
            y_hit = self.y_hit
            is_px_hit = self.is_pixel_hit
            base_z = self.base_z
            kx0 = self.kx0
            ky0 = self.ky0
            first_center = self.first_center
            pitch = self.pitch
            n = self.n
            r = self.r
            xC1 = self.xC1
            yC1 = self.yC1

            N = int(x_hit.shape[0])
            if N == 0:
                if not self._is_cancelled():
                    self.finished.emit(self.seq, np.array([]), np.array([]), np.array([]))
                return

            # Base validity (finite and not pixel-only)
            base_mask_global = np.isfinite(x_hit) & np.isfinite(y_hit) & (~is_px_hit)

            # Prepare chunks
            max_workers = max(1, (os.cpu_count() or 4) - 1)
            num_chunks = max_workers * 4
            indices = np.array_split(np.arange(N, dtype=np.int32), num_chunks)
            offsets = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
            # Aggregation window (canonical 3x3 around the central pad)
            x_min_win = xC1 - 1.5 * pitch
            x_max_win = xC1 + 1.5 * pitch
            y_min_win = yC1 - 1.5 * pitch
            y_max_win = yC1 + 1.5 * pitch

            # Grid resolution for fast rendering (tune as needed)
            bins_x = 200
            bins_y = 200
            inv_dx = float(bins_x) / max(x_max_win - x_min_win, 1e-12)
            inv_dy = float(bins_y) / max(y_max_win - y_min_win, 1e-12)

            def process_chunk(idx_arr: np.ndarray):
                if idx_arr.size == 0:
                    # Return empty partials
                    return (np.zeros((bins_y, bins_x), dtype=np.float64),
                            np.zeros((bins_y, bins_x), dtype=np.float64))
                xi = idx_arr
                x_local = x_hit[xi]
                y_local = y_hit[xi]
                z_local = base_z[xi]
                kx_local0 = kx0[xi]
                ky_local0 = ky0[xi]
                base_mask = base_mask_global[xi]

                sum_local = np.zeros((bins_y, bins_x), dtype=np.float64)
                cnt_local = np.zeros((bins_y, bins_x), dtype=np.float64)

                # First pass: count how many 3x3 windows each hit belongs to
                # This prevents over-counting hits that lie in overlapping windows
                win_count = np.zeros(x_local.shape, dtype=np.int16)

                for ox in offsets:
                    kx = kx_local0 + ox
                    valid_kx = (kx >= r) & (kx <= n - 1 - r)
                    x_center = first_center + kx.astype(np.float64) * pitch
                    cond_x = np.abs(x_local - x_center) <= (1.5 * pitch)

                    for oy in offsets:
                        ky = ky_local0 + oy
                        valid_ky = (ky >= r) & (ky <= n - 1 - r)
                        y_center = first_center + ky.astype(np.float64) * pitch
                        cond_y = np.abs(y_local - y_center) <= (1.5 * pitch)
                        c = base_mask & valid_kx & valid_ky & cond_x & cond_y
                        if not np.any(c):
                            continue
                        win_count[c] += 1

                    if self._is_cancelled():
                        break

                for ox in offsets:
                    kx = kx_local0 + ox
                    valid_kx = (kx >= r) & (kx <= n - 1 - r)
                    x_center = first_center + kx.astype(np.float64) * pitch
                    cond_x = np.abs(x_local - x_center) <= (1.5 * pitch)

                    for oy in offsets:
                        ky = ky_local0 + oy
                        valid_ky = (ky >= r) & (ky <= n - 1 - r)
                        y_center = first_center + ky.astype(np.float64) * pitch
                        cond_y = np.abs(y_local - y_center) <= (1.5 * pitch)
                        c = base_mask & valid_kx & valid_ky & cond_x & cond_y
                        if not np.any(c):
                            continue
                        xs_sub = x_local[c] - x_center[c] + xC1
                        ys_sub = y_local[c] - y_center[c] + yC1
                        z_sub = z_local[c]

                        # Bin indices
                        ix = np.floor((xs_sub - x_min_win) * inv_dx).astype(np.int32)
                        iy = np.floor((ys_sub - y_min_win) * inv_dy).astype(np.int32)
                        m = (ix >= 0) & (ix < bins_x) & (iy >= 0) & (iy < bins_y)
                        if not np.any(m):
                            continue
                        ixm = ix[m]
                        iym = iy[m]
                        zm = z_sub[m]
                        # Weight each hit by the number of windows it contributes to
                        wc = win_count[c][m]
                        # Numerical safety: ensure wc >= 1
                        wc = np.maximum(wc, 1)
                        weighted_vals = zm / wc
                        # Accumulate weighted sums and effective counts
                        np.add.at(sum_local, (iym, ixm), weighted_vals)
                        np.add.at(cnt_local, (iym, ixm), 1.0 / wc)

                    if self._is_cancelled():
                        break

                return (sum_local, cnt_local)

            sum_grid = np.zeros((bins_y, bins_x), dtype=np.float64)
            cnt_grid = np.zeros((bins_y, bins_x), dtype=np.float64)

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(process_chunk, idx) for idx in indices]
                for fut in futures:
                    if self._is_cancelled():
                        break
                    try:
                        s_loc, c_loc = fut.result()
                    except Exception:
                        # Propagate first error
                        raise
                    # Merge partials
                    sum_grid += s_loc
                    cnt_grid += c_loc

            if self._is_cancelled():
                return

            # Compute mean per bin; empty bins -> NaN
            with np.errstate(divide='ignore', invalid='ignore'):
                zgrid = np.divide(sum_grid, cnt_grid, out=np.full_like(sum_grid, np.nan, dtype=np.float64), where=(cnt_grid > 0))

            # Build bin edges
            x_edges = np.linspace(x_min_win, x_max_win, num=bins_x + 1, dtype=np.float64)
            y_edges = np.linspace(y_min_win, y_max_win, num=bins_y + 1, dtype=np.float64)

            if not self._is_cancelled():
                self.finished.emit(self.seq, x_edges, y_edges, zgrid)
        except Exception as e:
            if not self._is_cancelled():
                self.error.emit(self.seq, str(e))

@dataclass
class Geometry:
    pixel_size_mm: float = 0.0
    pixel_spacing_mm: float = 0.0
    pixel_corner_offset_mm: float = 0.0
    det_size_mm: float = 0.0
    num_per_side: int = 0
    neighborhood_radius: int = 1


class Mode:
    TwoD_Abs = 0
    TwoD_Signed = 1
    ThreeD_Abs = 2
    ThreeD_Signed = 3
    TwoD_vs_Pixel = 4
    ThreeD_vs_Pixel = 5
    TwoD3D_Combined = 6
    TwoD_vs_ThreeD = 7


MODE_LABELS = {
    Mode.TwoD_Abs: "2D abs",
    Mode.TwoD_Signed: "2D signed",
    Mode.ThreeD_Abs: "3D abs",
    Mode.ThreeD_Signed: "3D signed",
    Mode.TwoD_vs_Pixel: "2D vs Pixel",
    Mode.ThreeD_vs_Pixel: "3D vs Pixel",
    Mode.TwoD3D_Combined: "2D+3D combined",
    Mode.TwoD_vs_ThreeD: "2D vs 3D",
}


def read_named_double(file: uproot.ReadOnlyFile, key: str) -> float:
    obj = file.get(key)
    if obj is None:
        raise RuntimeError(f"Missing metadata object: '{key}'")
    # uproot returns a GenericObject for TNamed; access the fTitle member
    try:
        title = obj.member("fTitle")
    except Exception:
        # fallback: try attributes
        title = getattr(obj, "fTitle", None)
        if title is None:
            # try to interpret as a TObjString-like string
            title = str(obj)
    if title is None:
        raise RuntimeError(f"TNamed '{key}' has empty title")
    try:
        return float(title)
    except Exception as e:
        raise RuntimeError(f"Cannot parse '{key}' title to float: {title}") from e


def read_named_int(file: uproot.ReadOnlyFile, key: str) -> int:
    return int(round(read_named_double(file, key)))


class MatplotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 6), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.subplots_adjust(left=0.12, right=0.86, top=0.92, bottom=0.12)
        self.cbar = None

    def clear(self):
        # Remove any existing colorbar and its axes
        try:
            if self.cbar is not None:
                self.cbar.remove()
        except Exception:
            pass
        self.cbar = None
        # Also remove any extra axes that might have been created for colorbars
        for extra_ax in list(self.fig.axes):
            if extra_ax is not self.ax:
                try:
                    self.fig.delaxes(extra_ax)
                except Exception:
                    pass
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")


class PyColorGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("epicChargeSharing: Colorings")
        self.resize(1200, 900)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        vbox = QtWidgets.QVBoxLayout(central)

        # Top file controls
        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(QtWidgets.QLabel("ROOT file:"))
        self.file_edit = QtWidgets.QLineEdit()
        file_layout.addWidget(self.file_edit, 1)
        browse_btn = QtWidgets.QPushButton("Browse…")
        open_btn = QtWidgets.QPushButton("Open")
        file_layout.addWidget(browse_btn)
        file_layout.addWidget(open_btn)
        vbox.addLayout(file_layout)

        # Controls row
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        for k in sorted(MODE_LABELS.keys()):
            self.mode_combo.addItem(MODE_LABELS[k], k)
        controls.addWidget(self.mode_combo)

        controls.addWidget(QtWidgets.QLabel("Metric:"))
        self.metric_combo = QtWidgets.QComboBox()
        controls.addWidget(self.metric_combo)

        self.preview_btn = QtWidgets.QPushButton("Preview")
        self.rand_btn = QtWidgets.QPushButton("Random 3x3")
        self.save_btn = QtWidgets.QPushButton("Save…")
        self.show_fit_btn = QtWidgets.QPushButton("Show fit")
        self.show_fit_btn.setCheckable(True)
        self.aggregate_cb = QtWidgets.QCheckBox("Aggregate all 3x3s")
        controls.addWidget(self.preview_btn)
        controls.addWidget(self.rand_btn)
        controls.addWidget(self.save_btn)
        controls.addWidget(self.show_fit_btn)
        controls.addWidget(self.aggregate_cb)
        vbox.addLayout(controls)

        # Secondary controls row (color scaling and colormap)
        controls2 = QtWidgets.QHBoxLayout()
        controls2.addWidget(QtWidgets.QLabel("Scale:"))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItem("Auto", "auto")
        self.scale_combo.addItem("Robust (P2–P98)", "robust")
        self.scale_combo.addItem("Fixed", "fixed")
        self.scale_combo.addItem("Log (abs)", "log")
        self.scale_combo.addItem("SymLog (signed)", "symlog")
        controls2.addWidget(self.scale_combo)

        controls2.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmap_combo = QtWidgets.QComboBox()
        # First entry: Auto (viridis for abs, coolwarm for signed)
        self.cmap_combo.addItem("Auto", "auto")
        for name in [
            "viridis", "plasma", "inferno", "magma", "cividis",
            "turbo", "coolwarm", "RdBu_r", "Spectral", "RdYlBu_r"
        ]:
            self.cmap_combo.addItem(name, name)
        controls2.addWidget(self.cmap_combo)

        vbox.addLayout(controls2)

        # Canvas
        self.canvas = MatplotCanvas()
        vbox.addWidget(self.canvas, 1)

        # State
        self.file: Optional[uproot.ReadOnlyFile] = None
        self.file_path: str = ""
        self.geom = Geometry()
        self.mode: int = Mode.TwoD_Abs
        self.metric_index: int = 0
        self.i0: int = -1
        self.j0: int = -1
        self.hasRecon2D = False
        self.hasRecon2DSigned = False
        self.hasRecon3D = False
        self.hasRecon3DSigned = False
        self.hasPixelDeltas = False
        self.aggregate_all: bool = False
        # Options state
        self.scale_mode: str = "auto"
        self.cmap_choice: str = "auto"
        # Async aggregation state
        self._agg_seq: int = 0
        self._agg_thread: Optional[QtCore.QThread] = None
        self._agg_worker: Optional["_AggregateWorker"] = None
        self._agg_status_text = None
        # Show-fit interaction state
        self.show_fit_active: bool = False
        self._mpl_cid_click = None
        self._last_points = None  # np.ndarray shape (K,2) of (x,y)
        self._last_indices = None  # np.ndarray shape (K,) of event indices
        # Keep references to fit dialogs so they are not GC'ed
        self._fit_dialogs: List[QtWidgets.QDialog] = []
        # Store last drawn scatter colors and mapping for Show fit
        self._last_facecolors = None
        self._last_zvals = None
        self._last_norm = None
        self._last_cmap = None

        # Signals
        browse_btn.clicked.connect(self.on_browse)
        open_btn.clicked.connect(self.on_open)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.metric_combo.currentIndexChanged.connect(self.on_metric_changed)
        self.preview_btn.clicked.connect(self.on_preview)
        self.rand_btn.clicked.connect(self.on_random)
        self.save_btn.clicked.connect(self.on_save)
        self.aggregate_cb.stateChanged.connect(self.on_aggregate_changed)
        self.show_fit_btn.toggled.connect(self.on_show_fit_toggled)
        # Option signals
        self.scale_combo.currentIndexChanged.connect(self.on_options_changed)
        self.cmap_combo.currentIndexChanged.connect(self.on_options_changed)

        # Initial metric build
        self.rebuild_metric_combo()
        # Start with an empty canvas until a file is opened
        self.canvas.clear()
        self.canvas.draw()

    # ------------- UI handlers -------------
    def on_browse(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open ROOT file", self.file_edit.text() or os.getcwd(),
            "ROOT files (*.root);;All files (*)"
        )
        if fn:
            self.file_edit.setText(fn)

    def on_open(self):
        path = self.file_edit.text().strip()
        if not path:
            return
        if not os.path.exists(path):
            QtWidgets.QMessageBox.critical(self, "Open ROOT", f"File not found: {path}")
            return
        try:
            self.file = uproot.open(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open ROOT", f"Failed to open {path}: {e}")
            return
        self.file_path = path
        # Cancel any ongoing aggregation for previous file
        try:
            if self._agg_worker is not None:
                self._agg_worker.cancel()
        except Exception:
            pass
        self._agg_seq += 1
        self.detect_branch_availability()
        try:
            self.load_geometry()
        except Exception:
            pass
        self.update_mode_availability()
        self.rebuild_metric_combo()
        self.draw_current()
    def on_options_changed(self, *args, **kwargs):
        try:
            self.scale_mode = str(self.scale_combo.currentData())
        except Exception:
            self.scale_mode = "auto"
        try:
            self.cmap_choice = str(self.cmap_combo.currentData())
        except Exception:
            self.cmap_choice = "auto"
        self.draw_current()

    def on_mode_changed(self, idx: int):
        self.mode = self.mode_combo.currentData()
        self.rebuild_metric_combo()
        self.draw_current()

    def on_metric_changed(self, idx: int):
        self.metric_index = self.metric_combo.currentData()
        self.draw_current()

    def on_random(self):
        if self.geom.num_per_side <= 0:
            return
        if getattr(self, "aggregate_all", False):
            return
        # Choose only within inner region that guarantees a full neighborhood
        r = max(1, int(self.geom.neighborhood_radius))
        n = int(self.geom.num_per_side)
        lo = max(0, r - 1)
        hi = max(lo, n - r - 2)
        self.i0 = random.randint(lo, hi)
        self.j0 = random.randint(lo, hi)
        self.draw_current()

    def on_plot(self):
        self.draw_current()

    def on_save(self):
        if self.canvas is None:
            return
        fn, selected = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure", os.getcwd(),
            "SVG (*.svg);;PNG (*.png);;PDF (*.pdf)"
        )
        if fn:
            try:
                self.canvas.fig.savefig(fn, bbox_inches="tight")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save", f"Failed to save: {e}")
    def on_aggregate_changed(self, state: int):
        self.aggregate_all = (state == QtCore.Qt.Checked)
        try:
            self.rand_btn.setEnabled(not self.aggregate_all)
            # Disable Show fit when aggregating
            self.show_fit_btn.setEnabled(not self.aggregate_all)
            if self.aggregate_all and self.show_fit_btn.isChecked():
                self.show_fit_btn.setChecked(False)
        except Exception:
            pass
        # Cancel any ongoing aggregation and bump epoch
        try:
            if self._agg_worker is not None:
                self._agg_worker.cancel()
        except Exception:
            pass
        self._agg_seq += 1
        self.draw_current()

    def on_show_fit_toggled(self, checked: bool):
        self.show_fit_active = bool(checked)
        try:
            if self.show_fit_active and not getattr(self, 'aggregate_all', False):
                if self._mpl_cid_click is None:
                    self._mpl_cid_click = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
            else:
                if self._mpl_cid_click is not None:
                    try:
                        self.canvas.mpl_disconnect(self._mpl_cid_click)
                    except Exception:
                        pass
                    self._mpl_cid_click = None
        except Exception:
            pass
    def on_preview(self):
        # Show the entire detector grid with the current 3x3 highlighted
        try:
            self.load_geometry()
        except Exception:
            pass
        if self.geom.num_per_side <= 0:
            QtWidgets.QMessageBox.warning(self, "Preview", "Open a ROOT file first.")
            return
        n = self.geom.num_per_side
        i0 = self.i0 if (0 <= self.i0 <= n-3) else 0
        j0 = self.j0 if (0 <= self.j0 <= n-3) else 0

        half_det = self.geom.det_size_mm / 2.0
        first_center = -half_det + self.geom.pixel_corner_offset_mm + self.geom.pixel_size_mm / 2.0
        half_pitch = 0.5 * self.geom.pixel_spacing_mm

        xC0 = first_center + i0 * self.geom.pixel_spacing_mm
        xC1 = first_center + (i0+1) * self.geom.pixel_spacing_mm
        xC2 = first_center + (i0+2) * self.geom.pixel_spacing_mm
        yC0 = first_center + j0 * self.geom.pixel_spacing_mm
        yC1 = first_center + (j0+1) * self.geom.pixel_spacing_mm
        yC2 = first_center + (j0+2) * self.geom.pixel_spacing_mm

        x_min = -half_det
        x_max =  half_det
        y_min = -half_det
        y_max =  half_det

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("3x3 Preview")
        dlg.resize(800, 800)
        lay = QtWidgets.QVBoxLayout(dlg)
        fig = Figure(figsize=(6, 6), dpi=100)
        canvas = FigureCanvas(fig)
        lay.addWidget(canvas)
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        # Draw all pads
        ps = self.geom.pixel_size_mm
        color_all = (0.65, 0.80, 1.0, 0.25)
        color_line = (0.1, 0.3, 0.8, 1.0)
        for ii in range(n):
            xc = first_center + ii * self.geom.pixel_spacing_mm
            for jj in range(n):
                yc = first_center + jj * self.geom.pixel_spacing_mm
                x1 = xc - ps/2.0
                y1 = yc - ps/2.0
                rect = mpatches.Rectangle((x1, y1), ps, ps,
                                          facecolor=color_all, edgecolor=color_line, linewidth=0.8)
                ax.add_patch(rect)

        if getattr(self, "aggregate_all", False):
            # Draw boundaries for all 3x3 windows across the inner shooting region
            color_border = (1.0, 0.2, 0.2, 0.8)
            r = max(1, int(self.geom.neighborhood_radius))
            start = max(0, r - 1)
            end = max(start, n - r - 2)
            for ii0 in range(start, end + 1):
                xC0i = first_center + ii0 * self.geom.pixel_spacing_mm
                xC2i = first_center + (ii0 + 2) * self.geom.pixel_spacing_mm
                for jj0 in range(start, end + 1):
                    yC0j = first_center + jj0 * self.geom.pixel_spacing_mm
                    yC2j = first_center + (jj0 + 2) * self.geom.pixel_spacing_mm
                    xw1 = max(-half_det, xC0i - half_pitch)
                    xw2 = min( half_det, xC2i + half_pitch)
                    yw1 = max(-half_det, yC0j - half_pitch)
                    yw2 = min( half_det, yC2j + half_pitch)
                    border = mpatches.Rectangle((xw1, yw1), xw2-xw1, yw2-yw1,
                                                fill=False, edgecolor=color_border, linewidth=1.0, linestyle='-')
                    ax.add_patch(border)
        else:
            # Highlight selected 3x3 pads
            color_sel = (0.95, 0.55, 0.10, 0.55)
            for xc in (xC0, xC1, xC2):
                for yc in (yC0, yC1, yC2):
                    x1 = xc - ps/2.0
                    y1 = yc - ps/2.0
                    rect = mpatches.Rectangle((x1, y1), ps, ps,
                                              facecolor=color_sel, edgecolor=(0.75,0.25,0.05,1.0), linewidth=1.2)
                    ax.add_patch(rect)

            # Draw the 3x3 window boundary at half-pitch
            xw1 = max(-half_det, xC0 - half_pitch)
            xw2 = min( half_det, xC2 + half_pitch)
            yw1 = max(-half_det, yC0 - half_pitch)
            yw2 = min( half_det, yC2 + half_pitch)
            border = mpatches.Rectangle((xw1, yw1), xw2-xw1, yw2-yw1,
                                        fill=False, edgecolor="red", linewidth=2.0, linestyle='-')
            ax.add_patch(border)

        canvas.draw()
        # Keep dialog as an attribute so it doesn't get GC'ed
        self._preview_dialog = dlg
        self._preview_canvas = canvas
        dlg.show()

    # ------------- Data helpers -------------
    def detect_branch_availability(self):
        self.hasRecon2D = self.hasRecon2DSigned = False
        self.hasRecon3D = self.hasRecon3DSigned = False
        self.hasPixelDeltas = False
        try:
            hits = self.file["Hits"] if self.file is not None else None
        except Exception:
            hits = None
        if hits is None:
            return
        branches = set(hits.keys())
        self.hasRecon2D = ("ReconTrueDeltaX" in branches) and ("ReconTrueDeltaY" in branches)
        self.hasRecon2DSigned = ("ReconTrueDeltaX_Signed" in branches) and ("ReconTrueDeltaY_Signed" in branches)
        self.hasRecon3D = ("ReconTrueDeltaX_3D" in branches) and ("ReconTrueDeltaY_3D" in branches)
        self.hasRecon3DSigned = ("ReconTrueDeltaX_3D_Signed" in branches) and ("ReconTrueDeltaY_3D_Signed" in branches)
        self.hasPixelDeltas = ("PixelTrueDeltaX" in branches) and ("PixelTrueDeltaY" in branches)

    def update_mode_availability(self):
        # Rebuild mode combo to show only valid modes
        valid_modes: List[int] = []
        def add_if(cond: bool, m: int):
            if cond:
                valid_modes.append(m)

        add_if(self.hasRecon2D, Mode.TwoD_Abs)
        add_if(self.hasRecon2DSigned, Mode.TwoD_Signed)
        add_if(self.hasRecon3D, Mode.ThreeD_Abs)
        add_if(self.hasRecon3DSigned, Mode.ThreeD_Signed)
        add_if(self.hasRecon2D and self.hasPixelDeltas, Mode.TwoD_vs_Pixel)
        add_if(self.hasRecon3D and self.hasPixelDeltas, Mode.ThreeD_vs_Pixel)
        add_if(self.hasRecon2D and self.hasRecon3D, Mode.TwoD_vs_ThreeD)
        add_if(self.hasRecon2D and self.hasRecon3D, Mode.TwoD3D_Combined)

        current_mode = self.mode_combo.currentData()
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        for m in valid_modes:
            self.mode_combo.addItem(MODE_LABELS[m], m)
        # Select current if still valid, else first
        idx = next((i for i in range(self.mode_combo.count()) if self.mode_combo.itemData(i) == current_mode), 0)
        self.mode_combo.setCurrentIndex(max(0, idx))
        self.mode = self.mode_combo.currentData()
        self.mode_combo.blockSignals(False)

    def rebuild_metric_combo(self):
        metrics: List[Tuple[str, int]] = []
        if self.mode == Mode.TwoD_Abs:
            metrics = [
                ("|dx|", 0),
                ("|dy|", 1),
                ("(|dx|+|dy|)/2", 2),
                ("r2D = sqrt(dx^2+dy^2)", 3),
                ("max(|dx|,|dy|)", 4),
            ]
        elif self.mode == Mode.TwoD_Signed:
            metrics = [
                ("dx (signed)", 0),
                ("dy (signed)", 1),
                ("(dx+dy)/2 (signed)", 2),
            ]
        elif self.mode == Mode.ThreeD_Abs:
            metrics = [
                ("|dx_3D|", 0),
                ("|dy_3D|", 1),
                ("(|dx_3D|+|dy_3D|)/2", 2),
                ("r3D = sqrt(dx_3D^2+dy_3D^2)", 3),
                ("max(|dx_3D|,|dy_3D|)", 4),
            ]
        elif self.mode == Mode.ThreeD_Signed:
            metrics = [
                ("dx_3D (signed)", 0),
                ("dy_3D (signed)", 1),
                ("(dx_3D+dy_3D)/2 (signed)", 2),
            ]
        elif self.mode == Mode.TwoD_vs_Pixel:
            metrics = [
                ("|mean_2D - mean_pix|", 0),
                ("mean_2D - mean_pix (signed)", 1),
                ("abs(|dx|-|dx_pix|)", 2),
                ("|dx|-|dx_pix| (signed)", 3),
                ("abs(|dy|-|dy_pix|)", 4),
                ("|dy|-|dy_pix| (signed)", 5),
                ("|r2D - r_pix|", 6),
                ("r2D - r_pix (signed)", 7),
            ]
        elif self.mode == Mode.ThreeD_vs_Pixel:
            metrics = [
                ("|mean_3D - mean_pix|", 0),
                ("mean_3D - mean_pix (signed)", 1),
                ("abs(|dx_3D|-|dx_pix|)", 2),
                ("|dx_3D|-|dx_pix| (signed)", 3),
                ("abs(|dy_3D|-|dy_pix|)", 4),
                ("|dy_3D|-|dy_pix| (signed)", 5),
                ("|r3D - r_pix|", 6),
                ("r3D - r_pix (signed)", 7),
            ]
        elif self.mode == Mode.TwoD3D_Combined:
            metrics = [
                ("avg(|dx|,|dy|,|dx_3D|,|dy_3D|)", 0),
                ("avg(|dx|,|dx_3D|)", 1),
                ("avg(|dy|,|dy_3D|)", 2),
                ("avg(r2D,r3D)", 3),
            ]
        elif self.mode == Mode.TwoD_vs_ThreeD:
            metrics = [
                ("|mean_3D - mean_2D|", 0),
                ("mean_3D - mean_2D (signed)", 1),
                ("abs(|dx_3D|-|dx|)", 2),
                ("|dx_3D|-|dx| (signed)", 3),
                ("abs(|dy_3D|-|dy|)", 4),
                ("|dy_3D|-|dy| (signed)", 5),
                ("|r3D - r2D|", 6),
                ("r3D - r2D (signed)", 7),
            ]
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for label, mid in metrics:
            self.metric_combo.addItem(label, mid)
        self.metric_combo.setCurrentIndex(0)
        self.metric_index = self.metric_combo.currentData()
        self.metric_combo.blockSignals(False)

    def load_geometry(self):
        if self.file is None:
            return
        self.geom.pixel_size_mm = read_named_double(self.file, "GridPixelSize_mm")
        self.geom.pixel_spacing_mm = read_named_double(self.file, "GridPixelSpacing_mm")
        self.geom.pixel_corner_offset_mm = read_named_double(self.file, "GridPixelCornerOffset_mm")
        self.geom.det_size_mm = read_named_double(self.file, "GridDetectorSize_mm")
        self.geom.num_per_side = read_named_int(self.file, "GridNumBlocksPerSide")
        try:
            self.geom.neighborhood_radius = read_named_int(self.file, "NeighborhoodRadius")
        except Exception:
            # Default to 1 if metadata missing
            self.geom.neighborhood_radius = 1

    def _get_colorbar_label(self, sel: Optional[int] = None) -> str:
        try:
            metric_index = self.metric_combo.currentIndex() if sel is None else int(sel)
        except Exception:
            metric_index = 0
        m = self.mode
        def mm(expr: str) -> str:
            return expr + " [mm]"
        if m == Mode.TwoD_Abs:
            if metric_index == 0:
                return mm(r"$|x_{\mathrm{rec}}-x_{\mathrm{true}}|$")
            if metric_index == 1:
                return mm(r"$|y_{\mathrm{rec}}-y_{\mathrm{true}}|$")
            if metric_index == 2:
                return mm(r"$\frac{1}{2}(|x_{\mathrm{rec}}-x_{\mathrm{true}}|+|y_{\mathrm{rec}}-y_{\mathrm{true}}|)$")
            if metric_index == 3:
                return mm(r"$\sqrt{(x_{\mathrm{rec}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec}}-y_{\mathrm{true}})^2}$")
            return mm(r"$\max\{|x_{\mathrm{rec}}-x_{\mathrm{true}}|,|y_{\mathrm{rec}}-y_{\mathrm{true}}|\}$")
        if m == Mode.TwoD_Signed:
            if metric_index == 0:
                return mm(r"$x_{\mathrm{rec}}-x_{\mathrm{true}}$")
            if metric_index == 1:
                return mm(r"$y_{\mathrm{rec}}-y_{\mathrm{true}}$")
            return mm(r"$\frac{1}{2}((x_{\mathrm{rec}}-x_{\mathrm{true}})+(y_{\mathrm{rec}}-y_{\mathrm{true}}))$")
        if m == Mode.ThreeD_Abs:
            if metric_index == 0:
                return mm(r"$|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|$")
            if metric_index == 1:
                return mm(r"$|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|$")
            if metric_index == 2:
                return mm(r"$\frac{1}{2}(|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|+|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|)$")
            if metric_index == 3:
                return mm(r"$\sqrt{(x_{\mathrm{rec,3D}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec,3D}}-y_{\mathrm{true}})^2}$")
            return mm(r"$\max\{|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|,|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|\}$")
        if m == Mode.ThreeD_Signed:
            if metric_index == 0:
                return mm(r"$x_{\mathrm{rec,3D}}-x_{\mathrm{true}}$")
            if metric_index == 1:
                return mm(r"$y_{\mathrm{rec,3D}}-y_{\mathrm{true}}$")
            return mm(r"$\frac{1}{2}((x_{\mathrm{rec,3D}}-x_{\mathrm{true}})+(y_{\mathrm{rec,3D}}-y_{\mathrm{true}}))$")
        if m == Mode.TwoD_vs_Pixel:
            if metric_index == 0:
                return mm(r"$\left|\frac{1}{2}(|x_{\mathrm{rec}}-x_{\mathrm{true}}|+|y_{\mathrm{rec}}-y_{\mathrm{true}}|)-\frac{1}{2}(|x_{\mathrm{px}}-x_{\mathrm{true}}|+|y_{\mathrm{px}}-y_{\mathrm{true}}|)\right|$")
            if metric_index == 1:
                return mm(r"$\frac{1}{2}(|x_{\mathrm{rec}}-x_{\mathrm{true}}|+|y_{\mathrm{rec}}-y_{\mathrm{true}}|)-\frac{1}{2}(|x_{\mathrm{px}}-x_{\mathrm{true}}|+|y_{\mathrm{px}}-y_{\mathrm{true}}|)$")
            if metric_index == 2:
                return mm(r"$\left||x_{\mathrm{rec}}-x_{\mathrm{true}}|-|x_{\mathrm{px}}-x_{\mathrm{true}}|\right|$")
            if metric_index == 3:
                return mm(r"$|x_{\mathrm{rec}}-x_{\mathrm{true}}|-|x_{\mathrm{px}}-x_{\mathrm{true}}|$")
            if metric_index == 4:
                return mm(r"$\left||y_{\mathrm{rec}}-y_{\mathrm{true}}|-|y_{\mathrm{px}}-y_{\mathrm{true}}|\right|$")
            if metric_index == 5:
                return mm(r"$|y_{\mathrm{rec}}-y_{\mathrm{true}}|-|y_{\mathrm{px}}-y_{\mathrm{true}}|$")
            if metric_index == 6:
                return mm(r"$\left|\sqrt{(x_{\mathrm{rec}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec}}-y_{\mathrm{true}})^2}-\sqrt{(x_{\mathrm{px}}-x_{\mathrm{true}})^2+(y_{\mathrm{px}}-y_{\mathrm{true}})^2}\right|$")
            return mm(r"$\sqrt{(x_{\mathrm{rec}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec}}-y_{\mathrm{true}})^2}-\sqrt{(x_{\mathrm{px}}-x_{\mathrm{true}})^2+(y_{\mathrm{px}}-y_{\mathrm{true}})^2}$")
        if m == Mode.ThreeD_vs_Pixel:
            if metric_index == 0:
                return mm(r"$\left|\frac{1}{2}(|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|+|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|)-\frac{1}{2}(|x_{\mathrm{px}}-x_{\mathrm{true}}|+|y_{\mathrm{px}}-y_{\mathrm{true}}|)\right|$")
            if metric_index == 1:
                return mm(r"$\frac{1}{2}(|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|+|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|)-\frac{1}{2}(|x_{\mathrm{px}}-x_{\mathrm{true}}|+|y_{\mathrm{px}}-y_{\mathrm{true}}|)$")
            if metric_index == 2:
                return mm(r"$\left||x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|-|x_{\mathrm{px}}-x_{\mathrm{true}}|\right|$")
            if metric_index == 3:
                return mm(r"$|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|-|x_{\mathrm{px}}-x_{\mathrm{true}}|$")
            if metric_index == 4:
                return mm(r"$\left||y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|-|y_{\mathrm{px}}-y_{\mathrm{true}}|\right|$")
            if metric_index == 5:
                return mm(r"$|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|-|y_{\mathrm{px}}-y_{\mathrm{true}}|$")
            if metric_index == 6:
                return mm(r"$\left|\sqrt{(x_{\mathrm{rec,3D}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec,3D}}-y_{\mathrm{true}})^2}-\sqrt{(x_{\mathrm{px}}-x_{\mathrm{true}})^2+(y_{\mathrm{px}}-y_{\mathrm{true}})^2}\right|$")
            return mm(r"$\sqrt{(x_{\mathrm{rec,3D}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec,3D}}-y_{\mathrm{true}})^2}-\sqrt{(x_{\mathrm{px}}-x_{\mathrm{true}})^2+(y_{\mathrm{px}}-y_{\mathrm{true}})^2}$")
        if m == Mode.TwoD3D_Combined:
            if metric_index == 0:
                return mm(r"$\frac{1}{4}(|x_{\mathrm{rec}}-x_{\mathrm{true}}|+|y_{\mathrm{rec}}-y_{\mathrm{true}}|+|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|+|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|)$")
            if metric_index == 1:
                return mm(r"$\frac{1}{2}(|x_{\mathrm{rec}}-x_{\mathrm{true}}|+|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|)$")
            if metric_index == 2:
                return mm(r"$\frac{1}{2}(|y_{\mathrm{rec}}-y_{\mathrm{true}}|+|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|)$")
            return mm(r"$\frac{1}{2}(r_{2D}+r_{3D})$")
        if m == Mode.TwoD_vs_ThreeD:
            if metric_index == 0:
                return mm(r"$\left|\frac{1}{2}(|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|+|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|)-\frac{1}{2}(|x_{\mathrm{rec}}-x_{\mathrm{true}}|+|y_{\mathrm{rec}}-y_{\mathrm{true}}|)\right|$")
            if metric_index == 1:
                return mm(r"$\frac{1}{2}(|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|+|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|)-\frac{1}{2}(|x_{\mathrm{rec}}-x_{\mathrm{true}}|+|y_{\mathrm{rec}}-y_{\mathrm{true}}|)$")
            if metric_index == 2:
                return mm(r"$\left||x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|-|x_{\mathrm{rec}}-x_{\mathrm{true}}|\right|$")
            if metric_index == 3:
                return mm(r"$|x_{\mathrm{rec,3D}}-x_{\mathrm{true}}|-|x_{\mathrm{rec}}-x_{\mathrm{true}}|$")
            if metric_index == 4:
                return mm(r"$\left||y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|-|y_{\mathrm{rec}}-y_{\mathrm{true}}|\right|$")
            if metric_index == 5:
                return mm(r"$|y_{\mathrm{rec,3D}}-y_{\mathrm{true}}|-|y_{\mathrm{rec}}-y_{\mathrm{true}}|$")
            if metric_index == 6:
                return mm(r"$\left|\sqrt{(x_{\mathrm{rec,3D}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec,3D}}-y_{\mathrm{true}})^2}-\sqrt{(x_{\mathrm{rec}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec}}-y_{\mathrm{true}})^2}\right|$")
            return mm(r"$\sqrt{(x_{\mathrm{rec,3D}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec,3D}}-y_{\mathrm{true}})^2}-\sqrt{(x_{\mathrm{rec}}-x_{\mathrm{true}})^2+(y_{\mathrm{rec}}-y_{\mathrm{true}})^2}$")
        return mm("metric")

    # ------------- Color helpers -------------
    def _is_signed_metric(self, sel: int) -> bool:
        m = self.mode
        if m in (Mode.TwoD_Signed, Mode.ThreeD_Signed):
            return True
        if m == Mode.TwoD_vs_Pixel and (sel % 2 == 1):
            return True
        if m == Mode.ThreeD_vs_Pixel and (sel % 2 == 1):
            return True
        if m == Mode.TwoD_vs_ThreeD and (sel % 2 == 1):
            return True
        return False

    def _select_cmap(self, signed: bool) -> mcolors.Colormap:
        choice = getattr(self, "cmap_choice", "auto") or "auto"
        if choice == "auto":
            return plt.get_cmap("coolwarm") if signed else plt.get_cmap("viridis")
        # Prefer modern API (Matplotlib 3.7+), with fallback to pyplot for compatibility
        try:
            if hasattr(mpl, "colormaps") and hasattr(mpl.colormaps, "get_cmap"):
                return mpl.colormaps.get_cmap(choice)
        except Exception:
            pass
        try:
            return plt.get_cmap(choice)
        except Exception:
            return plt.get_cmap("coolwarm") if signed else plt.get_cmap("viridis")

    def _build_norm(self, z_view: np.ndarray, signed: bool, half_pitch: float) -> Tuple[mcolors.Normalize, float, float]:
        # Choose the pool of values to determine limits (view only)
        values = z_view
        # Fallback if empty
        if values is None or values.size == 0:
            vmin = -half_pitch if signed else 0.0
            vmax = half_pitch
            return (mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
                    if signed else mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax)

        # Remove NaNs/inf
        values = values[np.isfinite(values)]
        if values.size == 0:
            vmin = -half_pitch if signed else 0.0
            vmax = half_pitch
            return (mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
                    if signed else mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True), vmin, vmax)

        mode = getattr(self, "scale_mode", "auto")
        # Fixed
        if mode == "fixed":
            # Default fixed range: symmetric for signed, [0, half_pitch] for abs
            if signed:
                vmin, vmax = -half_pitch, half_pitch
                try:
                    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
                except Exception:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                return norm, vmin, vmax
            else:
                vmin, vmax = 0.0, half_pitch
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                return norm, vmin, vmax

        # Robust (percentiles)
        if mode == "robust":
            if signed:
                amax = float(np.percentile(np.abs(values), 98))
                if not np.isfinite(amax) or amax <= 0:
                    amax = half_pitch
                vmin, vmax = -amax, amax
                try:
                    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
                except Exception:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                return norm, vmin, vmax
            else:
                lo = float(np.percentile(values, 2))
                hi = float(np.percentile(values, 98))
                lo = max(0.0, lo)
                if not np.isfinite(hi) or hi <= 0:
                    hi = half_pitch
                norm = mcolors.Normalize(vmin=lo, vmax=hi, clip=True)
                return norm, lo, hi

        # Log (abs only)
        if mode == "log" and not signed:
            pos = values[values > 0]
            if pos.size == 0:
                vmin = 1e-6
                vmax = max(1e-3, half_pitch)
            else:
                vmin = float(np.percentile(pos, 2))
                vmax = float(np.percentile(pos, 98))
                vmin = max(vmin, 1e-9)
                if not np.isfinite(vmax) or vmax <= vmin:
                    vmax = max(1e-3, half_pitch)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            return norm, vmin, vmax

        # SymLog (signed only)
        if mode == "symlog" and signed:
            amax = float(np.max(np.abs(values)))
            if not np.isfinite(amax) or amax <= 0:
                amax = half_pitch
            vmin, vmax = -amax, amax
            linth = 0.07  # 70 micrometers default
            norm = mcolors.SymLogNorm(linthresh=max(1e-6, linth), vmin=vmin, vmax=vmax, base=10)
            return norm, vmin, vmax

        # Auto (default)
        if signed:
            amax = float(np.max(np.abs(values)))
            if not np.isfinite(amax) or amax <= 0:
                amax = half_pitch
            vmin, vmax = -amax, amax
            try:
                norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            except Exception:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            return norm, vmin, vmax
        else:
            vmax = float(np.max(values))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = half_pitch
            vmin = 0.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            return norm, vmin, vmax

    # ------------- Plotting -------------
    def draw_base_scene(self, x_min: float, y_min: float, x_max: float, y_max: float,
                         xC0: float, xC1: float, xC2: float,
                         yC0: float, yC1: float, yC2: float):
        ax = self.canvas.ax
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        # Draw 3x3 pads
        ps = self.geom.pixel_size_mm
        color_fill = (94/255, 113/255, 228/255, 0.35)  # approx Azure+alpha
        color_line = (0/255, 0/255, 180/255, 1.0)
        pad_centers = [xC0, xC1, xC2]
        pad_centers_y = [yC0, yC1, yC2]
        for xc in pad_centers:
            for yc in pad_centers_y:
                x1 = xc - ps/2.0
                y1 = yc - ps/2.0
                rect = mpatches.Rectangle((x1, y1), ps, ps,
                                          facecolor=color_fill, edgecolor=color_line, linewidth=1)
                ax.add_patch(rect)

        # Draw window border
        border = mpatches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                    fill=False, edgecolor="black", linewidth=3)
        ax.add_patch(border)

    def draw_current(self):
        # Keep empty until a file is opened
        if self.file is None:
            try:
                self.canvas.clear()
                self.canvas.draw()
            except Exception:
                pass
            return

        # Geometry
        try:
            self.load_geometry()
        except Exception:
            pass
        if self.geom.num_per_side < 3:
            QtWidgets.QMessageBox.critical(self, "Plot", "Invalid detector geometry (numPerSide<3)")
            return

        half_det = self.geom.det_size_mm / 2.0
        first_center = -half_det + self.geom.pixel_corner_offset_mm + self.geom.pixel_size_mm / 2.0
        half_pitch = 0.5 * self.geom.pixel_spacing_mm

        n = int(self.geom.num_per_side)
        default_i0 = max(0, (n - 3) // 2)
        default_j0 = max(0, (n - 3) // 2)
        i0 = self.i0 if (0 <= self.i0 <= n - 3) else default_i0
        j0 = self.j0 if (0 <= self.j0 <= n - 3) else default_j0
        i1, i2 = i0 + 1, i0 + 2
        j1, j2 = j0 + 1, j0 + 2

        xC0 = first_center + i0 * self.geom.pixel_spacing_mm
        xC1 = first_center + i1 * self.geom.pixel_spacing_mm
        xC2 = first_center + i2 * self.geom.pixel_spacing_mm
        yC0 = first_center + j0 * self.geom.pixel_spacing_mm
        yC1 = first_center + j1 * self.geom.pixel_spacing_mm
        yC2 = first_center + j2 * self.geom.pixel_spacing_mm

        x_min = max(-half_det, xC0 - half_pitch)
        x_max = min( half_det, xC2 + half_pitch)
        y_min = max(-half_det, yC0 - half_pitch)
        y_max = min( half_det, yC2 + half_pitch)

        # Clear canvas and draw base
        self.canvas.clear()
        self.draw_base_scene(x_min, y_min, x_max, y_max, xC0, xC1, xC2, yC0, yC1, yC2)

        # Prepare data
        try:
            hits = self.file["Hits"]
        except Exception:
            hits = None
        if hits is None:
            self.canvas.draw()
            return

        need2D = (self.mode in (Mode.TwoD_Abs, Mode.TwoD_Signed, Mode.TwoD_vs_Pixel, Mode.TwoD3D_Combined, Mode.TwoD_vs_ThreeD))
        need2DS = (self.mode == Mode.TwoD_Signed)
        need3D = (self.mode in (Mode.ThreeD_Abs, Mode.ThreeD_Signed, Mode.ThreeD_vs_Pixel, Mode.TwoD3D_Combined, Mode.TwoD_vs_ThreeD))
        need3DS = (self.mode == Mode.ThreeD_Signed)
        needPix = (self.mode in (Mode.TwoD_vs_Pixel, Mode.ThreeD_vs_Pixel))

        branches = ["TrueX", "TrueY", "isPixelHit"]
        if need2D:
            branches += ["ReconTrueDeltaX", "ReconTrueDeltaY"]
        if need2DS:
            branches += ["ReconTrueDeltaX_Signed", "ReconTrueDeltaY_Signed"]
        if need3D:
            branches += ["ReconTrueDeltaX_3D", "ReconTrueDeltaY_3D"]
        if need3DS:
            branches += ["ReconTrueDeltaX_3D_Signed", "ReconTrueDeltaY_3D_Signed"]
        if needPix:
            branches += ["PixelTrueDeltaX", "PixelTrueDeltaY"]

        try:
            arrs = hits.arrays(branches, library="np")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Plot", f"Failed reading branches: {e}")
            return

        x_hit = arrs.get("TrueX")
        y_hit = arrs.get("TrueY")
        is_pixel_hit = arrs.get("isPixelHit")
        if is_pixel_hit is None:
            is_pixel_hit = np.zeros_like(x_hit, dtype=bool)
        else:
            is_pixel_hit = is_pixel_hit.astype(bool)

        # Build selection mask and, if aggregating, map hits into canonical 3x3 window
        if getattr(self, "aggregate_all", False):
            # Start asynchronous, parallel aggregation to keep UI responsive
            # Cancel any previous worker by marking it cancelled
            try:
                if self._agg_worker is not None:
                    self._agg_worker.cancel()
            except Exception:
                pass
            # Clear last scatter mapping since we won't have clickable points
            self._last_facecolors = None
            self._last_zvals = None
            self._last_norm = None
            self._last_cmap = None

            n = int(self.geom.num_per_side)
            r = max(1, int(self.geom.neighborhood_radius))
            pitch = float(self.geom.pixel_spacing_mm)
            eps = 1e-9  # kept for future use if needed inside worker

            # Precompute metric per hit
            sel = self.metric_combo.currentIndex()
            def arr_or_zero(name: str):
                v = arrs.get(name)
                if v is None:
                    return np.zeros_like(x_hit, dtype=float)
                return v

            if self.mode == Mode.TwoD_Abs:
                dtx2 = np.abs(arr_or_zero("ReconTrueDeltaX"))
                dty2 = np.abs(arr_or_zero("ReconTrueDeltaY"))
                if sel == 0:
                    base_z = dtx2
                elif sel == 1:
                    base_z = dty2
                elif sel == 2:
                    base_z = 0.5 * (dtx2 + dty2)
                elif sel == 3:
                    base_z = np.sqrt(dtx2 * dtx2 + dty2 * dty2)
                else:
                    base_z = np.maximum(dtx2, dty2)
            elif self.mode == Mode.TwoD_Signed:
                dtx2s = arr_or_zero("ReconTrueDeltaX_Signed")
                dty2s = arr_or_zero("ReconTrueDeltaY_Signed")
                base_z = dtx2s if sel == 0 else (dty2s if sel == 1 else 0.5 * (dtx2s + dty2s))
            elif self.mode == Mode.ThreeD_Abs:
                dtx3 = np.abs(arr_or_zero("ReconTrueDeltaX_3D"))
                dty3 = np.abs(arr_or_zero("ReconTrueDeltaY_3D"))
                if sel == 0:
                    base_z = dtx3
                elif sel == 1:
                    base_z = dty3
                elif sel == 2:
                    base_z = 0.5 * (dtx3 + dty3)
                elif sel == 3:
                    base_z = np.sqrt(dtx3 * dtx3 + dty3 * dty3)
                else:
                    base_z = np.maximum(dtx3, dty3)
            elif self.mode == Mode.ThreeD_Signed:
                dtx3s = arr_or_zero("ReconTrueDeltaX_3D_Signed")
                dty3s = arr_or_zero("ReconTrueDeltaY_3D_Signed")
                base_z = dtx3s if sel == 0 else (dty3s if sel == 1 else 0.5 * (dtx3s + dty3s))
            elif self.mode == Mode.TwoD_vs_Pixel:
                dtx2 = np.abs(arr_or_zero("ReconTrueDeltaX"))
                dty2 = np.abs(arr_or_zero("ReconTrueDeltaY"))
                dpx = np.abs(arr_or_zero("PixelTrueDeltaX"))
                dpy = np.abs(arr_or_zero("PixelTrueDeltaY"))
                m2 = 0.5 * (dtx2 + dty2)
                mp = 0.5 * (dpx + dpy)
                if sel == 0:
                    base_z = np.abs(m2 - mp)
                elif sel == 1:
                    base_z = (m2 - mp)
                elif sel == 2:
                    base_z = np.abs(dtx2 - dpx)
                elif sel == 3:
                    base_z = (dtx2 - dpx)
                elif sel == 4:
                    base_z = np.abs(dty2 - dpy)
                elif sel == 5:
                    base_z = (dty2 - dpy)
                elif sel == 6:
                    r2 = np.sqrt(dtx2 * dtx2 + dty2 * dty2)
                    rp = np.sqrt(dpx * dpx + dpy * dpy)
                    base_z = np.abs(r2 - rp)
                else:
                    r2 = np.sqrt(dtx2 * dtx2 + dty2 * dty2)
                    rp = np.sqrt(dpx * dpx + dpy * dpy)
                    base_z = (r2 - rp)
            elif self.mode == Mode.ThreeD_vs_Pixel:
                dtx3 = np.abs(arr_or_zero("ReconTrueDeltaX_3D"))
                dty3 = np.abs(arr_or_zero("ReconTrueDeltaY_3D"))
                dpx = np.abs(arr_or_zero("PixelTrueDeltaX"))
                dpy = np.abs(arr_or_zero("PixelTrueDeltaY"))
                m3 = 0.5 * (dtx3 + dty3)
                mp = 0.5 * (dpx + dpy)
                if sel == 0:
                    base_z = np.abs(m3 - mp)
                elif sel == 1:
                    base_z = (m3 - mp)
                elif sel == 2:
                    base_z = np.abs(dtx3 - dpx)
                elif sel == 3:
                    base_z = (dtx3 - dpx)
                elif sel == 4:
                    base_z = np.abs(dty3 - dpy)
                elif sel == 5:
                    base_z = (dty3 - dpy)
                else:
                    r3 = np.sqrt(dtx3 * dtx3 + dty3 * dty3)
                    rp = np.sqrt(dpx * dpx + dpy * dpy)
                    if sel == 6:
                        base_z = np.abs(r3 - rp)
                    else:
                        base_z = (r3 - rp)
            elif self.mode == Mode.TwoD_vs_ThreeD:
                dtx2 = np.abs(arr_or_zero("ReconTrueDeltaX"))
                dty2 = np.abs(arr_or_zero("ReconTrueDeltaY"))
                dtx3 = np.abs(arr_or_zero("ReconTrueDeltaX_3D"))
                dty3 = np.abs(arr_or_zero("ReconTrueDeltaY_3D"))
                m2 = 0.5 * (dtx2 + dty2)
                m3 = 0.5 * (dtx3 + dty3)
                if sel == 0:
                    base_z = np.abs(m3 - m2)
                elif sel == 1:
                    base_z = (m3 - m2)
                elif sel == 2:
                    base_z = np.abs(dtx3 - dtx2)
                elif sel == 3:
                    base_z = (dtx3 - dtx2)
                elif sel == 4:
                    base_z = np.abs(dty3 - dty2)
                elif sel == 5:
                    base_z = (dty3 - dty2)
                else:
                    r2 = np.sqrt(dtx2 * dtx2 + dty2 * dty2)
                    r3 = np.sqrt(dtx3 * dtx3 + dty3 * dty3)
                    if sel == 6:
                        base_z = np.abs(r3 - r2)
                    else:
                        base_z = (r3 - r2)
            else:  # Mode.TwoD3D_Combined
                dtx2 = np.abs(arr_or_zero("ReconTrueDeltaX"))
                dty2 = np.abs(arr_or_zero("ReconTrueDeltaY"))
                dtx3 = np.abs(arr_or_zero("ReconTrueDeltaX_3D"))
                dty3 = np.abs(arr_or_zero("ReconTrueDeltaY_3D"))
                if sel == 0:
                    base_z = 0.25 * (dtx2 + dty2 + dtx3 + dty3)
                elif sel == 1:
                    base_z = 0.5 * (dtx2 + dtx3)
                elif sel == 2:
                    base_z = 0.5 * (dty2 + dty3)
                else:
                    r2 = np.sqrt(dtx2 * dtx2 + dty2 * dty2)
                    r3 = np.sqrt(dtx3 * dtx3 + dty3 * dty3)
                    base_z = 0.5 * (r2 + r3)

            # Discrete nearest pad indices for each hit
            xi = (x_hit - first_center) / max(pitch, 1e-12)
            yi = (y_hit - first_center) / max(pitch, 1e-12)
            kx0 = np.rint(xi).astype(np.int32)
            ky0 = np.rint(yi).astype(np.int32)

            # Increment sequence and show a status text
            self._agg_seq += 1
            seq = self._agg_seq
            try:
                if self._agg_status_text is not None:
                    self._agg_status_text.remove()
            except Exception:
                pass
            try:
                self._agg_status_text = self.canvas.ax.text(
                    0.5, 1.01, "Computing 3x3 aggregation…", transform=self.canvas.ax.transAxes,
                    ha='center', va='bottom', fontsize=9, color='gray')
                self.canvas.draw()
            except Exception:
                self._agg_status_text = None

            # Start worker thread
            worker = _AggregateWorker(
                x_hit=x_hit,
                y_hit=y_hit,
                is_pixel_hit=is_pixel_hit,
                base_z=base_z,
                kx0=kx0,
                ky0=ky0,
                first_center=float(first_center),
                pitch=float(pitch),
                n=int(n),
                r=int(r),
                xC1=float(xC1),
                yC1=float(yC1),
                sel=int(sel),
                seq=int(seq)
            )
            thread = QtCore.QThread(self)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._on_aggregate_done)
            worker.error.connect(self._on_aggregate_error)
            # Ensure thread cleanup
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            self._agg_worker = worker
            self._agg_thread = thread
            thread.start()
            return
        else:
            mask = np.isfinite(x_hit) & np.isfinite(y_hit)
            mask &= (x_hit >= x_min) & (x_hit <= x_max) & (y_hit >= y_min) & (y_hit <= y_max)
            mask &= (~is_pixel_hit)

            xs = x_hit[mask]
            ys = y_hit[mask]
            # Store mapping from plotted points to event indices for Show fit
            try:
                idxs = np.nonzero(mask)[0]
                self._last_points = np.stack([xs.astype(float), ys.astype(float)], axis=1) if xs.size else np.empty((0, 2), dtype=float)
                self._last_indices = idxs.astype(np.int64)
            except Exception:
                self._last_points = None
                self._last_indices = None

        # Prepare metric values
        zvals = None
        sel = self.metric_combo.currentIndex()
        def safe_get(name: str):
            v = arrs.get(name)
            return v[mask] if v is not None else None

        if self.mode == Mode.TwoD_Abs:
            dtx2 = np.abs(safe_get("ReconTrueDeltaX"))
            dty2 = np.abs(safe_get("ReconTrueDeltaY"))
            if sel == 0:
                zvals = dtx2
            elif sel == 1:
                zvals = dty2
            elif sel == 2:
                zvals = 0.5 * (dtx2 + dty2)
            elif sel == 3:
                zvals = np.sqrt(dtx2 * dtx2 + dty2 * dty2)
            else:
                zvals = np.maximum(dtx2, dty2)
        elif self.mode == Mode.TwoD_Signed:
            dtx2s = safe_get("ReconTrueDeltaX_Signed")
            dty2s = safe_get("ReconTrueDeltaY_Signed")
            if sel == 0:
                zvals = dtx2s
            elif sel == 1:
                zvals = dty2s
            else:
                zvals = 0.5 * (dtx2s + dty2s)
        elif self.mode == Mode.ThreeD_Abs:
            dtx3 = np.abs(safe_get("ReconTrueDeltaX_3D"))
            dty3 = np.abs(safe_get("ReconTrueDeltaY_3D"))
            if sel == 0:
                zvals = dtx3
            elif sel == 1:
                zvals = dty3
            elif sel == 2:
                zvals = 0.5 * (dtx3 + dty3)
            elif sel == 3:
                zvals = np.sqrt(dtx3 * dtx3 + dty3 * dty3)
            else:
                zvals = np.maximum(dtx3, dty3)
        elif self.mode == Mode.ThreeD_Signed:
            dtx3s = safe_get("ReconTrueDeltaX_3D_Signed")
            dty3s = safe_get("ReconTrueDeltaY_3D_Signed")
            if sel == 0:
                zvals = dtx3s
            elif sel == 1:
                zvals = dty3s
            else:
                zvals = 0.5 * (dtx3s + dty3s)
        elif self.mode == Mode.TwoD_vs_Pixel:
            dtx2 = np.abs(safe_get("ReconTrueDeltaX"))
            dty2 = np.abs(safe_get("ReconTrueDeltaY"))
            dpx = np.abs(safe_get("PixelTrueDeltaX"))
            dpy = np.abs(safe_get("PixelTrueDeltaY"))
            m2 = 0.5 * (dtx2 + dty2)
            mp = 0.5 * (dpx + dpy)
            if sel == 0:
                zvals = np.abs(m2 - mp)
            elif sel == 1:
                zvals = (m2 - mp)
            elif sel == 2:
                zvals = np.abs(dtx2 - dpx)
            elif sel == 3:
                zvals = (dtx2 - dpx)
            elif sel == 4:
                zvals = np.abs(dty2 - dpy)
            elif sel == 5:
                zvals = (dty2 - dpy)
            else:
                r2 = np.sqrt(dtx2 * dtx2 + dty2 * dty2)
                rp = np.sqrt(dpx * dpx + dpy * dpy)
                if sel == 6:
                    zvals = np.abs(r2 - rp)
                else:
                    zvals = (r2 - rp)
        elif self.mode == Mode.ThreeD_vs_Pixel:
            dtx3 = np.abs(safe_get("ReconTrueDeltaX_3D"))
            dty3 = np.abs(safe_get("ReconTrueDeltaY_3D"))
            dpx = np.abs(safe_get("PixelTrueDeltaX"))
            dpy = np.abs(safe_get("PixelTrueDeltaY"))
            m3 = 0.5 * (dtx3 + dty3)
            mp = 0.5 * (dpx + dpy)
            if sel == 0:
                zvals = np.abs(m3 - mp)
            elif sel == 1:
                zvals = (m3 - mp)
            elif sel == 2:
                zvals = np.abs(dtx3 - dpx)
            elif sel == 3:
                zvals = (dtx3 - dpx)
            elif sel == 4:
                zvals = np.abs(dty3 - dpy)
            elif sel == 5:
                zvals = (dty3 - dpy)
            else:
                r3 = np.sqrt(dtx3 * dtx3 + dty3 * dty3)
                rp = np.sqrt(dpx * dpx + dpy * dpy)
                if sel == 6:
                    zvals = np.abs(r3 - rp)
                else:
                    zvals = (r3 - rp)
        elif self.mode == Mode.TwoD3D_Combined:
            dtx2 = np.abs(safe_get("ReconTrueDeltaX"))
            dty2 = np.abs(safe_get("ReconTrueDeltaY"))
            dtx3 = np.abs(safe_get("ReconTrueDeltaX_3D"))
            dty3 = np.abs(safe_get("ReconTrueDeltaY_3D"))
            if sel == 0:
                zvals = 0.25 * (dtx2 + dty2 + dtx3 + dty3)
            elif sel == 1:
                zvals = 0.5 * (dtx2 + dtx3)
            elif sel == 2:
                zvals = 0.5 * (dty2 + dty3)
            else:
                r2 = np.sqrt(dtx2 * dtx2 + dty2 * dty2)
                r3 = np.sqrt(dtx3 * dtx3 + dty3 * dty3)
                zvals = 0.5 * (r2 + r3)
        elif self.mode == Mode.TwoD_vs_ThreeD:
            dtx2 = np.abs(safe_get("ReconTrueDeltaX"))
            dty2 = np.abs(safe_get("ReconTrueDeltaY"))
            dtx3 = np.abs(safe_get("ReconTrueDeltaX_3D"))
            dty3 = np.abs(safe_get("ReconTrueDeltaY_3D"))
            m2 = 0.5 * (dtx2 + dty2)
            m3 = 0.5 * (dtx3 + dty3)
            if sel == 0:
                zvals = np.abs(m3 - m2)
            elif sel == 1:
                zvals = (m3 - m2)
            elif sel == 2:
                zvals = np.abs(dtx3 - dtx2)
            elif sel == 3:
                zvals = (dtx3 - dtx2)
            elif sel == 4:
                zvals = np.abs(dty3 - dty2)
            elif sel == 5:
                zvals = (dty3 - dty2)
            else:
                r2 = np.sqrt(dtx2 * dtx2 + dty2 * dty2)
                r3 = np.sqrt(dtx3 * dtx3 + dty3 * dty3)
                if sel == 6:
                    zvals = np.abs(r3 - r2)
                else:
                    zvals = (r3 - r2)

        if zvals is None or len(zvals) == 0:
            zvals = np.array([], dtype=float)

        # Determine norm and colormap
        signed_range = self._is_signed_metric(sel)
        norm, zmin, zmax = self._build_norm(zvals, signed_range, half_pitch)
        cmap = self._select_cmap(signed_range)

        # Draw colored circles
        ax = self.canvas.ax

        # draw a compact scatter for performance, approximate radius
        # Approximate marker size so diameter ~ pixel_size_mm/3 in data units
        # Convert data span to pixels; assume 100 dpi and figure size
        try:
            bbox = self.canvas.fig.get_window_extent().transformed(self.canvas.fig.dpi_scale_trans.inverted())
            fig_w_px = bbox.width * self.canvas.fig.dpi
            fig_h_px = bbox.height * self.canvas.fig.dpi
            ax_bbox = ax.get_position()
            ax_w_px = (ax_bbox.width) * fig_w_px
            ax_h_px = (ax_bbox.height) * fig_h_px
            mm_per_px_x = (x_max - x_min) / max(ax_w_px, 1.0)
            # desired radius in pixels ~ (pixel_size_mm/6) / mm_per_px_x
            r_px = (self.geom.pixel_size_mm / 6.0) / max(mm_per_px_x, 1e-6)
            size_pts2 = max(10.0, (2.0 * r_px) ** 2)
        except Exception:
            size_pts2 = 36.0

        # Ensure previous colorbar is removed before adding a new one
        try:
            if self.canvas.cbar is not None:
                self.canvas.cbar.remove()
        except Exception:
            pass
        self.canvas.cbar = None

        sc = ax.scatter(xs, ys, c=zvals, cmap=cmap, norm=norm, s=size_pts2, marker='o', linewidths=0)
        cbar = self.canvas.fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        self.canvas.cbar = cbar

        # Colorbar title
        cbar.set_label(self._get_colorbar_label(sel))

        self.canvas.ax.tick_params(direction='in')
        self.canvas.draw()

        # Cache colors and mapping for Show fit color replication
        try:
            self._last_facecolors = sc.get_facecolors()
        except Exception:
            self._last_facecolors = None
        try:
            self._last_zvals = np.asarray(zvals) if zvals is not None else None
        except Exception:
            self._last_zvals = None
        try:
            self._last_norm = norm
            self._last_cmap = cmap
        except Exception:
            self._last_norm = None
            self._last_cmap = None

        # Enable/disable Show fit button based on availability
        try:
            can_show = (not getattr(self, 'aggregate_all', False)) and (self._last_points is not None) and (self._last_points.shape[0] > 0)
            self.show_fit_btn.setEnabled(bool(can_show))
        except Exception:
            pass


    @QtCore.pyqtSlot(int, object, object, object)
    def _on_aggregate_done(self, seq: int, xs_obj, ys_obj, zvals_obj):
        # Ignore stale results
        if seq != self._agg_seq:
            return
        # Remove status label if present
        try:
            if self._agg_status_text is not None:
                self._agg_status_text.remove()
                self._agg_status_text = None
        except Exception:
            self._agg_status_text = None

        try:
            x_edges = np.asarray(xs_obj)
            y_edges = np.asarray(ys_obj)
            zgrid = np.asarray(zvals_obj)
        except Exception:
            return

        # Determine norm/cmap from finite grid values
        half_pitch = 0.5 * self.geom.pixel_spacing_mm
        sel = self.metric_combo.currentIndex()
        signed_range = self._is_signed_metric(sel)
        finite_vals = zgrid[np.isfinite(zgrid)] if isinstance(zgrid, np.ndarray) else np.array([])
        norm, _, _ = self._build_norm(finite_vals, signed_range, half_pitch)
        cmap = self._select_cmap(signed_range)

        ax = self.canvas.ax

        try:
            if self.canvas.cbar is not None:
                self.canvas.cbar.remove()
        except Exception:
            pass
        self.canvas.cbar = None

        mesh = ax.pcolormesh(x_edges, y_edges, zgrid, cmap=cmap, norm=norm, shading='auto')
        cbar = self.canvas.fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        self.canvas.cbar = cbar
        cbar.set_label(self._get_colorbar_label(sel))
        self.canvas.ax.tick_params(direction='in')
        self.canvas.draw()

    @QtCore.pyqtSlot(int, str)
    def _on_aggregate_error(self, seq: int, msg: str):
        if seq != self._agg_seq:
            return
        try:
            if self._agg_status_text is not None:
                self._agg_status_text.remove()
                self._agg_status_text = None
        except Exception:
            pass
        QtWidgets.QMessageBox.critical(self, "Aggregate 3x3", f"Aggregation failed: {msg}")

    # ------------- Show-fit internals -------------
    def _on_canvas_click(self, event):
        if not self.show_fit_active:
            return
        try:
            if event is None or event.inaxes is not self.canvas.ax:
                return
            if self._last_points is None or self._last_indices is None or self._last_points.shape[0] == 0:
                QtWidgets.QMessageBox.information(self, "Show fit", "No selectable points in the current view.")
                return
            if event.xdata is None or event.ydata is None:
                return
            click_xy = np.array([float(event.xdata), float(event.ydata)], dtype=float)
            pts = self._last_points
            d2 = np.sum((pts - click_xy[None, :])**2, axis=1)
            idx_local = int(np.argmin(d2))
            evt_idx = int(self._last_indices[idx_local])
            # Determine the displayed color for the clicked point
            color_hex = None
            try:
                rgba = None
                if self._last_facecolors is not None and len(self._last_facecolors) > idx_local:
                    rgba = self._last_facecolors[idx_local]
                if rgba is None:
                    zvals = getattr(self, "_last_zvals", None)
                    norm = getattr(self, "_last_norm", None)
                    cmap = getattr(self, "_last_cmap", None)
                    if zvals is not None and norm is not None and cmap is not None and 0 <= idx_local < zvals.shape[0]:
                        rgba = cmap(norm(zvals[idx_local]))
                if rgba is not None:
                    color_hex = mcolors.to_hex(rgba, keep_alpha=False)
            except Exception:
                color_hex = None
            self._show_fit_for_event(evt_idx, color_hex=color_hex)
        except Exception as e:
            try:
                QtWidgets.QMessageBox.critical(self, "Show fit", f"Failed to show fit: {e}")
            except Exception:
                pass

    def _read_event(self, evt_idx: int):
        if self.file is None:
            return None
        try:
            hits = self.file["Hits"]
        except Exception:
            return None
        try:
            arrs = hits.arrays(["TrueX","TrueY","PixelX","PixelY","isPixelHit","F_i"],
                               entry_start=int(evt_idx), entry_stop=int(evt_idx)+1,
                               library="ak")
        except Exception:
            return None
        try:
            x_true = float(arrs["TrueX"][0])
            y_true = float(arrs["TrueY"][0])
            x_px = float(arrs["PixelX"][0])
            y_px = float(arrs["PixelY"][0])
            is_pixel = bool(arrs["isPixelHit"][0])
            Fi = np.asarray(arrs["F_i"][0], dtype=float)
        except Exception:
            return None
        if Fi is None or Fi.size == 0:
            return None
        Nf = int(round(float(np.sqrt(float(Fi.size)))))
        if Nf * Nf != Fi.size:
            return None
        return {
            "x_true": x_true,
            "y_true": y_true,
            "x_px": x_px,
            "y_px": y_px,
            "is_pixel": is_pixel,
            "Fi": Fi,
            "N": Nf,
            "R": (Nf - 1)//2,
        }

    def _show_fit_for_event(self, evt_idx: int, color_hex: Optional[str] = None):
        if getattr(self, "aggregate_all", False):
            QtWidgets.QMessageBox.information(self, "Show fit", "Disable aggregation to inspect individual events.")
            return
        try:
            self.load_geometry()
        except Exception:
            pass
        if self.geom.pixel_spacing_mm <= 0 or self.geom.pixel_size_mm <= 0:
            QtWidgets.QMessageBox.warning(self, "Show fit", "Invalid geometry (pixel spacing/size).")
            return
        ev = self._read_event(evt_idx)
        if ev is None:
            QtWidgets.QMessageBox.warning(self, "Show fit", f"Failed to read event #{evt_idx} or missing neighborhood.")
            return
        if ev["is_pixel"]:
            QtWidgets.QMessageBox.information(self, "Show fit", "Selected event is a pixel hit; non-pixel required.")
            return

        # Geometry and neighborhood
        N = int(ev["N"])
        R = int(ev["R"])
        Fi = ev["Fi"].reshape((N, N))
        x_px = ev["x_px"]
        y_px = ev["y_px"]
        x_true = ev["x_true"]
        y_true = ev["y_true"]
        pitch = float(self.geom.pixel_spacing_mm)
        psize = float(self.geom.pixel_size_mm)

        di_vals = np.arange(-R, R+1, dtype=int)
        dj_vals = np.arange(-R, R+1, dtype=int)
        # Central row (vary x at y=y_px -> dj=0)
        x_row = x_px + di_vals * pitch
        q_row = np.array([Fi[di+R, 0+R] for di in di_vals], dtype=float)
        # Central column (vary y at x=x_px -> di=0)
        y_col = y_px + dj_vals * pitch
        q_col = np.array([Fi[0+R, dj+R] for dj in dj_vals], dtype=float)

        # Baseline and weights (match ROOT macros: clamp baseline >= 0)
        B0_row = float(max(0.0, float(np.nanmin(q_row)))) if np.isfinite(q_row).any() else 0.0
        B0_col = float(max(0.0, float(np.nanmin(q_col)))) if np.isfinite(q_col).any() else 0.0
        w_row = np.clip(q_row - B0_row, 0.0, None)
        w_col = np.clip(q_col - B0_col, 0.0, None)

        def weighted_centroid(x, w):
            s = float(np.sum(w))
            return float(np.sum(w * x) / s) if s > 0 else float('nan')

        def weighted_sigma(x, w, mean):
            s = float(np.sum(w))
            if s <= 0 or not np.isfinite(mean):
                return max(0.25*pitch, 1e-6)
            var = float(np.sum(w * (x - mean)**2) / s)
            sig = float(np.sqrt(max(var, 1e-12)))
            sig = max(sig, max(1e-6, 0.02*pitch))
            sig = min(sig, 3.0*pitch)
            return sig

        mu_row = weighted_centroid(x_row, w_row)
        mu_col = weighted_centroid(y_col, w_col)
        sig_row = weighted_sigma(x_row, w_row, mu_row)
        sig_col = weighted_sigma(y_col, w_col, mu_col)
        A_row = float(max(1e-18, float(np.nanmax(q_row)) - B0_row))
        A_col = float(max(1e-18, float(np.nanmax(q_col)) - B0_col))

        # 3D-like parameters from moments
        xs_grid = np.array([x_px + di*pitch for di in di_vals], dtype=float)
        ys_grid = np.array([y_px + dj*pitch for dj in dj_vals], dtype=float)
        Xg, Yg = np.meshgrid(xs_grid, ys_grid, indexing='ij')
        B0 = float(np.nanmin(Fi))
        W = np.clip(Fi - B0, 0.0, None)
        wsum = float(np.sum(W))
        if wsum > 0:
            mux = float(np.sum(W * Xg) / wsum)
            muy = float(np.sum(W * Yg) / wsum)
            varx = float(np.sum(W * (Xg - mux)**2) / wsum)
            vary = float(np.sum(W * (Yg - muy)**2) / wsum)
            sigx = float(np.sqrt(max(varx, 1e-12)))
            sigy = float(np.sqrt(max(vary, 1e-12)))
            sigx = max(sigx, max(1e-6, 0.02*pitch)); sigx = min(sigx, 3.0*pitch)
            sigy = max(sigy, max(1e-6, 0.02*pitch)); sigy = min(sigy, 3.0*pitch)
        else:
            mux = mu_row; muy = mu_col; sigx = sig_row; sigy = sig_col
        A = float(max(1e-18, float(np.nanmax(Fi)) - B0))

        # Dialog with tabs for 2D/3D visuals
        dlg = QtWidgets.QDialog(self)
        try:
            title = f"Event {evt_idx} fits"
            if color_hex:
                title = f"{title} — {color_hex}"
            dlg.setWindowTitle(title)
        except Exception:
            dlg.setWindowTitle(f"Event {evt_idx} fits")
        dlg.resize(1100, 900)
        lay = QtWidgets.QVBoxLayout(dlg)
        tabs = QtWidgets.QTabWidget(dlg)
        lay.addWidget(tabs)

        # 2D tab
        w2d = QtWidgets.QWidget()
        l2d = QtWidgets.QVBoxLayout(w2d)
        fig2d = Figure(figsize=(10, 4.0), dpi=100)
        can2d = FigureCanvas(fig2d)
        l2d.addWidget(can2d)
        axC = fig2d.add_subplot(1, 2, 1)
        axR = fig2d.add_subplot(1, 2, 2)

        try:
            axC.set_title("Central column; y [mm]; F_i", color=(color_hex or None))
        except Exception:
            axC.set_title("Central column; y [mm]; F_i")
        axC.scatter(y_col, q_col, s=36, color='tab:blue')
        yMin = float(np.min(y_col) - 0.5*pitch)
        yMax = float(np.max(y_col) + 0.5*pitch)
        yy = np.linspace(yMin, yMax, 600)
        def gauss1d(x, A_, mu_, s_, B_):
            return A_ * np.exp(-0.5 * ((x - mu_) / max(s_, 1e-9))**2) + B_
        # Headroom and y-limits like macros
        try:
            dataMaxCol = float(np.nanmax(q_col)) if np.isfinite(q_col).any() else 1.0
        except Exception:
            dataMaxCol = 1.0
        yMaxCol_auto = 1.20 * max(dataMaxCol, A_col + B0_col)
        axC.set_ylim(0.0, yMaxCol_auto)
        # Fit curve in red (match macros)
        axC.plot(yy, gauss1d(yy, A_col, mu_col, sig_col, B0_col), color='r', lw=2, zorder=3)
        # Pixel-width boxes
        if np.isfinite(q_col).any():
            yPadMin = float(np.nanmin(q_col)); yPadMax = float(np.nanmax(q_col))
            halfHc = 0.015 * max(1e-9, (yPadMax - yPadMin))
            for (yc, qc) in zip(y_col, q_col):
                axC.add_patch(mpatches.Rectangle((yc - 0.5*psize, qc - halfHc), psize, 2*halfHc,
                                                 fill=False, edgecolor='gray', linewidth=1.0))
        # Draw vertical lines last with high zorder to ensure visibility
        axC.axvline(y_true, color='k', ls='--', lw=1.8, label='y_true', zorder=10)
        axC.axvline(mu_col, color='r', ls='--', lw=1.8, label='y_rec', zorder=10)
        axC.legend(loc='upper left', fontsize=8)

        try:
            axR.set_title("Central row; x [mm]; F_i", color=(color_hex or None))
        except Exception:
            axR.set_title("Central row; x [mm]; F_i")
        axR.scatter(x_row, q_row, s=36, color='tab:blue')
        xMin = float(np.min(x_row) - 0.5*pitch)
        xMax = float(np.max(x_row) + 0.5*pitch)
        xx = np.linspace(xMin, xMax, 600)
        # Headroom and y-limits for row
        try:
            dataMaxRow = float(np.nanmax(q_row)) if np.isfinite(q_row).any() else 1.0
        except Exception:
            dataMaxRow = 1.0
        yMaxRow_auto = 1.20 * max(dataMaxRow, A_row + B0_row)
        axR.set_ylim(0.0, yMaxRow_auto)
        # Fit curve in red (match macros)
        axR.plot(xx, gauss1d(xx, A_row, mu_row, sig_row, B0_row), color='r', lw=2, zorder=3)
        if np.isfinite(q_row).any():
            yPadMin = float(np.nanmin(q_row)); yPadMax = float(np.nanmax(q_row))
            halfH = 0.015 * max(1e-9, (yPadMax - yPadMin))
            for (xc, qc) in zip(x_row, q_row):
                axR.add_patch(mpatches.Rectangle((xc - 0.5*psize, qc - halfH), psize, 2*halfH,
                                                 fill=False, edgecolor='gray', linewidth=1.0))
        axR.axvline(x_true, color='k', ls='--', lw=1.8, label='x_true', zorder=10)
        axR.axvline(mu_row, color='r', ls='--', lw=1.8, label='x_rec', zorder=10)
        axR.legend(loc='upper left', fontsize=8)
        fig2d.tight_layout()
        can2d.draw()
        tabs.addTab(w2d, "2D")

        # 3D tab
        w3d = QtWidgets.QWidget()
        l3d = QtWidgets.QVBoxLayout(w3d)
        fig3d = Figure(figsize=(10, 8.0), dpi=100)
        can3d = FigureCanvas(fig3d)
        l3d.addWidget(can3d)
        ax11 = fig3d.add_subplot(2, 2, 1)
        ax12 = fig3d.add_subplot(2, 2, 2)
        ax21 = fig3d.add_subplot(2, 2, 3)
        ax22 = fig3d.add_subplot(2, 2, 4)

        xLo = x_px - (R + 0.5) * pitch
        xHi = x_px + (R + 0.5) * pitch
        yLo = y_px - (R + 0.5) * pitch
        yHi = y_px + (R + 0.5) * pitch
        im = ax11.imshow(Fi.T, origin='lower', extent=[xLo, xHi, yLo, yHi], aspect='equal', cmap='viridis')
        fig3d.colorbar(im, ax=ax11, fraction=0.046, pad=0.04)
        try:
            ax11.set_title("Data with fit contours; F_i", color=(color_hex or None))
        except Exception:
            ax11.set_title("Data with fit contours; F_i")
        # Model contours
        # Build model grid aligned with Fi shape (N x N)
        cx = np.linspace(xLo, xHi, N)
        cy = np.linspace(yLo, yHi, N)
        CX, CY = np.meshgrid(cx, cy, indexing='xy')
        Zm = A * np.exp(
            -0.5 * (
                ((CX - mux) / max(sigx, 1e-9))**2 +
                ((CY - muy) / max(sigy, 1e-9))**2
            )
        ) + B0
        # Match macro styling: red contours with width 2
        ax11.contour(CX, CY, Zm, levels=10, colors='r', linewidths=2.0)
        ax11.plot([x_true], [y_true], marker='X', color='k', ms=8, label='true')
        ax11.plot([x_px], [y_px], marker='o', color='tab:blue', ms=6, label='pixel')
        ax11.plot([mux], [muy], marker='*', color='r', ms=10, label='rec')
        ax11.legend(loc='upper left', fontsize=8)

        # Residuals
        Zres = Fi - Zm
        rmax = float(np.max(np.abs(Zres))) if np.isfinite(Zres).any() else 1.0
        im2 = ax12.imshow(Zres.T, origin='lower', extent=[xLo, xHi, yLo, yHi], aspect='equal', cmap='coolwarm', vmin=-rmax, vmax=rmax)
        fig3d.colorbar(im2, ax=ax12, fraction=0.046, pad=0.04)
        try:
            ax12.set_title("Residuals (data - model)", color=(color_hex or None))
        except Exception:
            ax12.set_title("Residuals (data - model)")

        # Row profile
        try:
            ax21.set_title("Row (y=y_px)", color=(color_hex or None))
        except Exception:
            ax21.set_title("Row (y=y_px)")
        ax21.scatter(x_row, q_row, s=24, color='tab:blue')
        xx = np.linspace(xLo, xHi, 600)
        Arow0 = A * np.exp(-0.5 * ((0.0 - muy)/max(sigy,1e-9))**2)
        ax21.plot(xx, gauss1d(xx, Arow0, mux, sigx, B0), color='r', lw=1.5, ls='--', label='slice @ y=0')
        ax21.axvline(x_true, color='k', ls='--', lw=1.0)
        ax21.axvline(mux, color='r', ls='--', lw=1.0)
        ax21.legend(loc='upper right', fontsize=8)

        # Column profile
        try:
            ax22.set_title("Column (x=x_px)", color=(color_hex or None))
        except Exception:
            ax22.set_title("Column (x=x_px)")
        ax22.scatter(y_col, q_col, s=24, color='tab:blue')
        yy = np.linspace(yLo, yHi, 600)
        Acol0 = A * np.exp(-0.5 * ((0.0 - mux)/max(sigx,1e-9))**2)
        ax22.plot(yy, gauss1d(yy, Acol0, muy, sigy, B0), color='r', lw=1.5, ls='--', label='slice @ x=0')
        ax22.axvline(y_true, color='k', ls='--', lw=1.0)
        ax22.axvline(muy, color='r', ls='--', lw=1.0)
        ax22.legend(loc='upper right', fontsize=8)

        fig3d.tight_layout()
        can3d.draw()
        tabs.addTab(w3d, "3D")

        # Keep a reference so it persists
        try:
            self._fit_dialogs.append(dlg)
        except Exception:
            pass
        dlg.show()

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = PyColorGUI()
    gui.show()

    # CLI arg: optional ROOT file path
    if len(sys.argv) > 1 and sys.argv[1].strip():
        gui.file_edit.setText(sys.argv[1].strip())
        gui.on_open()
    else:
        # Default ROOT path (if present)
        default_root = "/home/tom/Desktop/Putza/epicChargeSharing/build/epicChargeSharing.root"
        try:
            if os.path.exists(default_root):
                gui.file_edit.setText(default_root)
                gui.on_open()
        except Exception:
            pass

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


