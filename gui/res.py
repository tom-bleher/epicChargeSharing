#!/usr/bin/env python3
import sys
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import uproot


@dataclass
class Geometry:
    pixel_size_mm: float = 0.0
    pixel_spacing_mm: float = 0.0
    pixel_corner_offset_mm: float = 0.0
    det_size_mm: float = 0.0
    num_per_side: int = 0


def read_named_double(file: uproot.ReadOnlyFile, key: str) -> float:
    obj = file.get(key)
    if obj is None:
        raise RuntimeError(f"Missing metadata object: '{key}'")
    try:
        title = obj.member("fTitle")
    except Exception:
        title = getattr(obj, "fTitle", None)
        if title is None:
            title = str(obj)
    if title is None:
        raise RuntimeError(f"TNamed '{key}' has empty title")
    return float(title)


def read_named_int(file: uproot.ReadOnlyFile, key: str) -> int:
    return int(round(read_named_double(file, key)))


class MatplotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 6), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.subplots_adjust(left=0.12, right=0.86, top=0.92, bottom=0.12)

    def clear(self):
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")


class ResolutionGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("epicChargeSharing: Resolution Strip")
        self.resize(1200, 900)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # File controls
        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(QtWidgets.QLabel("ROOT file:"))
        self.file_edit = QtWidgets.QLineEdit()
        file_row.addWidget(self.file_edit, 1)
        self.browse_btn = QtWidgets.QPushButton("Browse…")
        self.open_btn = QtWidgets.QPushButton("Open")
        file_row.addWidget(self.browse_btn)
        file_row.addWidget(self.open_btn)
        vbox.addLayout(file_row)

        # Controls
        ctr = QtWidgets.QHBoxLayout()
        ctr.addWidget(QtWidgets.QLabel("Orientation:"))
        self.orient_combo = QtWidgets.QComboBox()
        self.orient_combo.addItem("Row (horizontal strip)", 0)
        self.orient_combo.addItem("Column (vertical strip)", 1)
        ctr.addWidget(self.orient_combo)

        ctr.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("2D abs", 0)
        self.mode_combo.addItem("3D abs", 1)
        self.mode_combo.addItem("2D signed", 2)
        self.mode_combo.addItem("3D signed", 3)
        ctr.addWidget(self.mode_combo)

        ctr.addWidget(QtWidgets.QLabel("Snap:"))
        self.snap_combo = QtWidgets.QComboBox()
        self.snap_combo.addItem("Pixel interiors", 0)
        self.snap_combo.addItem("Gaps between pixels", 1)
        self.snap_combo.addItem("Gap walls (near pad edges)", 2)
        ctr.addWidget(self.snap_combo)

        ctr.addWidget(QtWidgets.QLabel("Index:"))
        self.snap_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.snap_slider.setMinimum(0)
        self.snap_slider.setMaximum(2)  # default for interiors (3 positions)
        self.snap_slider.setSingleStep(1)
        self.snap_slider.setPageStep(1)
        self.snap_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.snap_slider.setTickInterval(1)
        ctr.addWidget(self.snap_slider, 1)

        # Aggregate option
        self.aggregate_check = QtWidgets.QCheckBox("All indices")
        ctr.addWidget(self.aggregate_check)
        # Aggregate across all 3x3 windows (canonical mapping)
        self.aggregate_all_cb = QtWidgets.QCheckBox("Aggregate all 3x3s")
        ctr.addWidget(self.aggregate_all_cb)

        ctr.addWidget(QtWidgets.QLabel("Band size [mm]:"))
        self.band_spin = QtWidgets.QDoubleSpinBox()
        self.band_spin.setDecimals(3)
        self.band_spin.setRange(0.01, 5.0)
        self.band_spin.setSingleStep(0.01)
        self.band_spin.setValue(0.20)
        ctr.addWidget(self.band_spin)

        # i0/j0 removed from GUI; selection can be randomized with the button

        self.rand_btn = QtWidgets.QPushButton("Random 3x3")
        self.plot_btn = QtWidgets.QPushButton("Plot")
        self.preview_btn = QtWidgets.QPushButton("Preview")
        ctr.addWidget(self.rand_btn)
        ctr.addWidget(self.plot_btn)
        ctr.addWidget(self.preview_btn)
        vbox.addLayout(ctr)

        # Canvas
        self.canvas = MatplotCanvas()
        vbox.addWidget(self.canvas, 1)

        # State
        self.file: Optional[uproot.ReadOnlyFile] = None
        self.file_path: str = ""
        self.geom = Geometry()
        self.hasRecon2D = False
        self.hasRecon3D = False
        self.hasRecon2DSigned = False
        self.hasRecon3DSigned = False
        self.i0: int = 0
        self.j0: int = 0
        self.aggregate_all: bool = False

        # Signals
        self.browse_btn.clicked.connect(self.on_browse)
        self.open_btn.clicked.connect(self.on_open)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.orient_combo.currentIndexChanged.connect(self.on_orient_changed)
        self.snap_combo.currentIndexChanged.connect(self.on_snap_changed)
        self.snap_slider.valueChanged.connect(self.on_snap_index_changed)
        self.band_spin.valueChanged.connect(self.on_band_changed)
        self.aggregate_check.toggled.connect(self.on_aggregate_toggled)
        self.aggregate_all_cb.toggled.connect(self.on_aggregate_all_toggled)
        self.rand_btn.clicked.connect(self.on_random)
        self.plot_btn.clicked.connect(self.on_plot)
        self.preview_btn.clicked.connect(self.on_preview)

        # Start blank until a file is selected by the user
        self._clamp_i0j0()
        self._update_snap_slider()
        self.draw_current()

    # -------- File / geometry --------
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
        self.detect_branch_availability()
        try:
            self.load_geometry()
        except Exception:
            pass
        # center default 3x3 after loading geometry
        self._set_default_center_indices()
        self._clamp_i0j0()
        self._update_snap_slider()
        self.draw_current()

    def detect_branch_availability(self):
        self.hasRecon2D = False
        self.hasRecon3D = False
        try:
            hits = self.file["Hits"] if self.file is not None else None
        except Exception:
            hits = None
        if hits is None:
            return
        branches = set(hits.keys())
        self.hasRecon2D = ("ReconTrueDeltaX" in branches) and ("ReconTrueDeltaY" in branches)
        self.hasRecon3D = ("ReconTrueDeltaX_3D" in branches) and ("ReconTrueDeltaY_3D" in branches)
        self.hasRecon2DSigned = ("ReconTrueDeltaX_Signed" in branches) and ("ReconTrueDeltaY_Signed" in branches)
        self.hasRecon3DSigned = ("ReconTrueDeltaX_3D_Signed" in branches) and ("ReconTrueDeltaY_3D_Signed" in branches)
        self._update_mode_availability()

    def load_geometry(self):
        if self.file is None:
            return
        self.geom.pixel_size_mm = read_named_double(self.file, "GridPixelSize_mm")
        self.geom.pixel_spacing_mm = read_named_double(self.file, "GridPixelSpacing_mm")
        self.geom.pixel_corner_offset_mm = read_named_double(self.file, "GridPixelCornerOffset_mm")
        self.geom.det_size_mm = read_named_double(self.file, "GridDetectorSize_mm")
        self.geom.num_per_side = read_named_int(self.file, "GridNumBlocksPerSide")

    # -------- UI change handlers --------
    def on_mode_changed(self, idx: int):
        self.draw_current()

    def on_orient_changed(self, idx: int):
        self._update_snap_slider()
        self.draw_current()

    def on_snap_changed(self, idx: int):
        self._update_snap_slider()
        self.draw_current()

    def on_snap_index_changed(self, idx: int):
        self.draw_current()

    def on_band_changed(self, v: float):
        self.draw_current()

    # i0/j0 GUI handlers removed

    def on_aggregate_toggled(self, checked: bool):
        # Disable index slider when aggregating across all indices
        self.snap_slider.setEnabled(not checked)
        self.draw_current()

    def on_aggregate_all_toggled(self, checked: bool):
        # Disable Random 3x3 when aggregating across all windows
        try:
            self.aggregate_all = bool(checked)
            self.rand_btn.setEnabled(not self.aggregate_all)
        except Exception:
            self.aggregate_all = bool(checked)
        self.draw_current()

    def on_random(self):
        if self.geom.num_per_side < 3:
            return
        if getattr(self, 'aggregate_all', False):
            return
        max_i = max(0, self.geom.num_per_side - 3)
        max_j = max(0, self.geom.num_per_side - 3)
        self.i0 = np.random.randint(0, max_i + 1)
        self.j0 = np.random.randint(0, max_j + 1)
        self.draw_current()

    # -------- Helpers --------
    def _clamp_i0j0(self):
        try:
            n = int(self.geom.num_per_side)
        except Exception:
            n = 0
        hi = max(0, n - 3)
        if self.i0 < 0:
            self.i0 = 0
        if self.j0 < 0:
            self.j0 = 0
        if self.i0 > hi:
            self.i0 = hi
        if self.j0 > hi:
            self.j0 = hi

    def _set_default_center_indices(self):
        try:
            n = int(self.geom.num_per_side)
        except Exception:
            n = 0
        if n >= 3:
            self.i0 = max(0, (n - 3) // 2)
            self.j0 = max(0, (n - 3) // 2)

    def _update_snap_slider(self):
        # Interiors -> 3 positions; Gaps -> 2 positions; Walls -> 2 positions
        mode = int(self.snap_combo.currentData())
        self.snap_slider.blockSignals(True)
        self.snap_slider.setMaximum(1 if mode in (1, 2) else 2)
        if self.snap_slider.value() > self.snap_slider.maximum():
            self.snap_slider.setValue(self.snap_slider.maximum())
        self.snap_slider.blockSignals(False)

    def _update_mode_availability(self):
        # Rebuild mode options to show only valid ones for the opened file
        items = []
        if self.hasRecon2D:
            items.append(("2D abs", 0))
        if self.hasRecon3D:
            items.append(("3D abs", 1))
        if self.hasRecon2DSigned:
            items.append(("2D signed", 2))
        if self.hasRecon3DSigned:
            items.append(("3D signed", 3))
        if not items:
            items = [("2D abs", 0)]
        cur = self.mode_combo.currentData()
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        for label, val in items:
            self.mode_combo.addItem(label, val)
        # Keep current selection if still present; otherwise select first
        idx = next((i for i in range(self.mode_combo.count()) if self.mode_combo.itemData(i) == cur), 0)
        self.mode_combo.setCurrentIndex(max(0, idx))
        self.mode_combo.blockSignals(False)

    def _centers_3x3(self) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
        half_det = self.geom.det_size_mm / 2.0
        first_center = -half_det + self.geom.pixel_corner_offset_mm + self.geom.pixel_size_mm / 2.0
        i0 = max(0, min(self.i0, self.geom.num_per_side - 3))
        j0 = max(0, min(self.j0, self.geom.num_per_side - 3))
        i1, i2 = i0 + 1, i0 + 2
        j1, j2 = j0 + 1, j0 + 2
        xC0 = first_center + i0 * self.geom.pixel_spacing_mm
        xC1 = first_center + i1 * self.geom.pixel_spacing_mm
        xC2 = first_center + i2 * self.geom.pixel_spacing_mm
        yC0 = first_center + j0 * self.geom.pixel_spacing_mm
        yC1 = first_center + j1 * self.geom.pixel_spacing_mm
        yC2 = first_center + j2 * self.geom.pixel_spacing_mm
        half_pitch = 0.5 * self.geom.pixel_spacing_mm
        x_min = max(-half_det, xC0 - half_pitch)
        x_max = min( half_det, xC2 + half_pitch)
        y_min = max(-half_det, yC0 - half_pitch)
        y_max = min( half_det, yC2 + half_pitch)
        return xC0, xC1, xC2, yC0, yC1, yC2, x_min, x_max, y_min, y_max, half_pitch, half_det, first_center, self.geom.pixel_spacing_mm

    def _map_hits_to_canonical_multi(self, x_hit: np.ndarray, y_hit: np.ndarray, ring: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Map each hit into the canonical 3x3 window around (xC1,yC1),
        # replicating across central pads within a square of side (2*ring+1).
        # Returns flattened x_map, y_map, and a boolean valid mask of same length.
        try:
            nside = int(self.geom.num_per_side)
            pitch = float(self.geom.pixel_spacing_mm)
            half_det = float(self.geom.det_size_mm) / 2.0
            first_center = -half_det + float(self.geom.pixel_corner_offset_mm) + float(self.geom.pixel_size_mm) / 2.0
        except Exception:
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)
        if x_hit is None or y_hit is None:
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)
        if nside < 3 or pitch <= 0:
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

        # Canonical centers
        _, xC1, _, _, yC1, _, _, _, _, _, *_ = self._centers_3x3()

        N = int(x_hit.shape[0])
        if N == 0:
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

        xi = (x_hit - first_center) / max(pitch, 1e-12)
        yi = (y_hit - first_center) / max(pitch, 1e-12)
        kx0 = np.rint(xi).astype(np.int32)
        ky0 = np.rint(yi).astype(np.int32)

        ring = int(max(0, ring))
        offsets = np.arange(-ring, ring + 1, dtype=np.int32)
        OX, OY = np.meshgrid(offsets, offsets, indexing='xy')
        OX = OX.ravel()
        OY = OY.ravel()
        K = int(OX.size)
        if K == 0:
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

        kx_c = (kx0[:, None] + OX[None, :]).astype(np.int32)
        ky_c = (ky0[:, None] + OY[None, :]).astype(np.int32)
        valid = (kx_c >= 1) & (kx_c <= (nside - 2)) & (ky_c >= 1) & (ky_c <= (nside - 2))

        x_center = first_center + kx_c.astype(np.float64) * pitch
        y_center = first_center + ky_c.astype(np.float64) * pitch
        x_map = (x_hit[:, None] - x_center) + xC1
        y_map = (y_hit[:, None] - y_center) + yC1

        return x_map.ravel().astype(float), y_map.ravel().astype(float), valid.ravel().astype(bool)

    def _pixel_hit_mask_from_coords(self, x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        # Determine whether coordinates fall inside any pixel pad (based on geometry)
        try:
            ps = float(self.geom.pixel_size_mm)
            pitch = float(self.geom.pixel_spacing_mm)
            half_det = float(self.geom.det_size_mm) / 2.0
            first_center = -half_det + float(self.geom.pixel_corner_offset_mm) + ps / 2.0
        except Exception:
            # If geometry is missing, conservatively return all False
            return np.zeros(0 if x_vals is None else x_vals.shape, dtype=bool)
        if x_vals is None or y_vals is None:
            return np.zeros(0, dtype=bool)
        # Nearest pad centers along x and y
        safe_pitch = max(pitch, 1e-12)
        kx = np.rint((x_vals - first_center) / safe_pitch)
        ky = np.rint((y_vals - first_center) / safe_pitch)
        xc = first_center + kx * safe_pitch
        yc = first_center + ky * safe_pitch
        half_ps = ps / 2.0
        inside_x = np.abs(x_vals - xc) <= half_ps
        inside_y = np.abs(y_vals - yc) <= half_ps
        return inside_x & inside_y

    def _current_strip_center(self) -> Tuple[float, bool]:
        # Returns (center, is_row) where center is y for row, x for column
        is_row = (self.orient_combo.currentData() == 0)
        snap_is_gaps = (self.snap_combo.currentData() == 1)
        idx = int(self.snap_slider.value())
        xC0, xC1, xC2, yC0, yC1, yC2, *_ = self._centers_3x3()
        if is_row:
            if snap_is_gaps:
                # gaps: between yC0-yC1 and yC1-yC2
                centers = [0.5 * (yC0 + yC1), 0.5 * (yC1 + yC2)]
            else:
                centers = [yC0, yC1, yC2]
        else:
            if snap_is_gaps:
                centers = [0.5 * (xC0 + xC1), 0.5 * (xC1 + xC2)]
            else:
                centers = [xC0, xC1, xC2]
        idx = max(0, min(idx, len(centers) - 1))
        return centers[idx], is_row

    # -------- Drawing --------
    def draw_current(self):
        if self.file is None:
            self.canvas.clear()
            self.canvas.draw()
            return
        try:
            self.load_geometry()
        except Exception:
            pass
        if self.geom.num_per_side < 3:
            self.canvas.clear()
            self.canvas.ax.text(0.5, 0.5, "Invalid geometry (numPerSide<3)", ha='center', va='center')
            self.canvas.draw()
            return

        xC0, xC1, xC2, yC0, yC1, yC2, x_min, x_max, y_min, y_max, *_ = self._centers_3x3()

        ax = self.canvas.ax
        self.canvas.clear()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        # Draw pads
        ps = self.geom.pixel_size_mm
        color_fill = (94/255, 113/255, 228/255, 0.35)
        color_line = (0/255, 0/255, 180/255, 1.0)
        for xc in (xC0, xC1, xC2):
            for yc in (yC0, yC1, yC2):
                x1 = xc - ps/2.0
                y1 = yc - ps/2.0
                rect = mpatches.Rectangle((x1, y1), ps, ps,
                                          facecolor=color_fill, edgecolor=color_line, linewidth=1)
                ax.add_patch(rect)

        # Window border
        border = mpatches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                    fill=False, edgecolor="black", linewidth=3)
        ax.add_patch(border)

        # Draw snap reference lines
        for yc in (yC0, yC1, yC2, 0.5*(yC0+yC1), 0.5*(yC1+yC2)):
            ax.axhline(y=yc, color=(0.5,0.5,0.5,0.25), linewidth=0.8, linestyle='--')
        for xc in (xC0, xC1, xC2, 0.5*(xC0+xC1), 0.5*(xC1+xC2)):
            ax.axvline(x=xc, color=(0.5,0.5,0.5,0.25), linewidth=0.8, linestyle='--')

        # Draw current selection (strips or walls)
        center, is_row = self._current_strip_center()
        band = float(self.band_spin.value())
        band = max(0.01, band)
        snap_mode = int(self.snap_combo.currentData())
        band_face = (1.0, 0.4, 0.1, 0.25)
        band_edge = (0.8, 0.2, 0.05, 0.9)
        if snap_mode == 2:
            # Gap walls: draw pairs of rectangles hugging pixel surfaces bounding the selected gap(s)
            ps_local = float(self.geom.pixel_size_mm)
            idx = int(self.snap_slider.value())
            def add_rect(rx, ry, rw, rh):
                # Clip within window
                rx = max(rx, x_min)
                ry = max(ry, y_min)
                rw = min(rx + rw, x_max) - rx
                rh = min(ry + rh, y_max) - ry
                if rw > 0 and rh > 0:
                    ax.add_patch(mpatches.Rectangle((rx, ry), rw, rh,
                                                    facecolor=band_face, edgecolor=band_edge, linewidth=1.2))
            if is_row:
                # Two gaps: [yC0,yC1] and [yC1,yC2]
                gaps = [
                    (yC0 + ps_local/2.0, yC1 - ps_local/2.0),
                    (yC1 + ps_local/2.0, yC2 - ps_local/2.0),
                ]
                pairs = gaps if self.aggregate_check.isChecked() else [gaps[max(0, min(idx, 1))]]
                for (low_edge, up_edge) in pairs:
                    # lower wall extends upward into gap; upper wall extends downward into gap
                    add_rect(x_min, low_edge, x_max - x_min, band)
                    add_rect(x_min, up_edge - band, x_max - x_min, band)
            else:
                # Two gaps: [xC0,xC1] and [xC1,xC2]
                gaps = [
                    (xC0 + ps_local/2.0, xC1 - ps_local/2.0),
                    (xC1 + ps_local/2.0, xC2 - ps_local/2.0),
                ]
                pairs = gaps if self.aggregate_check.isChecked() else [gaps[max(0, min(idx, 1))]]
                for (left_edge, right_edge) in pairs:
                    # left wall extends rightward into gap; right wall extends leftward into gap
                    add_rect(left_edge,  y_min, band, y_max - y_min)
                    add_rect(right_edge - band, y_min, band, y_max - y_min)
        else:
            snap_is_gaps = (snap_mode == 1)
            if self.aggregate_check.isChecked():
                if is_row:
                    centers = [0.5 * (yC0 + yC1), 0.5 * (yC1 + yC2)] if snap_is_gaps else [yC0, yC1, yC2]
                else:
                    centers = [0.5 * (xC0 + xC1), 0.5 * (xC1 + xC2)] if snap_is_gaps else [xC0, xC1, xC2]
            else:
                centers = [center]
            for c in centers:
                if is_row:
                    y1 = c - band/2.0
                    rect = mpatches.Rectangle((x_min, y1), x_max - x_min, band,
                                              facecolor=band_face, edgecolor=band_edge, linewidth=1.2)
                else:
                    x1 = c - band/2.0
                    rect = mpatches.Rectangle((x1, y_min), band, y_max - y_min,
                                              facecolor=band_face, edgecolor=band_edge, linewidth=1.2)
                ax.add_patch(rect)

        # Shade regions where the strip(s) are entirely within a pixel pad (blocked)
        # Not applicable for wall mode since selection is in gaps only
        if int(self.snap_combo.currentData()) != 2:
            blocked_face = (0.15, 0.15, 0.15, 0.30)
            ps = self.geom.pixel_size_mm
            snap_is_gaps = (self.snap_combo.currentData() == 1)
            if self.aggregate_check.isChecked():
                if is_row:
                    centers = [0.5 * (yC0 + yC1), 0.5 * (yC1 + yC2)] if snap_is_gaps else [yC0, yC1, yC2]
                else:
                    centers = [0.5 * (xC0 + xC1), 0.5 * (xC1 + xC2)] if snap_is_gaps else [xC0, xC1, xC2]
            else:
                centers = [center]
            if is_row:
                for c in centers:
                    y_low = c - band/2.0
                    y_high = c + band/2.0
                    # Check which pixel row contains the full band
                    for yc in (yC0, yC1, yC2):
                        py1 = yc - ps/2.0
                        py2 = yc + ps/2.0
                        if y_low >= py1 and y_high <= py2:
                            # Shade intersection with pixel pads across x in this row
                            for xc in (xC0, xC1, xC2):
                                px1 = xc - ps/2.0
                                px2 = xc + ps/2.0
                                ix1 = max(px1, x_min)
                                ix2 = min(px2, x_max)
                                if ix2 > ix1:
                                    overlay = mpatches.Rectangle((ix1, y_low), ix2 - ix1, band,
                                                                 facecolor=blocked_face, edgecolor=(0, 0, 0, 0))
                                    ax.add_patch(overlay)
                            break  # only one row can fully contain the band
            else:
                for c in centers:
                    x_low = c - band/2.0
                    x_high = c + band/2.0
                    # Check which pixel column contains the full band
                    for xc in (xC0, xC1, xC2):
                        px1 = xc - ps/2.0
                        px2 = xc + ps/2.0
                        if x_low >= px1 and x_high <= px2:
                            # Shade intersection with pixel pads across y in this column
                            for yc in (yC0, yC1, yC2):
                                py1 = yc - ps/2.0
                                py2 = yc + ps/2.0
                                iy1 = max(py1, y_min)
                                iy2 = min(py2, y_max)
                                if iy2 > iy1:
                                    overlay = mpatches.Rectangle((x_low, iy1), band, iy2 - iy1,
                                                                 facecolor=blocked_face, edgecolor=(0, 0, 0, 0))
                                    ax.add_patch(overlay)
                            break  # only one column can fully contain the band

        self.canvas.ax.tick_params(direction='in')
        self.canvas.draw()

    def on_preview(self):
        # Show the entire detector grid with the current 3x3 highlighted
        try:
            self.load_geometry()
        except Exception:
            pass
        if self.geom.num_per_side <= 0:
            QtWidgets.QMessageBox.warning(self, "Preview", "Open a ROOT file first.")
            return
        n = int(self.geom.num_per_side)
        # Ensure valid 3x3 anchor
        i0 = max(0, min(self.i0, n - 3))
        j0 = max(0, min(self.j0, n - 3))

        half_det = self.geom.det_size_mm / 2.0
        first_center = -half_det + self.geom.pixel_corner_offset_mm + self.geom.pixel_size_mm / 2.0
        half_pitch = 0.5 * self.geom.pixel_spacing_mm

        xC0 = first_center + i0 * self.geom.pixel_spacing_mm
        xC2 = first_center + (i0 + 2) * self.geom.pixel_spacing_mm
        yC0 = first_center + j0 * self.geom.pixel_spacing_mm
        yC2 = first_center + (j0 + 2) * self.geom.pixel_spacing_mm

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

        # Highlight selected 3x3 pads
        xC1 = first_center + (i0 + 1) * self.geom.pixel_spacing_mm
        yC1 = first_center + (j0 + 1) * self.geom.pixel_spacing_mm
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

    # -------- Plotting --------
    def _read_arrays(self, branches: List[str]) -> Optional[dict]:
        try:
            hits = self.file["Hits"]
        except Exception:
            return None
        try:
            arrs = hits.arrays(branches, library="np")
            return arrs
        except Exception:
            return None

    def on_plot(self):
        if self.file is None:
            QtWidgets.QMessageBox.warning(self, "Plot", "Open a ROOT file first.")
            return
        mode = int(self.mode_combo.currentData())
        is2Dabs = (mode == 0)
        is3Dabs = (mode == 1)
        is2Dsgn = (mode == 2)
        is3Dsgn = (mode == 3)
        if is2Dabs and not self.hasRecon2D:
            QtWidgets.QMessageBox.warning(self, "Plot", "2D abs branches not available in file.")
            return
        if is3Dabs and not self.hasRecon3D:
            QtWidgets.QMessageBox.warning(self, "Plot", "3D abs branches not available in file.")
            return
        if is2Dsgn and not self.hasRecon2DSigned:
            QtWidgets.QMessageBox.warning(self, "Plot", "2D signed branches not available in file.")
            return
        if is3Dsgn and not self.hasRecon3DSigned:
            QtWidgets.QMessageBox.warning(self, "Plot", "3D signed branches not available in file.")
            return

        xC0, xC1, xC2, yC0, yC1, yC2, x_min, x_max, y_min, y_max, *_ = self._centers_3x3()
        center, is_row = self._current_strip_center()
        band = float(self.band_spin.value())
        band = max(0.01, band)

        req_branches = ["TrueX", "TrueY", "isPixelHit"]
        if is2Dabs:
            req_branches += ["ReconTrueDeltaX", "ReconTrueDeltaY"]
        if is3Dabs:
            req_branches += ["ReconTrueDeltaX_3D", "ReconTrueDeltaY_3D"]
        if is2Dsgn:
            req_branches += ["ReconTrueDeltaX_Signed", "ReconTrueDeltaY_Signed"]
        if is3Dsgn:
            req_branches += ["ReconTrueDeltaX_3D_Signed", "ReconTrueDeltaY_3D_Signed"]
        arrs = self._read_arrays(req_branches)
        if arrs is None:
            QtWidgets.QMessageBox.critical(self, "Plot", "Failed reading branches from Hits.")
            return

        x_hit = arrs.get("TrueX")
        y_hit = arrs.get("TrueY")
        is_pixel_hit = arrs.get("isPixelHit")
        if is_pixel_hit is None:
            is_pixel_hit = np.zeros_like(x_hit, dtype=bool)
        else:
            is_pixel_hit = is_pixel_hit.astype(bool)

        # Optionally remap and REPLICATE all events into a canonical 3x3 window (aggregate across all 3x3s)
        aggregate_all = bool(getattr(self, 'aggregate_all', False))
        x_coord = x_hit
        y_coord = y_hit
        valid_multi = None
        if aggregate_all:
            # Use multi-mapping to replicate each hit across the 3x3 around its nearest center
            x_map, y_map, valid = self._map_hits_to_canonical_multi(x_hit, y_hit, ring=1)
            if x_map.size == 0:
                QtWidgets.QMessageBox.information(self, "Plot", "No mappable hits for aggregation.")
                return
            x_coord = x_map
            y_coord = y_map
            valid_multi = valid.astype(bool)
            # IMPORTANT: recompute pixel-hit mask using mapped coordinates, not the original mask
            is_pixel_hit = self._pixel_hit_mask_from_coords(x_coord, y_coord)
        else:
            valid_multi = np.ones_like(x_hit, dtype=bool)

        # Filter base region (use replicated mapped coordinates if aggregating)
        if aggregate_all:
            mask = np.isfinite(x_coord) & np.isfinite(y_coord)
            mask &= valid_multi
            mask &= (x_coord >= x_min) & (x_coord <= x_max) & (y_coord >= y_min) & (y_coord <= y_max)
        else:
            mask = np.isfinite(x_hit) & np.isfinite(y_hit)
            mask &= (x_hit >= x_min) & (x_hit <= x_max) & (y_hit >= y_min) & (y_hit <= y_max)
        # For resolution studies we always use non-pixel hits (charge sharing region)
        mask &= (~is_pixel_hit)
        snap_mode = int(self.snap_combo.currentData())
        if snap_mode == 2:
            # Gap walls: select paired rectangles hugging pad edges into the gap(s)
            ps_local = float(self.geom.pixel_size_mm)
            if is_row:
                gaps = [
                    (yC0 + ps_local/2.0, yC1 - ps_local/2.0),
                    (yC1 + ps_local/2.0, yC2 - ps_local/2.0),
                ]
                selected = gaps if self.aggregate_check.isChecked() else [gaps[max(0, min(int(self.snap_slider.value()), 1))]]
                wall_mask = np.zeros_like(mask)
                for (low_edge, up_edge) in selected:
                    wall_axis = (y_coord if aggregate_all else y_hit)
                    wall_mask |= ((wall_axis >= low_edge) & (wall_axis <= low_edge + band))
                    wall_mask |= ((wall_axis >= up_edge - band) & (wall_axis <= up_edge))
                mask &= wall_mask
            else:
                gaps = [
                    (xC0 + ps_local/2.0, xC1 - ps_local/2.0),
                    (xC1 + ps_local/2.0, xC2 - ps_local/2.0),
                ]
                selected = gaps if self.aggregate_check.isChecked() else [gaps[max(0, min(int(self.snap_slider.value()), 1))]]
                wall_mask = np.zeros_like(mask)
                for (left_edge, right_edge) in selected:
                    wall_axis = (x_coord if aggregate_all else x_hit)
                    wall_mask |= ((wall_axis >= left_edge) & (wall_axis <= left_edge + band))
                    wall_mask |= ((wall_axis >= right_edge - band) & (wall_axis <= right_edge))
                mask &= wall_mask
        else:
            # Apply strip selection: either current index or aggregate across all indices
            if self.aggregate_check.isChecked():
                snap_is_gaps = (snap_mode == 1)
                if is_row:
                    if snap_is_gaps:
                        centers = [0.5 * (yC0 + yC1), 0.5 * (yC1 + yC2)]
                    else:
                        centers = [yC0, yC1, yC2]
                    half_band = band / 2.0
                    y_axis = (y_coord if aggregate_all else y_hit)
                    if len(centers) == 0:
                        strip_mask = np.zeros_like(mask, dtype=bool)
                    else:
                        c_arr = np.asarray(centers, dtype=float)
                        strip_mask = np.any(np.abs(y_axis[:, None] - c_arr[None, :]) <= half_band, axis=1)
                else:
                    if snap_is_gaps:
                        centers = [0.5 * (xC0 + xC1), 0.5 * (xC1 + xC2)]
                    else:
                        centers = [xC0, xC1, xC2]
                    half_band = band / 2.0
                    x_axis = (x_coord if aggregate_all else x_hit)
                    if len(centers) == 0:
                        strip_mask = np.zeros_like(mask, dtype=bool)
                    else:
                        c_arr = np.asarray(centers, dtype=float)
                        strip_mask = np.any(np.abs(x_axis[:, None] - c_arr[None, :]) <= half_band, axis=1)
                mask &= strip_mask
            else:
                if is_row:
                    y_axis = (y_coord if aggregate_all else y_hit)
                    mask &= (np.abs(y_axis - center) <= (band/2.0))
                else:
                    x_axis = (x_coord if aggregate_all else x_hit)
                    mask &= (np.abs(x_axis - center) <= (band/2.0))

        if not np.any(mask):
            QtWidgets.QMessageBox.information(self, "Plot", "No hits in selected strip.")
            return

        xs = (x_coord if aggregate_all else x_hit)[mask]
        ys = (y_coord if aggregate_all else y_hit)[mask]
        # Precompute full dx/dy arrays (unmasked) for flexible selections below
        if is2Dabs:
            dx_base = np.abs(arrs.get("ReconTrueDeltaX"))
            dy_base = np.abs(arrs.get("ReconTrueDeltaY"))
        elif is3Dabs:
            dx_base = np.abs(arrs.get("ReconTrueDeltaX_3D"))
            dy_base = np.abs(arrs.get("ReconTrueDeltaY_3D"))
        elif is2Dsgn:
            dx_base = arrs.get("ReconTrueDeltaX_Signed")
            dy_base = arrs.get("ReconTrueDeltaY_Signed")
        else:  # is3Dsgn
            dx_base = arrs.get("ReconTrueDeltaX_3D_Signed")
            dy_base = arrs.get("ReconTrueDeltaY_3D_Signed")

        if aggregate_all:
            # Replicate dx/dy to align with replicated coordinates
            N = int(x_hit.shape[0]) if x_hit is not None else 0
            K = int(x_coord.size // max(1, N)) if N > 0 else 1
            dx_all = np.repeat(dx_base, K)
            dy_all = np.repeat(dy_base, K)
        else:
            dx_all = dx_base
            dy_all = dy_base

        axis_vals = (((x_coord if aggregate_all else x_hit)[mask]) if is_row else ((y_coord if aggregate_all else y_hit)[mask]))
        res_vals = (dx_all[mask] if is_row else dy_all[mask])
        axis_label = "x [mm]" if is_row else "y [mm]"
        if is2Dabs:
            res_label = ("|Δx| [mm]" if is_row else "|Δy| [mm]") + " (2D)"
        elif is3Dabs:
            res_label = ("|Δx_3D| [mm]" if is_row else "|Δy_3D| [mm]") + " (3D)"
        elif is2Dsgn:
            res_label = ("Δx (signed) [mm]" if is_row else "Δy (signed) [mm]") + " (2D)"
        else:
            res_label = ("Δx_3D (signed) [mm]" if is_row else "Δy_3D (signed) [mm]") + " (3D)"

        # Gap-wall mode: open three separate plots (bottom, top, both aggregated)
        if snap_mode == 2:
            # Prepare data and masks
            shade_color = (1.0, 0.78, 0.86, 0.35)  # light pink
            ps_local = float(self.geom.pixel_size_mm)
            axis_full = (x_coord if aggregate_all else x_hit) if is_row else (y_coord if aggregate_all else y_hit)
            res_full = dx_all if is_row else dy_all
            if is_row:
                pad_centers = (xC0, xC1, xC2)
                xmin_plot, xmax_plot = x_min, x_max
                gaps = [
                    (yC0 + ps_local/2.0, yC1 - ps_local/2.0),
                    (yC1 + ps_local/2.0, yC2 - ps_local/2.0),
                ]
                # For rows, wall selection is along y, not x
                wall_axis = (y_coord if aggregate_all else y_hit)
                mask_bottom_local = ((wall_axis >= gaps[0][0]) & (wall_axis <= gaps[0][0] + band)) | \
                                    ((wall_axis >= gaps[1][0]) & (wall_axis <= gaps[1][0] + band))
                mask_top_local = ((wall_axis >= gaps[0][1] - band) & (wall_axis <= gaps[0][1])) | \
                                 ((wall_axis >= gaps[1][1] - band) & (wall_axis <= gaps[1][1]))
                bottom_label = "bottom aggregated"
                top_label = "top aggregated"
            else:
                pad_centers = (yC0, yC1, yC2)
                xmin_plot, xmax_plot = y_min, y_max
                gaps = [
                    (xC0 + ps_local/2.0, xC1 - ps_local/2.0),
                    (xC1 + ps_local/2.0, xC2 - ps_local/2.0),
                ]
                # For columns, wall selection is along x
                wall_axis = (x_coord if aggregate_all else x_hit)
                mask_bottom_local = ((wall_axis >= gaps[0][0]) & (wall_axis <= gaps[0][0] + band)) | \
                                    ((wall_axis >= gaps[1][0]) & (wall_axis <= gaps[1][0] + band))
                mask_top_local = ((wall_axis >= gaps[0][1] - band) & (wall_axis <= gaps[0][1])) | \
                                 ((wall_axis >= gaps[1][1] - band) & (wall_axis <= gaps[1][1]))
                bottom_label = "left aggregated"
                top_label = "right aggregated"

            mask_bottom = mask & mask_bottom_local
            mask_top = mask & mask_top_local
            mask_both = mask_bottom | mask_top

            if not (np.any(mask_bottom) or np.any(mask_top) or np.any(mask_both)):
                QtWidgets.QMessageBox.information(self, "Plot", "No hits in selected wall bands.")
                return

            # Close previous non-modal plot dialogs if any
            try:
                prev_dlgs = getattr(self, "_current_plot_dialogs", [])
                for d in prev_dlgs:
                    try:
                        d.close()
                    except Exception:
                        pass
            except Exception:
                pass
            self._current_plot_dialogs = []
            self._current_plot_canvases = []

            def create_and_show_dialog(series_mask, series_title, color):
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("Strip resolution")
                dlg.resize(900, 700)
                lay = QtWidgets.QVBoxLayout(dlg)
                fig = Figure(figsize=(7, 5), dpi=100)
                canvas = FigureCanvas(fig)
                lay.addWidget(canvas, 1)
                ax = fig.add_subplot(111)
                ax.set_xlabel(axis_label)
                ax.set_ylabel("resolution [mm]")
                # Geometry shading and x-limits
                ax.set_xlim([xmin_plot, xmax_plot])
                for c in pad_centers:
                    ax.axvspan(c - ps_local/2.0, c + ps_local/2.0, facecolor=shade_color, edgecolor=None, linewidth=0, zorder=0)
                # Data points
                ax.scatter(axis_full[series_mask], res_full[series_mask], s=10.0, c=color, alpha=0.8, edgecolors='none')
                # Reference lines at pixel-size/sqrt(12) and 500um/sqrt(12)
                try:
                    y_ref_local = float(self.geom.pixel_size_mm) / np.sqrt(12.0)
                except Exception:
                    y_ref_local = None
                y_ref_500 = 0.5 / np.sqrt(12.0)
                ymin_cur, ymax_cur = ax.get_ylim()
                target_top = ymax_cur
                if y_ref_local is not None and np.isfinite(y_ref_local):
                    target_top = max(target_top, y_ref_local * 1.02)
                if np.isfinite(y_ref_500):
                    target_top = max(target_top, y_ref_500 * 1.02)
                if target_top > ymax_cur:
                    ax.set_ylim(ymin_cur, target_top)
                # Draw pad-span line segments for geometry-based reference
                if y_ref_local is not None and np.isfinite(y_ref_local):
                    for c in pad_centers:
                        ax.hlines(y=y_ref_local, xmin=c - ps_local/2.0, xmax=c + ps_local/2.0,
                                  colors='k', linewidth=2.0, zorder=3)
                # Draw full-width horizontal line for 500um/sqrt(12)
                ax.hlines(y=y_ref_500, xmin=xmin_plot, xmax=xmax_plot,
                          colors='r', linewidth=1.5, linestyles='--', zorder=3)
                if is2Dabs or is3Dabs:
                    ax.set_ylim(bottom=0.0)
                ax.grid(False)
                ax.set_title(res_label + f" (gap walls: {series_title})")
                ax.tick_params(direction='in')
                canvas.draw()
                # Save controls
                btn_row = QtWidgets.QHBoxLayout()
                save_btn = QtWidgets.QPushButton("Save…")
                close_btn = QtWidgets.QPushButton("Close")
                btn_row.addStretch(1)
                btn_row.addWidget(save_btn)
                btn_row.addWidget(close_btn)
                lay.addLayout(btn_row)
                def on_save_clicked():
                    fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                        dlg, "Save figure", os.getcwd(), "SVG (*.svg);;PNG (*.png);;PDF (*.pdf)"
                    )
                    if fn:
                        try:
                            fig.savefig(fn, bbox_inches="tight")
                        except Exception as e:
                            QtWidgets.QMessageBox.critical(dlg, "Save", f"Failed to save: {e}")
                save_btn.clicked.connect(on_save_clicked)
                close_btn.clicked.connect(dlg.accept)
                # Show non-modally and keep references to avoid GC
                dlg.show()
                self._current_plot_dialogs.append(dlg)
                self._current_plot_canvases.append(canvas)

            # Create the three separate plots
            create_and_show_dialog(mask_bottom, bottom_label, 'C0')
            create_and_show_dialog(mask_top,    top_label,    'C3')
            create_and_show_dialog(mask_both,   'both aggregated', 'C2')
            return

        # Non-wall modes: single plot as before
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Strip resolution")
        dlg.resize(900, 700)
        lay = QtWidgets.QVBoxLayout(dlg)
        fig = Figure(figsize=(7, 5), dpi=100)
        canvas = FigureCanvas(fig)
        lay.addWidget(canvas, 1)
        ax = fig.add_subplot(111)
        ax.set_xlabel(axis_label)
        ax.set_ylabel("resolution [mm]")
        # Shade pixel-pad regions along the plot x-axis (which is x for rows, y for columns)
        shade_color = (1.0, 0.78, 0.86, 0.35)
        ps_local = float(self.geom.pixel_size_mm)
        if is_row:
            pad_centers = (xC0, xC1, xC2)
            xmin_plot, xmax_plot = x_min, x_max
        else:
            pad_centers = (yC0, yC1, yC2)
            xmin_plot, xmax_plot = y_min, y_max
        ax.set_xlim([xmin_plot, xmax_plot])
        for c in pad_centers:
            ax.axvspan(c - ps_local/2.0, c + ps_local/2.0, facecolor=shade_color, edgecolor=None, linewidth=0, zorder=0)
        ax.scatter(axis_vals, res_vals, s=10.0, c='C0', alpha=0.8, edgecolors='none')
        # Reference lines at pixel-size/sqrt(12) and 500um/sqrt(12)
        try:
            y_ref = float(self.geom.pixel_size_mm) / np.sqrt(12.0)
        except Exception:
            y_ref = None
        y_ref_500 = 0.5 / np.sqrt(12.0)
        ymin_cur, ymax_cur = ax.get_ylim()
        target_top = ymax_cur
        if y_ref is not None and np.isfinite(y_ref):
            target_top = max(target_top, y_ref * 1.02)
        if np.isfinite(y_ref_500):
            target_top = max(target_top, y_ref_500 * 1.02)
        if target_top > ymax_cur:
            ax.set_ylim(ymin_cur, target_top)
        if y_ref is not None and np.isfinite(y_ref):
            for c in pad_centers:
                ax.hlines(y=y_ref, xmin=c - ps_local/2.0, xmax=c + ps_local/2.0,
                          colors='k', linewidth=2.0, zorder=3)
        # Full-width horizontal line for 500um/sqrt(12)
        ax.hlines(y=y_ref_500, xmin=xmin_plot, xmax=xmax_plot,
                  colors='r', linewidth=1.5, linestyles='--', zorder=3)
        if is2Dabs or is3Dabs:
            ax.set_ylim(bottom=0.0)
        ax.grid(False)
        if self.aggregate_check.isChecked():
            ax.set_title(res_label + " (all indices)")
        else:
            ax.set_title(res_label)
        ax.tick_params(direction='in')
        canvas.draw()
        # Save controls
        btn_row = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save…")
        close_btn = QtWidgets.QPushButton("Close")
        btn_row.addStretch(1)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)
        def on_save_clicked():
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save figure", os.getcwd(), "SVG (*.svg);;PNG (*.png);;PDF (*.pdf)"
            )
            if fn:
                try:
                    fig.savefig(fn, bbox_inches="tight")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(dlg, "Save", f"Failed to save: {e}")
        save_btn.clicked.connect(on_save_clicked)
        close_btn.clicked.connect(dlg.accept)
        dlg.exec_()

    def on_save_preview(self):
        # Removed: Save Preview option is no longer available.
        pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = ResolutionGUI()
    gui.show()
    # CLI arg: optional ROOT file
    if len(sys.argv) > 1 and sys.argv[1].strip():
        gui.file_edit.setText(sys.argv[1].strip())
        gui.on_open()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


