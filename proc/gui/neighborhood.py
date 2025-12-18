#!/usr/bin/env python3
import os
import math
from typing import Optional, Tuple, List

import numpy as np
import awkward as ak
import uproot

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages


# --- Helpers to read TNamed doubles (metadata) ---
def read_named_double(file: uproot.ReadOnlyFile, key: str) -> float:
    obj = file.get(key)
    if obj is None:
        raise RuntimeError(f"Missing metadata object: '{key}'")
    title = None
    try:
        title = obj.member("fTitle")
    except Exception:
        title = getattr(obj, "fTitle", None)
        if title is None:
            title = str(obj)
    if title is None:
        raise RuntimeError(f"TNamed '{key}' has empty title")
    try:
        return float(title)
    except Exception as e:
        raise RuntimeError(f"Cannot parse '{key}' title to float: {title}") from e


def read_named_int(file: uproot.ReadOnlyFile, key: str) -> int:
    return int(round(read_named_double(file, key)))


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 6), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.subplots_adjust(left=0.12, right=0.86, top=0.90, bottom=0.12)
        self._cbar = None

    def clear(self):
        # Remove any existing colorbar
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        # Remove all axes (including previously shrunken main axes)
        for ax in list(self.fig.axes):
            try:
                self.fig.delaxes(ax)
            except Exception:
                pass
        # Recreate a fresh main axes so layout doesn't accumulate shrink
        try:
            self.ax = self.fig.add_subplot(111)
        except Exception:
            # Fallback to ensure an axes exists
            self.ax = self.fig.add_axes([0.12, 0.12, 0.74, 0.78])
        self.ax.set_aspect("equal", adjustable="box")
        # Re-apply consistent subplot margins
        try:
            self.fig.subplots_adjust(left=0.12, right=0.86, top=0.90, bottom=0.12)
        except Exception:
            pass

    def set_colorbar(self, mappable, label: str):
        # Remove any previous colorbar and auxiliary axes (defensive)
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        for extra_ax in list(self.fig.axes):
            if extra_ax is not self.ax:
                try:
                    self.fig.delaxes(extra_ax)
                except Exception:
                    pass
        # Restore stable subplot margins before adding a new colorbar
        try:
            self.fig.subplots_adjust(left=0.12, right=0.86, top=0.90, bottom=0.12)
        except Exception:
            pass
        # Create a colorbar with fixed geometry to avoid cumulative layout drift
        self._cbar = self.fig.colorbar(mappable, ax=self.ax, fraction=0.046, pad=0.04)
        self._cbar.set_label(label)


class ChargeNeighborhoodGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Charge Neighborhood Viewer")
        self.resize(1100, 850)

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

        # Controls row
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Quantity:"))
        self.quantity_combo = QtWidgets.QComboBox()
        self.quantity_combo.addItem("Coulomb", "coulomb")
        self.quantity_combo.addItem("Fraction", "fraction")
        self.quantity_combo.addItem("d_i", "d_i")
        self.quantity_combo.addItem("alpha_i", "alpha_i")
        ctrl.addWidget(self.quantity_combo)

        ctrl.addWidget(QtWidgets.QLabel("Branch:"))
        self.branch_combo = QtWidgets.QComboBox()
        ctrl.addWidget(self.branch_combo)

        ctrl.addWidget(QtWidgets.QLabel("Event:"))
        self.event_spin = QtWidgets.QSpinBox()
        self.event_spin.setMinimum(0)
        self.event_spin.setMaximum(0)
        self.event_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.prev_btn = QtWidgets.QPushButton("◀")
        self.next_btn = QtWidgets.QPushButton("▶")
        ctrl.addWidget(self.event_spin)
        ctrl.addWidget(self.prev_btn)
        ctrl.addWidget(self.next_btn)

        ctrl.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmap_combo = QtWidgets.QComboBox()
        for name in ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "coolwarm", "RdBu_r"]:
            self.cmap_combo.addItem(name, name)
        ctrl.addWidget(self.cmap_combo)

        self.redraw_btn = QtWidgets.QPushButton("Redraw")
        ctrl.addWidget(self.redraw_btn)

        vbox.addLayout(ctrl)

        # PDF export row
        pdf_row = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export PDF…")
        pdf_row.addWidget(self.export_btn)
        pdf_row.addWidget(QtWidgets.QLabel("Pages:"))
        self.pages_spin = QtWidgets.QSpinBox()
        self.pages_spin.setRange(1, 10000)
        self.pages_spin.setValue(100)
        pdf_row.addWidget(self.pages_spin)
        pdf_row.addStretch(1)
        vbox.addLayout(pdf_row)

        # Canvas
        self.canvas = MplCanvas(self)
        vbox.addWidget(self.canvas, 1)

        # State
        self._root: Optional[uproot.ReadOnlyFile] = None
        self._tree: Optional[uproot.behaviors.TBranch.TBranch] = None
        self._n_entries: int = 0
        self._meta = {}

        # Signals
        self.browse_btn.clicked.connect(self._on_browse)
        self.open_btn.clicked.connect(self._on_open)
        self.prev_btn.clicked.connect(self._on_prev)
        self.next_btn.clicked.connect(self._on_next)
        self.export_btn.clicked.connect(self._on_export_pdf)
        self.redraw_btn.clicked.connect(self._draw_current)
        self.quantity_combo.currentIndexChanged.connect(self._draw_current)
        self.branch_combo.currentIndexChanged.connect(self._draw_current)
        self.cmap_combo.currentIndexChanged.connect(self._draw_current)
        self.event_spin.valueChanged.connect(self._draw_current)

        # Try a common default path
        default_candidates = [
            os.path.join(os.getcwd(), "epicChargeSharing.root"),
            os.path.join(os.getcwd(), "build", "epicChargeSharing.root"),
            os.path.join(os.path.dirname(os.getcwd()), "build", "epicChargeSharing.root"),
        ]
        for p in default_candidates:
            if os.path.exists(p):
                self.file_edit.setText(p)
                break

    # --- File open / load ---
    def _on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open ROOT file", os.getcwd(), "ROOT files (*.root);;All files (*)")
        if path:
            self.file_edit.setText(path)

    def _on_open(self):
        path = self.file_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Open", "Please choose a ROOT file.")
            return
        if not os.path.exists(path):
            QtWidgets.QMessageBox.critical(self, "Open", f"File not found:\n{path}")
            return
        try:
            if self._root is not None:
                try:
                    self._root.close()
                except Exception:
                    pass
            self._root = uproot.open(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open", f"Failed to open ROOT file:\n{e}")
            return

        try:
            self._meta = {
                "pixel_size": read_named_double(self._root, "GridPixelSize_mm"),
                "pixel_spacing": read_named_double(self._root, "GridPixelSpacing_mm"),
                "grid_offset": read_named_double(self._root, "GridOffset_mm"),
                "det_size": read_named_double(self._root, "GridDetectorSize_mm"),
                "num_per_side": read_named_int(self._root, "GridNumBlocksPerSide"),
            }
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Metadata", f"Failed to read grid metadata:\n{e}")
            self._meta = {}
            return

        # Optional metadata: neighborhood radius (fallback to 2 if absent)
        try:
            self._meta["neighborhood_radius"] = read_named_int(self._root, "NeighborhoodRadius")
        except Exception:
            self._meta["neighborhood_radius"] = 2

        try:
            self._tree = self._root["Hits"]
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open", f"'Hits' tree not found or unreadable:\n{e}")
            self._tree = None
            return

        try:
            self._n_entries = int(self._tree.num_entries)
        except Exception:
            self._n_entries = 0
        self.event_spin.setMaximum(max(0, self._n_entries - 1))
        self._populate_branches()

        # Pick first event with valid neighborhood
        start_evt = self._find_first_valid_event()
        if start_evt is None:
            QtWidgets.QMessageBox.warning(self, "Open", "No events with valid neighborhood vectors found.")
            start_evt = 0
        self.event_spin.setValue(start_evt)
        self._draw_current()

    def _populate_branches(self):
        # Clear and repopulate with explicit options. Default selection will be Qf if present.
        try:
            self.branch_combo.blockSignals(True)
        except Exception:
            pass
        try:
            self.branch_combo.clear()
        except Exception:
            pass
        available = set()
        try:
            if self._tree is not None:
                available = set(self._tree.keys())
        except Exception:
            available = set()

        # Base branches in preferred order
        for key in ["Qf", "Qi", "Qn", "Fi"]:
            if key in available:
                self.branch_combo.addItem(key, key)

        # Computed options (absolute differences)
        if ("Qi" in available) and ("Qf" in available):
            self.branch_combo.addItem("|Qi - Qf|", "delta_if")
        if ("Qn" in available) and ("Qf" in available):
            self.branch_combo.addItem("|Qn - Qf|", "delta_nf")

        # Outlier mask options (discrete binary coloring)
        if "Gauss3DMaskRemoved" in available:
            self.branch_combo.addItem("Outlier mask (3D)", "mask_3d")
        if "GaussRowMaskRemoved" in available:
            self.branch_combo.addItem("Outlier mask (row)", "mask_row")
        if "GaussColMaskRemoved" in available:
            self.branch_combo.addItem("Outlier mask (col)", "mask_col")

        # Default to Qf if present, else first entry
        try:
            idx_qf = next((i for i in range(self.branch_combo.count()) if self.branch_combo.itemData(i) == "Qf"), 0)
            self.branch_combo.setCurrentIndex(max(0, idx_qf))
        except Exception:
            pass
        try:
            self.branch_combo.blockSignals(False)
        except Exception:
            pass

    def _find_first_valid_event(self) -> Optional[int]:
        if self._tree is None or self._n_entries <= 0:
            return None
        # Use a concrete vector branch present in the file; prefer Qf
        key = None
        try:
            for k in ["Qf", "Qn", "Qi", "Fi", "d_i", "alpha_i"]:
                if self._tree is not None and k in self._tree.keys():
                    key = k
                    break
        except Exception:
            key = None
        if key is None:
            return None
        # Stream in chunks to avoid loading all
        step = 1024
        for start in range(0, self._n_entries, step):
            stop = min(self._n_entries, start + step)
            try:
                arrs = self._tree.arrays([key], entry_start=start, entry_stop=stop)
            except Exception:
                continue
            vecs = arrs[key]
            for i in range(stop - start):
                try:
                    v = vecs[i]
                    n = int(len(v))
                    if n >= 25:  # at least 5x5
                        return start + i
                except Exception:
                    pass
        return None

    def _selected_branch_spec(self):
        """Return a tuple describing the selected data source.

        Forms:
          ("direct", key) for a direct branch key
          ("delta_abs", ("Qi", "Qf"))
          ("delta_signed", ("Qi", "Qf"))
          ("mask", "mask_3d" | "mask_row" | "mask_col")
        If selection missing, fallback to first available direct key.
        """
        sel = None
        try:
            sel = self.branch_combo.currentData()
        except Exception:
            sel = None
        # Direct branches
        if sel in ("Qf", "Qi", "Qn", "Fi"):
            return ("direct", sel)
        # Computed absolute deltas
        if sel == "delta_if":
            return ("delta_abs", ("Qi", "Qf"))
        if sel == "delta_nf":
            return ("delta_abs", ("Qn", "Qf"))
        # Mask selections
        if sel in ("mask_3d", "mask_row", "mask_col"):
            return ("mask", sel)
        # Fallback: pick first available
        try:
            if self._tree is not None:
                for k in ["Qf", "Qi", "Qn", "Fi"]:
                    if k in self._tree.keys():
                        return ("direct", k)
        except Exception:
            pass
        return ("direct", None)

    # --- Navigation ---
    def _on_prev(self):
        v = self.event_spin.value()
        if v > 0:
            self.event_spin.setValue(v - 1)

    def _on_next(self):
        v = self.event_spin.value()
        if v < max(0, self._n_entries - 1):
            self.event_spin.setValue(v + 1)

    # --- Drawing ---
    def _draw_current(self):
        if self._tree is None or self._n_entries <= 0:
            return
        evt = self.event_spin.value()
        kind, spec = self._selected_branch_spec()
        # Determine which branches need to be read
        if kind == "direct":
            key = spec
        else:
            key = None
        if key is None and kind == "direct":
            QtWidgets.QMessageBox.warning(self, "Draw", "No neighborhood vector branch available.")
            return

        quantity = self.quantity_combo.currentData()
        cmap_name = self.cmap_combo.currentData()

        # Read one event worth of data
        branches = ["TrueX", "TrueY", "PixelX", "PixelY", "Edep", "isPixelHit"]
        mask_mode = (kind == "mask")
        mask_sel = spec if mask_mode else None
        shape_key = None
        if kind == "direct" and key is not None:
            branches.append(key)
        elif kind == "delta_abs":
            # delta requires both components
            try:
                a, b = spec
            except Exception:
                a, b = "Qi", "Qf"
            branches += [a, b]
        elif mask_mode:
            # Determine necessary branches for masks
            if mask_sel == "mask_3d":
                branches.append("Gauss3DMaskRemoved")
            elif mask_sel == "mask_row":
                branches.append("GaussRowMaskRemoved")
            elif mask_sel == "mask_col":
                branches.append("GaussColMaskRemoved")
            # We also need a shape source to obtain NxN and valid positions for mapping
            try:
                available = set(self._tree.keys()) if self._tree is not None else set()
            except Exception:
                available = set()
            for kshape in ["Qf", "Fi", "Qi", "Qn"]:
                if kshape in available:
                    shape_key = kshape
                    branches.append(kshape)
                    break
        # Include additional quantity branches if needed
        if quantity in ("d_i", "alpha_i"):
            branches.append(quantity)
        try:
            arrs = self._tree.arrays(branches, entry_start=evt, entry_stop=evt + 1)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Read", f"Failed to read event {evt}:\n{e}")
            return

        def scalar(name: str) -> float:
            try:
                x = arrs[name]
                if isinstance(x, ak.Array):
                    return float(x[0])
                return float(np.asarray(x)[0])
            except Exception:
                return float("nan")

        trueX = scalar("TrueX")
        trueY = scalar("TrueY")
        pixelX = scalar("PixelX")
        pixelY = scalar("PixelY")
        edep = scalar("Edep")
        try:
            is_pixel_hit = bool(arrs["isPixelHit"][0])
        except Exception:
            is_pixel_hit = False

        # Build the values array according to selection
        signed_values = False
        try:
            if mask_mode:
                # Build NxN array from mask branches using a shape source
                if shape_key is None:
                    values = np.array([], dtype=float)
                else:
                    shape_vec = np.asarray(arrs[shape_key][0], dtype=float)
                    if shape_vec.size == 0:
                        values = np.array([], dtype=float)
                    else:
                        dim_guess = int(round(math.sqrt(float(shape_vec.size))))
                        if dim_guess * dim_guess != int(shape_vec.size) or dim_guess < 3:
                            values = np.array([], dtype=float)
                        else:
                            N = dim_guess
                            R = N // 2
                            values = np.full(N * N, np.nan, dtype=float)
                            if mask_sel == "mask_3d":
                                mvec = np.asarray(arrs["Gauss3DMaskRemoved"][0], dtype=float)
                                k = 0
                                for di in range(-R, R + 1):
                                    for dj in range(-R, R + 1):
                                        idx_shape = (di + R) * N + (dj + R)
                                        q = shape_vec[idx_shape]
                                        if math.isfinite(q) and q >= 0.0:
                                            if k < mvec.size:
                                                values[idx_shape] = mvec[k]
                                                k += 1
                            elif mask_sel == "mask_row":
                                mvec = np.asarray(arrs["GaussRowMaskRemoved"][0], dtype=float)
                                k = 0
                                dj = 0
                                for di in range(-R, R + 1):
                                    idx_shape = (di + R) * N + (dj + R)
                                    q = shape_vec[idx_shape]
                                    if math.isfinite(q) and q >= 0.0:
                                        if k < mvec.size:
                                            values[idx_shape] = mvec[k]
                                            k += 1
                            elif mask_sel == "mask_col":
                                mvec = np.asarray(arrs["GaussColMaskRemoved"][0], dtype=float)
                                k = 0
                                di = 0
                                for dj in range(-R, R + 1):
                                    idx_shape = (di + R) * N + (dj + R)
                                    q = shape_vec[idx_shape]
                                    if math.isfinite(q) and q >= 0.0:
                                        if k < mvec.size:
                                            values[idx_shape] = mvec[k]
                                            k += 1
            elif kind == "direct" and key is not None:
                vec = arrs[key][0]
                values = np.asarray(vec, dtype=float)
            else:
                a_key, b_key = spec
                a_arr = np.asarray(arrs[a_key][0], dtype=float)
                b_arr = np.asarray(arrs[b_key][0], dtype=float)
                if a_arr.size == 0 or b_arr.size == 0:
                    values = np.array([], dtype=float)
                else:
                    d = a_arr - b_arr
                    values = np.abs(d)
        except Exception:
            values = np.array([], dtype=float)

        if values.size == 0:
            QtWidgets.QMessageBox.information(self, "Draw", f"Event {evt} has empty neighborhood vector '{key}'.")
            return

        dim_f = int(round(math.sqrt(float(values.size))))
        if dim_f * dim_f != int(values.size) or dim_f < 5:
            QtWidgets.QMessageBox.information(self, "Draw", f"Event {evt} vector size {values.size} is not a >=5x>=5 square.")
            return
        dim = dim_f

        # Transpose into grid[y][x]
        grid = np.full((dim, dim), np.nan, dtype=float)
        for di in range(dim):
            for dj in range(dim):
                idx = di * dim + dj
                grid[dj, di] = values[idx]

        c = dim // 2
        try:
            r_meta = int(self._meta.get("neighborhood_radius", 2))
        except Exception:
            r_meta = 2
        roi_r = max(1, min(c, r_meta))
        i0, i1 = c - roi_r, c + roi_r
        j0, j1 = c - roi_r, c + roi_r

        pixel_spacing = float(self._meta.get("pixel_spacing", 0.0))
        pixel_size = float(self._meta.get("pixel_size", 0.0))
        half_extent = (roi_r + 0.5) * pixel_spacing

        # q_total from edep
        e_charge = 1.602176634e-19
        pair_e_ev = 3.60
        q_total = 0.0
        if math.isfinite(edep) and edep > 0:
            q_total = edep * 1.0e6 / pair_e_ev * e_charge

        # Optionally build auxiliary grids for d_i / alpha_i
        aux_grid = None
        if quantity in ("d_i", "alpha_i"):
            try:
                aux_values = np.asarray(arrs[quantity][0], dtype=float)
            except Exception:
                aux_values = np.array([], dtype=float)
            if aux_values.size == values.size:
                aux_grid = np.full((dim, dim), np.nan, dtype=float)
                for di in range(dim):
                    for dj in range(dim):
                        idx = di * dim + dj
                        aux_grid[dj, di] = aux_values[idx]
            else:
                aux_grid = None

        # vmin/vmax on ROI
        usingQ = False if mask_mode else (True if kind != "direct" else (key in ("Qf", "Qn", "Qi")))
        roi_vals_for_range: List[float] = []
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                base_raw = grid[ii, jj]
                if not (math.isfinite(base_raw) and (signed_values or base_raw >= 0.0)):
                    continue
                if mask_mode:
                    roi_vals_for_range.append(base_raw)
                elif quantity == "fraction":
                    if math.isfinite(edep) and edep <= 0:
                        frac = 0.0
                    elif usingQ:
                        if q_total > 0:
                            frac = base_raw / q_total
                        else:
                            continue
                    else:
                        frac = base_raw
                    roi_vals_for_range.append(frac)
                elif quantity == "coulomb":
                    val = base_raw if usingQ else (base_raw * q_total)
                    roi_vals_for_range.append(val)
                elif quantity in ("d_i", "alpha_i"):
                    if aux_grid is None:
                        continue
                    aux_raw = aux_grid[ii, jj]
                    if math.isfinite(aux_raw):
                        roi_vals_for_range.append(aux_raw)

        if mask_mode:
            vmin, vmax = -0.5, 1.5
        elif quantity == "fraction":
            if signed_values and roi_vals_for_range:
                vmin, vmax = float(min(roi_vals_for_range)), float(max(roi_vals_for_range))
            else:
                vmin, vmax = 0.0, (max(roi_vals_for_range) if roi_vals_for_range else 1.0)
        else:
            if roi_vals_for_range:
                vmin, vmax = float(min(roi_vals_for_range)), float(max(roi_vals_for_range))
                if not (vmax > vmin):
                    vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = 0.0, 1.0

        # Prepare axes
        self.canvas.clear()
        ax = self.canvas.ax
        ax.set_xlim(pixelX - half_extent, pixelX + half_extent)
        ax.set_ylim(pixelY - half_extent, pixelY + half_extent)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        # Colormap and norm
        if mask_mode:
            cmap = mpl.colors.ListedColormap(["#bdbdbd", "#d32f2f"])  # 0=kept (gray), 1=removed (red)
            norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], ncolors=cmap.N)
        else:
            cmap = mpl.cm.get_cmap(cmap_name)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        # Valid mask and drawing
        alpha = 0.80
        valid = np.zeros((dim, dim), dtype=bool)
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                fval = grid[ii, jj]
                if not (math.isfinite(fval) and (signed_values or fval >= 0.0)):
                    continue
                valid[ii, jj] = True

                cx = pixelX + (jj - c) * pixel_spacing
                cy = pixelY + (ii - c) * pixel_spacing
                x1, x2 = cx - pixel_spacing / 2.0, cx + pixel_spacing / 2.0
                y1, y2 = cy - pixel_spacing / 2.0, cy + pixel_spacing / 2.0

                if mask_mode:
                    shown = fval
                elif quantity == "fraction":
                    if math.isfinite(edep) and edep <= 0:
                        shown = 0.0
                    else:
                        shown = (fval / q_total) if usingQ else fval
                elif quantity == "coulomb":
                    shown = fval if usingQ else (fval * q_total)
                elif quantity in ("d_i", "alpha_i"):
                    if aux_grid is None:
                        continue
                    shown = aux_grid[ii, jj]
                    if not math.isfinite(shown):
                        continue

                color = sm.to_rgba(shown)
                ax.add_patch(Rectangle((x1, y1), pixel_spacing, pixel_spacing, linewidth=1.0, edgecolor="black", facecolor=color, alpha=alpha))

                # Value label
                if mask_mode:
                    label = f"{int(round(shown))}"
                elif quantity == "coulomb":
                    label = f"{shown:.2e}"
                else:
                    label = f"{shown:.3f}"
                ax.text(cx, cy - pixel_size / 2.0 - 0.08, label, color="white", ha="center", va="center", fontsize=8)

        # Pixel squares overlay and borders where neighbor invalid
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                if not valid[ii, jj]:
                    continue
                cx = pixelX + (jj - c) * pixel_spacing
                cy = pixelY + (ii - c) * pixel_spacing
                x1, x2 = cx - pixel_size / 2.0, cx + pixel_size / 2.0
                y1, y2 = cy - pixel_size / 2.0, cy + pixel_size / 2.0
                ax.add_patch(Rectangle((x1, y1), pixel_size, pixel_size, linewidth=2.0, edgecolor="black", facecolor="none"))

        # Grid-like borders around missing neighbors
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                if not valid[ii, jj]:
                    continue
                cx = pixelX + (jj - c) * pixel_spacing
                cy = pixelY + (ii - c) * pixel_spacing
                x1, x2 = cx - pixel_spacing / 2.0, cx + pixel_spacing / 2.0
                y1, y2 = cy - pixel_spacing / 2.0, cy + pixel_spacing / 2.0
                # left
                if jj - 1 < 0 or not valid[ii, jj - 1]:
                    ax.plot([x1, x1], [y1, y2], color="black", linewidth=1.0)
                # right
                if jj + 1 >= dim or not valid[ii, jj + 1]:
                    ax.plot([x2, x2], [y1, y2], color="black", linewidth=1.0)
                # bottom
                if ii + 1 >= dim or not valid[ii + 1, jj]:
                    ax.plot([x1, x2], [y1, y1], color="black", linewidth=1.0)
                # top
                if ii - 1 < 0 or not valid[ii - 1, jj]:
                    ax.plot([x1, x2], [y2, y2], color="black", linewidth=1.0)

        # Hit marker
        hit_r = pixel_size / 6.0 if pixel_size > 0 else pixel_spacing / 6.0
        circ = plt.Circle((trueX, trueY), hit_r, color="red", fill=True, zorder=5)
        ax.add_patch(circ)

        # Title and colorbar
        if mask_mode:
            label = "Outlier removed (1=yes, 0=no)"
        else:
            label = {"coulomb": "Charge [C]", "fraction": "Charge Fraction", "d_i": "Distance [mm]", "alpha_i": "Alpha [rad]"}[quantity]
        ax.set_title(f"Event {evt} Charge Neighborhood\nHit: ({trueX:.3f}, {trueY:.3f}) mm,  Pixel: ({pixelX:.3f}, {pixelY:.3f}) mm   isPixelHit={int(is_pixel_hit)}")
        sm.set_array([])
        self.canvas.set_colorbar(sm, label)

        self.canvas.draw_idle()

    # --- PDF export ---
    def _on_export_pdf(self):
        if self._tree is None or self._n_entries <= 0:
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export PDF", os.getcwd(), "PDF (*.pdf)")
        if not out_path:
            return
        n_pages = self.pages_spin.value()
        start_evt = self.event_spin.value()
        try:
            self._export_pdf(out_path, start_evt, n_pages)
            QtWidgets.QMessageBox.information(self, "Export", f"Saved {n_pages} page(s) to:\n{out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed to export PDF:\n{e}")

    def _export_pdf(self, out_path: str, start_evt: int, n_pages: int):
        quantity = self.quantity_combo.currentData()
        kind, spec = self._selected_branch_spec()
        cmap_name = self.cmap_combo.currentData()
        pixel_spacing = float(self._meta.get("pixel_spacing", 0.0))
        pixel_size = float(self._meta.get("pixel_size", 0.0))
        try:
            r_meta = int(self._meta.get("neighborhood_radius", 2))
        except Exception:
            r_meta = 2
        roi_r = max(1, r_meta)
        half_pad = (roi_r + 0.5) * pixel_spacing

        with PdfPages(out_path) as pdf:
            for page in range(n_pages):
                evt = start_evt + page
                if evt >= self._n_entries:
                    break
                branches = ["TrueX", "TrueY", "PixelX", "PixelY", "Edep"]
                mask_mode = (kind == "mask")
                mask_sel = spec if mask_mode else None
                shape_key = None
                if kind == "direct" and spec is not None:
                    branches.append(spec)
                elif kind == "delta_abs":
                    try:
                        a, b = spec
                    except Exception:
                        a, b = "Qi", "Qf"
                    branches += [a, b]
                elif mask_mode:
                    if mask_sel == "mask_3d":
                        branches.append("Gauss3DMaskRemoved")
                    elif mask_sel == "mask_row":
                        branches.append("GaussRowMaskRemoved")
                    elif mask_sel == "mask_col":
                        branches.append("GaussColMaskRemoved")
                    # Shape source for 2D/3D masks
                    try:
                        available = set(self._tree.keys()) if self._tree is not None else set()
                    except Exception:
                        available = set()
                    for kshape in ["Qf", "Fi", "Qi", "Qn"]:
                        if kshape in available:
                            shape_key = kshape
                            branches.append(kshape)
                            break
                if quantity in ("d_i", "alpha_i"):
                    branches.append(quantity)
                arrs = self._tree.arrays(branches, entry_start=evt, entry_stop=evt + 1)

                def scalar(name: str) -> float:
                    x = arrs[name]
                    if isinstance(x, ak.Array):
                        return float(x[0])
                    return float(np.asarray(x)[0])

                trueX = scalar("TrueX")
                trueY = scalar("TrueY")
                pixelX = scalar("PixelX")
                pixelY = scalar("PixelY")
                edep = scalar("Edep")
                signed_values = False
                if mask_mode:
                    if shape_key is None:
                        values = np.array([], dtype=float)
                    else:
                        shape_vec = np.asarray(arrs[shape_key][0], dtype=float)
                        if shape_vec.size == 0:
                            values = np.array([], dtype=float)
                        else:
                            dim_guess = int(round(math.sqrt(float(shape_vec.size))))
                            if dim_guess * dim_guess != int(shape_vec.size) or dim_guess < 3:
                                values = np.array([], dtype=float)
                            else:
                                N = dim_guess
                                R = N // 2
                                values = np.full(N * N, np.nan, dtype=float)
                                if mask_sel == "mask_3d":
                                    mvec = np.asarray(arrs["Gauss3DMaskRemoved"][0], dtype=float)
                                    k = 0
                                    for di in range(-R, R + 1):
                                        for dj in range(-R, R + 1):
                                            idx_shape = (di + R) * N + (dj + R)
                                            q = shape_vec[idx_shape]
                                            if math.isfinite(q) and q >= 0.0:
                                                if k < mvec.size:
                                                    values[idx_shape] = mvec[k]
                                                    k += 1
                                elif mask_sel == "mask_row":
                                    mvec = np.asarray(arrs["GaussRowMaskRemoved"][0], dtype=float)
                                    k = 0
                                    dj = 0
                                    for di in range(-R, R + 1):
                                        idx_shape = (di + R) * N + (dj + R)
                                        q = shape_vec[idx_shape]
                                        if math.isfinite(q) and q >= 0.0:
                                            if k < mvec.size:
                                                values[idx_shape] = mvec[k]
                                                k += 1
                                elif mask_sel == "mask_col":
                                    mvec = np.asarray(arrs["GaussColMaskRemoved"][0], dtype=float)
                                    k = 0
                                    di = 0
                                    for dj in range(-R, R + 1):
                                        idx_shape = (di + R) * N + (dj + R)
                                        q = shape_vec[idx_shape]
                                        if math.isfinite(q) and q >= 0.0:
                                            if k < mvec.size:
                                                values[idx_shape] = mvec[k]
                                                k += 1
                elif kind == "direct" and spec is not None:
                    values = np.asarray(arrs[spec][0], dtype=float)
                else:
                    a_key, b_key = spec
                    a_arr = np.asarray(arrs[a_key][0], dtype=float)
                    b_arr = np.asarray(arrs[b_key][0], dtype=float)
                    d = a_arr - b_arr
                    values = np.abs(d)
                if values.size == 0:
                    continue
                dim = int(round(math.sqrt(float(values.size))))
                if dim * dim != int(values.size) or dim < 5:
                    continue

                grid = np.full((dim, dim), np.nan, dtype=float)
                for di in range(dim):
                    for dj in range(dim):
                        idx = di * dim + dj
                        grid[dj, di] = values[idx]
                c = dim // 2
                try:
                    r_meta = int(self._meta.get("neighborhood_radius", 2))
                except Exception:
                    r_meta = 2
                roi_r = max(1, min(c, r_meta))
                i0, i1 = c - roi_r, c + roi_r
                j0, j1 = c - roi_r, c + roi_r

                # q_total
                e_charge = 1.602176634e-19
                pair_e_ev = 3.60
                q_total = edep * 1.0e6 / pair_e_ev * e_charge if (edep > 0) else 0.0
                usingQ = False if mask_mode else (True if kind != "direct" else (spec in ("Qf", "Qn", "Qi")))

                # Optional aux grid for d_i / alpha_i
                aux_grid = None
                if quantity in ("d_i", "alpha_i"):
                    q_vals = np.asarray(arrs[quantity][0], dtype=float)
                    if q_vals.size == values.size:
                        aux_grid = np.full((dim, dim), np.nan, dtype=float)
                        for di in range(dim):
                            for dj in range(dim):
                                idx = di * dim + dj
                                aux_grid[dj, di] = q_vals[idx]

                # range
                roi_vals: List[float] = []
                for ii in range(i0, i1 + 1):
                    for jj in range(j0, j1 + 1):
                        raw = grid[ii, jj]
                        if not (math.isfinite(raw) and (signed_values or raw >= 0.0)):
                            continue
                        if quantity == "fraction":
                            if edep <= 0:
                                frac = 0.0
                            elif usingQ:
                                if q_total > 0:
                                    frac = raw / q_total
                                else:
                                    continue
                            else:
                                frac = raw
                            roi_vals.append(frac)
                        elif quantity == "coulomb":
                            roi_vals.append(raw if usingQ else (raw * q_total))
                        elif quantity in ("d_i", "alpha_i"):
                            if aux_grid is None:
                                continue
                            aux_raw = aux_grid[ii, jj]
                            if math.isfinite(aux_raw):
                                roi_vals.append(aux_raw)

                if mask_mode:
                    vmin, vmax = -0.5, 1.5
                elif quantity == "fraction":
                    if signed_values and roi_vals:
                        vmin, vmax = float(min(roi_vals)), float(max(roi_vals))
                    else:
                        vmin, vmax = 0.0, (max(roi_vals) if roi_vals else 1.0)
                else:
                    if roi_vals and max(roi_vals) > min(roi_vals):
                        vmin, vmax = float(min(roi_vals)), float(max(roi_vals))
                    else:
                        vmin, vmax = 0.0, 1.0

                # Figure
                fig = Figure(figsize=(6, 6), dpi=150)
                ax = fig.add_subplot(111)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(pixelX - half_pad, pixelX + half_pad)
                ax.set_ylim(pixelY - half_pad, pixelY + half_pad)
                ax.set_xlabel("x [mm]")
                ax.set_ylabel("y [mm]")
                fig.subplots_adjust(left=0.12, right=0.86, top=0.90, bottom=0.12)

                if mask_mode:
                    cmap = mpl.colors.ListedColormap(["#bdbdbd", "#d32f2f"])  # 0=kept, 1=removed
                    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], ncolors=cmap.N)
                else:
                    cmap = mpl.cm.get_cmap(cmap_name)
                    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

                valid = np.zeros((dim, dim), dtype=bool)
                for ii in range(i0, i1 + 1):
                    for jj in range(j0, j1 + 1):
                        fval = grid[ii, jj]
                        if not (math.isfinite(fval) and (signed_values or fval >= 0.0)):
                            continue
                        valid[ii, jj] = True
                        cx = pixelX + (jj - c) * pixel_spacing
                        cy = pixelY + (ii - c) * pixel_spacing
                        x1, y1 = cx - pixel_spacing / 2.0, cy - pixel_spacing / 2.0
                        if mask_mode:
                            shown = fval
                        elif quantity == "fraction":
                            if edep <= 0:
                                shown = 0.0
                            else:
                                shown = (fval / q_total) if usingQ else fval
                        elif quantity == "coulomb":
                            shown = fval if usingQ else (fval * q_total)
                        elif quantity in ("d_i", "alpha_i"):
                            if aux_grid is None:
                                continue
                            shown = aux_grid[ii, jj]
                            if not math.isfinite(shown):
                                continue
                        col = sm.to_rgba(shown)
                        ax.add_patch(Rectangle((x1, y1), pixel_spacing, pixel_spacing, linewidth=1.0, edgecolor="black", facecolor=col, alpha=0.80))

                # Overlay pixels
                for ii in range(i0, i1 + 1):
                    for jj in range(j0, j1 + 1):
                        if not valid[ii, jj]:
                            continue
                        cx = pixelX + (jj - c) * pixel_spacing
                        cy = pixelY + (ii - c) * pixel_spacing
                        x1, y1 = cx - pixel_size / 2.0, cy - pixel_size / 2.0
                        ax.add_patch(Rectangle((x1, y1), pixel_size, pixel_size, linewidth=2.0, edgecolor="black", facecolor="none"))

                # Hit
                hit_r = pixel_size / 6.0 if pixel_size > 0 else pixel_spacing / 6.0
                ax.add_patch(plt.Circle((trueX, trueY), hit_r, color="red", fill=True))

                if mask_mode:
                    label = "Outlier removed (1=yes, 0=no)"
                else:
                    label = {"coulomb": "Charge [C]", "fraction": "Charge Fraction", "d_i": "Distance [mm]", "alpha_i": "Alpha [rad]"}[quantity]
                ax.set_title(f"Event {evt} Charge Neighborhood")
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label(label)

                pdf.savefig(fig)
                plt.close(fig)


def main():
    app = QtWidgets.QApplication([])
    w = ChargeNeighborhoodGUI()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()


