#!/usr/bin/env python
"""Interactive S4M demo with a native PyQt5 GUI.

Left-drag  : pan
Wheel      : zoom to cursor
Left click : place major/minor points, then correction points
Right-drag : place a new major/minor pair anywhere
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "s4m-mpl"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as torch_f
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import register_all_modules
from mmengine.config import Config
from PyQt5.QtCore import QPointF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QKeyEvent, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from S4M.models.task_modules.prior_generators.prompt_encoder import EmbeddingIndex
from S4M.models.utils.custom_functional import multi_head_attention_forward

DEFAULT_IMAGE = "sample_dataset/sample_images/test.MMOTU_2d__00003.png"
DEFAULT_CONFIG = "S4M/configs/S4M/mmotu_majmin.py"
DEFAULT_CHECKPOINT = "UltraS4M_majmin.pth"

IMAGE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp *.pgm *.ppm "
    "*.jfif *.jp2 *.exr *.hdr *.pic *.dcm *.nrrd *.nii *.nii.gz *.mha *.mhd);;"
    "All files (*.*)"
)
PIL_CONVERT_SUFFIXES = {
    ".dcm", ".nrrd", ".nii", ".gz", ".mha", ".mhd", ".exr", ".hdr", ".pic", ".jp2"
}

MAJOR_COLOR = (31, 119, 180)
MINOR_COLOR = (174, 199, 232)
POS_COLOR = (0, 255, 0)
NEG_COLOR = (255, 0, 0)
MASK_COLOR = np.array([81, 112, 215], dtype=np.float32)

ZOOM_STEP = 1.25
ZOOM_MIN_FACTOR = 0.05
ZOOM_MAX_FACTOR = 40.0
POINT_RADIUS = 5
LINE_WIDTH = 2


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def needs_pil_conversion(image_path):
    suffix = image_path.suffix.lower()
    if suffix in PIL_CONVERT_SUFFIXES:
        return True
    try:
        return mmcv.imread(str(image_path), channel_order="bgr") is None
    except Exception:
        return True


def convert_with_pil(image_path, tmp_dir):
    from PIL import Image

    with Image.open(str(image_path)) as src:
        if src.mode in ("RGBA", "LA", "PA"):
            src = src.convert("RGBA")
            background = Image.new("RGBA", src.size, (0, 0, 0, 255))
            background.alpha_composite(src)
            src = background
        elif src.mode not in ("RGB", "L"):
            src = src.convert("RGB")
        out_path = Path(tmp_dir) / f"{image_path.stem or 'image'}.png"
        src.save(str(out_path))
    return out_path


def load_image_bgr(image_path, tmp_dir):
    image_path = Path(image_path)
    if needs_pil_conversion(image_path):
        try:
            image_path = convert_with_pil(image_path, tmp_dir)
        except Exception as exc:
            raise RuntimeError(f"Could not read {image_path}: {exc}") from exc
    image_bgr = mmcv.imread(str(image_path), channel_order="bgr")
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return image_bgr


def load_model(config, checkpoint, device):
    torch_f.multi_head_attention_forward = multi_head_attention_forward
    cfg_options = {"model.num_mask_refinements": 0}
    model = init_detector(
        str(config), str(checkpoint), device=device, cfg_options=cfg_options
    )
    model.eval()
    return model


def build_pipelines(config):
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config))
    image_steps = []
    pack_step = None
    annotation_section = False
    for step in cfg.test_pipeline:
        if step["type"] == "LoadAnnotations":
            annotation_section = True
            continue
        if step["type"].startswith("Pack"):
            pack_step = step
            continue
        if not annotation_section:
            image_steps.append(step)
    if pack_step is None:
        raise RuntimeError("Could not find a Pack* transform in the test pipeline.")
    return Compose(image_steps), Compose([pack_step])


def load_pipeline_image(image_pipeline, image_path):
    data = image_pipeline(dict(img_path=str(image_path), img_id=0))
    if data is None:
        raise RuntimeError(f"Could not load image through test pipeline: {image_path}")
    return data


def pair_length(pair):
    a, b = np.asarray(pair[0]), np.asarray(pair[1])
    return float(np.linalg.norm(a - b))


def order_major_minor(points):
    first_pair = points[:2]
    second_pair = points[2:4]
    if pair_length(first_pair) >= pair_length(second_pair):
        return np.asarray(first_pair + second_pair, dtype=np.float32)
    return np.asarray(second_pair + first_pair, dtype=np.float32)


def transform_points(image_data, points):
    points = np.asarray(points, dtype=np.float32)
    homography = image_data.get("homography_matrix")
    if homography is not None:
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.concatenate([points, ones], axis=1)
        transformed = points_h @ np.asarray(homography, dtype=np.float32).T
        return transformed[:, :2] / transformed[:, 2:3]
    scale = np.asarray(image_data["scale_factor"], dtype=np.float32)
    return points * scale


def bbox_from_points(points, width, height, padding=8.0):
    pts = np.asarray(points, dtype=np.float32)
    x1 = max(0.0, float(pts[:, 0].min() - padding))
    y1 = max(0.0, float(pts[:, 1].min() - padding))
    x2 = min(float(width - 1), float(pts[:, 0].max() + padding))
    y2 = min(float(height - 1), float(pts[:, 1].max() + padding))
    if x2 <= x1:
        x2 = min(float(width - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height - 1), y1 + 1.0)
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


def synthetic_mask(points, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) >= 3:
        hull = cv2.convexHull(np.round(pts).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
    else:
        x1, y1, x2, y2 = bbox_from_points(pts, width, height)[0].astype(np.int32)
        mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    if mask.sum() == 0:
        x, y = np.round(pts[0]).astype(np.int32)
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        mask[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = 1
    return BitmapMasks(mask[None], height, width)


def build_data(model, pack_pipeline, image_data, majmin_points, correction_points):
    ori_h, ori_w = image_data["ori_shape"]
    img_h, img_w = image_data["img_shape"]

    scaled_majmin = transform_points(image_data, order_major_minor(majmin_points))
    scaled_corrections = []
    correction_types = []
    for x, y, point_type in correction_points:
        scaled_corrections.append(transform_points(image_data, [(x, y)])[0])
        correction_types.append(point_type)

    results = image_data.copy()
    results["gt_bboxes"] = bbox_from_points(scaled_majmin, img_w, img_h)
    results["gt_bboxes_labels"] = np.zeros(1, dtype=np.int64)
    results["gt_masks"] = synthetic_mask(majmin_points, ori_w, ori_h)
    results["anatomical_pole_pools"] = scaled_majmin[None].astype(np.float32)

    packed = pack_pipeline(results)
    gt_instances = packed["data_samples"].gt_instances
    if scaled_corrections:
        gt_instances.interactive_points = torch.from_numpy(
            np.asarray(scaled_corrections, dtype=np.float32)[None]
        )
        gt_instances.interactive_points_types = torch.from_numpy(
            np.asarray(correction_types, dtype=np.int64)[None]
        )

    data = dict(inputs=[packed["inputs"]], data_samples=[packed["data_samples"]])
    return model.data_preprocessor(data, False)


@torch.no_grad()
def predict(model, pack_pipeline, image_data, majmin_points, correction_points):
    processed = build_data(
        model, pack_pipeline, image_data, majmin_points, correction_points
    )
    batch_inputs = processed["inputs"]
    batch_data_samples = processed["data_samples"]
    return model.predict(batch_inputs, batch_data_samples, rescale=True)[0]


def mask_to_numpy(data_sample):
    masks = data_sample.pred_instances.masks
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if masks.size == 0:
        return None
    return masks[0].astype(bool)


class InferenceWorker(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model, pack_pipeline, image_data, majmin, corrections):
        super().__init__()
        self.model = model
        self.pack_pipeline = pack_pipeline
        self.image_data = image_data
        self.majmin = majmin
        self.corrections = corrections

    def run(self):
        try:
            result = predict(
                self.model,
                self.pack_pipeline,
                self.image_data,
                self.majmin,
                self.corrections,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ImageCanvas(QWidget):
    """Displays a scaled image with pan (drag) and zoom (wheel) support."""

    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._mask_overlay = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._dragging = False
        self._drag_active = False
        self._drag_start_pos = None
        self._offset_start = QPointF(0, 0)
        self.click_callback = None
        self.major_minor_points = []
        self.correction_points = []
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 480)

    def set_image(self, image_rgb, major_minor, corrections):
        self._pixmap = numpy_to_pixmap(image_rgb)
        self.major_minor_points = major_minor
        self.correction_points = corrections
        self.fit()
        self.update()

    def set_scene(self, major_minor, corrections):
        self.major_minor_points = major_minor
        self.correction_points = corrections
        self.update()

    def fit(self):
        if self._pixmap is None:
            return
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._normalize_offset()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_at(self, widget_pos, factor):
        if self._pixmap is None:
            return
        new_zoom = float(np.clip(self._zoom * factor, ZOOM_MIN_FACTOR, ZOOM_MAX_FACTOR))
        actual = new_zoom / self._zoom
        if actual == 1.0:
            return
        # Keep the image point under the cursor fixed while zooming.
        # offset is the position of the image center relative to the widget center.
        pw, ph = self._pixmap_size()
        cx = self.width() / 2 + self._offset.x()
        cy = self.height() / 2 + self._offset.y()
        p = QPointF(widget_pos)
        self._offset = QPointF(
            (cx - p.x()) * actual + p.x() - self.width() / 2,
            (cy - p.y()) * actual + p.y() - self.height() / 2,
        )
        self._zoom = new_zoom
        self._normalize_offset()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def _normalize_offset(self):
        if self._pixmap is None:
            return
        pw, ph = self._pixmap_size()
        half_x, half_y = pw * self._zoom / 2, ph * self._zoom / 2
        view_half_x, view_half_y = self.width() / 2, self.height() / 2
        lo_x = min(view_half_x, half_x)
        hi_x = max(-view_half_x, -half_x)
        lo_y = min(view_half_y, half_y)
        hi_y = max(-view_half_y, -half_y)
        x = float(np.clip(self._offset.x(), min(hi_x, lo_x), max(hi_x, lo_x)))
        y = float(np.clip(self._offset.y(), min(hi_y, lo_y), max(hi_y, lo_y)))
        self._offset = QPointF(x, y)

    def _pixmap_size(self):
        if self._pixmap is None:
            return 0, 0
        return self._pixmap.width(), self._pixmap.height()

    def _image_origin(self):
        """Widget position of image pixel (0, 0)."""
        pw, ph = self._pixmap_size()
        return QPointF(
            self.width() / 2 + self._offset.x() - pw * self._zoom / 2,
            self.height() / 2 + self._offset.y() - ph * self._zoom / 2,
        )

    def widget_to_image(self, pos):
        if self._pixmap is None:
            return None
        p = QPointF(pos)
        origin = self._image_origin()
        x = (p.x() - origin.x()) / self._zoom
        y = (p.y() - origin.y()) / self._zoom
        w, h = self._pixmap_size()
        if not (0 <= x < w and 0 <= y < h):
            return None
        return x, y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._pixmap is None:
            painter.end()
            return
        pw, ph = self._pixmap_size()
        origin = self._image_origin()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.save()
        painter.translate(origin)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.restore()

        pen = QPen()
        pen.setWidthF(max(1.0, LINE_WIDTH * min(1.0, self._zoom)))
        pts = self.major_minor_points
        if len(pts) >= 2:
            ordered = (
                order_major_minor(pts)
                if len(pts) >= 4
                else np.asarray([pts[0], pts[1]], dtype=np.float32)
            )
            for pair, color, style in (
                (ordered[:2], MAJOR_COLOR, Qt.SolidLine),
                (
                    ordered[2:4],
                    MINOR_COLOR,
                    Qt.DashLine if len(pts) >= 4 else Qt.SolidLine,
                ),
            ):
                if len(pair) == 2:
                    pen.setColor(QColor_from_rgb(color))
                    painter.setPen(pen)
                    painter.drawLine(
                        int(pair[0][0]), int(pair[0][1]), int(pair[1][0]), int(pair[1][1])
                    )
        for idx, (x, y) in enumerate(pts):
            color = MINOR_COLOR
            if len(pts) == 4:
                ordered = order_major_minor(pts)
                color = (
                    MAJOR_COLOR
                    if any(np.allclose([x, y], p) for p in ordered[:2])
                    else MINOR_COLOR
                )
            self._draw_point(painter, x, y, QColor_from_rgb(color), label=None)
        for x, y, ptype in self.correction_points:
            color = POS_COLOR if ptype == EmbeddingIndex.POS.value else NEG_COLOR
            self._draw_point(painter, x, y, QColor_from_rgb(color), label=None)
        painter.end()

    def _draw_point(self, painter, x, y, color, label=None):
        origin = self._image_origin()
        cx = origin.x() + x * self._zoom
        cy = origin.y() + y * self._zoom
        r = POINT_RADIUS
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor_from_rgb((255, 255, 255)))
        painter.drawEllipse(QPointF(cx, cy), r + 2, r + 2)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._dragging = True
            self._drag_start_pos = event.pos()
            self._drag_active = False
            self._offset_start = QPointF(self._offset)
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            pos = self.widget_to_image(event.pos())
            if pos is not None:
                self._emit_click(pos, positive=not bool(event.modifiers() & Qt.ControlModifier))

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start_pos is not None:
            delta = event.pos() - self._drag_start_pos
            if not self._drag_active and (abs(delta.x()) + abs(delta.y())) > 6:
                self._drag_active = True
            if self._drag_active:
                self._offset = QPointF(
                    self._offset_start.x() + delta.x(),
                    self._offset_start.y() + delta.y(),
                )
                self._normalize_offset()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.MiddleButton):
            return
        was_active = self._drag_active
        self._dragging = False
        self._drag_active = False
        self._drag_start_pos = None
        self.setCursor(Qt.ArrowCursor)
        if event.button() == Qt.LeftButton and not was_active:
            pos = self.widget_to_image(event.pos())
            if pos is not None:
                self._emit_click(pos, positive=None)

    def _emit_click(self, pos, positive):
        callback = getattr(self, "click_callback", None)
        if callback is not None:
            callback(pos, positive)

    def mouseDoubleClickEvent(self, event):
        self.fit()

    def wheelEvent(self, event):
        factor = ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / ZOOM_STEP
        self.zoom_at(event.pos(), factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)


from PyQt5.QtCore import QRectF, QPointF, Qt, QThread, pyqtSignal  # noqa: E402
from PyQt5.QtGui import QColor  # noqa: E402


def QRect_from_center(center, w, h):
    return QRectF(center.x() - w / 2, center.y() - h / 2, w, h)


def QColor_from_rgb(rgb):
    return QColor(rgb[0], rgb[1], rgb[2])


def numpy_to_pixmap(image_rgb):
    h, w, c = image_rgb.shape
    if c == 3:
        image_rgb = np.ascontiguousarray(image_rgb)
        img = QImage(image_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(img.copy())
    if c == 4:
        image_rgb = np.ascontiguousarray(image_rgb)
        img = QImage(image_rgb.data, w, h, 4 * w, QImage.Format_RGBA8888)
        return QPixmap.fromImage(img.copy())
    raise ValueError(f"Unsupported channel count: {c}")


class OverlayComposer:
    """Renders base image + mask overlay into an RGB image."""

    @staticmethod
    def compose(image_rgb, mask):
        canvas = image_rgb.astype(np.float32).copy()
        if mask is not None:
            overlay = canvas.copy()
            overlay[mask] = overlay[mask] * 0.45 + MASK_COLOR * 0.55
            canvas = overlay
        return canvas.astype(np.uint8)


class MainWindow(QMainWindow):
    def __init__(self, model, image_pipeline, pack_pipeline, image_path, tmp_dir, device):
        super().__init__()
        self.model = model
        self.image_pipeline = image_pipeline
        self.pack_pipeline = pack_pipeline
        self.tmp_dir = Path(tmp_dir)
        self.device = device

        self.image_rgb = None
        self.image_data = None
        self.image_path = None
        self.mask = None
        self.major_minor_points = []
        self.correction_points = []
        self.positive_mode = True
        self.worker = None
        self.tmp_file_map = {}

        self.setWindowTitle("S4M — 4-points to Segment Anything")
        self.resize(1280, 900)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = ImageCanvas(central)
        self.canvas.click_callback = self.on_canvas_click
        layout.addWidget(self.canvas, 1)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(12, 8, 12, 8)

        self.positive_check = QCheckBox("Positive")
        self.negative_check = QCheckBox("Negative")
        self.positive_check.setChecked(True)
        self.positive_check.toggled.connect(self.on_mode_positive)
        self.negative_check.toggled.connect(self.on_mode_negative)

        self.segment_btn = QPushButton("Segment / Update")
        self.segment_btn.clicked.connect(self.on_segment)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.on_undo)
        self.reset_btn = QPushButton("Reset points")
        self.reset_btn.clicked.connect(self.on_reset)
        self.open_btn = QPushButton("Open image...")
        self.open_btn.clicked.connect(self.on_open)
        self.fit_btn = QPushButton("Fit view")
        self.fit_btn.clicked.connect(self.canvas.fit)
        self.save_btn = QPushButton("Save mask...")
        self.save_btn.clicked.connect(self.on_save_mask)

        controls_layout.addWidget(self.positive_check)
        controls_layout.addWidget(self.negative_check)
        controls_layout.addStretch(1)
        for btn in (
            self.segment_btn,
            self.undo_btn,
            self.reset_btn,
            self.open_btn,
            self.fit_btn,
            self.save_btn,
        ):
            controls_layout.addWidget(btn)

        layout.addWidget(controls)

        toolbar = QToolBar("main")
        self.addToolBar(toolbar)
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.setCentralWidget(central)
        self.status.showMessage(
            "Left-click pairs: points 1-2 (major) and 3-4 (minor) | wheel = zoom | "
            "drag = pan | double-click = fit | Ctrl+Right-click = force positive"
        )

    def on_mode_positive(self, checked):
        if checked:
            self.negative_check.blockSignals(True)
            self.negative_check.setChecked(False)
            self.negative_check.blockSignals(False)
            self.positive_mode = True

    def on_mode_negative(self, checked):
        if checked:
            self.positive_check.blockSignals(True)
            self.positive_check.setChecked(False)
            self.positive_check.blockSignals(False)
            self.positive_mode = False

    def on_canvas_click(self, pos, positive):
        x, y = pos
        if len(self.major_minor_points) < 4:
            self.major_minor_points.append((x, y))
            self.canvas.set_scene(self.major_minor_points, self.correction_points)
            if len(self.major_minor_points) == 4:
                self.on_segment()
            else:
                self.status.showMessage(
                    f"Major/minor point {len(self.major_minor_points)}/4 "
                    "(pairs: 1-2, 3-4)"
                )
            return
        if positive is None:
            positive = self.positive_mode
        ptype = EmbeddingIndex.POS.value if positive else EmbeddingIndex.NEG.value
        self.correction_points.append((x, y, ptype))
        self.canvas.set_scene(self.major_minor_points, self.correction_points)
        self.status.showMessage(
            f"Corrections: {len(self.correction_points)} "
            f"(+{sum(1 for p in self.correction_points if p[2] == EmbeddingIndex.POS.value)}"
            f"/-{sum(1 for p in self.correction_points if p[2] == EmbeddingIndex.NEG.value)})"
        )

    def on_segment(self):
        if self.worker is not None:
            return
        if len(self.major_minor_points) < 4 or self.image_data is None:
            self.status.showMessage("Click four major/minor points first.")
            return
        self.status.showMessage("Running S4M inference...")
        self.segment_btn.setEnabled(False)
        self.worker = InferenceWorker(
            self.model,
            self.pack_pipeline,
            self.image_data,
            list(self.major_minor_points),
            list(self.correction_points),
        )
        self.worker.finished_ok.connect(self.on_inference_done)
        self.worker.failed.connect(self.on_inference_failed)
        self.worker.start()

    def on_inference_done(self, data_sample):
        self.mask = mask_to_numpy(data_sample)
        self.refresh_overlay()
        self.segment_btn.setEnabled(True)
        self.worker = None
        self.status.showMessage("Segmentation updated.")

    def on_inference_failed(self, message):
        self.segment_btn.setEnabled(True)
        self.worker = None
        self.status.showMessage(f"S4M failed: {message}")

    def refresh_overlay(self):
        if self.image_rgb is None:
            return
        self.canvas.set_image(
            OverlayComposer.compose(self.image_rgb, self.mask),
            self.major_minor_points,
            self.correction_points,
        )
        self.canvas.set_scene(self.major_minor_points, self.correction_points)

    def on_undo(self):
        if self.worker is not None:
            return
        if self.correction_points:
            self.correction_points.pop()
        elif self.major_minor_points:
            self.major_minor_points.pop()
            self.mask = None
            self.refresh_overlay()
        self.canvas.set_scene(self.major_minor_points, self.correction_points)
        self.status.showMessage(
            f"Points: {len(self.major_minor_points)} major/minor, "
            f"{len(self.correction_points)} corrections"
        )

    def on_reset(self):
        if self.worker is not None:
            return
        self.major_minor_points = []
        self.correction_points = []
        self.mask = None
        self.refresh_overlay()
        self.status.showMessage("Reset. Click four major/minor points.")

    def on_open(self):
        if self.worker is not None:
            return
        start_dir = str(self.image_path.parent) if self.image_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", start_dir, IMAGE_FILTER
        )
        if not path:
            return
        self.load_image(Path(path))

    def on_save_mask(self):
        if self.mask is None:
            self.status.showMessage("No mask to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mask", "", "PNG image (*.png);;All files (*.*)"
        )
        if not path:
            return
        out = (self.mask.astype(np.uint8)) * 255
        cv2.imwrite(path, out)
        self.status.showMessage(f"Saved mask: {path}")

    def load_image(self, path):
        try:
            image_bgr = load_image_bgr(path, self.tmp_dir)
            image_data = load_pipeline_image(self.image_pipeline, path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.image_path = path
        self.image_rgb = mmcv.bgr2rgb(image_bgr)
        self.image_data = image_data
        self.major_minor_points = []
        self.correction_points = []
        self.mask = None
        self.refresh_overlay()
        self.status.showMessage(
            f"Loaded {path.name} ({image_data['ori_shape'][1]}x"
            f"{image_data['ori_shape'][0]}). Click four major/minor points."
        )

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.quit()
            self.worker.wait(2000)
        event.accept()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Image to segment.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    image_path = resolve_path(args.image)
    if not image_path.exists():
        path, _ = QFileDialog.getOpenFileName(
            None, "Open image", "", IMAGE_FILTER
        )
        if not path:
            return 0
        image_path = Path(path)

    with tempfile.TemporaryDirectory(prefix="s4m_qt_") as tmp_dir:
        image_pipeline, pack_pipeline = build_pipelines(config_path)
        image_bgr = load_image_bgr(image_path, tmp_dir)
        image_data = load_pipeline_image(image_pipeline, image_path)
        model = load_model(config_path, checkpoint_path, args.device)

        window = MainWindow(model, image_pipeline, pack_pipeline, image_path, tmp_dir, args.device)
        window.image_path = image_path
        window.image_rgb = mmcv.bgr2rgb(image_bgr)
        window.image_data = image_data
        window.refresh_overlay()
        window.show()
        return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())