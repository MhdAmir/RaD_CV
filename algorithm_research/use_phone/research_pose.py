import argparse
import csv
import dataclasses
import math
import time
import warnings
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore")

NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LEL, REL, LWR, RWR = 5, 6, 7, 8, 9, 10
LHIP, RHIP = 11, 12
LANK, RANK = 15, 16

GATE_KEYS = ["g1_walk", "g2_gaze", "g3_zone", "g4_no_cross", "g5_freeze"]
GATE_BADGES = ["WALK", "GAZE", "ZONE", "NO_CROSS", "FREEZE"]

GREEN = (80, 220, 60)
RED = (70, 70, 240)
ORANGE = (60, 160, 250)
BLUE = (245, 160, 50)
CYAN = (240, 240, 60)
YELLOW = (50, 230, 240)
WHITE = (250, 250, 250)
DARK = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX

TELEMETRY_HEADER = [
    "video", "frame", "t_sec", "track_id", "crop_file",
    "g1_walk", "g2_gaze", "g3_zone", "g4_no_cross", "g5_freeze", "pass_ratio_45",
    "walk_disp", "walk_depth", "face_conf", "gaze_cos_l", "gaze_cos_r",
    "near_ear_l", "near_ear_r", "gaze_ray_d_l", "gaze_ray_d_r",
    "nape_ratio", "elbow_deg_l", "elbow_deg_r",
    "wrist_above_hip_l", "wrist_above_hip_r", "zone_d_l", "zone_d_r",
    "cross_l", "cross_r", "var_l", "var_r", "arm_len_l", "arm_len_r",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "e2e_ms", "infer_ms", "fps",
]
VERBOSE_HEADER = [
    "video", "frame", "t_sec", "track_id",
    "g1_walk", "g2_gaze", "g3_zone", "g4_no_cross", "g5_freeze",
    "all5", "gate6_pass", "pass_ratio_45",
    "walk_disp", "walk_depth", "gait_swing", "face_conf", "var_l", "var_r",
    "gaze_cos_l", "gaze_cos_r", "near_ear_l", "near_ear_r",
    "gaze_ray_d_l", "gaze_ray_d_r",
    "zone_d_l", "zone_d_r", "nape_ratio", "yaw", "head_pitch", "e2e_ms",
]
SUMMARY_HEADER = [
    "run_tag", "video", "frames", "wall_s", "avg_fps", "avg_e2e_ms",
    "pre_ms", "infer_ms", "post_ms", "unique_tracks", "fired_tracks", "crop_events",
    "kpt_backend", "kpt_ms",
]

PROFILES = {
    "frontal": {},
    "oblique": {"zone_ratio": 0.85, "gait_swing": 0.020, "walk_disp": 0.35},
    "side": {"gaze_cos": 0.55, "side_view_frac": 0.10},
    "topdown": {"gait_swing": 0.0, "zone_ratio": 0.90},
}


@dataclass
class Config:
    model: str = r"C:\cctv-insight\ai\models\yolo26m-pose.pt"
    imgsz: int = 640
    quantize: int = 0
    conf: float = 0.25
    iou: float = 0.55
    device: int = 0
    tracker: str = "bytetrack.yaml"

    kpt_min: float = 0.30
    kpt_soft: float = 0.05
    min_bbox_h: float = 24.0
    min_torso: float = 25.0
    face_conf: float = 0.50
    gaze_cos: float = 0.60
    gaze_min_len: float = 0.06
    near_ear_ratio: float = 0.28
    gaze_ray_max: float = 0.25
    gaze_proj_min: float = 0.05
    nape_ratio: float = 0.22
    elbow_bend_deg: float = 100.0
    elbow_medial_frac: float = 0.45
    profile_bend_deg: float = 110.0
    pitch_min: float = 0.10
    yaw_max: float = 0.45
    side_view_frac: float = 0.20
    walk_win: int = 30
    walk_disp: float = 0.40
    walk_depth_ratio: float = 0.25
    gait_swing: float = 0.025
    walk_win_sec: float = 2.5
    zone_ratio: float = 0.75
    cross_frac: float = 0.30
    freeze_win: int = 30
    freeze_var: float = 0.008
    frozen_len_min: float = 0.20
    frozen_bend_deg: float = 135.0
    freeze_win_sec: float = 1.0
    slide_win: int = 45
    slide_ratio: float = 0.75
    slide_win_sec: float = 1.5
    missing_reset: int = 60
    max_gap: int = 90

    out_scale: float = 1.0
    display: bool = False
    save_video: bool = True
    max_frames: int = 0
    dump_kpts: bool = False

    kpt_backend: str = "yolo"
    rtm_model: str = str(Path.home() / ".cache/rtmlib/hub/checkpoints/"
                         "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx")
    rtm_input_w: int = 192
    rtm_input_h: int = 256
    rtm_trt: bool = False
    rtm_trt_cache: str = "trt_cache"


class RTMPoseBatch:
    """Top-down RTMPose on tracker boxes, all persons of a frame in one ONNX run."""

    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(self, cfg):
        import onnxruntime as ort
        from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
        from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

        self._cs = bbox_xyxy2cs
        self._affine = top_down_affine
        self._decode = get_simcc_maximum
        self.input_size = (cfg.rtm_input_w, cfg.rtm_input_h)
        providers = [("CUDAExecutionProvider", {"device_id": cfg.device}), "CPUExecutionProvider"]
        if cfg.rtm_trt:
            Path(cfg.rtm_trt_cache).mkdir(parents=True, exist_ok=True)
            providers.insert(0, ("TensorrtExecutionProvider", {
                "device_id": cfg.device, "trt_fp16_enable": True,
                "trt_engine_cache_enable": True, "trt_engine_cache_path": cfg.rtm_trt_cache,
                "trt_timing_cache_enable": True, "trt_timing_cache_path": cfg.rtm_trt_cache,
                # dynamic batch: build one engine covering 1..32 persons
                "trt_profile_min_shapes": f"input:1x3x{cfg.rtm_input_h}x{cfg.rtm_input_w}",
                "trt_profile_opt_shapes": f"input:8x3x{cfg.rtm_input_h}x{cfg.rtm_input_w}",
                "trt_profile_max_shapes": f"input:32x3x{cfg.rtm_input_h}x{cfg.rtm_input_w}",
            }))
        self.session = ort.InferenceSession(cfg.rtm_model, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.max_batch = 32 if cfg.rtm_trt else 64
        print(f"rtmpose={Path(cfg.rtm_model).name} provider={self.session.get_providers()[0]}")

    def __call__(self, frm, bboxes):
        if not bboxes:
            return np.zeros((0, 17, 3), dtype=np.float32)
        imgs, centers, scales = [], [], []
        for bb in bboxes:
            center, scale = self._cs(np.asarray(bb, dtype=np.float32), padding=1.25)
            img, scale = self._affine(self.input_size, scale, center, frm)
            imgs.append((img.astype(np.float32) - self.MEAN) / self.STD)
            centers.append(center)
            scales.append(scale)
        batch = np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2), dtype=np.float32)
        sx, sy = [], []
        for i in range(0, len(batch), self.max_batch):
            ox, oy = self.session.run(None, {self.input_name: batch[i:i + self.max_batch]})
            sx.append(ox)
            sy.append(oy)
        locs, scores = self._decode(np.concatenate(sx), np.concatenate(sy))
        scale = np.asarray(scales, dtype=np.float32)[:, None, :]
        center = np.asarray(centers, dtype=np.float32)[:, None, :]
        kpts = locs / 2.0 / np.asarray(self.input_size, dtype=np.float32) * scale + center - scale / 2
        # SimCC scores slightly exceed 1; clip so YOLO-calibrated thresholds stay meaningful
        conf = np.clip(scores, 0.0, 1.0)[..., None]
        return np.concatenate([kpts, conf], axis=-1).astype(np.float32)


class TrackState:
    def __init__(self, cfg, track_id, frame_idx):
        self.track_id = track_id
        self.centers = deque(maxlen=cfg.walk_win)
        self.ank_dy = deque(maxlen=cfg.walk_win)
        self.arm_vecs = deque(maxlen=cfg.freeze_win)
        self.nose_hist = deque(maxlen=15)
        self.gates_hist = deque(maxlen=cfg.slide_win)
        self.fired = False
        self.n_frames = 0
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.ready = False
        self.ratio = 0.0

    def reset(self, frame_idx):
        self.centers.clear()
        self.ank_dy.clear()
        self.arm_vecs.clear()
        self.nose_hist.clear()
        self.gates_hist.clear()
        self.n_frames = 0
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.ready = False
        self.ratio = 0.0


def conf_of(k, i):
    return float(k[i][2])


def xy_of(k, i, cmin):
    x, y, c = k[i]
    return (float(x), float(y)) if c >= cmin else None


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cos_sim(a, b):
    na = math.hypot(a[0], a[1])
    nb = math.hypot(b[0], b[1])
    if na < 1e-6 or nb < 1e-6:
        return -1.0
    return (a[0] * b[0] + a[1] * b[1]) / (na * nb)


def tri_angle(a, b, c):
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    na = math.hypot(bax, bay)
    nb = math.hypot(bcx, bcy)
    if na < 1e-6 or nb < 1e-6:
        return math.nan
    cosv = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (na * nb)))
    return math.degrees(math.acos(cosv))


def evaluate(k, x1, y1, x2, y2, state, cfg, m):
    g = {key: False for key in GATE_KEYS}

    def close_out():
        state.gates_hist.append(False)
        state.ready = len(state.gates_hist) >= cfg.slide_win
        state.ratio = float(np.mean(state.gates_hist))
        return g, False, state.ready, state.ratio

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = max(y2 - y1, 1.0)
    state.centers.append((cx, cy, bh))
    ank_ok = conf_of(k, LANK) >= cfg.kpt_min and conf_of(k, RANK) >= cfg.kpt_min
    state.ank_dy.append(float(k[LANK][1] - k[RANK][1]) if ank_ok else math.nan)
    if bh < cfg.min_bbox_h:
        return close_out()

    if len(state.centers) >= cfg.walk_win:
        x0, y0, bh0 = state.centers[0]
        mean_bh = sum(c[2] for c in state.centers) / len(state.centers)
        disp = math.hypot(cx - x0, cy - y0) / max(mean_bh, 1.0)
        depth = abs(bh - bh0) / max(bh0, 1.0)
        m["walk_disp"] = disp
        m["walk_depth"] = depth
        dy = np.asarray(state.ank_dy, dtype=float)
        swing = float(np.nanstd(dy)) / max(mean_bh, 1.0)
        m["gait_swing"] = swing
        gait_ok = (not math.isnan(swing) and cfg.gait_swing > 0
                   and swing > cfg.gait_swing)
        g["g1_walk"] = disp > cfg.walk_disp or depth > cfg.walk_depth_ratio or gait_ok

    sh_l = xy_of(k, LSH, cfg.kpt_min)
    sh_r = xy_of(k, RSH, cfg.kpt_min)
    hip_l = xy_of(k, LHIP, cfg.kpt_min)
    hip_r = xy_of(k, RHIP, cfg.kpt_min)
    if sh_l is None or sh_r is None:
        return close_out()

    sh_mid = (0.5 * (sh_l[0] + sh_r[0]), 0.5 * (sh_l[1] + sh_r[1]))
    shoulder_w = abs(sh_l[0] - sh_r[0])
    hip_pts = [p for p in (hip_l, hip_r) if p is not None]
    hip_mid = None
    torso = 0.0
    if hip_pts:
        hip_mid = (sum(p[0] for p in hip_pts) / len(hip_pts),
                   sum(p[1] for p in hip_pts) / len(hip_pts))
        if hip_mid[1] > sh_mid[1]:
            torso = dist(sh_mid, hip_mid)
    m["torso"] = torso

    chest_mid = None
    if torso >= cfg.min_torso and hip_mid is not None:
        chest_mid = (sh_mid[0] + 0.35 * (hip_mid[0] - sh_mid[0]),
                     sh_mid[1] + 0.35 * (hip_mid[1] - sh_mid[1]))
        m["chest_x"], m["chest_y"] = chest_mid

    nose = xy_of(k, NOSE, cfg.kpt_soft)
    eye_l = xy_of(k, LEYE, cfg.kpt_soft)
    eye_r = xy_of(k, REYE, cfg.kpt_soft)
    ear_l = xy_of(k, LEAR, cfg.kpt_soft)
    ear_r = xy_of(k, REAR, cfg.kpt_soft)
    face_c = float(np.mean([conf_of(k, NOSE), conf_of(k, LEYE), conf_of(k, REYE)]))
    m["face_conf"] = face_c
    if nose is not None:
        state.nose_hist.append(nose)

    ear_pts = [p for p in (ear_l, ear_r) if p is not None]
    eye_pts = [p for p in (eye_l, eye_r) if p is not None]
    head_ref = None
    if ear_pts:
        head_ref = (sum(p[0] for p in ear_pts) / len(ear_pts),
                    sum(p[1] for p in ear_pts) / len(ear_pts))
    elif eye_pts:
        head_ref = (sum(p[0] for p in eye_pts) / len(eye_pts),
                    sum(p[1] for p in eye_pts) / len(eye_pts))

    arms = {}
    for side, wi, ei, si in (("l", LWR, LEL, LSH), ("r", RWR, REL, RSH)):
        arms[side] = (xy_of(k, wi, cfg.kpt_min), xy_of(k, ei, cfg.kpt_soft),
                      xy_of(k, si, cfg.kpt_min))

    row = []
    for side in ("l", "r"):
        w, e, s = arms[side]
        if w and s and torso > 0:
            vx = (w[0] - s[0]) / torso
            vy = (w[1] - s[1]) / torso
            row.extend([vx, vy, math.hypot(vx, vy)])
        else:
            row.extend([math.nan, math.nan, math.nan])
    state.arm_vecs.append(tuple(row))

    frozen = {}
    if len(state.arm_vecs) >= cfg.freeze_win:
        arr = np.asarray(state.arm_vecs, dtype=float)
        for side, si_v, si_len in (("l", 0, 2), ("r", 3, 5)):
            var = float(np.nanvar(arr[:, si_v]) + np.nanvar(arr[:, si_v + 1]))
            mean_len = float(np.nanmean(arr[:, si_len]))
            w_cur, e_cur, s_cur = arms[side]
            bend_ok = True
            if e_cur and s_cur and w_cur:
                ang_now = tri_angle(s_cur, e_cur, w_cur)
                bend_ok = math.isnan(ang_now) or ang_now < cfg.frozen_bend_deg
            m[f"var_{side}"] = var
            m[f"arm_len_{side}"] = mean_len
            if mean_len >= cfg.frozen_len_min and bend_ok:
                frozen[side] = var < cfg.freeze_var
        if frozen:
            g["g5_freeze"] = any(frozen.values())

    gaze = None
    if nose is not None and head_ref is not None:
        gdx, gdy = nose[0] - head_ref[0], nose[1] - head_ref[1]
        if math.hypot(gdx, gdy) >= cfg.gaze_min_len * max(torso, 1.0):
            gaze = (gdx, gdy)

    yaw = None
    head_pitch = None
    if nose is not None and ear_l is not None and ear_r is not None:
        d_l = dist(nose, ear_l)
        d_r = dist(nose, ear_r)
        if d_l + d_r > 1e-3:
            yaw = (d_r - d_l) / (d_l + d_r)
    if nose is not None and head_ref is not None and torso >= 40:
        head_pitch = min(0.6, max(-0.3, (nose[1] - head_ref[1]) / torso))
    m["yaw"] = yaw
    m["head_pitch"] = head_pitch

    g2_facing = False
    if face_c >= cfg.face_conf and torso > 0 and chest_mid is not None and hip_mid is not None:
        for side in ("l", "r"):
            w, e, s = arms[side]
            if w and s:
                m[f"gaze_cos_{side}"] = cos_sim(gaze, (w[0] - s[0], w[1] - s[1])) if gaze else None
                near = min((dist(w, p) for p in ear_pts), default=None)
                if near is not None:
                    m[f"near_ear_{side}"] = near / torso
                ray_ok = False
                if gaze and nose is not None:
                    g_len = math.hypot(gaze[0], gaze[1])
                    unit = (gaze[0] / g_len, gaze[1] / g_len)
                    to_hx, to_hy = w[0] - nose[0], w[1] - nose[1]
                    proj = to_hx * unit[0] + to_hy * unit[1]
                    perp = math.hypot(to_hx - proj * unit[0], to_hy - proj * unit[1])
                    m[f"gaze_ray_d_{side}"] = perp / torso
                    m[f"gaze_proj_{side}"] = proj / torso
                    ray_ok = (proj / torso > cfg.gaze_proj_min
                              and perp / torso < cfg.gaze_ray_max)
                    m[f"gaze_ray_ok_{side}"] = float(ray_ok)
                if gaze and m[f"gaze_cos_{side}"] is not None and m[f"gaze_cos_{side}"] > cfg.gaze_cos:
                    g2_facing = True
                if near is not None and near / torso < cfg.near_ear_ratio:
                    g2_facing = True
                if e:
                    ang = tri_angle(s, e, w)
                    zone_d = min(dist(w, chest_mid),
                                 min((dist(w, p) for p in ear_pts), default=math.inf)) / torso
                    bowed = head_pitch is not None and head_pitch > cfg.pitch_min
                    if (not math.isnan(ang) and ang < cfg.profile_bend_deg
                            and w[1] < hip_mid[1] and zone_d < cfg.zone_ratio and bowed):
                        g2_facing = True

    g2_back = False
    if face_c < cfg.face_conf and ear_pts and torso > 0:
        nape = (head_ref[1] - sh_mid[1]) / torso
        m["nape_ratio"] = nape
        bent_any = False
        for side in ("l", "r"):
            w, e, s = arms[side]
            if e and s:
                ang = tri_angle(s, e, w) if w else math.nan
                m[f"elbow_deg_{side}"] = ang
                medial = shoulder_w > 8 and abs(e[0] - sh_mid[0]) <= cfg.elbow_medial_frac * shoulder_w
                if not math.isnan(ang) and ang < cfg.elbow_bend_deg and medial:
                    bent_any = True
        g2_back = nape > cfg.nape_ratio and bent_any

    yaw_reject = False
    if yaw is not None and abs(yaw) > cfg.yaw_max and torso > 0 and hip_mid is not None:
        turned_side = "r" if yaw < 0 else "l"
        w_t, e_t, s_t = arms[turned_side]
        hand_up = (w_t is not None and chest_mid is not None
                   and w_t[1] < hip_mid[1]
                   and min(dist(w_t, chest_mid), min((dist(w_t, p) for p in ear_pts),
                                                     default=math.inf)) / torso < cfg.zone_ratio)
        if not hand_up:
            yaw_reject = True
    if gaze and nose is not None:
        g_len = math.hypot(gaze[0], gaze[1])
        m["gaze_ux"] = gaze[0] / g_len
        m["gaze_uy"] = gaze[1] / g_len
        ray_any = m.get("gaze_ray_ok_l") == 1.0 or m.get("gaze_ray_ok_r") == 1.0
        if yaw_reject:
            m["gaze_draw"] = 3
        elif ray_any:
            m["gaze_draw"] = 2
        else:
            m["gaze_draw"] = 1
    g["g2_gaze"] = (g2_facing or g2_back) and not yaw_reject

    if torso > 0 and chest_mid is not None:
        hip_y = hip_mid[1]
        for side in ("l", "r"):
            w = arms[side][0]
            if w:
                above = w[1] < hip_y
                d_chest = dist(w, chest_mid) / torso
                d_ear = min((dist(w, p) for p in ear_pts), default=math.inf) / torso
                zone_d = min(d_chest, d_ear)
                m[f"wrist_above_hip_{side}"] = float(above)
                m[f"zone_d_{side}"] = zone_d
                if above and zone_d < cfg.zone_ratio:
                    g["g3_zone"] = True

    crossed = {}
    view_frontal = bh > 0 and shoulder_w >= cfg.side_view_frac * bh
    if view_frontal:
        basis = torso if torso >= cfg.min_torso else shoulder_w
        mid_x = sh_mid[0] if hip_mid is None else 0.5 * (sh_mid[0] + hip_mid[0])
        for side in ("l", "r"):
            w, e, s = arms[side]
            if w and e:
                d_w = w[0] - mid_x
                d_e = e[0] - mid_x
                cross = d_w * d_e < 0 and abs(d_w) > cfg.cross_frac * basis and abs(d_w) > abs(d_e)
                crossed[side] = cross
                m[f"cross_{side}"] = float(cross)
        g["g4_no_cross"] = not any(crossed.values())
    else:
        g["g4_no_cross"] = True
        m["g4_uneval"] = 1.0

    ok = all(g.values())
    state.gates_hist.append(ok)
    state.ready = len(state.gates_hist) >= cfg.slide_win
    state.ratio = float(np.mean(state.gates_hist))
    return g, ok, state.ready, state.ratio


def save_crop(frm, k, m, cfg, path, x1, y1, x2, y2):
    pts = []
    if k is not None:
        for i, cmin in ((LSH, cfg.kpt_min), (RSH, cfg.kpt_min),
                        (LEL, cfg.kpt_soft), (REL, cfg.kpt_soft),
                        (LWR, cfg.kpt_min), (RWR, cfg.kpt_min)):
            p = xy_of(k, i, cmin)
            if p is not None:
                pts.append(p)
        if m.get("chest_x") is not None:
            pts.append((m["chest_x"], m["chest_y"]))
    h, w = frm.shape[:2]
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        side = max(span_x, span_y, 48.0) + 2 * 0.35 * max(span_x, span_y, 48.0)
        cxx = 0.5 * (max(xs) + min(xs))
        cyy = 0.5 * (max(ys) + min(ys))
    else:
        side = max(x2 - x1, 0.6 * (y2 - y1))
        cxx = 0.5 * (x1 + x2)
        cyy = y1 + 0.55 * (y2 - y1)
    side = int(max(min(side, min(h, w)), 32))
    x0 = int(max(0, min(w - side, cxx - side / 2)))
    y0 = int(max(0, min(h - side, cyy - side / 2)))
    crop = frm[y0:min(h, y0 + side), x0:min(w, x0 + side)]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), crop)
    return path.name


def draw_person(frm, tid, x1, y1, x2, y2, g, ok, ratio, ready, fired, k, cfg, m, state):
    n_pass = sum(1 for key in GATE_KEYS if g[key])
    if fired:
        color = RED
    elif ok:
        color = GREEN
    elif n_pass >= 3:
        color = ORANGE
    else:
        color = BLUE
    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frm, (ix1, iy1), (ix2, iy2), color, 3 if fired else 2)

    nose_p = xy_of(k, NOSE, 0.05)
    for hx, hy in state.nose_hist:
        cv2.circle(frm, (int(hx), int(hy)), 2, (150, 220, 255), -1)
    if nose_p:
        gaze_draw = m.get("gaze_draw")
        ux, uy = m.get("gaze_ux"), m.get("gaze_uy")
        arrow_len = 0.35 * (m.get("torso") or 0.0)
        if gaze_draw and ux is not None and arrow_len > 20:
            tip = (int(nose_p[0] + ux * arrow_len), int(nose_p[1] + uy * arrow_len))
            acol = {1: YELLOW, 2: GREEN, 3: RED}[gaze_draw]
            cv2.arrowedLine(frm, (int(nose_p[0]), int(nose_p[1])), tip, acol, 2, tipLength=0.35)
        cv2.circle(frm, (int(nose_p[0]), int(nose_p[1])), 3, (150, 220, 255), -1)

    sh_l = xy_of(k, LSH, cfg.kpt_min)
    sh_r = xy_of(k, RSH, cfg.kpt_min)
    hip_l = xy_of(k, LHIP, cfg.kpt_min)
    hip_r = xy_of(k, RHIP, cfg.kpt_min)
    if sh_l and sh_r and hip_l and hip_r:
        cv2.line(frm, (int(sh_l[0]), int(sh_l[1])), (int(sh_r[0]), int(sh_r[1])), GREEN, 1)
        cv2.line(frm, (int(hip_l[0]), int(hip_l[1])), (int(hip_r[0]), int(hip_r[1])), (200, 120, 120), 1)
        cv2.line(frm, (int((sh_l[0] + sh_r[0]) / 2), int((sh_l[1] + sh_r[1]) / 2)),
                 (int((hip_l[0] + hip_r[0]) / 2), int((hip_l[1] + hip_r[1]) / 2)), GREEN, 1)
    for idx, rad, col in ((LWR, 4, CYAN), (RWR, 4, CYAN), (LEL, 2, (60, 180, 255)), (REL, 2, (60, 180, 255))):
        p = xy_of(k, idx, 0.05)
        if p:
            cv2.circle(frm, (int(p[0]), int(p[1])), rad, col, -1)

    h_img = frm.shape[0]
    y_badge = max(iy1 - 10, 52)
    y_id = max(y_badge - 17, 34)
    y_head = max(y_id - 17, 18)
    y_g6 = max(y_head - 17, 8)
    x_txt = ix1
    for i, label in enumerate(GATE_BADGES):
        txt = f" {label} "
        (tw, _), _ = cv2.getTextSize(txt, FONT, 0.40, 1)
        if x_txt + tw > frm.shape[1] - 4:
            break
        col = GREEN if g[GATE_KEYS[i]] else RED
        cv2.putText(frm, txt, (x_txt, y_badge), FONT, 0.40, col, 1, cv2.LINE_AA)
        x_txt += tw + 2
    if fired:
        viol_txt = f"ID{tid}  PELANGGARAN!"
        cv2.putText(frm, viol_txt, (ix1, min(y_id, h_img - 4)), FONT, 0.48, RED, 2, cv2.LINE_AA)
    else:
        cv2.putText(frm, f"ID{tid}", (ix1, min(y_id, h_img - 4)), FONT, 0.48, WHITE, 1, cv2.LINE_AA)
    yaw_s = f"{m['yaw']:.2f}" if m.get("yaw") is not None else "-"
    pitch_s = f"{m['head_pitch']:.2f}" if m.get("head_pitch") is not None else "-"
    cv2.putText(frm, f"yaw {yaw_s} pitch {pitch_s}", (ix1, min(y_head, h_img - 22)),
                FONT, 0.38, (200, 240, 250), 1, cv2.LINE_AA)
    if not fired and ready:
        g6txt = f"G6 {ratio * 100:.0f}%/75%"
        col6 = GREEN if ratio >= cfg.slide_ratio else ORANGE
        cv2.putText(frm, g6txt, (ix1, min(y_g6, h_img - 30)), FONT, 0.42, col6, 1, cv2.LINE_AA)


def draw_global(frm, text):
    cv2.putText(frm, text, (10, 24), FONT, 0.55, DARK, 3, cv2.LINE_AA)
    cv2.putText(frm, text, (10, 24), FONT, 0.55, (150, 255, 255), 1, cv2.LINE_AA)


def init_csv(path, header):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        with p.open("w", newline="") as f:
            csv.writer(f).writerow(header)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    if isinstance(v, float):
        return round(v, 4)
    return v


def run_video(video, cfg, model, pose_est, out_dir, telemetry_writer, telemetry_handle,
              verbose_writer, verbose_handle):
    vpath = Path(video)
    stem = vpath.stem
    cap = cv2.VideoCapture(str(vpath))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    cfg = replace(cfg,
                  walk_win=max(10, round(src_fps * cfg.walk_win_sec)) if cfg.walk_win_sec > 0 else cfg.walk_win,
                  freeze_win=max(8, round(src_fps * cfg.freeze_win_sec)) if cfg.freeze_win_sec > 0 else cfg.freeze_win,
                  slide_win=max(15, round(src_fps * cfg.slide_win_sec)) if cfg.slide_win_sec > 0 else cfg.slide_win)
    print(f"[{stem}] fps={src_fps:.0f} walk_win={cfg.walk_win}f freeze_win={cfg.freeze_win}f slide_win={cfg.slide_win}f")

    crop_dir = out_dir / "candidates_crop"
    crop_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    out_w = max(2, int(src_w * cfg.out_scale))
    out_h = max(2, int(src_h * cfg.out_scale))
    if cfg.save_video:
        writer = cv2.VideoWriter(str(out_dir / f"annotated_{stem}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (out_w, out_h))

    quantize_arg = cfg.quantize if cfg.quantize in (16,) else None
    gen = model.track(source=str(vpath), stream=True, persist=True, tracker=cfg.tracker,
                      classes=[0], conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz,
                      quantize=quantize_arg, device=cfg.device, verbose=False)

    dump = {"frame": [], "track_id": [], "bbox": [], "kpts": []}
    states = {}
    ids_seen = set()
    fired = set()
    timings = []
    sp_sum = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    sp_n = 0
    kpt_sum = 0.0
    n = 0
    events = 0
    t_iter = None
    t_start = time.perf_counter()

    for r in gen:
        now = time.perf_counter()
        e2e = (now - t_iter) * 1000.0 if t_iter is not None else math.nan
        t_iter = now
        n += 1
        frm = r.orig_img

        if n > 5:
            timings.append(e2e)
            if r.speed:
                sp_n += 1
                for key in sp_sum:
                    val = r.speed.get(key)
                    if val is not None:
                        sp_sum[key] += val

        boxes = r.boxes
        kps = np.zeros((0, 17, 3))
        if r.keypoints is not None:
            kd = r.keypoints.data
            kps = kd.cpu().numpy() if hasattr(kd, "cpu") else np.asarray(kd)

        ids = []
        if boxes is not None and boxes.id is not None:
            ids = [int(v) for v in boxes.id.tolist()]

        if pose_est is not None:
            t_k = time.perf_counter()
            kps = pose_est(frm, boxes.xyxy[:len(ids)].tolist() if ids else [])
            kpt_ms = (time.perf_counter() - t_k) * 1000.0
            if n > 5:
                kpt_sum += kpt_ms

        fps_now = 1000.0 * len(timings) / max(sum(timings), 1e-6) if timings else math.nan

        for i, tid in enumerate(ids):
            ids_seen.add(tid)
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            st = states.get(tid)
            if st is not None and n - st.last_frame > cfg.missing_reset:
                st.reset(n)
            elif st is None:
                st = TrackState(cfg, tid, n)
                states[tid] = st
            st.last_frame = n
            st.n_frames += 1

            m = {}
            k = kps[i] if i < len(kps) else None
            if cfg.dump_kpts and k is not None:
                dump["frame"].append(n)
                dump["track_id"].append(tid)
                dump["bbox"].append((x1, y1, x2, y2))
                dump["kpts"].append(k[:17, :3])
            if k is not None:
                g, ok, ready, ratio = evaluate(k, x1, y1, x2, y2, st, cfg, m)
            else:
                g = {key: False for key in GATE_KEYS}
                ok = False
                st.gates_hist.append(False)
                st.ready = len(st.gates_hist) >= st.gates_hist.maxlen
                st.ratio = float(np.mean(st.gates_hist))
                ready, ratio = st.ready, st.ratio

            infer_ms = r.speed.get("inference") if r.speed else None
            row_base = {
                "video": stem, "frame": n, "t_sec": fmt(n / src_fps), "track_id": tid,
                "g1_walk": int(g["g1_walk"]), "g2_gaze": int(g["g2_gaze"]),
                "g3_zone": int(g["g3_zone"]), "g4_no_cross": int(g["g4_no_cross"]),
                "g5_freeze": int(g["g5_freeze"]),
                "pass_ratio_45": fmt(ratio), "walk_disp": fmt(m.get("walk_disp")),
                "walk_depth": fmt(m.get("walk_depth")), "face_conf": fmt(m.get("face_conf")),
                "e2e_ms": fmt(e2e), "infer_ms": fmt(infer_ms), "fps": fmt(fps_now),
            }
            verbose_writer.writerow([
                row_base["video"], row_base["frame"], row_base["t_sec"], row_base["track_id"],
                row_base["g1_walk"], row_base["g2_gaze"], row_base["g3_zone"],
                row_base["g4_no_cross"], row_base["g5_freeze"], int(ok),
                int(ready and ratio >= cfg.slide_ratio), row_base["pass_ratio_45"],
                row_base["walk_disp"], row_base["walk_depth"], fmt(m.get("gait_swing")),
                row_base["face_conf"], fmt(m.get("var_l")), fmt(m.get("var_r")),
                fmt(m.get("gaze_cos_l")), fmt(m.get("gaze_cos_r")),
                fmt(m.get("near_ear_l")), fmt(m.get("near_ear_r")),
                fmt(m.get("gaze_ray_d_l")), fmt(m.get("gaze_ray_d_r")),
                fmt(m.get("zone_d_l")), fmt(m.get("zone_d_r")), fmt(m.get("nape_ratio")),
                fmt(m.get("yaw")), fmt(m.get("head_pitch")),
                row_base["e2e_ms"],
            ])

            if ready and ratio >= cfg.slide_ratio and not st.fired:
                st.fired = True
                fired.add(tid)
                events += 1
                crop_name = None
                if k is not None:
                    crop_name = save_crop(frm, k, m, cfg,
                                          crop_dir / f"{stem}_id{tid}_f{n:06d}.jpg",
                                          x1, y1, x2, y2)
                telemetry_writer.writerow([
                    row_base["video"], row_base["frame"], row_base["t_sec"], row_base["track_id"],
                    crop_name or "",
                    row_base["g1_walk"], row_base["g2_gaze"], row_base["g3_zone"],
                    row_base["g4_no_cross"], row_base["g5_freeze"], row_base["pass_ratio_45"],
                    row_base["walk_disp"], row_base["walk_depth"], row_base["face_conf"],
                    fmt(m.get("gaze_cos_l")), fmt(m.get("gaze_cos_r")),
                    fmt(m.get("near_ear_l")), fmt(m.get("near_ear_r")),
                    fmt(m.get("gaze_ray_d_l")), fmt(m.get("gaze_ray_d_r")),
                    fmt(m.get("nape_ratio")), fmt(m.get("elbow_deg_l")), fmt(m.get("elbow_deg_r")),
                    fmt(m.get("wrist_above_hip_l")), fmt(m.get("wrist_above_hip_r")),
                    fmt(m.get("zone_d_l")), fmt(m.get("zone_d_r")),
                    fmt(m.get("cross_l")), fmt(m.get("cross_r")),
                    fmt(m.get("var_l")), fmt(m.get("var_r")),
                    fmt(m.get("arm_len_l")), fmt(m.get("arm_len_r")),
                    fmt(x1), fmt(y1), fmt(x2), fmt(y2),
                    row_base["e2e_ms"], row_base["infer_ms"], row_base["fps"],
                ])

            if k is not None:
                draw_person(frm, tid, x1, y1, x2, y2, g, ok, ratio, ready, st.fired, k, cfg, m, st)

        for tid in [t for t, s in states.items() if n - s.last_frame > cfg.max_gap]:
            del states[tid]

        active = sum(1 for s in states.values() if s.last_frame == n)
        hud = (f"{stem}  f{n}  {fps_now:.1f} FPS  E2E {np.mean(timings[-30:]):.0f}ms"
               f"  track {active}  uniq {len(ids_seen)}  fired {len(fired)}")
        draw_global(frm, hud)

        disp = frm
        if cfg.out_scale != 1.0:
            disp = cv2.resize(frm, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if writer is not None:
            writer.write(disp)
        if cfg.display:
            cv2.imshow("pose_gate", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if cfg.max_frames and n >= cfg.max_frames:
            break

    wall = time.perf_counter() - t_start
    avg_ms = float(np.mean(timings)) if timings else 0.0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    sp_avg = {key: (sp_sum[key] / sp_n if sp_n else 0.0) for key in sp_sum}
    kpt_avg = kpt_sum / len(timings) if timings and pose_est is not None else 0.0
    telemetry_handle.flush()
    verbose_handle.flush()
    if writer is not None:
        writer.release()
    if cfg.dump_kpts:
        np.savez_compressed(out_dir / f"kpts_{stem}.npz",
                            frame=np.asarray(dump["frame"], dtype=np.int32),
                            track_id=np.asarray(dump["track_id"], dtype=np.int32),
                            bbox=np.asarray(dump["bbox"], dtype=np.float32).reshape(-1, 4),
                            kpts=np.asarray(dump["kpts"], dtype=np.float32).reshape(-1, 17, 3),
                            src_w=src_w, src_h=src_h, src_fps=src_fps)
    if cfg.display:
        cv2.destroyAllWindows()

    print(f"[{stem}] frames={n} wall={wall:.1f}s avgFPS={avg_fps:.1f} avgE2E={avg_ms:.1f}ms "
          f"pre={sp_avg['preprocess']:.2f} infer={sp_avg['inference']:.2f} post={sp_avg['postprocess']:.2f}ms "
          f"kpt[{cfg.kpt_backend}]={kpt_avg:.2f}ms uniq={len(ids_seen)} fired={len(fired)} events={events}")

    return {
        "video": stem, "frames": n, "wall_s": round(wall, 1),
        "avg_fps": round(avg_fps, 1), "avg_e2e_ms": round(avg_ms, 1),
        "pre_ms": round(sp_avg["preprocess"], 2), "infer_ms": round(sp_avg["inference"], 2),
        "post_ms": round(sp_avg["postprocess"], 2),
        "unique_tracks": len(ids_seen), "fired_tracks": len(fired), "crop_events": events,
        "kpt_backend": cfg.kpt_backend, "kpt_ms": round(kpt_avg, 2),
    }


def apply_set(cfg, pairs):
    for kv in pairs:
        key, val = kv.split("=", 1)
        ann = Config.__annotations__[key]
        if ann is bool:
            setattr(cfg, key, val.lower() in ("1", "true", "yes"))
        elif ann is int:
            setattr(cfg, key, int(val))
        elif ann is float:
            setattr(cfg, key, float(val))
        else:
            setattr(cfg, key, val)


def parse_args():
    out_base = Path(__file__).resolve().parent / "research_output"
    p = argparse.ArgumentParser(
        description="Phase-1 pose-gate research: YOLO26m-Pose + ByteTrack + 5 geometry gates + 45-frame sliding window")
    p.add_argument("--source", nargs="+",
                   default=[r"C:\cctv-dev-temp\DEV_Gerbang_Depan.mp4"])
    p.add_argument("--out-dir", default=str(out_base))
    p.add_argument("--model", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--quantize", type=int, choices=[0, 16], default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--display", action="store_true")
    p.add_argument("--no-save-video", action="store_true")
    p.add_argument("--kpt-backend", choices=["yolo", "rtmpose"], default=None,
                   help="keypoint source: YOLO pose head, or RTMPose top-down on tracker boxes")
    p.add_argument("--rtm-model", default=None, help="RTMPose ONNX path (default: rtmlib cache rtmpose-m)")
    p.add_argument("--rtm-trt", action="store_true", help="run RTMPose with TensorRT FP16")
    p.add_argument("--dump-kpts", action="store_true",
                   help="save per-track keypoints to kpts_<video>.npz for offline lifting")
    p.add_argument("--out-scale", type=float, default=None)
    p.add_argument("--camera-profile", choices=["none", "frontal", "oblique", "side", "topdown"],
                   default="none")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override Config, e.g. --set walk_disp=0.3 slide_ratio=0.7")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.model:
        cfg.model = args.model
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
    if args.quantize is not None:
        cfg.quantize = args.quantize
    if args.conf is not None:
        cfg.conf = args.conf
    if args.iou is not None:
        cfg.iou = args.iou
    if args.device is not None:
        cfg.device = args.device
    if args.out_scale is not None:
        cfg.out_scale = args.out_scale
    if args.max_frames is not None:
        cfg.max_frames = args.max_frames
    cfg.display = args.display
    if args.no_save_video:
        cfg.save_video = False
    cfg.dump_kpts = args.dump_kpts
    if args.kpt_backend is not None:
        cfg.kpt_backend = args.kpt_backend
    if args.rtm_model:
        cfg.rtm_model = args.rtm_model
    if args.rtm_trt:
        cfg.rtm_trt = True
    for kv, ov in PROFILES.get(args.camera_profile, {}).items():
        setattr(cfg, kv, ov)
    apply_set(cfg, args.set)

    if args.camera_profile == "topdown":
        print("WARNING: topdown profile is best-effort; COCO pose models degrade on bird's-eye views")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "telemetry_log.csv"
    verbose_path = out_dir / "verbose_gate_log.csv"
    summary_path = out_dir / "run_summary.csv"

    print(f"model={cfg.model} imgsz={cfg.imgsz} quantize={cfg.quantize} conf={cfg.conf} "
          f"iou={cfg.iou} tracker={cfg.tracker} out={out_dir}")
    print(f"thresholds: walk_disp={cfg.walk_disp} depth={cfg.walk_depth_ratio} gait={cfg.gait_swing} "
          f"gaze_cos={cfg.gaze_cos} near_ear={cfg.near_ear_ratio} nape={cfg.nape_ratio} "
          f"zone={cfg.zone_ratio} cross={cfg.cross_frac} freeze_var={cfg.freeze_var} "
          f"win_sec: walk={cfg.walk_win_sec} freeze={cfg.freeze_win_sec} slide={cfg.slide_win_sec} "
          f"slide_ratio={cfg.slide_ratio:.0%}")

    model = YOLO(cfg.model)
    pose_est = RTMPoseBatch(cfg) if cfg.kpt_backend == "rtmpose" else None
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    sum_rows = []

    for csv_path, header in ((telemetry_path, TELEMETRY_HEADER),
                             (verbose_path, VERBOSE_HEADER),
                             (summary_path, SUMMARY_HEADER)):
        init_csv(csv_path, header)

    with telemetry_path.open("a", newline="") as tf, verbose_path.open("a", newline="") as vf:
        tw = csv.writer(tf)
        vw = csv.writer(vf)
        for src in args.source:
            stats = run_video(src, cfg, model, pose_est, out_dir, tw, tf, vw, vf)
            sum_rows.append(stats)

    with summary_path.open("a", newline="") as sf:
        sw = csv.writer(sf)
        for row in sum_rows:
            sw.writerow([run_tag] + [row[col] for col in SUMMARY_HEADER[1:]])
        sf.flush()

    print("---- run summary ----")
    for row in sum_rows:
        print(f"[{row['video']}] fps={row['avg_fps']} e2e={row['avg_e2e_ms']}ms "
              f"infer={row['infer_ms']}ms tracks={row['unique_tracks']} "
              f"fired={row['fired_tracks']} events={row['crop_events']}")


if __name__ == "__main__":
    main()