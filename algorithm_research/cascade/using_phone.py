from collections import defaultdict, deque
import csv
from dataclasses import dataclass
from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO


# ==========================================
# 1. KONFIGURASI THRESHOLD RISET (MUDAH DI-TUNE)
# ==========================================
@dataclass
class ResearchConfig:
  model_path: str = "yolo26m-pose.pt"  # Bisa diganti yolov8n-pose.pt / .engine
  conf_person: float = 0.45
  kpt_conf: float = 0.40

  # Parameter Sliding Window (30 FPS)
  motion_window: int = 30  # 1.0 detik untuk cek jalan & kekakuan lengan
  decision_window: int = 45  # 1.5 detik untuk keputusan akhir
  vote_ratio_thresh: float = 0.72  # Minimal 72% frame dalam window harus positif

  # Gate 1: Syarat Berjalan (rasio perpindahan thd tinggi bbox selama 30 frame)
  min_walk_displacement: float = (
      0.08  # Naikkan jika orang berdiri masih terdeteksi
  )

  # Gate 2 & 3: Sudut Pandang & Zona Tangan (dinormalisasi thd panjang torso)
  min_gaze_cos: float = 0.65  # Cosine similarity pandangan ke tangan (~49 derajat)
  max_wrist_to_chest: float = (
      0.65  # Jarak maks tangan ke tengah dada ( x panjang torso )
  )
  max_wrist_to_ear: float = (
      0.40  # Jarak maks tangan ke telinga (kasus menelepon)
  )
  min_elbow_angle: float = 35.0  # Derajat minimal tekukan siku
  max_elbow_angle: float = (
      135.0  # Derajat maksimal tekukan siku (>140 = lengan lurus)
  )

  # Gate 5: Arm Motion Freeze (Variansi vektor lengan-bahu selama 30 frame)
  max_arm_variance: float = (
      0.018  # Semakin kecil = lengan harus semakin kaku/diam
  )

  # Output Riset
  output_dir: str = "research_output"
  crop_size: int = 224


# Indeks Keypoint COCO (17 titik)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHLDR, R_SHLDR, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16


# ==========================================
# 2. FUNGSI MATEMATIKA VEKTOR O(1)
# ==========================================
def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
  """Menghitung sudut interior (derajat) pada titik sendi b (a - b - c)."""
  ba = a - b
  bc = c - b
  norm_ba = np.linalg.norm(ba)
  norm_bc = np.linalg.norm(bc)
  if norm_ba < 1e-5 or norm_bc < 1e-5:
    return 180.0
  cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
  return float(np.degrees(np.arccos(cos_angle)))


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
  """Menghitung kemiripan arah dua vektor (-1.0 s/d 1.0)."""
  n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
  if n1 < 1e-5 or n2 < 1e-5:
    return -1.0
  return float(np.dot(v1, v2) / (n1 * n2))


# ==========================================
# 3. EVALUATOR PERILAKU PER TRACK_ID
# ==========================================
class TrackState:

  def __init__(self, cfg: ResearchConfig):
    self.center_hist = deque(maxlen=cfg.motion_window)
    self.l_arm_hist = deque(maxlen=cfg.motion_window)
    self.r_arm_hist = deque(maxlen=cfg.motion_window)
    self.vote_hist = deque(maxlen=cfg.decision_window)
    self.saved_crop_count = 0
    self.last_save_frame = -999


def evaluate_person_pose(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    bbox_xyxy: np.ndarray,
    state: TrackState,
    cfg: ResearchConfig,
) -> dict:
  """Mengevaluasi 5 aturan geometri & temporal tanpa classifier."""
  x1, y1, x2, y2 = bbox_xyxy
  bbox_h = max(y2 - y1, 1.0)
  center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
  state.center_hist.append(center)

  # Pastikan bahu dan pinggul terdeteksi untuk menghitung skala panjang torso
  if (
      kpts_conf[L_SHLDR] < cfg.kpt_conf and kpts_conf[R_SHLDR] < cfg.kpt_conf
  ) or (kpts_conf[L_HIP] < cfg.kpt_conf and kpts_conf[R_HIP] < cfg.kpt_conf):
    state.vote_hist.append(0)
    return {"valid": False}

  # Titik tengah bahu, dada, dan pinggul
  shldr_mid = (kpts_xy[L_SHLDR] + kpts_xy[R_SHLDR]) / 2.0
  hip_mid = (kpts_xy[L_HIP] + kpts_xy[R_HIP]) / 2.0
  chest_pt = shldr_mid + 0.35 * (hip_mid - shldr_mid)
  torso_len = max(float(np.linalg.norm(hip_mid - shldr_mid)), bbox_h * 0.28)
  shldr_width = max(
      float(np.linalg.norm(kpts_xy[L_SHLDR] - kpts_xy[R_SHLDR])), 10.0
  )

  # --------------------------------------------------
  # GATE 1: Locomotion (Apakah orang ini berjalan?)
  # --------------------------------------------------
  if len(state.center_hist) >= 10:
    disp = np.linalg.norm(state.center_hist[-1] - state.center_hist[0]) / bbox_h
    scale_factor = cfg.motion_window / len(state.center_hist)
    walk_score = disp * scale_factor
    is_walking = walk_score >= cfg.min_walk_displacement
  else:
    walk_score = 0.0
    is_walking = False

  # --------------------------------------------------
  # GATE 4: Anti-Crossed Arms (Cek melipat tangan di dada)
  # --------------------------------------------------
  is_crossed = False
  if kpts_conf[L_WRIST] > cfg.kpt_conf and kpts_conf[R_WRIST] > cfg.kpt_conf:
    wrist_dist_ratio = (
        np.linalg.norm(kpts_xy[L_WRIST] - kpts_xy[R_WRIST]) / shldr_width
    )
    # Cek apakah tangan kiri & kanan saling menyilang melewati sumbu X siku lawan
    l_to_r_elbow = (
        np.linalg.norm(kpts_xy[L_WRIST] - kpts_xy[R_ELBOW]) / torso_len
    )
    r_to_l_elbow = (
        np.linalg.norm(kpts_xy[R_WRIST] - kpts_xy[L_ELBOW]) / torso_len
    )
    if l_to_r_elbow < 0.32 and r_to_l_elbow < 0.32 and wrist_dist_ratio > 0.35:
      is_crossed = True

  # --------------------------------------------------
  # Evaluasi Lengan Kiri & Kanan (Gate 2, 3, 5)
  # --------------------------------------------------
  face_visible = (
      kpts_conf[NOSE] >= 0.50
      and kpts_conf[L_EYE] >= 0.45
      and kpts_conf[R_EYE] >= 0.45
  )
  eye_mid = (kpts_xy[L_EYE] + kpts_xy[R_EYE]) / 2.0
  gaze_vec = kpts_xy[NOSE] - eye_mid

  active_hand_found = False
  gaze_aligned = False
  arm_frozen = False
  best_wrist_xy = chest_pt
  best_arm_var = 999.0
  best_gaze_cos = 0.0

  sides = [
      ("L", L_SHLDR, L_ELBOW, L_WRIST, L_EAR, state.l_arm_hist),
      ("R", R_SHLDR, R_ELBOW, R_WRIST, R_EAR, state.r_arm_hist),
  ]

  for side_name, s_idx, e_idx, w_idx, ear_idx, arm_hist in sides:
    if kpts_conf[w_idx] < cfg.kpt_conf or kpts_conf[e_idx] < cfg.kpt_conf:
      continue

    shldr, elbow, wrist = kpts_xy[s_idx], kpts_xy[e_idx], kpts_xy[w_idx]

    # Simpan vektor relatif lengan-ke-bahu untuk dihitung variansinya
    rel_arm_vec = (wrist - shldr) / torso_len
    arm_hist.append(rel_arm_vec)

    # Gate 3: Sudut siku menekuk & tangan di atas pinggul (bukan di saku)
    elbow_angle = calc_angle(shldr, elbow, wrist)
    above_hip = wrist[1] < (hip_mid[1] - 0.05 * torso_len)
    dist_chest = np.linalg.norm(wrist - chest_pt) / torso_len
    dist_ear = (
        np.linalg.norm(wrist - kpts_xy[ear_idx]) / torso_len
        if kpts_conf[ear_idx] > 0.35
        else 99.0
    )

    in_phone_zone = (
        (cfg.min_elbow_angle <= elbow_angle <= cfg.max_elbow_angle)
        and above_hip
        and (
            dist_chest <= cfg.max_wrist_to_chest
            or dist_ear <= cfg.max_wrist_to_ear
        )
    )

    if not in_phone_zone:
      continue
    active_hand_found = True

    # Gate 2: Arah Pandang ke Tangan (atau Menelepon / Membelakangi Kamera)
    is_calling = dist_ear <= cfg.max_wrist_to_ear
    if face_visible:
      target_vec = wrist - kpts_xy[NOSE]
      g_cos = cosine_sim(gaze_vec, target_vec)
      # Valid jika mata menghadap tangan, atau posisi hidung menunduk mendekati dada, atau menelepon
      nose_drop = (kpts_xy[NOSE][1] - eye_mid[1]) / max(
          np.linalg.norm(kpts_xy[L_EYE] - kpts_xy[R_EYE]), 4.0
      )
      side_gaze_ok = (
          (g_cos >= cfg.min_gaze_cos) or (nose_drop > 0.55) or is_calling
      )
    else:
      # Kasus Membelakangi Kamera (Rear-View): Cek tengkuk menunduk (telinga turun mendekati bahu)
      g_cos = 0.0
      ears_y = (
          min(kpts_xy[L_EAR][1], kpts_xy[R_EAR][1])
          if (kpts_conf[L_EAR] > 0.3 or kpts_conf[R_EAR] > 0.3)
          else shldr_mid[1]
      )
      neck_dist = abs(shldr_mid[1] - ears_y) / torso_len
      side_gaze_ok = neck_dist < 0.38

    # Gate 5: Kekakuan Lengan (Arm Motion Freeze)
    if len(arm_hist) >= 12:
      arr = np.stack(arm_hist, axis=0)
      arm_var = float(np.mean(np.var(arr, axis=0)))
      side_frozen = arm_var <= cfg.max_arm_variance
    else:
      arm_var = 999.0
      side_frozen = False

    if side_gaze_ok:
      gaze_aligned = True
      best_gaze_cos = max(best_gaze_cos, g_cos)
    if side_frozen:
      arm_frozen = True

    if side_gaze_ok and side_frozen:
      best_wrist_xy = wrist
      best_arm_var = min(best_arm_var, arm_var)

  # Keputusan Frame Ini (Semua Gate Harus Lolos)
  frame_positive = (
      is_walking
      and active_hand_found
      and (not is_crossed)
      and gaze_aligned
      and arm_frozen
  )
  state.vote_hist.append(1 if frame_positive else 0)

  # Gate 6: Sliding Window Majority Vote (45 frame)
  vote_ratio = sum(state.vote_hist) / max(len(state.vote_hist), 1)
  is_candidate_alert = (
      len(state.vote_hist) >= int(cfg.decision_window * 0.65)
      and vote_ratio >= cfg.vote_ratio_thresh
  )

  return {
      "valid": True,
      "walk": is_walking,
      "zone": active_hand_found,
      "no_cross": not is_crossed,
      "gaze": gaze_aligned,
      "freeze": arm_frozen,
      "vote_ratio": vote_ratio,
      "alert": is_candidate_alert,
      "focus_pt": (best_wrist_xy + chest_pt) / 2.0,
      "arm_var": best_arm_var,
      "walk_score": walk_score,
  }


# ==========================================
# 4. PIPELINE UTAMA & AUTO-HARVESTER CROP
# ==========================================
def run_research_pipeline(
    video_source: str, cfg: ResearchConfig = ResearchConfig()
):
  out_dir = Path(cfg.output_dir)
  crop_dir = out_dir / "candidates_crop"
  crop_dir.mkdir(parents=True, exist_ok=True)

  model = YOLO(cfg.model_path)
  cap = cv2.VideoCapture(video_source)
  if not cap.isOpened():
    raise RuntimeError(f"Gagal membuka video: {video_source}")

  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

  out_video_path = str(out_dir / "debug_pose_gate.mp4")
  writer = cv2.VideoWriter(
      out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
  )

  csv_file = open(
      out_dir / "telemetry_log.csv", mode="w", newline="", encoding="utf-8"
  )
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow([
      "frame_idx",
      "track_id",
      "walk_score",
      "arm_var",
      "vote_ratio",
      "crop_filename",
  ])

  tracks = defaultdict(lambda: TrackState(cfg))
  frame_idx = 0
  fps_list = deque(maxlen=60)

  print(f"[INFO] Memulai riset Phase-1 (Pre-Classifier) pada: {video_source}")
  print(
      "[INFO] Crop kandidat yang lolos frame ke-45 akan disimpan di:"
      f" {crop_dir}"
  )

  while True:
    ret, raw_frame = cap.read()
    if not ret:
      break
    frame_idx += 1
    t0 = time.perf_counter()

    # Simpan salinan frame asli resolusi tinggi untuk keperluan Crop sebelum digambar
    clean_frame = raw_frame.copy()

    # 1. Single-Shot YOLO-Pose + ByteTrack
    results = model.track(
        raw_frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=cfg.conf_person,
        verbose=False,
    )
    res = results[0]

    if res.boxes is not None and res.boxes.id is not None and res.keypoints is not None:
      track_ids = res.boxes.id.int().cpu().tolist()
      bboxes = res.boxes.xyxy.cpu().numpy()
      kpts_xy = res.keypoints.xy.cpu().numpy()
      kpts_conf = (
          res.keypoints.conf.cpu().numpy()
          if res.keypoints.conf is not None
          else np.ones((len(track_ids), 17))
      )

      for tid, bbox, kxy, kcf in zip(track_ids, bboxes, kpts_xy, kpts_conf):
        state = tracks[tid]
        info = evaluate_person_pose(kxy, kcf, bbox, state, cfg)
        if not info["valid"]:
          continue

        x1, y1, x2, y2 = map(int, bbox)
        is_alert = info["alert"]
        color = (0, 0, 255) if is_alert else (0, 220, 0)

        # Gambar Bounding Box & Status HUD per Gate
        cv2.rectangle(raw_frame, (x1, y1), (x2, y2), color, 2)
        hud = (
            f"ID:{tid} "
            f"W:{int(info['walk'])} "
            f"Z:{int(info['zone'])} "
            f"G:{int(info['gaze'])} "
            f"F:{int(info['freeze'])} "
            f"Vote:{int(info['vote_ratio']*100)}%"
        )
        cv2.putText(
            raw_frame,
            hud,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        # Jika lolos sampai Gate 6 (Frame 45), simpan Crop Resolusi Asli untuk evaluasi/dataset
        if is_alert and (frame_idx - state.last_save_frame >= 30):
          state.last_save_frame = frame_idx
          state.saved_crop_count += 1

          fx, fy = int(info["focus_pt"][0]), int(info["focus_pt"][1])
          half_s = int(max((y2 - y1) * 0.28, cfg.crop_size // 2))
          cx1, cy1 = max(0, fx - half_s), max(0, fy - half_s)
          cx2, cy2 = min(w, fx + half_s), min(h, fy + half_s)

          crop_img = clean_frame[cy1:cy2, cx1:cx2]
          if crop_img.size > 0:
            crop_resized = cv2.resize(crop_img, (cfg.crop_size, cfg.crop_size))
            crop_name = f"tid{tid}_f{frame_idx}_v{int(info['vote_ratio']*100)}.jpg"
            cv2.imwrite(str(crop_dir / crop_name), crop_resized)
            csv_writer.writerow([
                frame_idx,
                tid,
                f"{info['walk_score']:.3f}",
                f"{info['arm_var']:.4f}",
                f"{info['vote_ratio']:.2f}",
                crop_name,
            ])

    dt = time.perf_counter() - t0
    fps_list.append(1.0 / max(dt, 1e-5))
    avg_fps = sum(fps_list) / len(fps_list)

    cv2.putText(
        raw_frame,
        f"FPS: {avg_fps:.1f} ({dt*1000:.1f} ms)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
    writer.write(raw_frame)

  cap.release()
  writer.release()
  csv_file.close()
  print(f"[SELESAI] Video debug tersimpan di: {out_video_path}")
  print(
      f"[SELESAI] Rata-rata kecepatan: {sum(fps_list)/max(len(fps_list),1):.1f}"
      " FPS"
  )


if __name__ == "__main__":
  import sys

  # Opsi 1: raw string (paling umum)
  video_input = sys.argv[1] if len(sys.argv) > 1 else r"C:\cctv-dev-temp\use_phone.MOV"
  run_research_pipeline(video_input)