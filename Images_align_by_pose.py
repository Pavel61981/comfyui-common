import torch
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

class ImagesAlignByPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "debug": ("BOOLEAN", {"default": True}),
                "povorot": ("BOOLEAN", {"default": True}),
                "obrezka": ([
                    "None",
                    "3:4 (Portrait)",
                    "9:16 (Portrait)",
                    "1:1 (Square)",
                    "4:3 (Landscape)",
                    "16:9 (Landscape)"
                ], {"default": "None"}),
                "side": ([
                    "по середине",
                    "сверху",
                    "снизу",
                    "слева",
                    "справа"
                ], {"default": "по середине"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned_image1", "aligned_image2", "debug_visualization")
    FUNCTION = "align_images"
    CATEGORY = "image/alignment"

    # Внутренные константы (не настраиваемые извне).
    MIN_TORSO_POINTS_FOR_USING_TORSO = 2
    TORSO_INDICES = [11, 12, 23, 24]  # left_shoulder, right_shoulder, left_hip, right_hip
    TORSO_WEIGHT = 5.0  # усиленный вес для точек торса
    MAX_TORSO_ROT_DEG = 20.0  # максимально допустимый поворот, основанный на торсе

    def tensor_to_cv2(self, tensor_image):
        img = tensor_image[0].cpu().numpy() * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def cv2_to_tensor(self, cv2_image):
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        return tensor.unsqueeze(0)

    def get_landmarks_full(self, cv2_image):
        h, w = cv2_image.shape[:2]
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            return None
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            # lm.x and lm.y are normalized [0,1] — переводим в пиксели
            landmarks.append([lm.x * w, lm.y * h, lm.visibility])
        return np.array(landmarks, dtype=np.float32)

    def is_point_valid(self, lm, w, h, visibility_threshold):
        if lm is None or len(lm) < 3:
            return False
        x, y, vis = lm[0], lm[1], lm[2]
        return (
            vis >= visibility_threshold and
            0 <= x < w and
            0 <= y < h
        )

    def compute_center_and_scale(self, landmarks, w, h, visibility_threshold):
        if landmarks is None or len(landmarks) == 0:
            return None, None

        valid_points = []
        weights = []
        for idx, lm in enumerate(landmarks):
            if self.is_point_valid(lm, w, h, visibility_threshold):
                base_weight = lm[2] if (len(lm) > 2 and lm[2] is not None) else 1.0
                weight = base_weight * (self.TORSO_WEIGHT if idx in self.TORSO_INDICES else 1.0)
                valid_points.append((idx, lm[:2]))
                weights.append(weight)

        if len(valid_points) == 0:
            return None, None

        # Торсовая логика: приоритет плечам и бёдрам
        valid_torso_points = []
        for t_idx in self.TORSO_INDICES:
            if t_idx < len(landmarks) and self.is_point_valid(landmarks[t_idx], w, h, visibility_threshold):
                valid_torso_points.append((t_idx, landmarks[t_idx][:2]))

        use_torso_only = len(valid_torso_points) >= self.MIN_TORSO_POINTS_FOR_USING_TORSO

        if use_torso_only:
            pts = np.array([p[1] for p in valid_torso_points])
            weights_arr = np.array([landmarks[p[0]][2] * self.TORSO_WEIGHT for p in valid_torso_points])
            center = np.average(pts, axis=0, weights=weights_arr)

            dists = []
            # межплечье
            if 11 < len(landmarks) and 12 < len(landmarks):
                if self.is_point_valid(landmarks[11], w, h, visibility_threshold) and self.is_point_valid(landmarks[12], w, h, visibility_threshold):
                    dists.append(np.linalg.norm(landmarks[11][:2] - landmarks[12][:2]))
            # межбедро
            if 23 < len(landmarks) and 24 < len(landmarks):
                if self.is_point_valid(landmarks[23], w, h, visibility_threshold) and self.is_point_valid(landmarks[24], w, h, visibility_threshold):
                    dists.append(np.linalg.norm(landmarks[23][:2] - landmarks[24][:2]))

            # высота торса: средние плеч и бёдра
            shoulder_candidates = []
            hip_candidates = []
            for si in (11, 12):
                if si < len(landmarks) and self.is_point_valid(landmarks[si], w, h, visibility_threshold):
                    shoulder_candidates.append(landmarks[si][:2])
            for hi in (23, 24):
                if hi < len(landmarks) and self.is_point_valid(landmarks[hi], w, h, visibility_threshold):
                    hip_candidates.append(landmarks[hi][:2])

            if len(shoulder_candidates) > 0 and len(hip_candidates) > 0:
                shoulder_mid = np.mean(shoulder_candidates, axis=0)
                hip_mid = np.mean(hip_candidates, axis=0)
                dists.append(np.linalg.norm(shoulder_mid - hip_mid))

            if len(dists) > 0:
                scale = float(np.median(dists))
            else:
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                scale = float(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2))

            return center, scale

        # иначе используем все валидные точки
        pts_all = np.array([p[1] for p in valid_points])
        weights_all = np.array(weights)
        center = np.average(pts_all, axis=0, weights=weights_all)

        pairs = [(11, 12), (23, 24), (13, 14), (25, 26)]
        dists = []
        for i, j in pairs:
            if i < len(landmarks) and j < len(landmarks):
                if self.is_point_valid(landmarks[i], w, h, visibility_threshold) and self.is_point_valid(landmarks[j], w, h, visibility_threshold):
                    d = np.linalg.norm(landmarks[i][:2] - landmarks[j][:2])
                    if d > 1e-5:
                        dists.append(d)

        if len(dists) > 0:
            scale = float(np.median(dists))
        else:
            x_min, y_min = np.min(pts_all, axis=0)
            x_max, y_max = np.max(pts_all, axis=0)
            scale = float(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2))

        return center, scale

    def compute_torso_angle(self, landmarks, w, h, visibility_threshold):
        """
        Вычисляем угол ориентации торса: вектор от средней точки плеч к средней точке бедер.
        Возвращаем (angle_radians, confidence) — confidence = число валидных точек торса (2..4).
        """
        if landmarks is None:
            return None, 0
        shoulder_pts = []
        hip_pts = []
        conf = 0
        for si in (11, 12):
            if si < len(landmarks) and self.is_point_valid(landmarks[si], w, h, visibility_threshold):
                shoulder_pts.append(landmarks[si][:2])
        for hi in (23, 24):
            if hi < len(landmarks) and self.is_point_valid(landmarks[hi], w, h, visibility_threshold):
                hip_pts.append(landmarks[hi][:2])
        conf = len(shoulder_pts) + len(hip_pts)
        if len(shoulder_pts) == 0 or len(hip_pts) == 0:
            return None, conf
        shoulder_mid = np.mean(shoulder_pts, axis=0)
        hip_mid = np.mean(hip_pts, axis=0)
        vec = hip_mid - shoulder_mid  # направление вниз по туловищу
        angle = float(np.arctan2(vec[1], vec[0]))  # угол в радианах
        return angle, conf

    def weighted_no_rotation_transform(self, src_pts, dst_pts, weights=None):
        if len(src_pts) < 1:
            return None
        if weights is None:
            weights = np.ones(len(src_pts), dtype=np.float64)
        else:
            weights = np.array(weights, dtype=np.float64)

        c_src = np.average(src_pts, axis=0, weights=weights)
        c_dst = np.average(dst_pts, axis=0, weights=weights)

        src_centered = src_pts - c_src
        dst_centered = dst_pts - c_dst

        denom = np.sum(weights * np.sum(dst_centered ** 2, axis=1))
        if denom < 1e-9:
            return None

        numer = np.sum(weights * np.sum(src_centered * dst_centered, axis=1))
        s = numer / denom
        if s <= 1e-6:
            return None

        t = c_src - s * c_dst
        M = np.zeros((2, 3), dtype=np.float32)
        M[0, 0] = s
        M[1, 1] = s
        M[:, 2] = t
        return M

    def estimate_similarity_transform(self, src_pts, dst_pts, weights=None):
        N = len(src_pts)
        if N < 2:
            return None
        if weights is None:
            w = np.ones(N, dtype=np.float64)
        else:
            w = np.array(weights, dtype=np.float64)
            w = np.maximum(w, 1e-9)

        w_sum = np.sum(w)
        if w_sum <= 0:
            return None

        mu_src = np.sum(src_pts * w[:, None], axis=0) / w_sum
        mu_dst = np.sum(dst_pts * w[:, None], axis=0) / w_sum

        src_cent = src_pts - mu_src
        dst_cent = dst_pts - mu_dst

        Sigma = (dst_cent * w[:, None]).T @ src_cent / w_sum

        var_dst = np.sum(w * np.sum(dst_cent ** 2, axis=1)) / w_sum
        if var_dst <= 1e-12:
            return None

        try:
            U, D, Vt = np.linalg.svd(Sigma)
        except Exception:
            return None

        S = np.eye(2)
        if np.linalg.det(U @ Vt) < 0:
            S[1, 1] = -1

        R = U @ S @ Vt
        scale = np.sum(D * np.diag(S)) / var_dst
        if scale <= 1e-9:
            return None

        t = mu_src - scale * (R @ mu_dst)
        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = (scale * R).astype(np.float32)
        M[:, 2] = t.astype(np.float32)
        return M

    def compute_residuals(self, M, src_pts, dst_pts):
        if M is None:
            return np.array([])
        dst_aug = np.concatenate([dst_pts, np.ones((len(dst_pts), 1), dtype=np.float32)], axis=1)
        transformed = (M @ dst_aug.T).T
        residuals = np.linalg.norm(transformed - src_pts, axis=1)
        return residuals

    def refine_transform_via_inliers(self, src_pts, dst_pts, inlier_mask, allow_rotation=True, weights=None):
        idx = np.where(np.array(inlier_mask).flatten() > 0)[0]
        if len(idx) < 2:
            return None

        pts_src = src_pts[idx].astype(np.float64)
        pts_dst = dst_pts[idx].astype(np.float64)

        w = None
        if weights is not None:
            w = np.array(weights)[idx]

        if allow_rotation:
            M = self.estimate_similarity_transform(pts_src, pts_dst, weights=w)
            return M
        else:
            M = self.weighted_no_rotation_transform(pts_src.astype(np.float32), pts_dst.astype(np.float32), weights=w)
            return M

    def _auto_parameters(self, pts1, pts2, w1, h1, w2, h2, vis_values):
        n = max(1, len(pts1))
        med_vis = float(np.median(vis_values)) if len(vis_values) > 0 else 0.6
        visibility_threshold = float(np.clip(med_vis * 0.85, 0.25, 0.75))
        padding = int(max(0, round(min(w1, h1) * 0.02)))

        pair_dists = np.linalg.norm(pts1 - pts2, axis=1) if len(pts1) > 0 else np.array([0.0])
        median_pt_dist = float(np.median(pair_dists)) if pair_dists.size > 0 else max(w1, h1) * 0.1

        img_diag = np.sqrt(w1 * w1 + h1 * h1)
        residual_threshold_px = int(np.clip(max(8.0, median_pt_dist * 0.5, 0.01 * img_diag), 6, max(w1, h1) * 0.25))

        no_rot_accept_median_res_px = residual_threshold_px
        no_rot_min_inlier_ratio = float(np.clip(0.55 + 0.02 * (n - 4), 0.45, 0.85))

        ransac = True
        ransac_threshold = int(max(2, residual_threshold_px))
        ransac_iters = int(np.clip(100 + 20 * (n - 4), 100, 2000))

        if len(pts1) >= 3:
            vecs = pts1 - pts2
            angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            ang_std = float(np.std(angles))
            allow_rotation = not (ang_std < 0.2)
        else:
            allow_rotation = True

        scale_clip_min = 0.7
        scale_clip_max = 1.4

        return {
            "visibility_threshold": visibility_threshold,
            "padding": padding,
            "residual_threshold_px": residual_threshold_px,
            "no_rot_accept_median_res_px": no_rot_accept_median_res_px,
            "no_rot_min_inlier_ratio": no_rot_min_inlier_ratio,
            "ransac": ransac,
            "ransac_threshold": ransac_threshold,
            "ransac_iters": ransac_iters,
            "allow_rotation": allow_rotation,
            "scale_clip_min": scale_clip_min,
            "scale_clip_max": scale_clip_max,
        }

    def align_images(self, image1, image2, debug=True, povorot=True, obrezka="None", side="по середине"):
        img1_cv2 = self.tensor_to_cv2(image1)
        img2_cv2 = self.tensor_to_cv2(image2)

        landmarks1 = self.get_landmarks_full(img1_cv2)
        landmarks2 = self.get_landmarks_full(img2_cv2)

        if landmarks1 is None or landmarks2 is None:
            print("⚠️ Не удалось обнаружить позу. Возвращаем оригиналы.")
            return (image1, image2, image1)

        h1, w1 = img1_cv2.shape[:2]
        h2, w2 = img2_cv2.shape[:2]

        vis_vals = []
        for lm in np.vstack((landmarks1, landmarks2)):
            if len(lm) > 2:
                vis_vals.append(lm[2])

        soft_vis = 0.3
        candidate_idx = []
        pts1_cand = []
        pts2_cand = []
        vis_cand = []

        for idx in range(min(len(landmarks1), len(landmarks2))):
            lm1 = landmarks1[idx]
            lm2 = landmarks2[idx]
            if lm1[2] >= soft_vis and lm2[2] >= soft_vis:
                pts1_cand.append(lm1[:2])
                pts2_cand.append(lm2[:2])
                vis_cand.append((lm1[2] + lm2[2]) / 2.0)
                candidate_idx.append(idx)

        if len(pts1_cand) < 2:
            print("⚠️ Недостаточно общих ключевых точек для выравнивания. Возвращаем оригиналы.")
            return (image1, image2, image1)

        pts1_cand = np.array(pts1_cand, dtype=np.float32)
        pts2_cand = np.array(pts2_cand, dtype=np.float32)
        vis_cand = np.array(vis_cand, dtype=np.float32)

        auto = self._auto_parameters(pts1_cand, pts2_cand, w1, h1, w2, h2, vis_vals)
        visibility_threshold = auto["visibility_threshold"]
        padding = auto["padding"]
        residual_threshold_px = auto["residual_threshold_px"]
        no_rot_accept_median_res_px = auto["no_rot_accept_median_res_px"]
        no_rot_min_inlier_ratio = auto["no_rot_min_inlier_ratio"]
        ransac = auto["ransac"]
        ransac_threshold = auto["ransac_threshold"]
        ransac_iters = auto["ransac_iters"]
        allow_rotation = auto["allow_rotation"]
        scale_clip_min = auto["scale_clip_min"]
        scale_clip_max = auto["scale_clip_max"]

        valid_idx = []
        pts1 = []
        pts2 = []
        weights = []

        for i, idx in enumerate(candidate_idx):
            lm1 = landmarks1[idx]
            lm2 = landmarks2[idx]
            if self.is_point_valid(lm1, w1, h1, visibility_threshold) and self.is_point_valid(lm2, w2, h2, visibility_threshold):
                pts1.append(lm1[:2])
                pts2.append(lm2[:2])
                w_vis = float((lm1[2] + lm2[2]) / 2.0)
                # усиливаем вклад торса
                weight = w_vis * (self.TORSO_WEIGHT if idx in self.TORSO_INDICES else 1.0)
                weights.append(weight)
                valid_idx.append(idx)

        if len(pts1) < 2:
            print("⚠️ После фильтрации недостаточно качественных соответствий. Возвращаем оригиналы.")
            return (image1, image2, image1)

        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)
        weights_arr = np.array(weights, dtype=np.float32)

        # Если поворот выключен — запрещаем любую ротацию.
        if not povorot:
            # 1) Попытка: чистый no-rotation (масштаб + перевод)
            M_no_rot = self.weighted_no_rotation_transform(pts1, pts2, weights=weights_arr)
            best_M = M_no_rot
            best_info = "no_rotation_forced"

            # Если торс есть в обеих картинках, подвинем второй по центру торса (чтобы "лицо наклонено" не влиял)
            center1, scale1 = self.compute_center_and_scale(landmarks1, w1, h1, visibility_threshold)
            center2, scale2 = self.compute_center_and_scale(landmarks2, w2, h2, visibility_threshold)
            if best_M is not None and center1 is not None and center2 is not None:
                # корректируем трансляцию так, чтобы центры тела совпали при текущем масштабе
                s = float(best_M[0, 0]) if best_M is not None else 1.0
                best_M[:, 2] = (np.array(center1, dtype=np.float32) - s * np.array(center2, dtype=np.float32)).astype(np.float32)

        else:
            # 1) Попытка: no-rotation (быстро)
            M_no_rot = self.weighted_no_rotation_transform(pts1, pts2, weights=weights_arr)
            accept_no_rot = False
            info_notes = []

            def score_median_res(M):
                if M is None:
                    return float("inf"), 0.0
                residuals = self.compute_residuals(M, pts1, pts2)
                if residuals.size == 0:
                    return float("inf"), 0.0
                med = float(np.median(residuals))
                inlier_ratio = float(np.mean(residuals <= float(residual_threshold_px)))
                return med, inlier_ratio

            med_no_rot, inlier_ratio_no_rot = score_median_res(M_no_rot)
            info_notes.append(f"no_rot_med={med_no_rot:.2f}px inl_ratio={inlier_ratio_no_rot:.2f}")

            if M_no_rot is not None:
                if med_no_rot <= float(no_rot_accept_median_res_px) or inlier_ratio_no_rot >= float(no_rot_min_inlier_ratio):
                    accept_no_rot = True

            best_M = None
            best_score = float("inf")
            best_info = "none"

            if accept_no_rot:
                best_M = M_no_rot
                best_score = med_no_rot
                best_info = f"no_rot_accepted ({info_notes[-1]})"
            else:
                best_M = M_no_rot
                best_score = med_no_rot
                best_info = f"no_rot_rejected ({info_notes[-1]})"

                run_ransac = ransac and (len(pts1) >= 3)
                if run_ransac:
                    try:
                        M_ransac, inliers = cv2.estimateAffinePartial2D(pts2, pts1,
                                                                        method=cv2.RANSAC,
                                                                        ransacReprojThreshold=float(ransac_threshold),
                                                                        maxIters=int(ransac_iters),
                                                                        confidence=0.99,
                                                                        refineIters=10)
                        if M_ransac is not None:
                            med_ransac, inlier_ratio_ransac = score_median_res(M_ransac)
                            if med_ransac < best_score:
                                best_M = M_ransac
                                best_score = med_ransac
                                best_info = f"ransac (med={med_ransac:.2f}px inl={inlier_ratio_ransac:.2f})"
                            else:
                                best_info += f" -> ransac_worse (med_ransac={med_ransac:.2f}px)"
                        else:
                            best_info += " -> ransac_failed"
                    except Exception as e:
                        best_info += f" -> ransac_error:{e}"
                else:
                    best_info += " -> ransac_skipped"

            if best_M is None:
                print("⚠️ Не удалось оценить трансформацию. Возвращаем оригиналы.")
                return (image1, image2, image1)

            # Дополнительный шаг: если торс найден в обеих картинках, использовать его угол как приоритет
            torso_angle1, conf1 = self.compute_torso_angle(landmarks1, w1, h1, visibility_threshold)
            torso_angle2, conf2 = self.compute_torso_angle(landmarks2, w2, h2, visibility_threshold)

            if torso_angle1 is not None and torso_angle2 is not None and (conf1 + conf2) >= 3:
                desired_angle = torso_angle1 - torso_angle2
                max_rot = np.deg2rad(self.MAX_TORSO_ROT_DEG)
                confidence_factor = np.clip((conf1 + conf2) / 4.0, 0.5, 1.5)
                max_rot *= confidence_factor
                if desired_angle > max_rot:
                    desired_angle = max_rot
                elif desired_angle < -max_rot:
                    desired_angle = -max_rot

                a = best_M[0, 0]
                b = best_M[0, 1]
                scale_est = float(np.sqrt(a * a + b * b))
                c = float(np.cos(desired_angle))
                s = float(np.sin(desired_angle))
                R_torso = np.array([[c, -s], [s, c]], dtype=np.float64)
                R_scaled = (scale_est * R_torso).astype(np.float32)
                center1, _ = self.compute_center_and_scale(landmarks1, w1, h1, visibility_threshold)
                center2, _ = self.compute_center_and_scale(landmarks2, w2, h2, visibility_threshold)
                if center1 is not None and center2 is not None:
                    t = np.array(center1, dtype=np.float64) - R_scaled @ np.array(center2, dtype=np.float64)
                    best_M[:2, :2] = R_scaled
                    best_M[:, 2] = t.astype(np.float32)
                    best_info += f" -> torso_rotation_applied({np.rad2deg(desired_angle):.1f}deg)"

        # Outlier rejection + refine (после возможного применения торса)
        residuals_best = self.compute_residuals(best_M, pts1, pts2)
        inlier_mask_post = (residuals_best <= float(residual_threshold_px))

        if np.sum(inlier_mask_post) >= 2 and np.sum(inlier_mask_post) < len(pts1):
            M_refined = self.refine_transform_via_inliers(pts1, pts2, inlier_mask_post.reshape(-1, 1),
                                                          allow_rotation=povorot, weights=weights_arr)
            if M_refined is not None:
                def med_of(M):
                    r = self.compute_residuals(M, pts1, pts2)
                    return float(np.median(r)) if r.size > 0 else float("inf")

                score_before = med_of(best_M)
                score_refined = med_of(M_refined)
                if score_refined <= score_before + 1e-6:
                    best_M = M_refined
                    best_score = score_refined
                    best_info += f" -> refined (med={score_refined:.2f}px)"
                else:
                    best_info += f" -> refine_no_improve (before={score_before:.2f} ref={score_refined:.2f})"

        # Контроль масштаба: извлечь масштаб и клиппинг
        a = best_M[0, 0]
        b = best_M[0, 1]
        scale_est = float(np.sqrt(a * a + b * b))
        scale_clipped = float(np.clip(scale_est, scale_clip_min, scale_clip_max))

        if scale_est <= 0:
            print("⚠️ Оценён нулевой/отрицательный масштаб. Возвращаем оригиналы.")
            return (image1, image2, image1)

        if abs(scale_clipped - scale_est) > 1e-6:
            center1, scale1 = self.compute_center_and_scale(landmarks1, w1, h1, visibility_threshold)
            center2, scale2 = self.compute_center_and_scale(landmarks2, w2, h2, visibility_threshold)
            if center1 is None or center2 is None:
                factor = scale_clipped / (scale_est + 1e-9)
                best_M[:2, :2] *= factor
            else:
                R = best_M[:2, :2].astype(np.float64)
                R = R * (scale_clipped / (scale_est + 1e-9))
                t = np.array(center1, dtype=np.float64) - R @ np.array(center2, dtype=np.float64)
                best_M[:2, :2] = R.astype(np.float32)
                best_M[:, 2] = t.astype(np.float32)

        # Применим трансформацию ко второму изображению
        aligned_img2 = cv2.warpAffine(img2_cv2, best_M, (w1, h1))

        # Обрезка: рассчитываем пересечение области трансформированного второго изображения и первого
        corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        transformed_corners = cv2.transform(np.array([corners]), best_M)[0]
        inter_x_min = int(max(0, np.min(transformed_corners[:, 0])) + padding)
        inter_y_min = int(max(0, np.min(transformed_corners[:, 1])) + padding)
        inter_x_max = int(min(w1, np.max(transformed_corners[:, 0])) - padding)
        inter_y_max = int(min(h1, np.max(transformed_corners[:, 1])) - padding)

        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            print("⚠️ Пересечение пустое после трансформации. Возвращаем оригиналы.")
            return (image1, image2, image1)

        # Получаем центры объектов в координатах img1
        center1, _ = self.compute_center_and_scale(landmarks1, w1, h1, visibility_threshold)
        center2, _ = self.compute_center_and_scale(landmarks2, w2, h2, visibility_threshold)
        center2_trans = None
        if center2 is not None:
            c2_aug = np.array([center2[0], center2[1], 1.0], dtype=np.float32)
            center2_trans = (best_M @ c2_aug)[:2]

        # Выбираем центр обрезки: усреднённый по центрам (если доступны), иначе центр пересечения
        if center1 is not None and center2_trans is not None:
            crop_center_x = float((center1[0] + center2_trans[0]) / 2.0)
            crop_center_y = float((center1[1] + center2_trans[1]) / 2.0)
        elif center1 is not None:
            crop_center_x, crop_center_y = float(center1[0]), float(center1[1])
        elif center2_trans is not None:
            crop_center_x, crop_center_y = float(center2_trans[0]), float(center2_trans[1])
        else:
            crop_center_x = float((inter_x_min + inter_x_max) / 2.0)
            crop_center_y = float((inter_y_min + inter_y_max) / 2.0)

        # Параметры обрезки в зависимости от выбора obrezka
        selected = obrezka
        side_opt = side

        if selected == "None":
            # Целевая ширина/высота — по размеру наименьшего оригинала
            target_w = min(w1, w2)
            target_h = min(h1, h2)
            # Поправим, чтобы укладывалось в пересечение
            avail_w = inter_x_max - inter_x_min
            avail_h = inter_y_max - inter_y_min
            target_w = min(target_w, avail_w)
            target_h = min(target_h, avail_h)
            # Если целевые размеры пустые — используем доступную область
            if target_w <= 0 or target_h <= 0:
                target_w = avail_w
                target_h = avail_h
        else:
            # Разбор соотношения сторон
            ratio_map = {
                "3:4 (Portrait)": 3.0 / 4.0,
                "9:16 (Portrait)": 9.0 / 16.0,
                "1:1 (Square)": 1.0,
                "4:3 (Landscape)": 4.0 / 3.0,
                "16:9 (Landscape)": 16.0 / 9.0,
            }
            aspect = ratio_map.get(selected, 1.0)
            # Доступная область
            avail_w = inter_x_max - inter_x_min
            avail_h = inter_y_max - inter_y_min
            if avail_w <= 0 or avail_h <= 0:
                print("⚠️ Недостаточная область для обрезки. Возвращаем оригиналы.")
                return (image1, image2, image1)
            # Подбираем максимальную область с заданным соотношением внутри доступной области
            # Сначала пробуем максимальную по высоте
            crop_h = min(avail_h, int(avail_w / aspect))
            crop_w = int(round(crop_h * aspect))
            # Если по ширине не влезло — используем максимальную по ширине
            if crop_w > avail_w:
                crop_w = avail_w
                crop_h = int(round(crop_w / aspect))
            target_w = int(max(1, crop_w))
            target_h = int(max(1, crop_h))

        # Обеспечим, что целевые размеры не превышают доступной пересечения
        avail_w = inter_x_max - inter_x_min
        avail_h = inter_y_max - inter_y_min
        target_w = int(min(target_w, avail_w))
        target_h = int(min(target_h, avail_h))

        # Получаем центры объектов в координатах img1 (повторно, чтобы точно иметь значения)
        center1, _ = self.compute_center_and_scale(landmarks1, w1, h1, visibility_threshold)
        center2, _ = self.compute_center_and_scale(landmarks2, w2, h2, visibility_threshold)
        center2_trans = None
        if center2 is not None:
            c2_aug = np.array([center2[0], center2[1], 1.0], dtype=np.float32)
            center2_trans = (best_M @ c2_aug)[:2]

        # Определяем опорные координаты для центрирования объекта
        obj_cx = None
        obj_cy = None
        if center1 is not None and center2_trans is not None:
            obj_cx = float((center1[0] + center2_trans[0]) / 2.0)
            obj_cy = float((center1[1] + center2_trans[1]) / 2.0)
        elif center1 is not None:
            obj_cx, obj_cy = float(center1[0]), float(center1[1])
        elif center2_trans is not None:
            obj_cx, obj_cy = float(center2_trans[0]), float(center2_trans[1])
        else:
            obj_cx = float((inter_x_min + inter_x_max) / 2.0)
            obj_cy = float((inter_y_min + inter_y_max) / 2.0)

        # Вычисляем центр обрезки в зависимости от выбора стороны.
        if side_opt == "по середине":
            crop_center_x = obj_cx
            crop_center_y = obj_cy
        elif side_opt == "сверху":
            crop_center_x = obj_cx
            crop_center_y = float(inter_y_min + target_h / 2.0)
        elif side_opt == "снизу":
            crop_center_x = obj_cx
            crop_center_y = float(inter_y_max - target_h / 2.0)
        elif side_opt == "слева":
            crop_center_y = obj_cy
            crop_center_x = float(inter_x_min + target_w / 2.0)
        elif side_opt == "справа":
            crop_center_y = obj_cy
            crop_center_x = float(inter_x_max - target_w / 2.0)
        else:
            crop_center_x = obj_cx
            crop_center_y = obj_cy

        # Сдвигаем центр обрезки так, чтобы прямоугольник уместился в пределах пересечения
        x0 = int(round(crop_center_x - target_w / 2.0))
        y0 = int(round(crop_center_y - target_h / 2.0))
        x1 = x0 + target_w
        y1 = y0 + target_h

        # Корректируем выход за границы пересечения, при этом стараемся сохранить якорь со стороны
        if x0 < inter_x_min:
            x1 += (inter_x_min - x0)
            x0 = inter_x_min
        if x1 > inter_x_max:
            x0 -= (x1 - inter_x_max)
            x1 = inter_x_max
        if y0 < inter_y_min:
            y1 += (inter_y_min - y0)
            y0 = inter_y_min
        if y1 > inter_y_max:
            y0 -= (y1 - inter_y_max)
            y1 = inter_y_max

        # Финальная корректировка (если после сдвига вышли за границы изображения)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w1, x1)
        y1 = min(h1, y1)

        if x0 >= x1 or y0 >= y1:
            print("⚠️ Не удалось корректно посчитать область обрезки. Возвращаем оригиналы.")
            return (image1, image2, image1)
        avail_w = inter_x_max - inter_x_min
        avail_h = inter_y_max - inter_y_min
        target_w = int(min(target_w, avail_w))
        target_h = int(min(target_h, avail_h))

        # Сдвигаем центр обрезки так, чтобы прямоугольник уместился в пределах пересечения
        x0 = int(round(crop_center_x - target_w / 2.0))
        y0 = int(round(crop_center_y - target_h / 2.0))
        x1 = x0 + target_w
        y1 = y0 + target_h

        if x0 < inter_x_min:
            x1 += (inter_x_min - x0)
            x0 = inter_x_min
        if x1 > inter_x_max:
            x0 -= (x1 - inter_x_max)
            x1 = inter_x_max
        if y0 < inter_y_min:
            y1 += (inter_y_min - y0)
            y0 = inter_y_min
        if y1 > inter_y_max:
            y0 -= (y1 - inter_y_max)
            y1 = inter_y_max

        # Финальная корректировка (если после сдвига вышли за границы изображения)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w1, x1)
        y1 = min(h1, y1)

        if x0 >= x1 or y0 >= y1:
            print("⚠️ Не удалось корректно посчитать область обрезки. Возвращаем оригиналы.")
            return (image1, image2, image1)

        # Debug визуализация
        if debug:
            pts2_transformed = (best_M @ np.concatenate([pts2, np.ones((len(pts2), 1), dtype=np.float32)], axis=1).T).T
            debug1 = img1_cv2.copy()
            debug2 = aligned_img2.copy()
            final_residuals = np.linalg.norm(pts2_transformed - pts1, axis=1)
            for i, idx in enumerate(valid_idx):
                x1p, y1p = int(pts1[i, 0]), int(pts1[i, 1])
                x2t, y2t = int(pts2_transformed[i, 0]), int(pts2_transformed[i, 1])
                color = (0, 255, 0) if final_residuals[i] <= residual_threshold_px else (0, 0, 255)
                thickness = 3 if idx in self.TORSO_INDICES else -1
                radius = 7 if idx in self.TORSO_INDICES else 4
                cv2.circle(debug1, (x1p, y1p), radius, color, thickness)
                cv2.circle(debug2, (x2t, y2t), radius, color, thickness)
                cv2.line(debug1, (x1p, y1p), (x2t, y2t), (255, 165, 0), 1)

            cv2.rectangle(debug1, (x0, y0), (x1, y1), (255, 165, 0), 2)
            cv2.rectangle(debug2, (x0, y0), (x1, y1), (255, 165, 0), 2)
            cv2.putText(debug1, f"Method: {best_info}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(debug2, f"Method: {best_info}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            h1d, w1d = debug1.shape[:2]
            h2d, w2d = debug2.shape[:2]
            if h1d != h2d:
                new_w2 = int(w2d * h1d / h2d)
                debug2 = cv2.resize(debug2, (new_w2, h1d))
            debug_combined = np.hstack((debug1, debug2))
        else:
            debug_combined = img1_cv2

        # Обрезка final — применяем одинаковую область к обоим изображениям
        cropped_img1 = img1_cv2[y0:y1, x0:x1]
        cropped_img2 = aligned_img2[y0:y1, x0:x1]

        # Если размеры по каким-то причинам отличаются (маловероятно) — привести к одному размеру
        if cropped_img1.shape[:2] != cropped_img2.shape[:2]:
            h_c, w_c = min(cropped_img1.shape[0], cropped_img2.shape[0]), min(cropped_img1.shape[1], cropped_img2.shape[1])
            cropped_img1 = cropped_img1[0:h_c, 0:w_c]
            cropped_img2 = cropped_img2[0:h_c, 0:w_c]

        tensor1 = self.cv2_to_tensor(cropped_img1)
        tensor2 = self.cv2_to_tensor(cropped_img2)
        tensor_debug = self.cv2_to_tensor(debug_combined)

        print(f"✅ Выравнивание и обрезка выполнены ({best_info}). Область: x0={x0} y0={y0} x1={x1} y1={y1}")
        return (tensor1, tensor2, tensor_debug)


NODE_CLASS_MAPPINGS = {
    "ImagesAlignByPose": ImagesAlignByPose
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagesAlignByPose": "🎯 Images align by pose"
}
