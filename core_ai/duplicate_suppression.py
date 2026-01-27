import math

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0


def suppress_duplicates(face_states, iou_thresh=0.5, center_dist_thresh=50):
    kept = []

    for face in face_states:
        x, y, w, h = face["bbox"]
        cx = x + w / 2
        cy = y + h / 2

        duplicate = False
        for k in kept:
            kx, ky, kw, kh = k["bbox"]
            kcx = kx + kw / 2
            kcy = ky + kh / 2

            if iou((x, y, w, h), (kx, ky, kw, kh)) > iou_thresh:
                duplicate = True
                break

            dist = math.hypot(cx - kcx, cy - kcy)
            if dist < center_dist_thresh:
                duplicate = True
                break

        if not duplicate:
            kept.append(face)

    return kept
