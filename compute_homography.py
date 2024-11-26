import numpy as np
import cv2

def compute_transforms(p1, p2, p3, p4, q1, q2, q3, q4):
    P = np.asarray([p1, p2, p3, p4], dtype="float32")
    Q = np.asarray([q1, q2, q3, q4], dtype="float32")
    H_forward = cv2.getPerspectiveTransform(P, Q)
    H_reverse = cv2.getPerspectiveTransform(Q, P)
    return H_forward, H_reverse

def apply_forward(H, p):
    p_t = np.concatenate([np.asarray(p), np.array([1])])
    q_t = H @ p_t
    return (q_t/q_t[2])[:2]

def apply_reverse(H, q):
    q_t = np.concatenate([np.asarray(q), np.array([1])])
    p_t = H @ q_t
    return (p_t/p_t[2])[:2]