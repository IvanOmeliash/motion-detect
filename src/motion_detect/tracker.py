"""SORT tracker: Kalman filter per track + Hungarian assignment.

Ported unchanged from the research notebook (behaviour-preserving refactor).
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1=max(bb_test[0],bb_gt[0]); yy1=max(bb_test[1],bb_gt[1])
    xx2=min(bb_test[2],bb_gt[2]); yy2=min(bb_test[3],bb_gt[3])
    w=max(0.,xx2-xx1); h=max(0.,yy2-yy1); inter=w*h
    union=((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])+
           (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])-inter)
    return inter/union if union>0 else 0.0

def center_distance(bb_test, bb_gt):
    cx1=(bb_test[0]+bb_test[2])/2; cy1=(bb_test[1]+bb_test[3])/2
    cx2=(bb_gt[0]+bb_gt[2])/2;     cy2=(bb_gt[1]+bb_gt[3])/2
    return np.sqrt((cx1-cx2)**2+(cy1-cy2)**2)

class Track:
    def __init__(self, bbox, track_id):
        self.track_id=track_id; self.hits=1; self.hit_streak=1
        self.kf=cv2.KalmanFilter(4,2)
        self.kf.measurementMatrix=np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kf.transitionMatrix=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kf.processNoiseCov=np.diag([1.,1.,10.,10.]).astype(np.float32)
        self.kf.measurementNoiseCov=np.eye(2,dtype=np.float32)*5.0
        self.kf.errorCovPost=np.eye(4,dtype=np.float32)*100.0
        cx=(bbox[0]+bbox[2])/2; cy=(bbox[1]+bbox[3])/2
        self.kf.statePre=np.array([[cx],[cy],[0],[0]],np.float32)
        self.kf.statePost=np.array([[cx],[cy],[0],[0]],np.float32)
        self.time_since_update=0; self.history=deque(maxlen=60)
        self.bbox=list(bbox); self.w=bbox[2]-bbox[0]; self.h=bbox[3]-bbox[1]

    def predict(self):
        pred=self.kf.predict()
        cx,cy=float(pred[0,0]),float(pred[1,0])
        self.bbox=[cx-self.w/2,cy-self.h/2,cx+self.w/2,cy+self.h/2]
        self.time_since_update+=1; return self.bbox

    def update(self, bbox):
        self.time_since_update=0; self.hits+=1; self.hit_streak+=1
        self.bbox=list(bbox); self.w=bbox[2]-bbox[0]; self.h=bbox[3]-bbox[1]
        cx=(bbox[0]+bbox[2])/2; cy=(bbox[1]+bbox[3])/2
        self.history.append((int(cx),int(cy)))
        self.kf.correct(np.array([[cx],[cy]],np.float32))

    def miss(self): self.hit_streak=0

class SortTracker:
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.05, max_dist=80):
        self.max_age=max_age; self.min_hits=min_hits
        self.iou_threshold=iou_threshold; self.max_dist=max_dist
        self.tracks=[]; self.track_id_counter=1

    def update(self, dets):
        for t in self.tracks: t.predict()
        matched,unmatched_dets,unmatched_trks=self._assign(dets,self.tracks)
        for d,t in matched:   self.tracks[t].update(dets[d])
        for t in unmatched_trks: self.tracks[t].miss()
        for d in unmatched_dets:
            self.tracks.append(Track(dets[d],self.track_id_counter))
            self.track_id_counter+=1
        self.tracks=[t for t in self.tracks if t.time_since_update<=self.max_age]
        return self.tracks

    def _assign(self, dets, tracks):
        if not tracks: return [], list(range(len(dets))), []
        if not dets:   return [], [], list(range(len(tracks)))
        trk_boxes=[t.bbox for t in tracks]
        iou_mat=np.zeros((len(dets),len(trk_boxes)),np.float32)
        dist_mat=np.zeros((len(dets),len(trk_boxes)),np.float32)
        for d,det in enumerate(dets):
            for t,trk in enumerate(trk_boxes):
                iou_mat[d,t]=iou(det,trk); dist_mat[d,t]=center_distance(det,trk)
        ri,ci=linear_sum_assignment(-iou_mat)
        matched,md,mt=[],set(),set()
        for r,c in zip(ri,ci):
            if iou_mat[r,c]>=self.iou_threshold or dist_mat[r,c]<=self.max_dist:
                matched.append((r,c)); md.add(r); mt.add(c)
        return matched,[d for d in range(len(dets)) if d not in md],[t for t in range(len(tracks)) if t not in mt]

# ================================================================
