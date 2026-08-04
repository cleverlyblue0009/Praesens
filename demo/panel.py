"""Milestone 8: panel mode.

A guided demo a sceptical panelist can drive themselves, entirely by
keypress: [1] live verification, [2] switch to an injected feed, [3]
toggle the light pattern off (the control, proving the score depends on
the pattern rather than the person), [4] a stored side-by-side genuine-vs-
injected comparison. Nothing is typed and nothing can hang -- every
transition is a state flip or an instant capture-handle swap (Milestone
7), never a blocking call.

Panel mode uses a much shorter rolling score window than the general live
dashboard (Milestone 6's default 6s vs. this module's ~1.5s) so switching
to an injected feed visibly collapses the score within about a second
instead of taking up to the full window to fully evict the old genuine
samples. That's a real trade: a 1.5s window at a 5Hz chip rate is only
~7-8 chips, noisier than the 20s sessions Milestone 4/5 use for the
paper's numbers. It's the right trade here -- a demo needs the audience to
SEE the collapse happen, not wait it out.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import yaml

from demo.attack import AttackDashboard
from demo.live import LiveDashboard, REPO_ROOT, GREEN, RED, GRAY, WHITE, LIGHT_GRAY, BLUE, ORANGE, _plot_series
from eval.analyse import load_sessions


class PanelState(str, Enum):
    LIVE = "live"
    ATTACK = "attack"
    REPLAY = "replay"


def load_reference_traces(logs_dir: Path):
    """Best available stored genuine and injected sessions for the
    side-by-side comparison screen, or (None, None) if no data has been
    collected yet -- must not crash the demo before Milestone 4/5 data
    collection has happened."""
    records = load_sessions(logs_dir)
    bonafide = [r for r in records if r["condition"] == "bonafide" and r.get("timestamps")]
    attacks = [r for r in records if r["condition"] in ("replay", "swap") and r.get("timestamps")]
    genuine_ref = max(bonafide, key=lambda r: r["score"], default=None)
    attack_ref = min(attacks, key=lambda r: r["score"], default=None)
    return genuine_ref, attack_ref


class PanelDashboard(AttackDashboard):
    def __init__(self, raw_config: dict, max_camera_probe: int = 6):
        super().__init__(raw_config, max_camera_probe=max_camera_probe)

        # Trade statistical power for a fast, visible collapse -- see module docstring.
        self.dconfig.rolling_window_s = raw_config["demo"].get(
            "panel_rolling_window_s", self.dconfig.rolling_window_s
        )

        self.primary_index = self.oconfig.camera_index
        self.state = PanelState.LIVE
        self.status_message = ""

        logs_dir = REPO_ROOT / raw_config["eval"]["logs_dir"]
        self.genuine_ref, self.attack_ref = load_reference_traces(logs_dir)
        if self.genuine_ref is None or self.attack_ref is None:
            print("NOTE: no stored bona fide/attack session logs found yet -- the [4] "
                  "side-by-side comparison screen will show a placeholder until "
                  "praesens/session.py has been run to collect real sessions.")

    # -- scenario transitions, each a single keypress, none of which block --

    def enter_live(self) -> None:
        self.switch_camera(self.primary_index)
        self.state = PanelState.LIVE
        self.status_message = ""

    def enter_attack(self) -> None:
        alternates = [idx for idx in self.camera_slots if idx != self.oconfig.camera_index]
        if not alternates:
            self.status_message = "no injected feed source found -- start OBS Virtual Camera and relaunch"
            print(f"WARNING: {self.status_message}")
            return
        self.switch_camera(alternates[0])
        self.state = PanelState.ATTACK
        self.status_message = ""

    def toggle_pattern(self) -> None:
        self.emitter.set_enabled(not self.emitter.is_enabled())
        self.status_message = ""

    def enter_replay(self) -> None:
        self.state = PanelState.REPLAY
        self.status_message = ""

    # -- key handling --------------------------------------------------------

    def _handle_key(self, key: int, frame) -> bool:
        if key == ord('1'):
            self.enter_live()
            return True
        elif key == ord('2'):
            self.enter_attack()
            return True
        elif key == ord('3'):
            self.toggle_pattern()
            return True
        elif key == ord('4'):
            self.enter_replay()
            return True
        # Deliberately bypass AttackDashboard's raw digit->camera-slot
        # mapping (it would collide with 1-4 above); keep only q/e/r/s.
        return LiveDashboard._handle_key(self, key, frame)

    # -- rendering ------------------------------------------------------------

    def _render_dashboard(self):
        if self.state == PanelState.REPLAY:
            return self._render_replay_screen()
        return super()._render_dashboard()

    def _draw_banner(self, canvas, x, y, w, h):
        """Same physics, panel-friendly wording: ACCEPT/REJECT instead of
        Milestone 6's LIVE HUMAN PRESENT/INJECTED FEED DETECTED."""
        if not self.face_recently_detected:
            color, title, reason = GRAY, "NO FACE DETECTED", "waiting for a face in frame"
        elif self.current_score >= self.dconfig.score_threshold:
            color, title = GREEN, "ACCEPT"
            reason = f"screen pattern reflected on face (r={self.current_score:.2f})"
        else:
            color, title = RED, "REJECT"
            if not np.isnan(self.current_snr_db) and self.current_snr_db < self.oconfig.snr_floor_db:
                reason = f"insufficient signal to measure (snr={self.current_snr_db:.1f}dB) -- move to better light"
            else:
                reason = f"screen pattern not reflected on face (r={self.current_score:.2f})"

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        title_scale = h / 110.0
        (tw, _th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_scale, 5)
        cv2.putText(canvas, title, (x + (w - tw) // 2, y + int(h * 0.55)),
                    cv2.FONT_HERSHEY_DUPLEX, title_scale, WHITE, 5, cv2.LINE_AA)
        reason_scale = h / 400.0
        (rw, _rh), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, reason_scale, 2)
        cv2.putText(canvas, reason, (x + (w - rw) // 2, y + int(h * 0.80)),
                    cv2.FONT_HERSHEY_SIMPLEX, reason_scale, WHITE, 2, cv2.LINE_AA)
        return y + h

    def _draw_status_row(self, canvas, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (12, 12, 12), -1)
        keys = "[1] LIVE   [2] SWITCH TO INJECTED FEED   [3] TOGGLE LIGHT PATTERN   [4] COMPARE STORED   [Q] QUIT"
        cv2.putText(canvas, keys, (x + 20, y + int(h * 0.42)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
        state_text = (f"state={self.state.value}   FPS: {self._fps():.1f}   "
                      f"lag: {self.current_lag_ms:.0f}ms   {self._camera_name}   "
                      f"pattern: {'ON' if self.emitter.is_enabled() else 'OFF'}")
        if self.status_message:
            state_text += f"   -- {self.status_message}"
        cv2.putText(canvas, state_text, (x + 20, y + int(h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
        return y + h

    def _render_replay_screen(self):
        w, h = self.dconfig.dashboard_width, self.dconfig.dashboard_height
        canvas = np.full((h, w, 3), (15, 15, 15), dtype=np.uint8)

        header_h = 70
        cv2.putText(canvas, "STORED COMPARISON: genuine vs. injected", (30, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

        status_h = int(h * 0.10)
        body_h = h - header_h - status_h

        if self.genuine_ref is None or self.attack_ref is None:
            msg = "No stored session data yet -- run praesens/session.py to collect sessions first"
            (mw, _mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(canvas, msg, ((w - mw) // 2, header_h + body_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, LIGHT_GRAY, 2, cv2.LINE_AA)
        else:
            half_w = w // 2
            self._draw_reference_panel(canvas, 0, header_h, half_w, body_h,
                                        self.genuine_ref, "GENUINE (bona fide)", GREEN)
            self._draw_reference_panel(canvas, half_w, header_h, w - half_w, body_h,
                                        self.attack_ref, "INJECTED (attack)", RED)
            cv2.line(canvas, (half_w, header_h), (half_w, header_h + body_h), (60, 60, 60), 2)

        self._draw_status_row(canvas, 0, header_h + body_h, w, status_h)
        return canvas

    def _draw_reference_panel(self, canvas, x, y, w, h, record, label, color):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (28, 28, 28), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + 40), color, -1)
        cv2.putText(canvas, f"{label}  (score={record['score']:.2f})", (x + 15, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

        pad = 30
        rect = (x + pad, y + 60, x + w - pad, y + h - pad)
        ts = np.array(record["timestamps"])
        if len(ts) >= 2:
            ts = ts - ts[0]
            _plot_series(canvas, rect, ts, record["trace_emitted"], (-3, 3), BLUE, 3)
            _plot_series(canvas, rect, ts, record["trace_measured"], (-3, 3), ORANGE, 3)

        lx, ly = rect[0] + 10, rect[1] + 20
        cv2.line(canvas, (lx, ly), (lx + 25, ly), BLUE, 4)
        cv2.putText(canvas, "emitted", (lx + 32, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.line(canvas, (lx + 140, ly), (lx + 165, ly), ORANGE, 4)
        cv2.putText(canvas, "measured", (lx + 172, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milestone 8 panel mode")
    parser.add_argument("--max-seconds", type=float, default=None,
                         help="auto-stop after N seconds (smoke-test use; omit for real demo use)")
    parser.add_argument("--max-camera-probe", type=int, default=6)
    args = parser.parse_args()

    with open(REPO_ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f)

    dashboard = PanelDashboard(raw, max_camera_probe=args.max_camera_probe)
    print("Panel mode running. Keys: [1] live, [2] injected feed, [3] toggle pattern, "
          "[4] stored comparison, [S] screenshot, [Q] quit.")
    dashboard.run(max_seconds=args.max_seconds)
