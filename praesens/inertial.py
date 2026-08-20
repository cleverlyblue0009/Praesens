"""Milestone 10: inertial (IMU) lane stub.

Most laptops -- including the ASUS TUF A16 this project was developed and
measured on -- expose no accelerometer/gyroscope reading Python can reach
without a vendor-specific driver or sensor-fusion service most consumer
machines don't ship. Rather than fake a signal, poll a sensor that isn't
there, or block the pipeline waiting for one, this module honestly reports
no_evidence whenever no IMU is actually available -- which, on most current
hardware, is always (rule 7: a lane that can't produce evidence says so; a
zero score would mean something entirely different -- "measured and
failed," not "nothing to measure").

Documented as a real interface so a host that DOES expose a usable IMU is
wiring this module up to that host's sensor API, not a rewrite of the lane
or of anything that consumes InertialResult.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InertialResult:
    status: str  # "no_evidence" on every host this project has been run on
    subscore: float | None = None
    diagnostics: str = ""


def imu_available() -> bool:
    """Runtime check for a usable IMU. Always False on hosts without a
    vendor-specific sensor API this project integrates -- which is most
    laptops, including the one this project's numbers were measured on.
    Extend this (and read_imu_sample below) if a specific host's sensor
    API becomes available to integrate against; do not fake a positive
    result to make a lane "work" when nothing is actually being read."""
    return False


def read_imu_sample() -> InertialResult:
    if not imu_available():
        return InertialResult(status="no_evidence", subscore=None,
                               diagnostics="no IMU exposed on this host")
    raise NotImplementedError(
        "imu_available() reported True but no reader is implemented yet -- "
        "wire this host's actual sensor API in here rather than leaving "
        "this raise in place."
    )


if __name__ == "__main__":
    result = read_imu_sample()
    print(f"status={result.status}  subscore={result.subscore}  diagnostics={result.diagnostics!r}")
