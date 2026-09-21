#!/usr/bin/env python3
"""Record MPU9250 motion data on a Raspberry Pi over I2C."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

HEADER = (
    "timestamp_utc",
    "acceleration_x_g",
    "acceleration_y_g",
    "acceleration_z_g",
    "gyroscope_x_dps",
    "gyroscope_y_dps",
    "gyroscope_z_dps",
    "magnetometer_x_ut",
    "magnetometer_y_ut",
    "magnetometer_z_ut",
)


def create_sensor():
    from mpu9250_jmdev.mpu_9250 import MPU9250
    from mpu9250_jmdev.registers import (
        AFS_8G,
        AK8963_ADDRESS,
        AK8963_BIT_16,
        AK8963_MODE_C100HZ,
        GFS_1000,
        MPU9050_ADDRESS_68,
    )

    sensor = MPU9250(
        address_ak=AK8963_ADDRESS,
        address_mpu_master=MPU9050_ADDRESS_68,
        address_mpu_slave=None,
        bus=1,
        gfs=GFS_1000,
        afs=AFS_8G,
        mfs=AK8963_BIT_16,
        mode=AK8963_MODE_C100HZ,
    )
    sensor.configure()
    return sensor


def output_path(directory: Path, subject: str, activity: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_subject = subject.strip().replace(" ", "-")
    safe_activity = activity.strip().replace(" ", "-")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{safe_activity}_{safe_subject}_{timestamp}.csv"


def collect(sensor, destination: Path, rate_hz: float, duration: float | None) -> int:
    interval = 1.0 / rate_hz
    started = time.monotonic()
    next_sample = started
    count = 0

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        try:
            while duration is None or time.monotonic() - started < duration:
                acceleration = sensor.readAccelerometerMaster()
                gyroscope = sensor.readGyroscopeMaster()
                magnetometer = sensor.readMagnetometerMaster()
                writer.writerow(
                    (
                        datetime.now(timezone.utc).isoformat(),
                        *acceleration,
                        *gyroscope,
                        *magnetometer,
                    )
                )
                count += 1
                next_sample += interval
                time.sleep(max(0.0, next_sample - time.monotonic()))
        except KeyboardInterrupt:
            pass
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="participant identifier")
    parser.add_argument("--activity", required=True, help="recorded activity")
    parser.add_argument("--output", type=Path, default=Path("recordings"))
    parser.add_argument("--rate", type=float, default=100.0, help="samples per second")
    parser.add_argument("--duration", type=float, help="recording length in seconds")
    args = parser.parse_args()

    destination = output_path(args.output, args.subject, args.activity)
    count = collect(create_sensor(), destination, args.rate, args.duration)
    print(f"Saved {count} samples to {destination}")


if __name__ == "__main__":
    main()
