"""Calculate Allan deviation from IMU data in rosbag2 files."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from rosbags.rosbag2 import Reader  # type: ignore[import-not-found]
from rosbags.typesys import Stores, get_typestore  # type: ignore[import-not-found]


def allan_dev_overlapping(x: np.ndarray, fs: float, num_taus: int = 60) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fs
    n = len(x)
    m_max = max(1, n // 10)
    m_vals = np.unique(np.logspace(0, np.log10(m_max), num_taus).astype(int))
    taus = m_vals * dt

    # cumulative sum for fast moving averages
    c = np.cumsum(np.insert(x, 0, 0.0))

    adev = np.empty_like(taus, dtype=float)
    for i, m in enumerate(m_vals):
        avgs = (c[m:] - c[:-m]) / m
        diffs = avgs[m:] - avgs[:-m]
        adev[i] = np.sqrt(0.5 * np.mean(diffs * diffs))
    return taus, adev


def fit_sigma_w(
    taus: np.ndarray, adev: np.ndarray, slope_target: float = -0.5, tol: float = 0.12
) -> tuple[float | None, tuple[float, float] | None]:
    # white noise: adev ~ sigma_w / sqrt(2*tau)
    slopes = np.diff(np.log(adev)) / np.diff(np.log(taus))
    idx = np.where(np.abs(slopes - slope_target) < tol)[0]
    if len(idx) < 3:
        return None, None
    # use middle chunk
    i0, i1 = idx[len(idx) // 4], idx[3 * len(idx) // 4]
    sel = slice(i0, i1 + 1)
    sigma_w = np.mean(adev[sel] * np.sqrt(2 * taus[sel]))
    return sigma_w, (taus[i0], taus[i1 + 1])


def fit_sigma_b_rw(
    taus: np.ndarray, adev: np.ndarray, slope_target: float = +0.5, tol: float = 0.12
) -> tuple[float | None, tuple[float, float] | None]:
    # bias random walk: adev ~ sigma_b * sqrt(tau/3)
    slopes = np.diff(np.log(adev)) / np.diff(np.log(taus))
    idx = np.where(np.abs(slopes - slope_target) < tol)[0]
    if len(idx) < 3:
        return None, None
    i0, i1 = idx[len(idx) // 4], idx[3 * len(idx) // 4]
    sel = slice(i0, i1 + 1)
    sigma_b = np.mean(adev[sel] * np.sqrt(3 / taus[sel]))
    return sigma_b, (taus[i0], taus[i1 + 1])


def resample_uniform(t: np.ndarray, y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    t0, t1 = t[0], t[-1]
    dt = 1.0 / fs
    tu = np.arange(t0, t1, dt)
    yu = np.interp(tu, t, y)
    return tu, yu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="Path to rosbag2 folder (contains metadata.yaml)")
    ap.add_argument("--topic", default="/imu/data_raw")
    ap.add_argument("--fs", type=float, default=0.0, help="Resample rate Hz (0=auto from median dt)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    typestore = get_typestore(Stores.ROS2_JAZZY)

    t_list: list[float] = []
    w: list[list[float]] = [[], [], []]
    a: list[list[float]] = [[], [], []]

    with Reader(args.bag) as reader:
        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            raise SystemExit(f"No connections found for topic {args.topic}")

        for conn, ts, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)

            # Prefer header stamp (sensor time). Fall back to bag time if stamp is zero.
            sec = int(msg.header.stamp.sec)
            nsec = int(msg.header.stamp.nanosec)
            timestamp = sec + 1e-9 * nsec
            if timestamp == 0.0:
                timestamp = ts * 1e-9  # bag timestamp is in ns

            t_list.append(timestamp)

            w[0].append(float(msg.angular_velocity.x))
            w[1].append(float(msg.angular_velocity.y))
            w[2].append(float(msg.angular_velocity.z))

            a[0].append(float(msg.linear_acceleration.x))
            a[1].append(float(msg.linear_acceleration.y))
            a[2].append(float(msg.linear_acceleration.z))

    t_arr = np.asarray(t_list, dtype=float)
    order = np.argsort(t_arr)
    t: np.ndarray = t_arr[order]
    w = [np.asarray(ch, dtype=float)[order] for ch in w]
    a = [np.asarray(ch, dtype=float)[order] for ch in a]

    # Drop duplicate / non-increasing timestamps
    keep = np.concatenate([[True], np.diff(t) > 0])
    t = t[keep]
    w = [ch[keep] for ch in w]
    a = [ch[keep] for ch in a]

    # Estimate fs if needed
    if args.fs <= 0:
        dt_med = np.median(np.diff(t))
        fs = 1.0 / dt_med
        fs = round(fs)  # good enough
    else:
        fs = args.fs

    # Resample uniformly
    tu, _ = resample_uniform(t, w[0], fs)

    results = {}

    def process(name: str, chans: list[np.ndarray], units: str) -> None:
        nonlocal results
        print(f"\n=== {name} ({units}) ===")
        sig_w = []
        sig_b = []

        for i, ch in enumerate(chans):
            # subtract mean (helps numerics; doesn’t change Allan slopes)
            _, yu = resample_uniform(t, ch - np.mean(ch), fs)
            taus, adev = allan_dev_overlapping(yu, fs)

            sigma_w, tau_rng_w = fit_sigma_w(taus, adev)
            sigma_b, tau_rng_b = fit_sigma_b_rw(taus, adev)

            axis = "xyz"[i]
            print(f"Axis {axis}:")
            if sigma_w is not None and tau_rng_w is not None:
                print(
                    f"  noise_density ≈ {sigma_w:.6e}  ({units}/√Hz)   fit τ∈[{tau_rng_w[0]:.3g},{tau_rng_w[1]:.3g}] s"
                )
                sig_w.append(sigma_w)
            else:
                print("  noise_density: could not auto-detect −1/2 slope region")

            if sigma_b is not None and tau_rng_b is not None:
                # units here are (units/s)/√Hz = units/(s*√Hz)
                # for gyro units=rad/s => rad/s^2/√Hz; accel units=m/s^2 => m/s^3/√Hz
                print(
                    f"  random_walk   ≈ {sigma_b:.6e}  ({units}/s/√Hz)  fit τ∈[{tau_rng_b[0]:.3g},{tau_rng_b[1]:.3g}] s"
                )
                sig_b.append(sigma_b)
            else:
                print("  random_walk: could not auto-detect +1/2 slope region (try longer log)")

            if args.plot:
                plt.figure()
                plt.loglog(taus, adev)
                plt.xlabel("τ (s)")
                plt.ylabel("Allan deviation")
                plt.title(f"{name} axis {axis}")
                plt.grid(True, which="both")
                plt.show()

        results[name] = {
            "fs_used": fs,
            "noise_density_avg": float(np.mean(sig_w)) if sig_w else None,
            "random_walk_avg": float(np.mean(sig_b)) if sig_b else None,
        }

    process("gyro", w, "rad/s")
    process("accel", a, "m/s^2")

    print("\n=== Suggested cuVSLAM / Isaac ROS params (average of axes) ===")
    print(f"fs_used: {results['gyro']['fs_used']} Hz")
    print(f"gyroscope_noise_density:      {results['gyro']['noise_density_avg']}")
    print(f"gyroscope_random_walk:        {results['gyro']['random_walk_avg']}")
    print(f"accelerometer_noise_density:  {results['accel']['noise_density_avg']}")
    print(f"accelerometer_random_walk:    {results['accel']['random_walk_avg']}")


if __name__ == "__main__":
    main()
