import numpy as np
import matplotlib.pyplot as plt

# Constants
T0 = 37.0  # Initial body temperature (°C)
T_env = 20.0  # Environmental temperature (°C)
T_target = 36.8  # Target body temperature (°C)
metabolic_rate = 0.008  # °C per minute (physiologically realistic)
kp = 0.3  # Proportional control gain

def simulate(k, cooling=False, duration=120):
    """
    Simulates body temperature regulation in microgravity with Lunelle intervention.

    Features:
    - Newtonian cooling + metabolic heat generation
    - Sensor noise + moving average filtering
    - Closed-loop proportional control
    - State-based triggering (temperature + stability)
    - Timeout fallback to prevent deadlock
    """

    time = np.arange(0, duration)
    T = [T0]
    N = 5  # moving average window
    time_above_target = 0  # fallback counter

    for t in range(1, len(time)):
        current = T[-1]

        # ---- Measurement (filtered + noisy) ----
        if len(T) >= N:
            avg_temp = np.mean(T[-N:])
        else:
            avg_temp = np.mean(T)

        current_measured = avg_temp + np.random.normal(0, 0.05)

        # ---- Stability check ----
        if t > 1:
            delta = abs(T[-1] - T[-2])
        else:
            delta = 0

        # ---- Physics ----
        dT = -k * (current - T_env) + metabolic_rate

        # ---- Timeout fallback tracking ----
        if current_measured > T_target:
            time_above_target += 1
        else:
            time_above_target = 0

        # ---- Control logic ----
        if cooling and (
            (current_measured > T_target and delta < 0.01)
            or
            (time_above_target > 15)
        ):
            error = current_measured - T_target
            dT -= kp * error

        # ---- Update ----
        T.append(current + dT)

    return np.array(T)


if __name__ == "__main__":
    time = np.arange(0, 120)

    # Earth vs Space vs Lunelle
    k_earth = 0.05
    k_space = 0.015

    earth = simulate(k_earth, cooling=False)
    space = simulate(k_space, cooling=False)
    lunelle = simulate(k_space, cooling=True)

    plt.figure(figsize=(10, 6))
    plt.plot(time, earth, label="Earth (Normal Cooling)")
    plt.plot(time, space, label="Microgravity (Poor Cooling)")
    plt.plot(time, lunelle, label="Lunelle Intervention")

    plt.axhline(y=T_target, linestyle='--', label="Sleep Threshold (36.8°C)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Body Temperature (°C)")
    plt.title("Lunelle v0.1 — Closed-Loop Thermal Regulation in Microgravity")
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig("/mnt/data/thermal_plot.png")
    plt.show()
