import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from satsim.utilities.consts import R_EARTH


class Visualizer:

    def __init__(self, history, config):
        self.history = history
        self.config = config
        self.save_dir = config.get("save_dir")
        self.show = config.get("show", True)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    #entry point
    def run(self):
        #generate plots where config is flagged true
        plot_methods = {
            "altitude":     self.plot_altitude,
            "orbit_2d":     self.plot_orbit_2d,
            "orbit_3d":     self.plot_orbit_3d,
            "ground_track": self.plot_ground_track,
        }
        for name, method in plot_methods.items():
            if self.config.get(name, False):
                print(f"[viz] generating {name}")
                method()

    #save/show/close logic
    def _finalize(self, fig, name):
        if self.save_dir:
            fig.savefig(os.path.join(self.save_dir, f"{name}.png"), dpi=150)
        if self.show:
            plt.show()
        plt.close(fig)

    #the plots
    def plot_altitude(self):
        t = self.history["t"] / 60.0
        alt_km = (np.linalg.norm(self.history["r"], axis=1) - R_EARTH) / 1000.0

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, alt_km, linewidth=1.2)
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Altitude [km]")
        ax.set_title("Spacecraft altitude")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._finalize(fig, "altitude")

    def plot_orbit_2d(self):
        r_km = self.history["r"] / 1000.0
        x, y = r_km[:, 0], r_km[:, 1]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(x, y, linewidth=0.8)
        ax.scatter(x[0],  y[0],  color="green", s=40, label="start", zorder=5)
        ax.scatter(x[-1], y[-1], color="red",   s=40, label="end",   zorder=5)

        theta = np.linspace(0, 2 * np.pi, 200)
        R = R_EARTH / 1000.0
        ax.fill(R * np.cos(theta), R * np.sin(theta),
                color="lightblue", alpha=0.6)

        ax.set_xlabel("X [km, ECI]")
        ax.set_ylabel("Y [km, ECI]")
        ax.set_title("Orbit, equatorial projection")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        self._finalize(fig, "orbit_2d")

    def plot_orbit_3d(self):
        r_km = self.history["r"] / 1000.0

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(r_km[:, 0], r_km[:, 1], r_km[:, 2], linewidth=0.8, color="C0")
        ax.scatter(*r_km[0],  color="green", s=40, label="start")
        ax.scatter(*r_km[-1], color="red",   s=40, label="end")

        R = R_EARTH / 1000.0
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        ax.plot_wireframe(R * np.cos(u) * np.sin(v),
                          R * np.sin(u) * np.sin(v),
                          R * np.cos(v),
                          color="lightblue", alpha=0.3, linewidth=0.4)

        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_zlabel("Z [km]")
        ax.set_title("Orbit, 3D")
        ax.legend()

        max_range = np.max(np.abs(r_km)) * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        fig.tight_layout()
        self._finalize(fig, "orbit_3d")

    def plot_ground_track(self):
        try:
            from utilities.coords import eci_to_ecef, ecef_to_geodetic
        except ImportError:
            print("[plot_ground_track] skipped: utilities.coords not implemented yet")
            return

        t = self.history["t"]
        r_eci = self.history["r"]

        lats, lons = [], []
        for ti, ri in zip(t, r_eci):
            r_ecef = eci_to_ecef(ri, ti)
            lat, lon, _ = ecef_to_geodetic(r_ecef)
            lats.append(np.degrees(lat))
            lons.append(np.degrees(lon))

        lats = np.array(lats)
        lons = np.array(lons)

        fig, ax = plt.subplots(figsize=(11, 5))
        seg_starts = [0] + list(np.where(np.abs(np.diff(lons)) > 180)[0] + 1) + [len(lons)]
        for s, e in zip(seg_starts[:-1], seg_starts[1:]):
            ax.plot(lons[s:e], lats[s:e], color="C0", linewidth=1)
        ax.scatter(lons[0],  lats[0],  color="green", s=40, label="start", zorder=5)
        ax.scatter(lons[-1], lats[-1], color="red",   s=40, label="end",   zorder=5)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title("Ground track")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        self._finalize(fig, "ground_track")