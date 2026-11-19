"""Gera uma curva de secagem esquemática com dados arbitrários."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class DryingCurveParameters:
    initial_moisture: float = 1.20
    critical_moisture: float = 0.55
    equilibrium_moisture: float = 0.10
    initial_rate: float = 0.125
    constant_rate: float = 0.15
    warmup_end_time: float = 1.5
    initial_temperature: float = 20.0
    wet_bulb_temperature: float = 35.0
    air_temperature: float = 70.0

    def __post_init__(self):
        if not (0 <= self.equilibrium_moisture < self.critical_moisture):
            raise ValueError("A umidade crítica deve ser maior que a de equilíbrio.")
        if not (0 <= self.initial_rate < self.constant_rate):
            raise ValueError("A taxa inicial deve ser menor que a taxa constante.")
        if self.warmup_end_time <= 0:
            raise ValueError("O tempo de aquecimento deve ser positivo.")
        warmup_loss = (
            (self.initial_rate + self.constant_rate)
            * self.warmup_end_time / 2
        )
        if self.initial_moisture - warmup_loss <= self.critical_moisture:
            raise ValueError("A fase de taxa constante deve ter duração positiva.")
        if not (
            self.initial_temperature < self.wet_bulb_temperature
            < self.air_temperature
        ):
            raise ValueError("As temperaturas devem aumentar em cada fase.")


@dataclass(frozen=True)
class DryingCurve:
    time: np.ndarray
    moisture: np.ndarray
    rate: np.ndarray
    temperature: np.ndarray
    critical_time: float


def critical_time(p):
    warmup_loss = (p.initial_rate + p.constant_rate) * p.warmup_end_time / 2
    return p.warmup_end_time + (
        p.initial_moisture - warmup_loss - p.critical_moisture
    ) / p.constant_rate


def moisture_at(p, time):
    time = np.asarray(time)
    warmup_loss = (p.initial_rate + p.constant_rate) * p.warmup_end_time / 2
    warmup = (
        p.initial_moisture - p.initial_rate * time
        - (p.constant_rate - p.initial_rate) * time**2
        / (2 * p.warmup_end_time)
    )
    constant = (
        p.initial_moisture - warmup_loss
        - p.constant_rate * (time - p.warmup_end_time)
    )
    tc = critical_time(p)
    decay = p.constant_rate / (p.critical_moisture - p.equilibrium_moisture)
    falling = p.equilibrium_moisture + (
        p.critical_moisture - p.equilibrium_moisture
    ) * np.exp(-decay * np.maximum(time - tc, 0))
    return np.where(time < p.warmup_end_time, warmup,
                    np.where(time < tc, constant, falling))


def drying_rate_at(p, time):
    time = np.asarray(time)
    warmup = p.initial_rate + (
        p.constant_rate - p.initial_rate
    ) * time / p.warmup_end_time
    tc = critical_time(p)
    decay = p.constant_rate / (p.critical_moisture - p.equilibrium_moisture)
    falling = p.constant_rate * np.exp(-decay * np.maximum(time - tc, 0))
    return np.where(time < p.warmup_end_time, warmup,
                    np.where(time < tc, p.constant_rate, falling))


def temperature_at(p, time):
    time = np.asarray(time)
    warmup = p.initial_temperature + (
        p.wet_bulb_temperature - p.initial_temperature
    ) * time / p.warmup_end_time
    tc = critical_time(p)
    decay = p.constant_rate / (p.critical_moisture - p.equilibrium_moisture)
    falling = p.air_temperature - (
        p.air_temperature - p.wet_bulb_temperature
    ) * np.exp(-decay * np.maximum(time - tc, 0))
    return np.where(time < p.warmup_end_time, warmup,
                    np.where(time < tc, p.wet_bulb_temperature, falling))


def compute_drying_curve(p, n_points=1001):
    if n_points < 2:
        raise ValueError("São necessários pelo menos dois pontos.")
    tc = critical_time(p)
    decay = p.constant_rate / (p.critical_moisture - p.equilibrium_moisture)
    time = np.linspace(0, tc + 4.5 / decay, n_points)
    return DryingCurve(
        time, moisture_at(p, time), drying_rate_at(p, time),
        temperature_at(p, time), tc,
    )


def default_output_dir():
    return Path(__file__).resolve().parents[1] / "src" / "drying_curve"


def build_figure(p):
    curve = compute_drying_curve(p)
    fig, moisture_ax = plt.subplots(figsize=(12, 6.8))
    fig.subplots_adjust(left=0.11, right=0.76, bottom=0.20, top=0.82)
    rate_ax = moisture_ax.twinx()
    temperature_ax = moisture_ax.twinx()
    temperature_ax.spines["right"].set_position(("axes", 1.20))

    moisture_line, = moisture_ax.plot(
        curve.time, curve.moisture, color="#0072B2", lw=2.7,
        label="Teor de umidade",
    )
    rate_line, = rate_ax.plot(
        curve.time, curve.rate, color="#D55E00", lw=2.5, ls="--",
        label="Taxa de secagem",
    )
    temp_line, = temperature_ax.plot(
        curve.time, curve.temperature, color="#009E73", lw=2.5, ls="-.",
        label="Temperatura do produto",
    )
    moisture_ax.set(
        xlabel="Tempo de secagem (h)",
        ylabel="Teor de umidade (kg de água/kg de sólido seco)",
        xlim=(0, curve.time[-1]), ylim=(0, 1.35),
    )
    rate_ax.set_ylabel("Taxa de secagem (kg de água/(kg de sólido seco·h))")
    rate_ax.set_ylim(0, 0.22)
    temperature_ax.set_ylabel("Temperatura do produto (°C)")
    temperature_ax.set_ylim(0, 75)
    moisture_ax.grid(True, color="#d9d9d9", lw=0.6)
    moisture_ax.set_axisbelow(True)

    moisture_ax.axvline(p.warmup_end_time, color="0.5", lw=1, ls=":")
    moisture_ax.axvline(curve.critical_time, color="0.5", lw=1, ls=":")
    moisture_ax.axhline(p.equilibrium_moisture, color="#0072B2", lw=1, ls=":")
    moisture_ax.scatter(
        [curve.critical_time], [p.critical_moisture],
        color="#0072B2", marker="o", zorder=5,
    )
    moisture_ax.annotate(
        "Umidade crítica", (curve.critical_time, p.critical_moisture),
        xytext=(curve.critical_time + 2.3, p.critical_moisture + 0.18),
        arrowprops={"arrowstyle": "->", "color": "0.25"},
        fontsize=10,
    )
    moisture_ax.annotate(
        "Umidade de equilíbrio", (curve.time[-1] * 0.72, p.equilibrium_moisture),
        xytext=(curve.time[-1] * 0.58, p.equilibrium_moisture + 0.13),
        arrowprops={"arrowstyle": "->", "color": "0.25"},
        fontsize=10,
    )
    for start, end, label in (
        (0, p.warmup_end_time, "Inicial"),
        (p.warmup_end_time, curve.critical_time, "Taxa constante"),
        (curve.critical_time, curve.time[-1], "Taxa decrescente"),
    ):
        moisture_ax.text(
            (start + end) / 2, 1.02, label,
            transform=moisture_ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=10,
        )

    fig.suptitle("Curva de secagem genérica", fontsize=16, y=0.96)
    fig.legend(
        handles=[moisture_line, rate_line, temp_line], loc="lower center",
        bbox_to_anchor=(0.45, 0.07), ncol=3, frameon=False, fontsize=10,
    )
    fig.text(
        0.45, 0.025, "Esquema ilustrativo — dados arbitrários; "
        "equilíbrio aproximado assintoticamente.",
        ha="center", fontsize=9,
    )
    return fig


def generate_drying_curve_figure(output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(DryingCurveParameters())
    paths = [output_dir / "curva_de_secagem.png",
             output_dir / "curva_de_secagem.pdf"]
    try:
        for path in paths:
            figure.savefig(path, dpi=300, facecolor="white")
    finally:
        plt.close(figure)
    return paths


if __name__ == "__main__":
    for output in generate_drying_curve_figure():
        print(f"Figura salva em: {output}")