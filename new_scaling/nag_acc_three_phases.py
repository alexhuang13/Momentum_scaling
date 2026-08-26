import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# Global plotting settings
# =============================================================================

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (5, 4),
    "legend.fontsize": 10,
    "legend.handlelength": 1,
})

# =============================================================================
# 1. Global configuration
# =============================================================================

BETA = 6.0
S_VALUE = 0.3
SIGMA = 0.1

DIM = 4096
N_RUNS = 50
D_LIST = [2000, 4000, 8000, 16000, 32000, 64000, 128000]
BASE_SEED = 20260725
TAIL_FRACTION = 0.05

# =============================================================================
# USER-EDITABLE BATCH-SIZE PARAMETERS
# =============================================================================
# For every phase and optimizer,
#     B = round(batch_coefficient * D**batch_exponent).
# The exponent is shared within a phase, while the coefficient can be tuned
# independently for SGD, Polyak, and Nesterov.

# Small batch: a_small = 0, so these are the actual constant batch sizes.
SMALL_BATCH_EXPONENT = 0.0
SMALL_BATCH_COEFFICIENTS = {
    "sgd": 10.0,
    "hb": 10.0,
    "nag": 10.0,
}

# Large batch: B_A = c_A * D^a_large for each optimizer A.
LARGE_BATCH_EXPONENT = (
    (2.0 * S_VALUE * BETA + 1.0)
    / (2.0 * S_VALUE * BETA + BETA + 1.0)
    - 0.05
)
LARGE_BATCH_COEFFICIENTS = {
    "sgd": 0.5,
    "hb": 0.5,
    "nag": 0.5,
}

# Super-large batch: B_A = c_A * D^0.8 for each optimizer A.
SUPER_BATCH_EXPONENT = 0.8
SUPER_BATCH_COEFFICIENTS = {
    "sgd": 0.08,
    "hb": 0.08,
    "nag": 0.08,
}

# Strict slack in a+b<1 for the super-large phase.
TRANSIENT_SLACK = 0.01

CURRENT_DIR = Path.cwd()

# All runs still live under the original nag_acc_three_phases folder.
BASE_OUTPUT_DIR = (
    CURRENT_DIR
    if CURRENT_DIR.name == "nag_acc_three_phases"
    else CURRENT_DIR / "nag_acc_three_phases"
)

# Separate different (s, beta) experiments while keeping every original
# output filename unchanged, e.g. nag_acc_three_phases/s0.3beta4/.
RUN_SUBDIR = f"s{S_VALUE:g}beta{BETA:g}"
OUTPUT_DIR = BASE_OUTPUT_DIR / RUN_SUBDIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["sgd", "hb", "nag"]
GROUP_LABELS = {
    "sgd": "SGD",
    "hb": "Polyak",
    "nag": "Nesterov",
}
MARKERS = {
    "sgd": "o",
    "hb": "^",
    "nag": "s",
}

# Fourth point is the middle of the seven D values.
THEORY_ANCHOR_INDEX = 3

BASE_OPTIMAL_EXPONENT = S_VALUE / (S_VALUE + 1.0)
POLYAK_MAX_BATCH_EXPONENT = (
    2.0 * S_VALUE + 1.0
) / (
    2.0 * S_VALUE + 2.0
)
NAG_MAX_BATCH_EXPONENT = (
    2.0 * S_VALUE * BETA + 1.0
) / (
    2.0 * S_VALUE * BETA + BETA + 1.0
)

if not 0.0 < LARGE_BATCH_EXPONENT < 1.0:
    raise ValueError("Invalid large-batch exponent.")
if not POLYAK_MAX_BATCH_EXPONENT < SUPER_BATCH_EXPONENT < 1.0:
    raise ValueError("Invalid super-large-batch exponent.")
if not 0.0 < TRANSIENT_SLACK < 1.0 - SUPER_BATCH_EXPONENT:
    raise ValueError("TRANSIENT_SLACK is incompatible with a_super.")

print(f"beta={BETA:g}, s={S_VALUE:g}, sigma={SIGMA:g}")
print(f"dim={DIM}, nruns={N_RUNS}")
print(f"D values={D_LIST}")
print(f"output directory={OUTPUT_DIR.resolve()}")
print()
print(f"SGD critical exponent      = {BASE_OPTIMAL_EXPONENT:.9f}")
print(f"Polyak critical exponent   = {POLYAK_MAX_BATCH_EXPONENT:.9f}")
print(f"Nesterov crossover exponent= {NAG_MAX_BATCH_EXPONENT:.9f}")
print(f"large-batch exponent       = {LARGE_BATCH_EXPONENT:.9f}")
print(f"super-large exponent       = {SUPER_BATCH_EXPONENT:.9f}")
print()
for group in GROUP_ORDER:
    print(
        f"small/{GROUP_LABELS[group]}: "
        f"B = {SMALL_BATCH_COEFFICIENTS[group]:g} "
        f"* D^{SMALL_BATCH_EXPONENT:.9f}"
    )
for group in GROUP_ORDER:
    print(
        f"large/{GROUP_LABELS[group]}: "
        f"B = {LARGE_BATCH_COEFFICIENTS[group]:g} "
        f"* D^{LARGE_BATCH_EXPONENT:.9f}"
    )
for group in GROUP_ORDER:
    print(
        f"super/{GROUP_LABELS[group]}: "
        f"B = {SUPER_BATCH_COEFFICIENTS[group]:g} "
        f"* D^{SUPER_BATCH_EXPONENT:.9f}"
    )



# =============================================================================
# 2. Construct all phase configurations
# =============================================================================

def make_algorithm_config(
    *,
    a,
    regime,
    batch_constants,
    fixed_batches=None,
):
    """Return theory-scaled (a,b,c,p) choices for all three algorithms."""
    a = float(a)

    # SGD.
    sgd_c = max(BASE_OPTIMAL_EXPONENT - a, 0.0)
    sgd_p = min(
        S_VALUE * (1.0 - a - sgd_c),
        a + sgd_c,
    )

    # Polyak.
    if regime == "small":
        hb_b = 0.0
        hb_c = BASE_OPTIMAL_EXPONENT - a
        hb_p = BASE_OPTIMAL_EXPONENT
    elif regime == "large":
        hb_c = 0.0
        hb_b = a - BASE_OPTIMAL_EXPONENT
        hb_p = BASE_OPTIMAL_EXPONENT
    elif regime == "super":
        hb_c = 0.0
        hb_b = 1.0 - a - TRANSIENT_SLACK
        hb_p = S_VALUE * (1.0 - a - hb_c + hb_b)
    else:
        raise ValueError(f"Unknown regime: {regime}.")

    # Nesterov.
    if regime == "small":
        nag_b = 0.0
        nag_c = BASE_OPTIMAL_EXPONENT - a
        nag_p = BASE_OPTIMAL_EXPONENT
    elif regime == "large":
        nag_c = 0.0
        nag_x = (
            S_VALUE * BETA
            - (BETA - 1.0) * a
        ) / (
            S_VALUE * BETA + 1.0
        )
        nag_b = a - nag_x
        nag_p = S_VALUE * (1.0 - nag_x)
    elif regime == "super":
        nag_c = 0.0
        nag_b = 1.0 - a - TRANSIENT_SLACK
        nag_p = S_VALUE * (1.0 - a - nag_c + nag_b)
    else:
        raise ValueError(f"Unknown regime: {regime}.")

    required_groups = set(GROUP_ORDER)
    if set(batch_constants) != required_groups:
        raise ValueError(
            "batch_constants must contain exactly: "
            f"{sorted(required_groups)}"
        )
    if fixed_batches is None:
        fixed_batches = {group: None for group in GROUP_ORDER}
    if set(fixed_batches) != required_groups:
        raise ValueError(
            "fixed_batches must contain exactly: "
            f"{sorted(required_groups)}"
        )

    def batch_fields(group):
        coefficient = float(batch_constants[group])
        if coefficient <= 0:
            raise ValueError(
                f"Batch coefficient for {group} must be positive."
            )
        return {
            "a": a,
            "batch_constant": coefficient,
            "fixed_batch": fixed_batches[group],
        }

    return {
        "sgd": {
            **batch_fields("sgd"),
            "b": 0.0,
            "c": sgd_c,
            "p": sgd_p,
        },
        "hb": {
            **batch_fields("hb"),
            "b": hb_b,
            "c": hb_c,
            "p": hb_p,
        },
        "nag": {
            **batch_fields("nag"),
            "b": nag_b,
            "c": nag_c,
            "p": nag_p,
        },
    }


PHASE_CONFIGS = {
    "small_batch": {
        "label": "Small batch",
        "regime": "small",
        "a": SMALL_BATCH_EXPONENT,
        "batch_constants": SMALL_BATCH_COEFFICIENTS,
        "fixed_batches": {group: None for group in GROUP_ORDER},
        "eta_constants": {
            "sgd": 0.6,
            "hb": 0.4,
            "nag": 0.41,
        },
        "gap_constants": {
            "sgd": 0.25,
            "hb": 0.25,
            "nag": 0.25,
        },
        "filename": "small_batch",
    },
    "large_batch": {
        "label": "Large batch",
        "regime": "large",
        "a": LARGE_BATCH_EXPONENT,
        "batch_constants": LARGE_BATCH_COEFFICIENTS,
        "fixed_batches": {group: None for group in GROUP_ORDER},
        "eta_constants": {
            "sgd": 0.1,
            "hb": 0.10,
            "nag": 0.11,
        },
        "gap_constants": {
            "sgd": 1.0,
            "hb": 1.0,
            "nag": 1.0,
        },
        "filename": "large_batch",
    },
    "super_large_batch": {
        "label": "Super-large batch",
        "regime": "super",
        "a": SUPER_BATCH_EXPONENT,
        "batch_constants": SUPER_BATCH_COEFFICIENTS,
        "fixed_batches": {group: None for group in GROUP_ORDER},
        "eta_constants": {
            "sgd": 0.30,
            "hb": 0.1,
            "nag": 0.11,
        },
        "gap_constants": {
            "sgd": 1.0,
            "hb": 1.0,
            "nag": 1.0,
        },
        "filename": "super_large_batch",
    },
}

for phase_name, phase in PHASE_CONFIGS.items():
    phase["algorithms"] = make_algorithm_config(
        a=phase["a"],
        regime=phase["regime"],
        batch_constants=phase["batch_constants"],
        fixed_batches=phase["fixed_batches"],
    )

# -----------------------------------------------------------------------------
# Theoretical checks.
# -----------------------------------------------------------------------------

small = PHASE_CONFIGS["small_batch"]["algorithms"]
large = PHASE_CONFIGS["large_batch"]["algorithms"]
super_large = PHASE_CONFIGS["super_large_batch"]["algorithms"]

assert np.isclose(small["sgd"]["p"], small["hb"]["p"])
assert np.isclose(small["hb"]["p"], small["nag"]["p"])

assert large["sgd"]["p"] < large["hb"]["p"] < large["nag"]["p"]

assert super_large["sgd"]["p"] < super_large["hb"]["p"]
assert np.isclose(super_large["hb"]["p"], super_large["nag"]["p"])
assert np.isclose(
    super_large["hb"]["b"],
    1.0 - SUPER_BATCH_EXPONENT - TRANSIENT_SLACK,
)

phase_rows = []
for phase_name, phase in PHASE_CONFIGS.items():
    for group in GROUP_ORDER:
        cfg = phase["algorithms"][group]
        phase_rows.append({
            "phase": phase_name,
            "algorithm": GROUP_LABELS[group],
            "a": cfg["a"],
            "b": cfg["b"],
            "c": cfg["c"],
            "theory exponent": cfg["p"],
            "batch coefficient": cfg["batch_constant"],
            "eta constant": phase["eta_constants"][group],
            "gap constant": phase["gap_constants"][group],
            "fixed batch": cfg["fixed_batch"],
        })

PHASE_PARAMETER_TABLE = pd.DataFrame(phase_rows)
print(PHASE_PARAMETER_TABLE)


# =============================================================================
# 3. Power-law kernel-regression problem
# =============================================================================

def build_problem(
    dim=DIM,
    beta=BETA,
    s=S_VALUE,
    dtype=np.float32,
):
    """
    Spectrum:
        lambda_j = j^{-beta}

    Source condition:
        lambda_j * theta_star_j^2 = j^{-1-beta*s}

    Hence:
        theta_star_j^2 = j^{-1-beta*(s-1)}.
    """
    if dim < 1:
        raise ValueError("DIM must be positive.")
    if beta <= 1:
        raise ValueError("BETA must be greater than 1.")
    if s <= 0:
        raise ValueError("S_VALUE must be positive.")

    j = np.arange(1, dim + 1, dtype=np.float64)
    lam = j ** (-beta)
    theta_squared = j ** (
        -(1.0 + beta * (s - 1.0))
    )
    theta_star = np.sqrt(theta_squared)

    return (
        lam.astype(dtype),
        theta_star.astype(dtype),
    )


def excess_loss_batch(delta, lam):
    """Return one excess loss for every row of delta."""
    return 0.5 * np.sum(
        lam[None, :] * delta * delta,
        axis=1,
        dtype=np.float64,
    )


def excess_loss_single(delta, lam):
    return float(
        0.5 * np.sum(
            lam * delta * delta,
            dtype=np.float64,
        )
    )


LAM, THETA_STAR = build_problem()

print(f"lambda_1={LAM[0]:.6e}")
print(f"lambda_DIM={LAM[-1]:.6e}")
print(
    "initial excess loss="
    f"{excess_loss_single(-THETA_STAR, LAM):.6e}"
)


# =============================================================================
# 4. Vectorized SGD / Heavy-Ball / Nesterov simulator
# =============================================================================

def stable_seed(*items):
    text = "|".join(str(item) for item in items)
    return (
        BASE_SEED
        + zlib.crc32(text.encode("utf-8"))
    ) % (2**32 - 1)


def tail_length(num_steps, tail_fraction=TAIL_FRACTION):
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must lie in (0,1].")

    return max(
        1,
        int(np.ceil(tail_fraction * num_steps)),
    )


def run_optimizer(
    algo,
    batch_size,
    num_steps,
    eta,
    rho,
    lam=LAM,
    theta_star=THETA_STAR,
    sigma=SIGMA,
    n_runs=N_RUNS,
    seed=0,
    tail_fraction=TAIL_FRACTION,
):
    """
    Run N_RUNS independent covariance-surrogate trajectories.

    Return one metric per run: the average excess loss over the final
    ceil(tail_fraction * num_steps) optimization steps.
    """
    if algo not in {"sgd", "hb", "nag"}:
        raise ValueError(f"Unknown algorithm: {algo}.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if n_runs < 1:
        raise ValueError("n_runs must be positive.")
    if not (0.0 < eta < 1.0):
        raise ValueError(f"eta must lie in (0,1), got {eta}.")
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must lie in [0,1), got {rho}.")

    rng = np.random.default_rng(seed)
    runs = int(n_runs)
    dim = lam.size

    delta = np.broadcast_to(
        -theta_star,
        (runs, dim),
    ).copy()
    delta_previous = delta.copy()

    lam_row = lam[None, :]
    sqrt_lam_row = np.sqrt(lam).astype(
        np.float32
    )[None, :]

    num_tail_steps = tail_length(
        num_steps,
        tail_fraction,
    )
    first_tail_step = (
        num_steps - num_tail_steps + 1
    )

    tail_loss_sum = np.zeros(
        runs,
        dtype=np.float64,
    )

    for step in range(1, num_steps + 1):
        if algo == "nag":
            evaluation_delta = (
                delta
                + rho * (delta - delta_previous)
            )
        else:
            evaluation_delta = delta

        evaluation_loss = excess_loss_batch(
            evaluation_delta,
            lam,
        )

        gaussian = rng.standard_normal(
            size=(runs, dim)
        ).astype(
            np.float32,
            copy=False,
        )

        noise_scale = np.sqrt(
            (evaluation_loss + sigma**2)
            / batch_size
        ).astype(np.float32)

        noise = (
            noise_scale[:, None]
            * sqrt_lam_row
            * gaussian
        )
        gradient = (
            lam_row * evaluation_delta
            + noise
        )

        if algo == "sgd":
            next_delta = delta - eta * gradient
        elif algo == "hb":
            next_delta = (
                delta
                - eta * gradient
                + rho * (delta - delta_previous)
            )
        else:
            next_delta = (
                evaluation_delta
                - eta * gradient
            )

        delta_previous, delta = (
            delta,
            next_delta.astype(
                np.float32,
                copy=False,
            ),
        )

        if step >= first_tail_step:
            tail_loss_sum += excess_loss_batch(
                delta,
                lam,
            )

    tail_mean_losses = (
        tail_loss_sum / num_tail_steps
    )

    if np.any(~np.isfinite(tail_mean_losses)):
        raise FloatingPointError(
            "Non-finite tail-average loss for "
            f"algo={algo}, B={batch_size}, "
            f"K={num_steps}, eta={eta}, rho={rho}."
        )

    return tail_mean_losses


# =============================================================================
# 5. Finite-D schedules
# =============================================================================

def make_base_schedule(D, algorithm_config):
    """Construct optimizer-specific B = round(batch_constant * D**a)."""
    if D < 1:
        raise ValueError("D must be positive.")

    fixed_batch = algorithm_config["fixed_batch"]
    if fixed_batch is not None:
        batch_size = int(fixed_batch)
    else:
        batch_size = max(
            1,
            int(round(
                algorithm_config["batch_constant"]
                * D ** algorithm_config["a"]
            )),
        )

    num_steps = max(
        1,
        int(D // batch_size),
    )

    used_budget = batch_size * num_steps

    return {
        "D": int(D),
        "B": int(batch_size),
        "K": int(num_steps),
        "used_budget": int(used_budget),
        "used_fraction": float(used_budget / D),
        "tail_fraction": float(TAIL_FRACTION),
        "tail_steps": int(tail_length(num_steps)),
        "a": float(algorithm_config["a"]),
        "batch_constant": float(
            algorithm_config["batch_constant"]
        ),
        "fixed_batch": fixed_batch,
    }


def make_schedule(phase_name, group, D):
    phase = PHASE_CONFIGS[phase_name]
    cfg = phase["algorithms"][group]
    schedule = make_base_schedule(D, cfg)

    eta = float(
        phase["eta_constants"][group]
        * D ** (-cfg["c"])
    )

    if group == "sgd":
        one_minus_rho = 1.0
        rho = 0.0
    else:
        one_minus_rho = min(
            1.0,
            phase["gap_constants"][group]
            * D ** (-cfg["b"]),
        )
        rho = 1.0 - one_minus_rho

    schedule.update({
        "phase": phase_name,
        "phase_label": phase["label"],
        "group": group,
        "algorithm": group,
        "eta": eta,
        "rho": float(rho),
        "one_minus_rho": float(one_minus_rho),
        "b": float(cfg["b"]),
        "c": float(cfg["c"]),
        "p_theory": float(cfg["p"]),
        "K_times_gap": float(
            schedule["K"] * one_minus_rho
        ),
        "eta_over_gap_squared": float(
            eta / one_minus_rho**2
        ),
    })

    # Finite-D stability diagnostics.
    if group == "hb":
        schedule["stability_ratio"] = float(
            eta / min(1.0, schedule["B"] * one_minus_rho)
        )
    elif group == "nag":
        schedule["stability_ratio"] = float(
            eta / min(
                1.0,
                schedule["B"] ** BETA * one_minus_rho,
            )
        )
    else:
        schedule["stability_ratio"] = float(eta)

    return schedule


SCHEDULE_PREVIEW = pd.concat(
    [
        pd.DataFrame([
            make_schedule(phase_name, group, D)
            for D in D_LIST
        ])
        for phase_name in PHASE_CONFIGS
        for group in GROUP_ORDER
    ],
    ignore_index=True,
)

print(SCHEDULE_PREVIEW)


# =============================================================================
# 6. Run, summarize, plot, and save one phase
# =============================================================================

def fit_exponent(D_values, mean_losses):
    D_values = np.asarray(D_values, dtype=np.float64)
    mean_losses = np.asarray(mean_losses, dtype=np.float64)

    if np.any(~np.isfinite(mean_losses)):
        raise ValueError("Mean losses must be finite.")
    if np.any(mean_losses <= 0):
        raise ValueError("Mean losses must be positive.")

    slope, intercept = np.polyfit(
        np.log(D_values),
        np.log(mean_losses),
        deg=1,
    )
    return -float(slope), float(intercept)


def plot_phase(
    phase_name,
    mean_curves,
    theory_exponents,
):
    phase = PHASE_CONFIGS[phase_name]
    D_array = np.asarray(D_LIST, dtype=np.float64)

    if not 0 <= THEORY_ANCHOR_INDEX < len(D_array):
        raise IndexError(
            "THEORY_ANCHOR_INDEX must identify an existing D point."
        )

    plt.figure()

    for group in GROUP_ORDER:
        mean_losses = np.asarray(
            mean_curves[group],
            dtype=np.float64,
        )

        point_handle, = plt.plot(
            D_array,
            mean_losses,
            linestyle="None",
            marker=MARKERS[group],
            label=GROUP_LABELS[group],
            zorder=3,
        )

        point_color = point_handle.get_color()
        prediction_exponent = float(theory_exponents[group])

        anchor_D = D_array[THEORY_ANCHOR_INDEX]
        anchor_loss = mean_losses[THEORY_ANCHOR_INDEX]
        prediction_losses = anchor_loss * (
            D_array / anchor_D
        ) ** (-prediction_exponent)

        plt.plot(
            D_array,
            prediction_losses,
            linestyle="--",
            color=point_color,
            label="_nolegend_",
            zorder=2,
        )

    title_map = {
        "small_batch": "Small batch",
        "large_batch": "Large batch",
        "super_large_batch": "Ultra-large batch",
    }

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Data D")
    plt.ylabel("Loss")
    plt.title(title_map.get(phase_name, phase["label"]))

    prediction_legend = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Prediction",
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(prediction_legend)
    labels.append("Prediction")
    plt.legend(handles, labels)
    plt.tight_layout()

    pdf_path = OUTPUT_DIR / f"{phase['filename']}.pdf"
    png_path = OUTPUT_DIR / f"{phase['filename']}.png"

    for output_path in (pdf_path, png_path):
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()
    return pdf_path, png_path


def run_phase(phase_name):
    phase = PHASE_CONFIGS[phase_name]
    results = {group: [] for group in GROUP_ORDER}
    schedules = {group: [] for group in GROUP_ORDER}
    runtime_rows = []

    print("=" * 80)
    print(f"Running phase: {phase['label']} ({phase_name})")
    print("=" * 80)

    for group in GROUP_ORDER:
        for D in D_LIST:
            schedule = make_schedule(phase_name, group, D)
            start_time = time.time()

            tail_mean_losses = run_optimizer(
                algo=group,
                batch_size=schedule["B"],
                num_steps=schedule["K"],
                eta=schedule["eta"],
                rho=schedule["rho"],
                n_runs=N_RUNS,
                seed=stable_seed(
                    "three_batch_phases_tail_1pct",
                    phase_name,
                    group,
                    D,
                ),
                tail_fraction=TAIL_FRACTION,
            )

            elapsed = time.time() - start_time
            mean_tail_loss = float(np.mean(tail_mean_losses))

            results[group].append(tail_mean_losses)
            schedules[group].append(schedule)
            runtime_rows.append({
                "phase": phase_name,
                "group": group,
                "D": int(D),
                "tail steps": schedule["tail_steps"],
                "elapsed seconds": elapsed,
            })

            print(
                f"{group.upper():3s}: "
                f"D={D:6d} | "
                f"B={schedule['B']:4d} | "
                f"K={schedule['K']:6d} | "
                f"tail={schedule['tail_steps']:4d} | "
                f"eta={schedule['eta']:.6f} | "
                f"rho={schedule['rho']:.6f} | "
                f"mean last-1%-steps loss={mean_tail_loss:.6e} | "
                f"{elapsed:.2f}s"
            )

    mean_curves = {}
    theory_reference_curves = {}
    summary_rows = []
    result_rows = []
    theory_rows = []
    D_array = np.asarray(D_LIST, dtype=np.float64)

    for group in GROUP_ORDER:
        mean_losses = np.asarray([
            float(np.mean(x))
            for x in results[group]
        ], dtype=np.float64)
        mean_curves[group] = mean_losses

        measured_exponent, intercept = fit_exponent(
            D_LIST,
            mean_losses,
        )
        theory_exponent = phase["algorithms"][group]["p"]

        anchor_index = THEORY_ANCHOR_INDEX
        theory_reference = (
            mean_losses[anchor_index]
            * (D_array / D_array[anchor_index]) ** (-theory_exponent)
        )
        theory_reference_curves[group] = theory_reference

        summary_rows.append({
            "phase": phase_name,
            "group": group,
            "label": GROUP_LABELS[group],
            "theory exponent": theory_exponent,
            "measured exponent": measured_exponent,
            "log-fit intercept": intercept,
            "metric": "mean loss over final 1% of steps",
        })

        for D, run_losses, schedule in zip(
            D_LIST,
            results[group],
            schedules[group],
        ):
            result_rows.append({
                **schedule,
                "label": GROUP_LABELS[group],
                "mean last-1%-steps loss": float(np.mean(run_losses)),
            })

        for D, predicted_loss in zip(D_LIST, theory_reference):
            theory_rows.append({
                "phase": phase_name,
                "group": group,
                "D": int(D),
                "theory reference loss": float(predicted_loss),
                "theory exponent": theory_exponent,
            })

    result_table = pd.DataFrame(result_rows)
    slope_summary = pd.DataFrame(summary_rows)
    theory_reference_table = pd.DataFrame(theory_rows)
    runtime_table = pd.DataFrame(runtime_rows)
    schedule_table = pd.concat(
        [pd.DataFrame(schedules[group]) for group in GROUP_ORDER],
        ignore_index=True,
    )

    print("\nMeasured and predicted exponents:")
    print(slope_summary)

    pdf_path, png_path = plot_phase(
        phase_name,
        mean_curves,
        {
            group: phase["algorithms"][group]["p"]
            for group in GROUP_ORDER
        },
    )

    # Only save the phase PDF and PNG; no CSV/NPZ files are written.

    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")

    return {
        "results": results,
        "schedules": schedules,
        "mean_curves": mean_curves,
        "theory_reference_curves": theory_reference_curves,
        "result_table": result_table,
        "slope_summary": slope_summary,
        "schedule_table": schedule_table,
        "theory_reference_table": theory_reference_table,
        "runtime_table": runtime_table,
        "pdf_path": pdf_path,
        "png_path": png_path,
    }


# =============================================================================
# 7. Execute all three phases
# =============================================================================

PHASES_TO_RUN = [
    "small_batch",
    "large_batch",
    "super_large_batch",
]

ALL_PHASE_RESULTS = {}

for phase_name in PHASES_TO_RUN:
    ALL_PHASE_RESULTS[phase_name] = run_phase(phase_name)

print("\nAll requested figures:")
for phase_name in PHASES_TO_RUN:
    print(
        f"  {phase_name}: "
        f"{ALL_PHASE_RESULTS[phase_name]['pdf_path']}"
    )


# =============================================================================
# 8. Combined slope summary
# =============================================================================

COMBINED_SLOPE_SUMMARY = pd.concat(
    [
        ALL_PHASE_RESULTS[phase_name]["slope_summary"]
        for phase_name in PHASES_TO_RUN
    ],
    ignore_index=True,
)

print(COMBINED_SLOPE_SUMMARY)

print(f"Saved only 3 PDF files and 3 PNG files in: {OUTPUT_DIR.resolve()}")
