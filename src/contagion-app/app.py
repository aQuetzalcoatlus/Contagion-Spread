from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# Core simulation code (adapted from your notebook)
# =========================================================


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.5
    state: str = "healthy"  # "healthy", "infected", "immunized"
    steps_infected: int = 0
    infected_this_step: bool = (
        False  # has this particle already infected someone this step?
    )


def initialize_particles(
    num_particles: int,
    immunized_percentage: float,
    box_size: float,
    rng: np.random.Generator,
) -> List[Particle]:
    """Create a list of Particle objects with random positions and velocities."""
    particles: List[Particle] = []
    immunized_count = int(num_particles * immunized_percentage / 100)

    for i in range(num_particles):
        x, y = rng.uniform(0, box_size, size=2)
        vx, vy = rng.uniform(-0.5, 0.5, size=2)
        state = "immunized" if i < immunized_count else "healthy"
        particles.append(Particle(x=x, y=y, vx=vx, vy=vy, state=state))

    # Infect one random particle at t=0
    patient_zero_idx = rng.integers(0, num_particles)
    particles[patient_zero_idx].state = "infected"
    return particles


def elastic_collision(p1: Particle, p2: Particle) -> None:
    """2D elastic collision for equal-mass particles."""
    r1 = np.array([p1.x, p1.y])
    r2 = np.array([p2.x, p2.y])
    v1 = np.array([p1.vx, p1.vy])
    v2 = np.array([p2.vx, p2.vy])

    d_vec = r1 - r2
    d_sq = np.dot(d_vec, d_vec)
    if d_sq == 0:
        return  # avoid division by zero for identical positions

    u1 = v1 - np.dot(v1 - v2, d_vec) / d_sq * d_vec
    u2 = v2 - np.dot(v2 - v1, -d_vec) / d_sq * (-d_vec)

    p1.vx, p1.vy = u1
    p2.vx, p2.vy = u2


def update_positions(
    particles: List[Particle], box_size: float, timestep: float = 1.0
) -> None:
    """Move particles and reflect off box boundaries."""
    for p in particles:
        p.x += p.vx * timestep
        p.y += p.vy * timestep

        # Reflective boundaries in x
        if p.x < 0 or p.x > box_size:
            p.vx *= -1
            p.x = np.clip(p.x, 0, box_size)

        # Reflective boundaries in y
        if p.y < 0 or p.y > box_size:
            p.vy *= -1
            p.y = np.clip(p.y, 0, box_size)


def check_and_handle_collisions(
    particles: List[Particle],
    breakthrough: float,
    rng: np.random.Generator,
) -> None:
    """Handle elastic collisions and contagion spread."""
    num_particles = len(particles)

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            p1, p2 = particles[i], particles[j]

            dx, dy = p1.x - p2.x, p1.y - p2.y
            distance = np.hypot(dx, dy)
            overlap = p1.radius + p2.radius - distance

            if overlap > 0:
                # Separate overlapping particles
                if distance == 0:
                    # Random tiny nudge to avoid division by zero
                    direction = rng.normal(size=2)
                    direction /= np.linalg.norm(direction)
                else:
                    direction = np.array([dx, dy]) / distance

                separation = overlap / 2
                p1.x += direction[0] * separation
                p1.y += direction[1] * separation
                p2.x -= direction[0] * separation
                p2.y -= direction[1] * separation

                # Elastic collision
                elastic_collision(p1, p2)

                # Contagion logic (each infected can infect at most one person per step)
                def try_infect(source: Particle, target: Particle) -> None:
                    if source.state != "infected" or source.infected_this_step:
                        return
                    if target.state == "healthy":
                        target.state = "infected"
                        source.infected_this_step = True
                    elif target.state == "immunized":
                        if rng.uniform() <= breakthrough:
                            target.state = "infected"
                            source.infected_this_step = True

                try_infect(p1, p2)
                try_infect(p2, p1)


def update_states(particles: List[Particle], infection_duration: int) -> None:
    """Update infection timers and convert recovered to immunized."""
    for p in particles:
        if p.state == "infected":
            p.steps_infected += 1
            if p.steps_infected >= infection_duration:
                p.state = "immunized"
                p.steps_infected = 0


def run_simulation(
    num_particles: int,
    immunized_percentage: float,
    box_size: float,
    num_steps: int,
    breakthrough: float,
    infection_duration: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one simulation and return (trajectory, state) arrays.

    trajectory: shape (num_steps, num_particles, 2)
    state_data: shape (num_steps, num_particles) with string states
    """
    rng = np.random.default_rng(seed)
    particles = initialize_particles(num_particles, immunized_percentage, box_size, rng)

    trajectory = np.zeros((num_steps, num_particles, 2), dtype=float)
    state_data = np.zeros((num_steps, num_particles), dtype=object)

    for step in range(num_steps):
        # reset infection flags at the start of the step
        for p in particles:
            p.infected_this_step = False

        update_positions(particles, box_size)
        check_and_handle_collisions(particles, breakthrough, rng)
        update_states(particles, infection_duration)

        for i, p in enumerate(particles):
            trajectory[step, i, :] = (p.x, p.y)
            state_data[step, i] = p.state

    return trajectory, state_data


def run_multiple_immunization_sims(
    immunization_rates: List[int],
    num_particles: int,
    box_size: float,
    num_steps: int,
    breakthrough: float,
    infection_duration: int,
    seed: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Run one simulation per immunization rate and collect results."""
    results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, rate in enumerate(immunization_rates):
        # Different seed per scenario for variety but repeatable pattern
        traj, states = run_simulation(
            num_particles=num_particles,
            immunized_percentage=rate,
            box_size=box_size,
            num_steps=num_steps,
            breakthrough=breakthrough,
            infection_duration=infection_duration,
            seed=seed + idx,
        )
        results[rate] = (traj, states)
    return results


# =========================================================
# Streamlit UI
# =========================================================


st.set_page_config(
    page_title="Contagion Agent Simulation",
    layout="wide",
)

st.image(
    "./resources/Contagion_repo_image_README.png",
    # caption="Schwarzwald Forest Change Explorer",
    width=200,
)
st.title("Agent-based Simulation of Contagion Spread")
st.caption(
    "Particles move in a box, collide elastically, and may infect each other on contact. "
    "Some particles start immunized; infected particles eventually become immunized."
)


with st.expander("What is this app showing?"):
    st.markdown(
        """
In this simulation, we model how an infectious disease spreads through a population, where individuals are represented by moving particles, called "agents".

We start by implementing particles to act as individuals who can be healthy, infected, or immunized against an infection.

These particles are then set in motion within a box with no Periodic Boundary Conditions (PBC), observing how they collide and potentially spread the infection over time.

Lastly, we analyze the outcomes to see how different levels of immunization in the population affect the spread of the disease.

Each dot is an individual:

- **Green** → healthy
- **Red** → currently infected
- **Blue** → immunized (either from the start or after recovery)

The key knobs you can change:

- Initial **immunized percentage** of the population  
- **Breakthrough probability**: chance that immunized agents still get infected  
- **Infection duration**: how many steps an agent stays infected before becoming immunized  
- Number of particles, box size, and number of simulation steps
"""
    )

# ----------------------------- Sidebar controls ---------------------------------

st.sidebar.header("Simulation parameters")

num_particles = st.sidebar.slider("Number of particles", 20, 150, 50, step=5)
box_size = st.sidebar.slider("Box size", 20.0, 80.0, 50.0, step=5.0)
num_steps = st.sidebar.slider("Number of time steps", 200, 5000, 2000, step=200)
infection_duration = st.sidebar.slider(
    "Infection duration (steps until immunized)", 50, 1000, 500, step=50
)

immunized_percentage_single = st.sidebar.slider(
    "Initial immunized population (%) – single run", 0, 95, 20, step=5
)

breakthrough = st.sidebar.slider(
    "Breakthrough infection probability", 0.0, 1.0, 0.2, step=0.05
)

seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=10_000, value=42, step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Tip: use fewer steps or particles if the app feels slow. "
    "You can increase them again once you’ve understood the dynamics."
)

# ----------------------------- Tabs: single vs multi ----------------------------

tab_single, tab_multi = st.tabs(["Single simulation", "Compare immunization rates"])

# ===== Single simulation tab =====

with tab_single:
    st.subheader("Single run")

    run_button = st.button("Run / Rerun simulation", type="primary")

    # Current parameter set for this tab
    params = {
        "num_particles": num_particles,
        "immunized_percentage_single": immunized_percentage_single,
        "box_size": box_size,
        "num_steps": num_steps,
        "breakthrough": breakthrough,
        "infection_duration": infection_duration,
        "seed": seed,
    }

    if "single_params" not in st.session_state:
        st.session_state.single_params = None

    need_new_run = (
        run_button
        or "single_result" not in st.session_state
        or st.session_state.single_params != params
    )

    if need_new_run:
        traj, states = run_simulation(
            num_particles=num_particles,
            immunized_percentage=immunized_percentage_single,
            box_size=box_size,
            num_steps=num_steps,
            breakthrough=breakthrough,
            infection_duration=infection_duration,
            seed=seed,
        )
        st.session_state.single_result = (traj, states)
        st.session_state.single_params = params

    traj, states = st.session_state.single_result
    timesteps = np.arange(num_steps)

    # --- Metrics ---
    infected_counts = np.sum(states == "infected", axis=1)
    total_ever_infected = np.sum(np.any(states == "infected", axis=0))

    zero_indices = np.where(infected_counts == 0)[0]
    time_to_eradication = zero_indices[0] if len(zero_indices) > 0 else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Peak infected at once", int(infected_counts.max()))
    col2.metric("Agents ever infected", int(total_ever_infected))
    if time_to_eradication is not None:
        col3.metric("Time to eradication (steps)", int(time_to_eradication))
    else:
        col3.metric("Time to eradication (steps)", "Not reached")

    st.markdown("### Infection curve over time")

    # build a proper DataFrame for Altair
    df_curve_data = pd.DataFrame(
        {
            "step": timesteps,
            "infected": infected_counts,
        }
    )

    df_curve = (
        alt.Chart(df_curve_data)
        .mark_line()
        .encode(
            x=alt.X("step:Q", title="Time step"),
            y=alt.Y("infected:Q", title="Number of infected agents"),
            tooltip=["step:Q", "infected:Q"],
        )
        .properties(height=300)
    )

    st.altair_chart(df_curve, use_container_width=True)

    st.markdown("### Snapshot of particle positions")

    step_to_view = st.slider(
        "Choose a time step to view", 0, num_steps - 1, 0, step=max(num_steps // 50, 1)
    )

    x = traj[step_to_view, :, 0]
    y = traj[step_to_view, :, 1]
    colors = np.array(
        [
            "green" if s == "healthy" else "red" if s == "infected" else "blue"
            for s in states[step_to_view]
        ]
    )

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title(f"Agent positions at step {step_to_view}")
    ax.grid(True)

    # Custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Healthy",
            markerfacecolor="green",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Infected",
            markerfacecolor="red",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Immunized",
            markerfacecolor="blue",
            markersize=8,
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    st.pyplot(fig)

# ===== Multiple immunization rates tab =====

with tab_multi:
    st.subheader("Compare different initial immunization percentages")

    default_rates = [20, 50, 80, 90]
    chosen_rates = st.multiselect(
        "Initial immunized percentages to simulate",
        options=list(range(0, 101, 10)),
        default=default_rates,
        help="Each selected percentage runs a full simulation with the same other parameters.",
    )
    chosen_rates = sorted(set(int(r) for r in chosen_rates))
    if not chosen_rates:
        st.info("Select at least one immunization rate to run the comparison.")
    else:
        run_multi_button = st.button("Run comparison", key="run_multi")

        if run_multi_button or "multi_result" not in st.session_state:
            results = run_multiple_immunization_sims(
                immunization_rates=chosen_rates,
                num_particles=num_particles,
                box_size=box_size,
                num_steps=num_steps,
                breakthrough=breakthrough,
                infection_duration=infection_duration,
                seed=seed + 10_000,
            )
            st.session_state.multi_result = results

        results = st.session_state.multi_result

        # Build Altair chart for infection curves
        records = []
        for rate, (_traj, state_arr) in results.items():
            infected_counts = np.sum(state_arr == "infected", axis=1)
            for step, infected in enumerate(infected_counts):
                records.append(
                    {
                        "step": step,
                        "infected": int(infected),
                        "immunized_rate": f"{rate}%",
                    }
                )

        df_all = alt.Data(values=records)

        chart_multi = (
            alt.Chart(df_all)
            .mark_line()
            .encode(
                x=alt.X("step:Q", title="Time step"),
                y=alt.Y("infected:Q", title="Number of infected agents"),
                color=alt.Color(
                    "immunized_rate:N", title="Initial immunized population"
                ),
                tooltip=["step:Q", "infected:Q", "immunized_rate:N"],
            )
            .properties(height=350)
        )

        st.altair_chart(chart_multi, use_container_width=True)

        st.caption(
            "Higher initial immunization usually lowers and delays the infection peak. "
            "When breakthrough probability is small enough, contagion may die out completely."
        )

# ----------------------------- Reflection questions ------------------------------

st.markdown("---")
st.subheader("Questions to check your understanding")

st.markdown(
    """
1. **Vary the initial immunized percentage while keeping the breakthrough probability fixed.**  
   How does increasing the immunized percentage change:
   - the *peak* of the infection curve, and  
   - whether the disease becomes endemic or dies out?

2. **Now fix the immunized percentage and change the breakthrough probability.**  
   For which values does the infection quickly die out, and for which values does it persist
   throughout the simulation? How does this relate to the idea of *vaccine efficacy*?

3. **Experiment with the infection duration (steps until an infected agent becomes immunized).**  
   How does making infections last longer or shorter affect the number of infected agents over time,
   even if immunization percentage and breakthrough probability stay the same?
"""
)

st.info(
    "Try to answer these in your own words after playing with the sliders and comparing scenarios in both tabs. "
    "The goal is to build intuition for how immunization and vaccine efficacy interact in an agent-based model."
)
