"""

Displays:
- Grid with Bob (red) and Doctor (cyan)
- Current beliefs, desires, goals panel
- lime green for gym
- orange for nutrition center
- blue for clinic
"""

import solara
from matplotlib.lines import Line2D

try:
    from .agents import DoctorAgent, LocationAgent, UserAgent
    from .model import BDIRecommenderModel
except ImportError:
    from agents import DoctorAgent, LocationAgent, UserAgent
    from model import BDIRecommenderModel
from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle


def post_process(ax):
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=10, label="Bob (User)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="cyan",
               markersize=10, label="Doctor"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="limegreen",
               markersize=10, label="Gym"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="orange",
               markersize=10, label="Nutrition Centre"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="dodgerblue",
               markersize=10, label="Health Clinic"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize="small")


def agent_portrayal(agent):
    """Define how agents are displayed on the grid."""
    if isinstance(agent, UserAgent):
        return AgentPortrayalStyle(color="red", size=100, marker="o", zorder=3)
    elif isinstance(agent, DoctorAgent):
        return AgentPortrayalStyle(color="cyan", size=120, marker="s", zorder=2)
    elif isinstance(agent, LocationAgent):
        colors = {
            "Gym": "limegreen",
            "Nutrition Centre": "orange",
            "Health Clinic": "dodgerblue",
        }
        return AgentPortrayalStyle(
            color=colors.get(agent.location_name, "gray"),
            size=150,
            marker="s",
            zorder=1,
        )
    return AgentPortrayalStyle()


def get_bdi_summary(model):
    """Display Bob's BDI state summary."""
    user = model.user
    goals = ", ".join(user.goals.keys()) if user.goals else "None"
    beliefs = ", ".join(f"{k}={v:.1f}" for k, v in user.beliefs.items())
    desires = ", ".join(f"{k}={v:.1f}" for k, v in user.desires.items())
    intentions = ", ".join(i[0] for i in user.intentions) if user.intentions else "None"
    dest_idx = user.current_destination_index
    total_dest = len(user.destinations)

    return solara.Markdown(f"""
**Time:** {model.time}<br>
**Beliefs:** {beliefs}<br>
**Desires:** {desires}<br>
**Goals:** {goals}<br>
**Intentions:** {intentions}<br>
**Progress:** {dest_idx}/{total_dest} destinations
""")


model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "user_desire_pa": {
        "type": "SliderFloat",
        "value": 0.8,
        "label": "User Desire: PA (Physical Activity)",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
    },
    "user_desire_wr": {
        "type": "SliderFloat",
        "value": 0.8,
        "label": "User Desire: WR (Weight Reduction)",
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
    },
    "doctor_proposal_tick": {
        "type": "SliderInt",
        "value": 3,
        "label": "Doctor Proposal Tick",
        "min": 1,
        "max": 40,
        "step": 1,
    },
}

model = BDIRecommenderModel(rng=42)

renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(agent_portrayal)
renderer.post_process = post_process
renderer.draw_agents()

page = SolaraViz(
    model,
    renderer,
    components=[get_bdi_summary],
    model_params=model_params,
    name="BDI Recommender Agent",
)
page  # noqa
