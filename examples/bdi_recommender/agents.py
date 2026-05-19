"""BDI Agent classes for the Recommender example.

This module defines:
- BDIAgent: Mixin providing BDI (Belief-Desire-Intention) capabilities with reactive state
- UserAgent: Bob - a mobile agent seeking health recommendations
- DoctorAgent: A fixed agent providing medical advice

"""

from collections import deque
from typing import TYPE_CHECKING, Any

from mesa.discrete_space import Cell, FixedAgent, Grid2DMovingAgent
from mesa.experimental.mesa_signals import HasEmitters, Observable, computed_property

if TYPE_CHECKING:
    try:
        from .model import BDIRecommenderModel
    except ImportError:
        from model import BDIRecommenderModel


class BDIAgent(HasEmitters):
    """Mixin providing BDI (Belief-Desire-Intention) capabilities.

    Beliefs and trust are Observable, enabling reactive updates when changed.
    Goals are computed automatically from beliefs and desires.

    Attributes:
        beliefs: Observable dict mapping belief names to strength values
        desires: Dict mapping desire names to priority values
        trust: Observable trust level (0.0 to 1.0)
        intentions: List of (action_name, args) tuples to execute
        incoming_queue: Queue of received messages
    """

    beliefs = Observable()
    trust = Observable()

    def __init__(self, *args, **kwargs):
        """Initialize BDI agent with empty mental state."""
        super().__init__(*args, **kwargs)
        self.beliefs: dict[str, float] = {}
        self.desires: dict[str, float] = {}
        self.trust: float = 0.5
        self.intentions: list[tuple[str, tuple]] = []
        self.incoming_queue: deque[dict[str, Any]] = deque()

    @computed_property
    def goals(self) -> dict[str, float]:
        """Elect goals from beliefs and desires using BDI logic.

        Goals are desires that are possible given current beliefs.
        A desire is blocked if there's a contradicting belief (Not-X blocks X).
        Returns the highest priority achievable desires.
        """

        current_desires = self.desires.copy()

        while current_desires:
            max_priority = max(current_desires.values())
            candidates = {k: v for k, v in current_desires.items() if v == max_priority}

            goals = {}
            for desire_name, priority in candidates.items():
                blocking_belief = f"Not-{desire_name}"
                if (
                    blocking_belief in self.beliefs
                    and self.beliefs[blocking_belief] > 0
                ):
                    continue
                goals[desire_name] = priority

            if goals:
                return goals

            for key in candidates:
                del current_desires[key]

        return {}

    def get_message(self) -> dict[str, Any] | None:
        return self.incoming_queue.popleft() if self.incoming_queue else None


class UserAgent(Grid2DMovingAgent, BDIAgent):
    """Bob - the user seeking health recommendations.

    A mobile BDI agent that:
    - Maintains beliefs about health conditions
    - Has desires for health outcomes
    - Computes goals from beliefs/desires
    - Receives recommendations from doctor
    - Moves to destinations based on recommendations
    """

    _DIRECTIONS = {
        (-1, 0): "n",
        (1, 0): "s",
        (0, 1): "e",
        (0, -1): "w",
        (-1, 1): "ne",
        (-1, -1): "nw",
        (1, 1): "se",
        (1, -1): "sw",
    }

    def __init__(
        self,
        model: BDIRecommenderModel,
        cell: Cell,
        initial_beliefs: dict[str, float] | None = None,
        initial_desires: dict[str, float] | None = None,
    ):
        """Initialize Bob with default beliefs and desires.

        Args:
            model: The BDIRecommenderModel instance
            cell: Starting cell position
            initial_beliefs: Optional dictionary of initial beliefs
            initial_desires: Optional dictionary of initial desires
        """
        super().__init__(model)
        self.cell = cell
        self.doctor: DoctorAgent | None = None

        # Initial beliefs (from NetLogo model or dynamic input)
        # Default: "Not-eh": Not experiencing health issues (0.4 confidence)
        # Default: "bs": Blood sugar concerns (0.9 confidence)
        self.beliefs = (
            initial_beliefs
            if initial_beliefs is not None
            else {"Not-eh": 0.4, "bs": 0.9}
        )

        # Initial desires (from NetLogo model or dynamic input)
        # Default: pa: physical activity, wr: weight reduction
        # Default: eh: exercise habit, w: wellness
        self.desires = (
            initial_desires
            if initial_desires is not None
            else {"pa": 0.8, "wr": 0.8, "eh": 0.9, "w": 0.75}
        )

        self.trust = 0.5
        self.destinations: list[tuple[int, int]] = []
        self.current_destination_index = 0

    def step(self):
        """Execute one step of Bob's behavior."""
        self.listen_to_messages()

        # If we have a destination to reach, move toward it
        if self.destinations and self.current_destination_index < len(
            self.destinations
        ):
            self.move_toward_destination()
        elif not self.intentions:
            # Compute goals and get recommendation if no current plan
            goals = self.goals
            if goals:
                self.get_recommendation(goals)

        self.execute_intentions()

    def listen_to_messages(self):
        """Process all incoming messages."""
        while msg := self.get_message():
            performative = msg.get("performative")
            if performative == "proposal":
                self.evaluate_proposal(msg)

    def evaluate_proposal(self, msg: dict[str, Any]):
        """Evaluate a proposal from another agent based on trust.

        Args:
            msg: Message dict with 'sender' and 'content' keys
        """
        sender = msg.get("sender")
        if sender and sender.trust > self.trust:
            # Accept proposal from trusted sender
            belief_update = msg.get("content", {})
            self.update_beliefs(belief_update, sender_trust=sender.trust)

            # Send acceptance
            response = {
                "performative": "accept",
                "sender": self,
                "receiver": sender,
                "content": belief_update,
            }
            sender.incoming_queue.append(response)
        # If not trusted, silently reject (no response)

    def update_beliefs(
        self, belief_update: dict[str, float], sender_trust: float = 0.5
    ):
        """Update beliefs with new information.

        Args:
            belief_update: Dict of belief name -> new value
            sender_trust: Trust level of the agent who sent the update
        """
        new_beliefs = self.beliefs.copy()

        for belief, value in belief_update.items():
            # Weight by sender's trust
            weighted_value = min(value, sender_trust)
            new_beliefs[belief] = weighted_value

        self.beliefs = new_beliefs

        # Recalculate goals and intentions
        self.intentions.clear()
        self.destinations.clear()
        self.current_destination_index = 0

        if self.goals:
            self.get_recommendation(self.goals)

    def get_recommendation(self, goals: dict[str, float]):
        """Get health recommendation based on current goals.

        Based on NetLogo model's get-recommendation procedure.

        Args:
            goals: Current computed goals
        """
        # Check if physical activity (pa) and weight reduction (wr) are goals
        if "pa" in goals and "wr" in goals:
            # Recommend visiting multiple health locations
            self.destinations = [
                self.model.gym.cell.coordinate,
                self.model.nutrition_centre.cell.coordinate,
                self.model.clinic.cell.coordinate,
            ]
            self.intentions.append(("visit_locations", ()))
        else:
            # Special diet recommendation - visit doctor
            if self.doctor and self.doctor.cell:
                coord = self.doctor.cell.coordinate
                self.destinations = [(coord[0], coord[1])]
                self.intentions.append(("visit_doctor", ()))

        self.current_destination_index = 0

    def move_toward_destination(self):
        """Move one step toward the current destination."""
        if self.current_destination_index >= len(self.destinations):
            return

        dest = self.destinations[self.current_destination_index]
        current = self.cell.coordinate

        # move in direction of destination
        dx = 0 if dest[0] == current[0] else (1 if dest[0] > current[0] else -1)
        dy = 0 if dest[1] == current[1] else (1 if dest[1] > current[1] else -1)

        if dx != 0 or dy != 0:
            if direction := self._DIRECTIONS.get((dx, dy)):
                self.move(direction)

        # Check if arrived at destination
        if self.cell.coordinate == dest:
            self.current_destination_index += 1

    def execute_intentions(self):
        """Execute pending intentions."""

        if self.current_destination_index >= len(self.destinations) and self.intentions:
            # All destinations visited, clear intentions
            self.intentions.clear()


class DoctorAgent(FixedAgent, BDIAgent):
    """The doctor who provides health recommendations.

    A fixed BDI agent that:
    - Has high trust level
    - Sends proposals to connected users
    - Provides belief updates about health conditions
    """

    def __init__(
        self,
        model: BDIRecommenderModel,
        cell: Cell,
        initial_beliefs: dict[str, float] | None = None,
    ):
        """Initialize the doctor at a fixed position.

        Args:
            model: The BDIRecommenderModel instance
            cell: Fixed cell position
            initial_beliefs: Optional dictionary of initial beliefs
        """
        super().__init__(model)
        self.cell = cell

        # Doctor's beliefs (from NetLogo model or dynamic input)
        # Default: "Not-pa": 1.0 (Knows patient lacks physical activity)
        self.beliefs = (
            initial_beliefs if initial_beliefs is not None else {"Not-pa": 1.0}
        )
        self.desires = {}
        self.trust = 0.9  # High trust level
        self.proposal_sent = False

    location_name = "Doctor"

    def step(self):
        """Execute one step of doctor's behavior."""
        self.listen_to_messages()

    def listen_to_messages(self):
        """Process incoming messages."""
        while self.get_message():
            # Process accept messages (currently just consume them)
            pass

    def send_proposal(self):
        """Send a health proposal to the connected user."""
        # Find connected user (Bob)
        user = None
        for agent in self.model.agents:
            if isinstance(agent, UserAgent) and agent.doctor is self:
                user = agent
                break

        if user:
            # Send proposal with belief about physical activity
            belief_to_share = next(iter(self.beliefs.items()))
            proposal = {
                "performative": "proposal",
                "sender": self,
                "receiver": user,
                "content": {belief_to_share[0]: belief_to_share[1]},
            }
            user.incoming_queue.append(proposal)
            self.proposal_sent = True


class LocationAgent(FixedAgent):
    """A fixed location that Bob can visit (Gym, Clinic, etc.)."""

    def __init__(
        self,
        model: BDIRecommenderModel,
        cell: Cell,
        name: str = "Location",
    ):
        super().__init__(model)
        self.cell = cell
        self.location_name = name
