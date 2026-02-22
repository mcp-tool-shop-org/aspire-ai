"""
ASPIRE for Isaac Gym/Lab - Teaching robots to develop physical intuition.

This integration brings ASPIRE's adversarial learning to robotics:
- Motion teachers critique trajectories for safety, efficiency, and grace
- Trajectory critics internalize physical intuition
- Robots self-refine actions before execution

Example:
    from aspire.integrations.isaac import (
        MotionTeacher,
        TrajectoryCritic,
        AspireIsaacTrainer,
    )

    # Create teacher that evaluates motion quality
    teacher = MotionTeacher(
        personas=["safety_inspector", "efficiency_expert", "grace_coach"],
        strategy="vote",
    )

    # Train robot to internalize good motion judgment
    trainer = AspireIsaacTrainer(
        env="FrankaCubeStack-v0",
        teacher=teacher,
        critic=TrajectoryCritic(),
    )
    trainer.train(epochs=100)
"""

from .config import IsaacAspireConfig
from .isaac_wrapper import (
    AspireIsaacEnv,
    StateActionPair,
    TrajectoryBuffer,
)
from .motion_teacher import (
    EfficiencyExpert,
    GraceCoach,
    MotionTeacher,
    PhysicsOracle,
    SafetyInspector,
)
from .trainer import AspireIsaacTrainer
from .trajectory_critic import (
    MotionCriticHead,
    TrajectoryCritic,
    TrajectoryEncoder,
)

__all__ = [
    # Teachers
    "MotionTeacher",
    "SafetyInspector",
    "EfficiencyExpert",
    "GraceCoach",
    "PhysicsOracle",
    # Critics
    "TrajectoryCritic",
    "TrajectoryEncoder",
    "MotionCriticHead",
    # Environment
    "AspireIsaacEnv",
    "TrajectoryBuffer",
    "StateActionPair",
    # Training
    "AspireIsaacTrainer",
    "IsaacAspireConfig",
]
