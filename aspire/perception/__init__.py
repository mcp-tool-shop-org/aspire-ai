"""
ASPIRE Perception Module - Enhancing agent awareness and cognitive empathy.

This module adds four key perception capabilities:

1. Theory of Mind (ToM) - Mental state tracking and perspective-taking
2. Controlled Chaos - Adversarial training for robustness
3. Character Persistence - Stable identity and value anchoring
4. Meta-Cognition - Uncertainty tracking and self-reflection

These work together to create agents with genuine perceptual depth,
not just pattern matching but actual understanding of context, users, and self.
"""

from aspire.perception.theory_of_mind import (
    MentalStateTracker,
    BeliefState,
    IntentState,
    EmotionalState,
    KnowledgeState,
    EmotionType,
    IntentCategory,
    EmotionalValence,
)
from aspire.perception.controlled_chaos import (
    ChaosGenerator,
    ChaosConfig,
    ChaosType,
    ChaosSeverity,
    ChaosInjection,
    NoiseInjector,
    AmbiguityGenerator,
    AdversarialScenarioGenerator,
    apply_chaos_to_batch,
)
from aspire.perception.character import (
    CharacterCore,
    ValueAnchor,
    BehavioralTrait,
    AutobiographicalMemory,
    MemoryEntry,
    ValueType,
    TraitDimension,
    create_socratic_character,
    create_compassionate_character,
    create_analytical_character,
)
from aspire.perception.metacognition import (
    UncertaintyEstimator,
    ConfidenceCalibrator,
    ReflectiveLoop,
    MetaCognitionModule,
    UncertaintyType,
    ConfidenceLevel,
    UncertaintyEstimate,
    ReflectiveInsight,
)
from aspire.perception.empathy_evaluation import (
    PerceptionEvaluator,
    PerceptionEvaluation,
    PerceptionDimension,
    PerceptionScore,
    PERCEPTION_DIMENSION_CRITERIA,
)
from aspire.perception.syntropy import (
    SyntropicEngine,
    SyntropicMeasurement,
    SyntropicDimension,
    SyntropicState,
    CoherenceField,
    SyntropicResonanceDetector,
    SyntropicIntegrator,
    SyntropicFlowTracker,
    SyntropicEmpathyEvaluator,
    compute_negentropy_approximation,
    compute_semantic_coherence,
    compute_empathic_syntropy,
)
from aspire.perception.integration import (
    PerceptionModule,
    PerceptionConfig,
    PerceptionState,
    PerceptionLoss,
    integrate_perception_with_trainer,
)

__all__ = [
    # Theory of Mind
    "MentalStateTracker",
    "BeliefState",
    "IntentState",
    "EmotionalState",
    "KnowledgeState",
    "EmotionType",
    "IntentCategory",
    "EmotionalValence",
    # Controlled Chaos
    "ChaosGenerator",
    "ChaosConfig",
    "ChaosType",
    "ChaosSeverity",
    "ChaosInjection",
    "NoiseInjector",
    "AmbiguityGenerator",
    "AdversarialScenarioGenerator",
    "apply_chaos_to_batch",
    # Character
    "CharacterCore",
    "ValueAnchor",
    "BehavioralTrait",
    "AutobiographicalMemory",
    "MemoryEntry",
    "ValueType",
    "TraitDimension",
    "create_socratic_character",
    "create_compassionate_character",
    "create_analytical_character",
    # Meta-Cognition
    "UncertaintyEstimator",
    "ConfidenceCalibrator",
    "ReflectiveLoop",
    "MetaCognitionModule",
    "UncertaintyType",
    "ConfidenceLevel",
    "UncertaintyEstimate",
    "ReflectiveInsight",
    # Evaluation
    "PerceptionEvaluator",
    "PerceptionEvaluation",
    "PerceptionDimension",
    "PerceptionScore",
    "PERCEPTION_DIMENSION_CRITERIA",
    # Syntropy
    "SyntropicEngine",
    "SyntropicMeasurement",
    "SyntropicDimension",
    "SyntropicState",
    "CoherenceField",
    "SyntropicResonanceDetector",
    "SyntropicIntegrator",
    "SyntropicFlowTracker",
    "SyntropicEmpathyEvaluator",
    "compute_negentropy_approximation",
    "compute_semantic_coherence",
    "compute_empathic_syntropy",
    # Integration
    "PerceptionModule",
    "PerceptionConfig",
    "PerceptionState",
    "PerceptionLoss",
    "integrate_perception_with_trainer",
]
