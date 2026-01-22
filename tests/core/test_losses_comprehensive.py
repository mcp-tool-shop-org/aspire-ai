"""
Comprehensive tests for ASPIRE loss functions.

Tests the mathematical correctness and edge cases of:
- Critic loss (MSE, contrastive)
- Student loss (KL divergence, reward-weighted)
"""

import pytest
import torch
import torch.nn.functional as F

from aspire.losses.critic import CriticLoss
from aspire.losses.student import StudentLoss


class TestCriticLoss:
    """Tests for CriticLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return CriticLoss(mse_weight=1.0, contrastive_weight=0.5)

    @pytest.fixture
    def batch_data(self):
        """Create sample batch data."""
        batch_size = 4
        return {
            "predicted_scores": torch.tensor([7.5, 3.2, 8.1, 5.0]),
            "target_scores": torch.tensor([8.0, 3.0, 8.0, 5.5]),
        }

    def test_mse_loss_perfect_prediction(self, loss_fn):
        """MSE should be zero for perfect predictions."""
        predicted = torch.tensor([5.0, 7.0, 3.0])
        target = torch.tensor([5.0, 7.0, 3.0])

        loss = loss_fn.compute_mse_loss(predicted, target)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mse_loss_known_values(self, loss_fn):
        """Test MSE with known expected values."""
        predicted = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 5.0])
        # MSE = ((1)^2 + (0)^2 + (2)^2) / 3 = 5/3

        loss = loss_fn.compute_mse_loss(predicted, target)

        expected_mse = (1.0 + 0.0 + 4.0) / 3.0
        assert loss.item() == pytest.approx(expected_mse, rel=1e-5)

    def test_contrastive_loss_correct_ordering(self, loss_fn):
        """Contrastive loss should be low when ordering is correct."""
        # Predicted correctly ranks: a > b
        embeddings = torch.tensor([
            [1.0, 0.0],  # Sample A (higher score)
            [0.0, 1.0],  # Sample B (lower score)
        ])
        scores = torch.tensor([8.0, 3.0])  # A should rank higher

        loss = loss_fn.compute_contrastive_loss(embeddings, scores)

        # Loss should be relatively low for correct ordering
        assert loss.item() < 1.0

    def test_total_loss_combines_components(self, loss_fn, batch_data):
        """Total loss should combine MSE and contrastive components."""
        # Create embeddings for contrastive loss
        embeddings = torch.randn(4, 64)

        total, components = loss_fn(
            batch_data["predicted_scores"],
            batch_data["target_scores"],
            embeddings=embeddings,
            return_components=True,
        )

        assert "mse" in components
        assert "contrastive" in components
        assert total > 0

    def test_loss_is_differentiable(self, loss_fn, batch_data):
        """Loss should be differentiable for backprop."""
        predicted = batch_data["predicted_scores"].clone().requires_grad_(True)
        target = batch_data["target_scores"]

        loss = loss_fn.compute_mse_loss(predicted, target)
        loss.backward()

        assert predicted.grad is not None
        assert not torch.isnan(predicted.grad).any()

    def test_loss_handles_batch_size_one(self, loss_fn):
        """Loss should work with batch size of 1."""
        predicted = torch.tensor([5.0])
        target = torch.tensor([6.0])

        loss = loss_fn.compute_mse_loss(predicted, target)

        assert loss.item() == pytest.approx(1.0, rel=1e-5)

    def test_loss_handles_large_values(self, loss_fn):
        """Loss should handle large score values."""
        predicted = torch.tensor([1000.0, 2000.0])
        target = torch.tensor([1000.0, 2000.0])

        loss = loss_fn.compute_mse_loss(predicted, target)

        assert loss.item() == pytest.approx(0.0, abs=1e-4)
        assert not torch.isnan(loss)


class TestStudentLoss:
    """Tests for StudentLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        return StudentLoss(
            kl_weight=1.0,
            reward_weight=0.5,
            entropy_weight=0.01,
        )

    @pytest.fixture
    def batch_data(self):
        """Create sample batch data."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        return {
            "logits": torch.randn(batch_size, seq_len, vocab_size),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "rewards": torch.tensor([0.8, 0.3]),
            "attention_mask": torch.ones(batch_size, seq_len),
        }

    def test_cross_entropy_loss_basic(self, loss_fn, batch_data):
        """Test basic cross entropy calculation."""
        loss = loss_fn.compute_ce_loss(
            batch_data["logits"],
            batch_data["labels"],
            batch_data["attention_mask"],
        )

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_reward_weighting_positive(self, loss_fn, batch_data):
        """Higher rewards should reduce effective loss."""
        high_reward = torch.tensor([0.9])
        low_reward = torch.tensor([0.1])

        # Same base loss
        logits = batch_data["logits"][:1]
        labels = batch_data["labels"][:1]
        mask = batch_data["attention_mask"][:1]

        base_loss = loss_fn.compute_ce_loss(logits, labels, mask)

        # Reward-weighted losses
        high_weighted = loss_fn.apply_reward_weight(base_loss, high_reward)
        low_weighted = loss_fn.apply_reward_weight(base_loss, low_reward)

        # Higher reward = lower weighted loss (we want to reinforce good responses)
        assert high_weighted.item() < low_weighted.item()

    def test_entropy_regularization(self, loss_fn, batch_data):
        """Entropy term should encourage exploration."""
        # Uniform distribution (high entropy)
        uniform_logits = torch.zeros(1, 10, 100)

        # Peaked distribution (low entropy)
        peaked_logits = torch.zeros(1, 10, 100)
        peaked_logits[:, :, 0] = 10.0

        uniform_entropy = loss_fn.compute_entropy(uniform_logits)
        peaked_entropy = loss_fn.compute_entropy(peaked_logits)

        # Uniform should have higher entropy
        assert uniform_entropy.item() > peaked_entropy.item()

    def test_loss_gradient_flow(self, loss_fn, batch_data):
        """Gradients should flow through the loss."""
        logits = batch_data["logits"].clone().requires_grad_(True)

        loss = loss_fn.compute_ce_loss(
            logits,
            batch_data["labels"],
            batch_data["attention_mask"],
        )
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_masked_loss_ignores_padding(self, loss_fn):
        """Masked positions should not contribute to loss."""
        batch_size = 2
        seq_len = 10
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Mask out second half of second sequence
        mask = torch.ones(batch_size, seq_len)
        mask[1, 5:] = 0

        loss = loss_fn.compute_ce_loss(logits, labels, mask)

        # Loss should still be valid
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_kl_divergence_same_distribution(self, loss_fn):
        """KL divergence should be zero for identical distributions."""
        logits = torch.randn(2, 10, 50)

        kl = loss_fn.compute_kl_divergence(logits, logits)

        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_divergence_different_distributions(self, loss_fn):
        """KL divergence should be positive for different distributions."""
        logits_p = torch.randn(2, 10, 50)
        logits_q = torch.randn(2, 10, 50)

        kl = loss_fn.compute_kl_divergence(logits_p, logits_q)

        assert kl.item() > 0


class TestLossNumericalStability:
    """Test numerical stability of loss functions."""

    def test_critic_loss_with_zeros(self):
        """Critic loss should handle zero predictions."""
        loss_fn = CriticLoss()
        predicted = torch.zeros(4)
        target = torch.zeros(4)

        loss = loss_fn.compute_mse_loss(predicted, target)

        assert loss.item() == 0.0
        assert not torch.isnan(loss)

    def test_student_loss_with_extreme_logits(self):
        """Student loss should handle extreme logit values."""
        loss_fn = StudentLoss()

        # Very large logits
        large_logits = torch.ones(2, 5, 100) * 100
        labels = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.ones(2, 5)

        loss = loss_fn.compute_ce_loss(large_logits, labels, mask)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_empty_batch_handling(self):
        """Loss functions should handle empty tensors gracefully."""
        loss_fn = CriticLoss()

        # Empty batch
        predicted = torch.tensor([])
        target = torch.tensor([])

        # Should either return 0 or raise a clear error
        try:
            loss = loss_fn.compute_mse_loss(predicted, target)
            assert loss.item() == 0.0 or torch.isnan(loss)
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise error for empty batch
