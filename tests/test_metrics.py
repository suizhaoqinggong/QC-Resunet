import pytest
import torch


def test_metric_collection():
    """Test metric collection computes and resets."""
    from training_framework.metrics.accuracy import AccuracyMetric

    metric = AccuracyMetric(threshold=0.5)
    preds = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    metric.update(preds, targets)
    results = metric.compute()
    assert "accuracy" in results
    assert results["accuracy"] == 1.0

    metric.reset()
    results = metric.compute()
    assert results["accuracy"] == 0.0


def test_segmentation_overlap_metrics_do_not_sync_during_update(monkeypatch):
    """Dice/IoU update should avoid CPU copies and scalar syncs in the batch loop."""
    from training_framework.metrics.dice import DiceMetric
    from training_framework.metrics.iou import IoUMetric

    def fail_cpu(self):
        raise AssertionError("metric update should not force tensors to CPU")

    def fail_item(self):
        raise AssertionError("metric update should not call Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)
    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    preds = torch.randn(1, 4, 2, 2, 2)
    targets = torch.randint(0, 4, (1, 2, 2, 2))

    dice = DiceMetric(num_classes=4)
    iou = IoUMetric(num_classes=4)

    dice.update(preds, targets)
    iou.update(preds, targets)


@pytest.mark.parametrize(
    "scores,expected_roc,expected_pr", [([0.9, 0.1], 1.0, 1.0), ([0.1, 0.9], 0.0, 0.5), ([0.5, 0.5], 0.5, 0.5)]
)
def test_auc_endpoints_and_tied_scores(scores, expected_roc, expected_pr):
    from training_framework.metrics.auc import AUCMetric

    metric = AUCMetric(num_classes=1)
    metric.update(torch.tensor(scores).reshape(2, 1), torch.tensor([[1.0], [0.0]]))
    results = metric.compute()
    assert results["auroc"] == pytest.approx(expected_roc)
    assert results["auprc"] == pytest.approx(expected_pr)
