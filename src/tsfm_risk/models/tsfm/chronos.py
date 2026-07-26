"""Chronos / Chronos-Bolt wrapper (Amazon).

Chronos treats a series as a language: values are mean-scaled by the context,
quantized into a fixed vocabulary, and modelled by a T5 encoder-decoder.
Chronos-Bolt replaces autoregressive decoding with direct multi-step patch
forecasting and a quantile head, which is what makes it fast enough to run on
CPU over ~100k origins.

Two consequences matter for this study and are stated in the paper:

1. **Instance normalisation is a free volatility estimate.** The context is
   divided by its mean absolute value before entering the model, so a
   RiskMetrics-like scale is handed to the model before the first attention
   layer. Any apparent skill at "volatility forecasting" must be judged against
   an EWMA baseline, not against a constant.
2. **The native quantile grid stops at 0.1/0.9.** VaR at 1-5% is therefore not
   directly readable, which is why the vol-path is the primary mapping
   (``docs/amendments.md``, amendment 1). Requested levels outside the native
   grid are still returned — the model interpolates internally — but they are
   flagged via :attr:`native_levels` so the direct path can refuse them.

Input convention: we feed **log-returns**, not prices. The study models the
return distribution, and a price-level context would make the model spend its
capacity on the (near-unpredictable) drift.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from tsfm_risk.models.tsfm.base import DEFAULT_LEVELS, TSFMForecaster

# checkpoint release dates: conservative upper bounds on pretraining data
_CUTOFFS = {
    "amazon/chronos-bolt-tiny": dt.date(2024, 11, 27),
    "amazon/chronos-bolt-mini": dt.date(2024, 11, 27),
    "amazon/chronos-bolt-small": dt.date(2024, 11, 27),
    "amazon/chronos-bolt-base": dt.date(2024, 11, 27),
}

_BOLT_NATIVE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ChronosForecaster(TSFMForecaster):
    """Zero-shot quantile forecaster backed by a Chronos checkpoint."""

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-base",
        device: str = "cpu",
        max_context: int = 2048,
        batch_size: int = 64,
    ):
        self.model_id = model_id
        self.device = device
        self.max_context = max_context
        self.batch_size = batch_size
        self.name = model_id.split("/")[-1]
        self.data_cutoff = _CUTOFFS.get(model_id)
        self.native_levels = _BOLT_NATIVE_LEVELS if "bolt" in model_id else None
        self._pipeline = None

    # ------------------------------------------------------------ lifecycle

    @property
    def pipeline(self):
        """Lazily loaded so importing the module never downloads weights."""
        if self._pipeline is None:
            import torch
            from chronos import BaseChronosPipeline

            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
            )
        return self._pipeline

    # ------------------------------------------------------------- forecast

    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        levels: tuple[float, ...] = DEFAULT_LEVELS,
    ) -> np.ndarray:
        return self.predict_batch([context], horizon, levels)[0]

    def predict_batch(
        self,
        contexts: list[np.ndarray],
        horizon: int,
        levels: tuple[float, ...] = DEFAULT_LEVELS,
    ) -> np.ndarray:
        """Quantiles for many origins at once: shape (n, horizon, n_levels).

        Batching is what makes the full grid feasible on CPU — the per-call
        overhead dominates single-origin inference.
        """
        import torch

        self.check_native(levels)
        prepared = []
        for c in contexts:
            x = self._validate(c, horizon, levels)
            prepared.append(torch.tensor(self.truncate_context(x), dtype=torch.float32))

        out = np.empty((len(prepared), horizon, len(levels)), dtype=float)
        for start in range(0, len(prepared), self.batch_size):
            chunk = prepared[start : start + self.batch_size]
            # positional: the argument is named `context` in chronos 1.x and
            # `inputs` in 2.x, so bind it positionally to work with both
            q, _mean = self.pipeline.predict_quantiles(
                chunk,
                prediction_length=horizon,
                quantile_levels=list(levels),
            )
            out[start : start + len(chunk)] = q.detach().cpu().numpy()
        return self._enforce_monotone(out)
