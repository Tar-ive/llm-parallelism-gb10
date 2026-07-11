"""Push per-step training metrics to the local Prometheus Pushgateway.

The monitoring stack (metrics/monitoring/docker-compose.yml) runs a
Pushgateway on :9091; Prometheus scrapes it every 2s with honor_labels so
the labels set here survive. Grafana's "LLM Training" dashboard reads:

    llm_train_step_ms, llm_train_tokens_per_s, llm_train_tflops,
    llm_train_mfu_pct, llm_train_loss, llm_train_mem_gib, llm_train_step_total

All metrics are gauges labelled by run tag (and static run config in
llm_train_info). Failures are swallowed: training must never die because
the monitoring stack is down.
"""

import time

try:
    from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway, delete_from_gateway
    _HAVE_PROM = True
except ImportError:
    _HAVE_PROM = False

GATEWAY = "localhost:9091"


class TrainingMetrics:
    """Per-run metric pusher. One Pushgateway group per (job, tag)."""

    def __init__(self, tag: str, config: dict | None = None, gateway: str = GATEWAY):
        self.enabled = _HAVE_PROM
        self.tag = tag
        self.gateway = gateway
        if not self.enabled:
            return
        self.registry = CollectorRegistry()
        lbl = ["tag"]
        self.step_ms = Gauge("llm_train_step_ms", "Wall time of the last optimizer step (ms)", lbl, registry=self.registry)
        self.tokens_per_s = Gauge("llm_train_tokens_per_s", "Training throughput (tokens/s)", lbl, registry=self.registry)
        self.tflops = Gauge("llm_train_tflops", "Achieved model TFLOPS", lbl, registry=self.registry)
        self.mfu = Gauge("llm_train_mfu_pct", "Model FLOPs utilization (% of assumed peak)", lbl, registry=self.registry)
        self.loss = Gauge("llm_train_loss", "Training loss", lbl, registry=self.registry)
        self.mem = Gauge("llm_train_mem_gib", "torch.cuda.max_memory_allocated (GiB)", lbl, registry=self.registry)
        self.steps = Counter("llm_train_step", "Optimizer steps completed", lbl, registry=self.registry)
        self.info = Gauge("llm_train_info", "Static run configuration", lbl + sorted(config or {}), registry=self.registry)
        if config is not None:
            self.info.labels(tag=tag, **{k: str(v) for k, v in sorted(config.items())}).set(1)
        self._last_push = 0.0

    def record_step(self, step_ms: float, tokens_per_s: float, tflops: float,
                    mfu_pct: float, loss: float, mem_gib: float,
                    min_push_interval_s: float = 1.0):
        if not self.enabled:
            return
        self.step_ms.labels(tag=self.tag).set(step_ms)
        self.tokens_per_s.labels(tag=self.tag).set(tokens_per_s)
        self.tflops.labels(tag=self.tag).set(tflops)
        self.mfu.labels(tag=self.tag).set(mfu_pct)
        self.loss.labels(tag=self.tag).set(loss)
        self.mem.labels(tag=self.tag).set(mem_gib)
        self.steps.labels(tag=self.tag).inc()
        now = time.time()
        if now - self._last_push >= min_push_interval_s:
            self._push()
            self._last_push = now

    def _push(self):
        try:
            push_to_gateway(self.gateway, job="llm_train", grouping_key={"tag": self.tag}, registry=self.registry)
        except Exception:
            pass

    def finish(self, clear: bool = False):
        """Final push. With clear=True the run's series disappear from the gateway."""
        if not self.enabled:
            return
        if clear:
            try:
                delete_from_gateway(self.gateway, job="llm_train", grouping_key={"tag": self.tag})
            except Exception:
                pass
        else:
            self._push()
