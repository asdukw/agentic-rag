"""DeepSeek usage aggregation and CNY cost estimates.

DeepSeek chat-completions responses expose token usage, including separate
prompt-cache hit and miss counts.  They do not expose a bill amount, so this
module deliberately labels every calculated amount as an estimate derived
from that response usage and the locally configured price table.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeepSeekCostStatus(StrEnum):
    """Whether a DeepSeek CNY estimate is usable."""

    NOT_APPLICABLE = "not_applicable"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class _StrictCostModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class DeepSeekUsage(_StrictCostModel):
    """Aggregate response usage for one operation/model pair.

    ``cache_breakdown_complete`` is intentionally explicit.  A provider can
    report total prompt tokens without the cache split; treating those tokens
    as cache misses would materially overstate a cached request's cost.
    """

    operation: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=256)
    calls: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    cache_hit_tokens: int | None = Field(default=None, ge=0)
    cache_miss_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int = Field(ge=0)
    cache_breakdown_complete: bool

    @model_validator(mode="after")
    def validate_cache_breakdown(self) -> DeepSeekUsage:
        if self.cache_breakdown_complete:
            if self.cache_hit_tokens is None or self.cache_miss_tokens is None:
                raise ValueError("complete cache usage requires hit and miss token counts")
            if self.cache_hit_tokens + self.cache_miss_tokens != self.prompt_tokens:
                raise ValueError("cache hit and miss tokens must equal prompt_tokens")
        elif self.cache_hit_tokens is not None or self.cache_miss_tokens is not None:
            raise ValueError("incomplete cache usage must not expose partial token counts")
        return self


class DeepSeekCostSummary(_StrictCostModel):
    """A serializable CNY estimate together with its observable token usage."""

    status: DeepSeekCostStatus
    currency: Literal["CNY"] = "CNY"
    cost_cny: float | None = Field(default=None, ge=0.0)
    usage: tuple[DeepSeekUsage, ...] = ()
    price_assumption: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_status(self) -> DeepSeekCostSummary:
        if self.status is DeepSeekCostStatus.NOT_APPLICABLE:
            if self.cost_cny != 0.0 or self.usage:
                raise ValueError("not_applicable cost requires no usage and a zero amount")
        elif self.status is DeepSeekCostStatus.ESTIMATED:
            if self.cost_cny is None or not self.usage:
                raise ValueError("estimated cost requires an amount and response usage")
        elif self.status is DeepSeekCostStatus.UNKNOWN and self.cost_cny is not None:
            raise ValueError("unknown cost must not provide an amount")
        return self


@dataclass(frozen=True, slots=True)
class DeepSeekModelPricing:
    """One model tier's three CNY prices, each per million tokens."""

    cache_hit_input_cny_per_million_tokens: Decimal
    cache_miss_input_cny_per_million_tokens: Decimal
    output_cny_per_million_tokens: Decimal


@dataclass(frozen=True, slots=True)
class DeepSeekPricing:
    """The configured Flash and Pro price table.

    Model recognition is exact by design.  An unrecognised response model is
    retained in usage output but is never silently charged as either tier.
    """

    flash: DeepSeekModelPricing
    pro: DeepSeekModelPricing

    def estimate(self, records: Iterable[DeepSeekUsage]) -> DeepSeekCostSummary:
        usage = aggregate_deepseek_usage(records)
        if not usage:
            return DeepSeekCostSummary(
                status=DeepSeekCostStatus.NOT_APPLICABLE,
                cost_cny=0.0,
                price_assumption="本次操作没有可计费的 DeepSeek 响应 usage。",
            )

        unknown_reasons: list[str] = []
        total = Decimal("0")
        used_tiers: list[str] = []
        for item in usage:
            pricing = self._pricing_for_model(item.model)
            if pricing is None:
                unknown_reasons.append(f"未识别的实际响应模型 {item.model!r}")
                continue
            if not item.cache_breakdown_complete:
                unknown_reasons.append(
                    f"{item.operation}/{item.model} 缺少完整的缓存命中/未命中 token 用量"
                )
                continue
            assert item.cache_hit_tokens is not None
            assert item.cache_miss_tokens is not None
            total += (
                Decimal(item.cache_hit_tokens) * pricing.cache_hit_input_cny_per_million_tokens
                + Decimal(item.cache_miss_tokens) * pricing.cache_miss_input_cny_per_million_tokens
                + Decimal(item.completion_tokens) * pricing.output_cny_per_million_tokens
            ) / Decimal(1_000_000)
            tier = self._tier_for_model(item.model)
            if tier is not None and tier not in used_tiers:
                used_tiers.append(tier)

        if unknown_reasons:
            return DeepSeekCostSummary(
                status=DeepSeekCostStatus.UNKNOWN,
                cost_cny=None,
                usage=usage,
                price_assumption=(
                    "Cannot estimate CNY cost from complete DeepSeek response usage: "
                    + "; ".join(unknown_reasons)
                ),
            )

        return DeepSeekCostSummary(
            status=DeepSeekCostStatus.ESTIMATED,
            cost_cny=float(total),
            usage=usage,
            price_assumption=(
                "Estimated from DeepSeek response usage (not a provider bill): "
                + "; ".join(self._tier_description(tier) for tier in used_tiers)
            ),
        )

    def _pricing_for_model(self, model: str) -> DeepSeekModelPricing | None:
        tier = self._tier_for_model(model)
        return getattr(self, tier) if tier is not None else None

    @staticmethod
    def _tier_for_model(model: str) -> Literal["flash", "pro"] | None:
        model_name = model.casefold()
        if model_name == "deepseek-v4-flash":
            return "flash"
        if model_name == "deepseek-v4-pro":
            return "pro"
        return None

    def _tier_description(self, tier: Literal["flash", "pro"]) -> str:
        pricing = getattr(self, tier)
        label = "Flash" if tier == "flash" else "Pro"
        return (
            f"{label} (cache-hit input CNY "
            f"{_decimal_text(pricing.cache_hit_input_cny_per_million_tokens)}/M, "
            f"cache-miss input CNY "
            f"{_decimal_text(pricing.cache_miss_input_cny_per_million_tokens)}/M, "
            f"output CNY {_decimal_text(pricing.output_cny_per_million_tokens)}/M)"
        )


def deepseek_usage(
    *,
    operation: str,
    model: str,
    prompt_tokens: int,
    cache_hit_tokens: int | None,
    cache_miss_tokens: int | None,
    completion_tokens: int,
) -> DeepSeekUsage:
    """Build one response usage record without inventing a cache split."""

    complete = (
        cache_hit_tokens is not None
        and cache_miss_tokens is not None
        and cache_hit_tokens + cache_miss_tokens == prompt_tokens
    )
    if prompt_tokens == 0 and cache_hit_tokens is None and cache_miss_tokens is None:
        complete = True
        cache_hit_tokens = 0
        cache_miss_tokens = 0
    return DeepSeekUsage(
        operation=operation,
        model=model,
        calls=1,
        prompt_tokens=prompt_tokens,
        cache_hit_tokens=cache_hit_tokens if complete else None,
        cache_miss_tokens=cache_miss_tokens if complete else None,
        completion_tokens=completion_tokens,
        cache_breakdown_complete=complete,
    )


def aggregate_deepseek_usage(records: Iterable[DeepSeekUsage]) -> tuple[DeepSeekUsage, ...]:
    """Group response usage deterministically by operation and actual model."""

    grouped: dict[tuple[str, str], list[DeepSeekUsage]] = defaultdict(list)
    for record in records:
        grouped[(record.operation, record.model)].append(record)
    values: list[DeepSeekUsage] = []
    for (operation, model), items in sorted(grouped.items()):
        cache_complete = all(item.cache_breakdown_complete for item in items)
        values.append(
            DeepSeekUsage(
                operation=operation,
                model=model,
                calls=sum(item.calls for item in items),
                prompt_tokens=sum(item.prompt_tokens for item in items),
                cache_hit_tokens=(
                    sum(item.cache_hit_tokens or 0 for item in items) if cache_complete else None
                ),
                cache_miss_tokens=(
                    sum(item.cache_miss_tokens or 0 for item in items) if cache_complete else None
                ),
                completion_tokens=sum(item.completion_tokens for item in items),
                cache_breakdown_complete=cache_complete,
            )
        )
    return tuple(values)


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")
