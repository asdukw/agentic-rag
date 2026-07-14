from __future__ import annotations

from decimal import Decimal

import pytest

from hybrid_rag.deepseek_costs import (
    DeepSeekCostStatus,
    DeepSeekModelPricing,
    DeepSeekPricing,
    deepseek_usage,
)


def _pricing() -> DeepSeekPricing:
    return DeepSeekPricing(
        flash=DeepSeekModelPricing(
            cache_hit_input_cny_per_million_tokens=Decimal("0.02"),
            cache_miss_input_cny_per_million_tokens=Decimal("1"),
            output_cny_per_million_tokens=Decimal("2"),
        ),
        pro=DeepSeekModelPricing(
            cache_hit_input_cny_per_million_tokens=Decimal("0.025"),
            cache_miss_input_cny_per_million_tokens=Decimal("3"),
            output_cny_per_million_tokens=Decimal("6"),
        ),
    )


def test_flash_cost_uses_cache_hit_miss_and_output_prices() -> None:
    summary = _pricing().estimate(
        (
            deepseek_usage(
                operation="keyword",
                model="deepseek-v4-flash",
                prompt_tokens=100,
                cache_hit_tokens=25,
                cache_miss_tokens=75,
                completion_tokens=50,
            ),
        )
    )

    assert summary.status is DeepSeekCostStatus.ESTIMATED
    assert summary.currency == "CNY"
    assert summary.cost_cny == pytest.approx((25 * 0.02 + 75 + 50 * 2) / 1_000_000)
    assert summary.usage[0].cache_breakdown_complete


def test_pro_cost_uses_its_own_three_prices() -> None:
    summary = _pricing().estimate(
        (
            deepseek_usage(
                operation="judge",
                model="deepseek-v4-pro",
                prompt_tokens=200,
                cache_hit_tokens=80,
                cache_miss_tokens=120,
                completion_tokens=40,
            ),
        )
    )

    assert summary.status is DeepSeekCostStatus.ESTIMATED
    assert summary.cost_cny == pytest.approx((80 * 0.025 + 120 * 3 + 40 * 6) / 1_000_000)


@pytest.mark.parametrize(
    ("model", "cache_hit_tokens", "cache_miss_tokens"),
    (
        ("deepseek-v4-flash", None, None),
        ("deepseek-v4-flash", 10, 89),
        ("other-model", 10, 90),
    ),
)
def test_cost_is_unknown_when_usage_cannot_be_priced_exactly(
    model: str,
    cache_hit_tokens: int | None,
    cache_miss_tokens: int | None,
) -> None:
    summary = _pricing().estimate(
        (
            deepseek_usage(
                operation="answer",
                model=model,
                prompt_tokens=100,
                cache_hit_tokens=cache_hit_tokens,
                cache_miss_tokens=cache_miss_tokens,
                completion_tokens=20,
            ),
        )
    )

    assert summary.status is DeepSeekCostStatus.UNKNOWN
    assert summary.cost_cny is None
