"""
Screener Service — low-level data-fetching helpers for TradingView analysis.

All functions call TradingView APIs and return normalised Row / MultiRow lists.
They are intentionally free of MCP concerns so they can be unit-tested directly.
"""
from __future__ import annotations

from typing import Any, List, Optional

from tradingview_mcp.core.types import (
    IndicatorMap, MultiRow, Row,
    percent_change, tf_to_tv_resolution,
)
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.services.indicators import compute_metrics
from tradingview_mcp.core.utils.validators import (
    EXCHANGE_SCREENER,
    get_market_type,
    get_tv_exchange_prefix,
    is_stock_exchange,
)

try:
    # Patched: route through resilience layer (retry + 60s TTL cache).
    import tradingview_ta  # noqa: F401  presence check
    from tradingview_mcp.core.services.screener_provider import (
        resilient_get_multiple_analysis as get_multiple_analysis,
    )
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

try:
    from tradingview_screener import Query
    from tradingview_screener.column import Column
    _SCREENER_AVAILABLE = True
except ImportError:
    _SCREENER_AVAILABLE = False


# ── Bollinger / trending fetchers ──────────────────────────────────────────────

def fetch_bollinger_analysis(
    exchange: str,
    timeframe: str = "4h",
    limit: int = 50,
    bbw_filter: float = None,
) -> List[Row]:
    """
    Fetch analysis using tradingview_ta with Bollinger Band squeeze logic.

    Args:
        exchange:   Exchange identifier (e.g. KUCOIN, BINANCE, EGX).
        timeframe:  TradingView interval string (5m, 15m, 1h, 4h, 1D, 1W, 1M).
        limit:      Maximum rows to return.
        bbw_filter: Exclude rows where BBW >= this value (squeeze detector).

    Returns:
        List of Row dicts sorted by changePercent descending.
    """
    if not _TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is missing; run `uv sync`.")

    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")

    symbols = symbols[: limit * 2]
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    try:
        analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=symbols)
    except Exception as exc:
        raise RuntimeError(f"Analysis failed: {exc}") from exc

    rows: List[Row] = []
    for key, value in analysis.items():
        try:
            if value is None:
                continue
            indicators = value.indicators
            metrics = compute_metrics(indicators)
            if not metrics or metrics.get("bbw") is None:
                continue
            if bbw_filter is not None and (metrics["bbw"] >= bbw_filter or metrics["bbw"] <= 0):
                continue
            if not (indicators.get("EMA50") and indicators.get("RSI")):
                continue

            rows.append(
                Row(
                    symbol=key,
                    changePercent=metrics["change"],
                    indicators=IndicatorMap(
                        open=metrics.get("open"),
                        close=metrics.get("price"),
                        SMA20=indicators.get("SMA20"),
                        BB_upper=indicators.get("BB.upper"),
                        BB_lower=indicators.get("BB.lower"),
                        EMA50=indicators.get("EMA50"),
                        RSI=indicators.get("RSI"),
                        volume=indicators.get("volume"),
                    ),
                )
            )
        except (TypeError, ZeroDivisionError, KeyError):
            continue

    rows.sort(key=lambda x: x["changePercent"], reverse=True)
    return rows[:limit]


def fetch_trending_analysis(
    exchange: str,
    timeframe: str = "5m",
    filter_type: str = "",
    rating_filter: int = None,
    limit: int = 50,
) -> List[Row]:
    """
    Fetch trending coins across all available symbols in batches of 200.

    Args:
        exchange:      Exchange identifier.
        timeframe:     TradingView interval string.
        filter_type:   Optional filter mode ('rating').
        rating_filter: BB rating value to match when filter_type == 'rating'.
        limit:         Maximum rows to return.

    Returns:
        List of Row dicts sorted by changePercent descending.
    """
    if not _TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is missing; run `uv sync`.")

    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    batch_size = 200
    all_coins: List[Row] = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=batch)
        except Exception:
            continue

        for key, value in analysis.items():
            try:
                if value is None:
                    continue
                indicators = value.indicators
                metrics = compute_metrics(indicators)
                if not metrics or metrics.get("bbw") is None:
                    continue
                if filter_type == "rating" and rating_filter is not None:
                    if metrics["rating"] != rating_filter:
                        continue

                all_coins.append(
                    Row(
                        symbol=key,
                        changePercent=metrics["change"],
                        indicators=IndicatorMap(
                            open=metrics.get("open"),
                            close=metrics.get("price"),
                            SMA20=indicators.get("SMA20"),
                            BB_upper=indicators.get("BB.upper"),
                            BB_lower=indicators.get("BB.lower"),
                            EMA50=indicators.get("EMA50"),
                            RSI=indicators.get("RSI"),
                            volume=indicators.get("volume"),
                        ),
                    )
                )
            except (TypeError, ZeroDivisionError, KeyError):
                continue

    all_coins.sort(key=lambda x: x["changePercent"], reverse=True)
    return all_coins[:limit]


# ── Multi-timeframe screener ───────────────────────────────────────────────────

def fetch_multi_changes(
    exchange: str,
    timeframes: Optional[List[str]],
    base_timeframe: str = "4h",
    limit: Optional[int] = None,
    cookies: Any = None,
) -> List[MultiRow]:
    """
    Fetch open/close data across multiple timeframes using tradingview-screener.

    Args:
        exchange:       Exchange identifier (empty string = all markets).
        timeframes:     List of timeframe strings; defaults to [15m, 1h, 4h, 1D].
        base_timeframe: Primary timeframe for indicator columns.
        limit:          Maximum rows from screener (None = no cap).
        cookies:        Optional cookies for authenticated screener requests.

    Returns:
        List of MultiRow dicts with per-timeframe change percentages.
    """
    if not _SCREENER_AVAILABLE:
        raise RuntimeError("tradingview-screener missing; run `uv sync`.")

    tfs = timeframes or ["15m", "1h", "4h", "1D"]
    suffix_map: dict[str, str] = {}
    for tf in tfs:
        s = tf_to_tv_resolution(tf)
        if s:
            suffix_map[tf] = s
    if not suffix_map:
        suffix_map = {base_timeframe: tf_to_tv_resolution(base_timeframe) or "240"}

    base_suffix = tf_to_tv_resolution(base_timeframe) or next(iter(suffix_map.values()))
    cols: list[str] = []
    seen: set[str] = set()
    for tf, s in suffix_map.items():
        for c in (f"open|{s}", f"close|{s}"):
            if c not in seen:
                cols.append(c)
                seen.add(c)
    for c in (
        f"SMA20|{base_suffix}",
        f"BB.upper|{base_suffix}",
        f"BB.lower|{base_suffix}",
        f"volume|{base_suffix}",
    ):
        if c not in seen:
            cols.append(c)
            seen.add(c)

    market = get_market_type(exchange) if exchange else "crypto"
    q = Query().set_markets(market).select(*cols)
    if exchange:
        q = q.where(Column("exchange") == exchange.upper())
    if limit:
        q = q.limit(int(limit))

    _total, df = q.get_scanner_data(cookies=cookies)
    if df is None or df.empty:
        return []

    out: List[MultiRow] = []
    for _, r in df.iterrows():
        symbol = r.get("ticker")
        changes: dict[str, Optional[float]] = {}
        for tf, s in suffix_map.items():
            o = r.get(f"open|{s}")
            c = r.get(f"close|{s}")
            changes[tf] = percent_change(o, c)
        base_ind = IndicatorMap(
            open=r.get(f"open|{base_suffix}"),
            close=r.get(f"close|{base_suffix}"),
            SMA20=r.get(f"SMA20|{base_suffix}"),
            BB_upper=r.get(f"BB.upper|{base_suffix}"),
            BB_lower=r.get(f"BB.lower|{base_suffix}"),
            volume=r.get(f"volume|{base_suffix}"),
        )
        out.append(MultiRow(symbol=symbol, changes=changes, base_indicators=base_ind))
    return out


# ── Candle pattern analysis ────────────────────────────────────────────────────

def calculate_candle_pattern_score(
    indicators: dict,
    pattern_length: int,
    min_increase: float,
) -> dict:
    """
    Score a candle pattern based on body ratio, momentum, volume, and RSI.

    Args:
        indicators:     Raw indicators dict from tradingview_ta.
        pattern_length: Number of consecutive periods being analysed.
        min_increase:   Minimum price change percentage threshold.

    Returns:
        Dict with 'detected' bool, 'score' int, 'details' list, and computed fields.
    """
    try:
        open_price = indicators.get("open", 0)
        close_price = indicators.get("close", 0)
        high_price = indicators.get("high", 0)
        low_price = indicators.get("low", 0)
        volume = indicators.get("volume", 0)
        rsi = indicators.get("RSI", 50)

        if not all([open_price, close_price, high_price, low_price]):
            return {"detected": False, "score": 0}

        candle_body = abs(close_price - open_price)
        candle_range = high_price - low_price
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        price_change = ((close_price - open_price) / open_price) * 100

        score = 0
        details: list[str] = []

        if body_ratio > 0.7:
            score += 2
            details.append("Strong candle body")
        elif body_ratio > 0.5:
            score += 1
            details.append("Moderate candle body")

        if abs(price_change) >= min_increase:
            score += 2
            details.append(f"Strong momentum ({price_change:.1f}%)")
        elif abs(price_change) >= min_increase / 2:
            score += 1
            details.append(f"Moderate momentum ({price_change:.1f}%)")

        if volume > 5000:
            score += 1
            details.append("Good volume")

        if (price_change > 0 and 50 < rsi < 80) or (price_change < 0 and 20 < rsi < 50):
            score += 1
            details.append("RSI momentum aligned")

        ema50 = indicators.get("EMA50", close_price)
        if (price_change > 0 and close_price > ema50) or (price_change < 0 and close_price < ema50):
            score += 1
            details.append("Trend alignment")

        return {
            "detected": score >= 3,
            "score": score,
            "details": details,
            "price": round(close_price, 6),
            "total_change": round(price_change, 3),
            "body_ratio": round(body_ratio, 3),
            "volume": volume,
        }
    except Exception as exc:
        return {"detected": False, "score": 0, "error": str(exc)}


def fetch_multi_timeframe_patterns(
    exchange: str,
    symbols: List[str],
    base_tf: str,
    length: int,
    min_increase: float,
) -> List[dict]:
    """
    Fetch multi-timeframe pattern data using tradingview-screener.

    Args:
        exchange:     Exchange identifier.
        symbols:      Symbol list to query.
        base_tf:      Base timeframe string (e.g. '15m').
        length:       Pattern length for scoring.
        min_increase: Minimum percentage increase for pattern detection.

    Returns:
        List of pattern result dicts sorted by pattern_score descending.
    """
    if not _SCREENER_AVAILABLE:
        return []
    try:
        tf_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1D": "1D"}
        tv_interval = tf_map.get(base_tf, "15")

        cols = [
            f"open|{tv_interval}",
            f"close|{tv_interval}",
            f"high|{tv_interval}",
            f"low|{tv_interval}",
            f"volume|{tv_interval}",
            "RSI",
        ]

        market = get_market_type(exchange)
        q = Query().set_markets(market).select(*cols)
        q = q.where(Column("exchange") == exchange.upper())
        q = q.limit(len(symbols))

        _total, df = q.get_scanner_data()
        if df is None or df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            symbol = row.get("ticker", "")
            try:
                ind = {
                    "open": row.get(f"open|{tv_interval}"),
                    "close": row.get(f"close|{tv_interval}"),
                    "high": row.get(f"high|{tv_interval}"),
                    "low": row.get(f"low|{tv_interval}"),
                    "volume": row.get(f"volume|{tv_interval}", 0),
                    "RSI": row.get("RSI", 50),
                }
                if not all([ind["open"], ind["close"], ind["high"], ind["low"]]):
                    continue

                pattern_score = calculate_candle_pattern_score(ind, length, min_increase)
                if pattern_score["detected"]:
                    results.append(
                        {
                            "symbol": symbol,
                            "pattern_score": pattern_score["score"],
                            "price": pattern_score["price"],
                            "change": pattern_score["total_change"],
                            "body_ratio": pattern_score["body_ratio"],
                            "volume": ind["volume"],
                            "rsi": round(ind["RSI"], 2),
                            "details": pattern_score["details"],
                        }
                    )
            except Exception:
                continue

        return sorted(results, key=lambda x: x["pattern_score"], reverse=True)
    except Exception:
        return []


# ── Coin analysis (single asset) ───────────────────────────────────────────────

def _fetch_coin_screener_extras(full_symbol: str, exchange: str) -> dict:
    """Fetch additional screener columns (52w levels, performance, pivots, etc.).

    Returns a dict mapping column name -> value (empty dict on failure or when
    tradingview_screener is unavailable). Stock-only columns (sector, pivots,
    earnings, base depth) are only requested for stock exchanges.
    """
    if not _SCREENER_AVAILABLE:
        return {}

    market = get_market_type(exchange)
    cols = [
        'name',
        'average_volume_30d_calc', 'average_volume_10d_calc',
        'relative_volume_10d_calc',
        'price_52_week_high', 'price_52_week_low', 'High.All',
        'Perf.1M', 'Perf.3M', 'Perf.6M',
        'ATR', 'beta_1_year', 'Volatility.D',
        'ADX+DI', 'ADX-DI',
        'EMA50', 'EMA200', 'SMA150',
        'Recommend.MA', 'SMA200',
        'Perf.W', 'gap',
    ]
    if is_stock_exchange(exchange):
        cols.extend([
            'sector', 'industry',
            'Pivot.M.Classic.R1', 'Pivot.M.Classic.R2', 'Pivot.M.Classic.R3',
            'Pivot.M.Classic.S1', 'Pivot.M.Classic.S2', 'Pivot.M.Classic.S3',
            'Pivot.M.Classic.Middle',
            'Low.3M', 'Low.6M', 'High.3M', 'High.6M',
            'earnings_release_next_date',
        ])

    try:
        q = Query().set_markets(market).select(*cols).set_tickers(full_symbol).limit(1)
        _t, df = q.get_scanner_data()
        if df is not None and not df.empty:
            return df.iloc[0].to_dict()
    except Exception:
        return {}
    return {}


def _fetch_spy_performance() -> dict:
    """Fetch SPY 1M/3M/6M performance for relative-strength calculation.

    Returns a dict with keys ``perf_1m``, ``perf_3m``, ``perf_6m`` (all 0 on
    failure or when tradingview_screener is unavailable).
    """
    if not _SCREENER_AVAILABLE:
        return {"perf_1m": 0, "perf_3m": 0, "perf_6m": 0}
    try:
        q = (
            Query()
            .select('Perf.1M', 'Perf.3M', 'Perf.6M')
            .where(Column('name') == 'SPY')
            .set_markets('america')
            .limit(1)
        )
        _t, df = q.get_scanner_data()
        if df is not None and not df.empty:
            row = df.iloc[0]
            return {
                "perf_1m": row.get('Perf.1M', 0) or 0,
                "perf_3m": row.get('Perf.3M', 0) or 0,
                "perf_6m": row.get('Perf.6M', 0) or 0,
            }
    except Exception:
        pass
    return {"perf_1m": 0, "perf_3m": 0, "perf_6m": 0}


def _build_coin_screener_sections(
    screener_data: dict,
    exchange: str,
    indicators: dict,
    current_price: float,
    change_percent: float,
    volume: float,
) -> dict:
    """Build the additional output sections from screener data.

    Returns a dict whose keys are merged into the final analyze_coin response:
    additional price_data fields, technical_indicators fields, plus new top-level
    sections (performance, relative_strength, ma_analysis, sector_info,
    pivot_analysis, base_analysis, breakout_check, earnings, gap_analysis,
    climax_check, risk_data, trend_data, stage2_check).
    """
    from datetime import datetime, date

    is_stock = is_stock_exchange(exchange)

    # Volume from screener
    avg_volume = screener_data.get('average_volume_30d_calc', 0) or 0
    avg_volume_10d = screener_data.get('average_volume_10d_calc', 0) or 0
    rel_volume = screener_data.get('relative_volume_10d_calc', 0) or 0
    volume_ratio = round(volume / avg_volume, 2) if avg_volume and avg_volume > 0 else None

    # 52-week and all-time data
    high_52w = screener_data.get('price_52_week_high', 0) or 0
    low_52w = screener_data.get('price_52_week_low', 0) or 0
    all_time_high = screener_data.get('High.All', 0) or 0
    pct_from_high = round(((high_52w - current_price) / high_52w) * 100, 2) if high_52w else None
    pct_above_low = round(((current_price - low_52w) / low_52w) * 100, 2) if low_52w else None

    # Performance
    perf_1m = screener_data.get('Perf.1M', 0) or 0
    perf_3m = screener_data.get('Perf.3M', 0) or 0
    perf_6m = screener_data.get('Perf.6M', 0) or 0

    # Risk data
    atr = screener_data.get('ATR', 0) or 0
    beta = screener_data.get('beta_1_year') or 0
    volatility = screener_data.get('Volatility.D', 0) or 0
    atr_stop = round(current_price - (2 * atr), 2) if atr else None
    atr_stop_pct = round((2 * atr / current_price) * 100, 2) if atr and current_price else None

    # Trend direction from ADX +/- DI
    adx_plus = screener_data.get('ADX+DI', 0) or 0
    adx_minus = screener_data.get('ADX-DI', 0) or 0
    trend_direction = (
        "BULLISH" if adx_plus > adx_minus
        else "BEARISH" if adx_minus > adx_plus
        else "NEUTRAL"
    )

    # Moving averages for Stage 2 check
    ema50 = screener_data.get('EMA50', 0) or indicators.get("EMA50", 0) or 0
    sma150 = screener_data.get('SMA150', 0) or 0
    ema200 = screener_data.get('EMA200', 0) or indicators.get("EMA200", 0) or 0

    # 200MA trend detection
    ma_recommendation = screener_data.get('Recommend.MA', 0) or 0
    sma200 = screener_data.get('SMA200', 0)
    sma200_trending_up = ma_recommendation > 0

    # RS vs SPY (US stocks only)
    if is_stock:
        spy = _fetch_spy_performance()
        spy_perf_1m = spy["perf_1m"]
        spy_perf_3m = spy["perf_3m"]
        spy_perf_6m = spy["perf_6m"]
        rs_1m = round(perf_1m - spy_perf_1m, 2) if perf_1m else None
        rs_3m = round(perf_3m - spy_perf_3m, 2) if perf_3m else None
        rs_6m = round(perf_6m - spy_perf_6m, 2) if perf_6m else None
    else:
        spy_perf_1m = spy_perf_3m = spy_perf_6m = None
        rs_1m = rs_3m = rs_6m = None

    # Pivot points (stocks only)
    if is_stock:
        pivot_r1 = screener_data.get('Pivot.M.Classic.R1', 0) or 0
        pivot_r2 = screener_data.get('Pivot.M.Classic.R2', 0) or 0
        pivot_r3 = screener_data.get('Pivot.M.Classic.R3', 0) or 0
        pivot_s1 = screener_data.get('Pivot.M.Classic.S1', 0) or 0
        pivot_s2 = screener_data.get('Pivot.M.Classic.S2', 0) or 0
        pivot_s3 = screener_data.get('Pivot.M.Classic.S3', 0) or 0
        pivot_middle = screener_data.get('Pivot.M.Classic.Middle', 0) or 0
        distance_to_r1_pct = round(((pivot_r1 - current_price) / current_price) * 100, 2) if pivot_r1 and current_price else None
        distance_to_r2_pct = round(((pivot_r2 - current_price) / current_price) * 100, 2) if pivot_r2 and current_price else None
        distance_to_s1_pct = round(((current_price - pivot_s1) / current_price) * 100, 2) if pivot_s1 and current_price else None
    else:
        pivot_r1 = pivot_r2 = pivot_r3 = pivot_s1 = pivot_s2 = pivot_s3 = pivot_middle = 0
        distance_to_r1_pct = distance_to_r2_pct = distance_to_s1_pct = None

    # Base depth (stocks only)
    if is_stock:
        low_3m = screener_data.get('Low.3M', 0) or 0
        low_6m = screener_data.get('Low.6M', 0) or 0
        high_3m = screener_data.get('High.3M', 0) or 0
        high_6m = screener_data.get('High.6M', 0) or 0
        base_depth_3m_pct = round(((high_3m - low_3m) / high_3m) * 100, 2) if high_3m and low_3m else None
        base_depth_6m_pct = round(((high_6m - low_6m) / high_6m) * 100, 2) if high_6m and low_6m else None
        valid_base_depth = base_depth_3m_pct <= 40 if base_depth_3m_pct is not None else None
        position_in_base_pct = round(((high_3m - current_price) / (high_3m - low_3m)) * 100, 2) if high_3m and low_3m and (high_3m - low_3m) > 0 else None
    else:
        low_3m = low_6m = high_3m = high_6m = 0
        base_depth_3m_pct = base_depth_6m_pct = None
        valid_base_depth = None
        position_in_base_pct = None

    # Earnings (stocks only)
    earnings_date = None
    days_until_earnings = None
    earnings_within_7_days = None
    if is_stock:
        earnings_date_raw = screener_data.get('earnings_release_next_date')
        if earnings_date_raw is not None and earnings_date_raw != 0:
            try:
                if isinstance(earnings_date_raw, (int, float)):
                    # TradingView returns timestamps either in seconds or ms.
                    if earnings_date_raw > 946684800000:
                        earnings_date = datetime.fromtimestamp(earnings_date_raw / 1000).date()
                    elif earnings_date_raw > 946684800:
                        earnings_date = datetime.fromtimestamp(earnings_date_raw).date()
                elif isinstance(earnings_date_raw, str) and earnings_date_raw:
                    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                        try:
                            earnings_date = datetime.strptime(earnings_date_raw, fmt).date()
                            break
                        except ValueError:
                            continue
                if earnings_date:
                    days_until_earnings = (earnings_date - date.today()).days
                    earnings_within_7_days = 0 <= days_until_earnings <= 7
            except Exception:
                earnings_date = None
                days_until_earnings = None
                earnings_within_7_days = None

    # Gap detection
    gap_pct = screener_data.get('gap', 0) or 0
    if gap_pct > 1:
        gap_type = "GAP_UP"
    elif gap_pct < -1:
        gap_type = "GAP_DOWN"
    else:
        gap_type = "NONE"

    # Weekly perf
    perf_w = screener_data.get('Perf.W', 0) or 0

    # Climax detection
    sma20 = indicators.get("SMA20", 0) or 0
    pct_above_20ma = round(((current_price - sma20) / sma20) * 100, 2) if sma20 and sma20 > 0 else None
    pct_above_50ma = round(((current_price - ema50) / ema50) * 100, 2) if ema50 and ema50 > 0 else None
    climax_above_20ma = pct_above_20ma > 30 if pct_above_20ma is not None else False
    climax_above_50ma = pct_above_50ma > 30 if pct_above_50ma is not None else False
    climax_weekly_gain = perf_w > 25 if perf_w else False
    climax_daily_gain = change_percent > 10 if change_percent else False
    climax_count = sum([climax_above_20ma, climax_above_50ma, climax_weekly_gain, climax_daily_gain])

    return {
        "price_data_extra": {
            "avg_volume": avg_volume,
            "avg_volume_10d": avg_volume_10d,
            "volume_ratio": volume_ratio,
            "relative_volume": round(rel_volume, 2) if rel_volume else None,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "all_time_high": all_time_high,
            "pct_from_high": pct_from_high,
            "pct_above_low": pct_above_low,
        },
        "ma_extra": {
            "ema50_screener": round(ema50, 6) if ema50 else None,
            "sma150": round(sma150, 6) if sma150 else None,
            "ema200_screener": round(ema200, 6) if ema200 else None,
        },
        "performance": {
            "perf_w": round(perf_w, 2) if perf_w else None,
            "perf_1m": round(perf_1m, 2) if perf_1m else None,
            "perf_3m": round(perf_3m, 2) if perf_3m else None,
            "perf_6m": round(perf_6m, 2) if perf_6m else None,
        },
        "relative_strength": {
            "spy_perf_1m": round(spy_perf_1m, 2) if spy_perf_1m is not None else None,
            "spy_perf_3m": round(spy_perf_3m, 2) if spy_perf_3m is not None else None,
            "spy_perf_6m": round(spy_perf_6m, 2) if spy_perf_6m is not None else None,
            "rs_1m": rs_1m,
            "rs_3m": rs_3m,
            "rs_6m": rs_6m,
            "outperforming_spy": rs_3m > 0 if rs_3m is not None else None,
        },
        "ma_analysis": {
            "sma200": round(sma200, 2) if sma200 else None,
            "ma_recommendation": round(ma_recommendation, 2),
            "sma200_trending_up": sma200_trending_up,
        },
        "sector_info": {
            "sector": screener_data.get('sector') if is_stock else None,
            "industry": screener_data.get('industry') if is_stock else None,
        },
        "pivot_analysis": {
            "pivot_r1": round(pivot_r1, 2) if pivot_r1 else None,
            "pivot_r2": round(pivot_r2, 2) if pivot_r2 else None,
            "pivot_r3": round(pivot_r3, 2) if pivot_r3 else None,
            "pivot_middle": round(pivot_middle, 2) if pivot_middle else None,
            "pivot_s1": round(pivot_s1, 2) if pivot_s1 else None,
            "pivot_s2": round(pivot_s2, 2) if pivot_s2 else None,
            "pivot_s3": round(pivot_s3, 2) if pivot_s3 else None,
            "distance_to_r1_pct": distance_to_r1_pct,
            "distance_to_r2_pct": distance_to_r2_pct,
            "distance_to_s1_pct": distance_to_s1_pct,
            "breakout_proximity": (
                "CLOSE" if distance_to_r1_pct is not None and distance_to_r1_pct <= 3
                else "FAR" if distance_to_r1_pct else None
            ),
        },
        "base_analysis": {
            "high_3m": round(high_3m, 2) if high_3m else None,
            "low_3m": round(low_3m, 2) if low_3m else None,
            "high_6m": round(high_6m, 2) if high_6m else None,
            "low_6m": round(low_6m, 2) if low_6m else None,
            "base_depth_3m_pct": base_depth_3m_pct,
            "base_depth_6m_pct": base_depth_6m_pct,
            "valid_base_depth": valid_base_depth,
            "position_in_base_pct": position_in_base_pct,
            "near_base_top": position_in_base_pct <= 15 if position_in_base_pct is not None else None,
        },
        "breakout_check": {
            "above_pivot_r1": current_price > pivot_r1 if pivot_r1 else None,
            "above_52w_high": current_price > high_52w if high_52w else None,
            "above_3m_high": current_price > high_3m if high_3m else None,
            "volume_confirmed": volume_ratio >= 1.5 if volume_ratio else False,
            "valid_breakout": (current_price > pivot_r1 if pivot_r1 else False) and (volume_ratio >= 1.5 if volume_ratio else False),
        },
        "earnings": {
            "next_date": str(earnings_date) if earnings_date else None,
            "days_until": days_until_earnings,
            "within_7_days": earnings_within_7_days,
            "warning": earnings_within_7_days is True,
        },
        "gap_analysis": {
            "gap_pct": round(gap_pct, 2) if gap_pct else 0,
            "gap_type": gap_type,
            "significant_gap": abs(gap_pct) > 3 if gap_pct else False,
        },
        "climax_check": {
            "pct_above_20ma": pct_above_20ma,
            "pct_above_50ma": pct_above_50ma,
            "above_30pct_20ma": climax_above_20ma,
            "above_30pct_50ma": climax_above_50ma,
            "weekly_gain_over_25pct": climax_weekly_gain,
            "daily_gain_over_10pct": climax_daily_gain,
            "climax_score": climax_count,
            "climax_warning": climax_count >= 2,
        },
        "risk_data": {
            "atr": round(atr, 2) if atr else None,
            "atr_stop": atr_stop,
            "atr_stop_pct": atr_stop_pct,
            "beta": round(beta, 2) if beta else None,
            "volatility_daily": round(volatility, 2) if volatility else None,
        },
        "trend_data": {
            "adx_plus": round(adx_plus, 2) if adx_plus else None,
            "adx_minus": round(adx_minus, 2) if adx_minus else None,
            "trend_direction": trend_direction,
        },
        "stage2_check": {
            "price_above_50ma": current_price > ema50 if ema50 else None,
            "price_above_150ma": current_price > sma150 if sma150 else None,
            "price_above_200ma": current_price > ema200 if ema200 else None,
            "ma_stacked": (ema50 > sma150 > ema200) if all([ema50, sma150, ema200]) else None,
            "sma200_trending_up": sma200_trending_up,
            "within_25pct_high": pct_from_high <= 25 if pct_from_high is not None else None,
            "above_25pct_low": pct_above_low >= 25 if pct_above_low is not None else None,
            "outperforming_spy": rs_3m > 0 if rs_3m is not None else None,
        },
    }


def analyze_coin(
    symbol: str,
    exchange: str,
    timeframe: str,
) -> dict:
    """
    Full technical analysis for a single coin/stock.

    Args:
        symbol:    Validated symbol string (with exchange prefix).
        exchange:  Validated exchange identifier.
        timeframe: Validated TradingView interval string.

    Returns:
        Dict containing price data, all extended indicators, market sentiment,
        and (for stocks) stock score + trade setup.
    """
    from tradingview_mcp.core.services.indicators import (
        extract_extended_indicators,
        analyze_timeframe_context,
        compute_stock_score,
        compute_trade_setup,
        compute_trade_quality,
    )
    from tradingview_mcp.core.utils.validators import is_stock_exchange

    if not _TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    full_symbol = symbol.upper() if ":" in symbol else f"{get_tv_exchange_prefix(exchange)}:{symbol.upper()}"
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    try:
        analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=[full_symbol])

        if full_symbol not in analysis or analysis[full_symbol] is None:
            return {"error": f"No data found for {symbol} on {exchange}", "symbol": symbol, "exchange": exchange, "timeframe": timeframe}

        data = analysis[full_symbol]
        indicators = data.indicators
        metrics = compute_metrics(indicators)

        if not metrics:
            return {"error": f"Could not compute metrics for {symbol}", "symbol": symbol, "exchange": exchange, "timeframe": timeframe}

        volume = indicators.get("volume", 0)
        high = indicators.get("high", 0)
        low = indicators.get("low", 0)
        open_price = indicators.get("open", 0)
        close_price = indicators.get("close", 0)

        extended = extract_extended_indicators(indicators)
        tf_context = analyze_timeframe_context(indicators, timeframe)

        # Additional screener-sourced sections (volume_ratio, 52w levels, VCP,
        # RS vs SPY, pivots, base depth, earnings, gap, climax, stage2, etc.)
        screener_data = _fetch_coin_screener_extras(full_symbol, exchange)
        extras = _build_coin_screener_sections(
            screener_data=screener_data,
            exchange=exchange,
            indicators=indicators,
            current_price=metrics["price"],
            change_percent=metrics["change"],
            volume=volume,
        )
        price_data_extra = extras.pop("price_data_extra", {})
        ma_extra = extras.pop("ma_extra", {})

        trade_data: dict = {}
        if is_stock_exchange(exchange):
            score_result = compute_stock_score(indicators)
            if score_result:
                trade_data["stock_score"] = score_result["score"]
                trade_data["grade"] = score_result["grade"]
                trade_data["trend_state"] = score_result["trend_state"]
                setup = compute_trade_setup(indicators)
                if setup:
                    trade_data["trade_setup"] = {
                        "setup_types": setup["setup_types"],
                        "entry_points": setup["entry_points"],
                        "stop_loss": setup["stop_loss"],
                        "stop_distance_pct": setup["stop_distance_pct"],
                        "targets": setup["targets"],
                        "risk_reward": setup["risk_reward"],
                        "supports": setup["supports"],
                        "resistances": setup["resistances"],
                    }
                    quality = compute_trade_quality(indicators, score_result["score"], setup)
                    if quality:
                        trade_data["trade_quality_score"] = quality["trade_quality_score"]
                        trade_data["trade_quality"] = quality["quality"]
                        trade_data["trade_notes"] = quality["notes"]

        return {
            "symbol": full_symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "timestamp": "real-time",
            "price_data": {
                "current_price": metrics["price"],
                "open": round(open_price, 6) if open_price else None,
                "high": round(high, 6) if high else None,
                "low": round(low, 6) if low else None,
                "close": round(close_price, 6) if close_price else None,
                "change_percent": metrics["change"],
                "volume": volume,
                **price_data_extra,
            },
            "ma_screener": ma_extra,
            "timeframe_context": tf_context,
            "rsi": extended["rsi"],
            "macd": extended["macd"],
            "sma": extended["sma"],
            "ema": extended["ema"],
            "bollinger_bands": extended["bollinger_bands"],
            "atr": extended["atr"],
            "volume_analysis": extended["volume"],
            "obv": extended["obv"],
            "support_resistance": extended["support_resistance"],
            "stochastic": extended["stochastic"],
            "adx": extended["adx"],
            "market_structure": extended["market_structure"],
            **({"vwap": extended["vwap"]} if "vwap" in extended else {}),
            "market_sentiment": {
                "overall_rating": metrics["rating"],
                "buy_sell_signal": metrics["signal"],
                "volatility": (
                    "High" if metrics["bbw"] and metrics["bbw"] > 0.05
                    else "Medium" if metrics["bbw"] and metrics["bbw"] > 0.02
                    else "Low"
                ),
                "momentum": "Bullish" if metrics["change"] > 0 else "Bearish",
            },
            **extras,
            **trade_data,
        }
    except Exception as exc:
        return {"error": f"Analysis failed: {exc}", "symbol": symbol, "exchange": exchange, "timeframe": timeframe}


# ── Consecutive candle pattern scan ────────────────────────────────────────────

def scan_consecutive_candles(
    exchange: str,
    timeframe: str,
    pattern_type: str,
    candle_count: int,
    min_growth: float,
    limit: int,
) -> dict:
    """
    Scan for coins with consecutive growing/shrinking candle patterns.

    Args:
        exchange:     Validated exchange identifier.
        timeframe:    Validated TradingView interval.
        pattern_type: 'bullish' or 'bearish'.
        candle_count: Number of consecutive candles (2-5).
        min_growth:   Minimum growth percentage per candle.
        limit:        Maximum results.

    Returns:
        Dict with pattern_type, total_found, and data list.
    """
    if not _TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    symbols = load_symbols(exchange)
    if not symbols:
        return {"error": f"No symbols found for exchange: {exchange}", "exchange": exchange, "timeframe": timeframe}

    symbols = symbols[: min(limit * 3, 200)]
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")

    try:
        analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=symbols)
    except Exception as exc:
        return {"error": f"Pattern analysis failed: {exc}", "exchange": exchange, "timeframe": timeframe}

    pattern_coins: list[dict] = []

    for symbol, data in analysis.items():
        if data is None:
            continue
        try:
            indicators = data.indicators
            open_price = indicators.get("open")
            close_price = indicators.get("close")
            high_price = indicators.get("high")
            low_price = indicators.get("low")
            volume = indicators.get("volume", 0)

            if not all([open_price, close_price, high_price, low_price]):
                continue

            current_change = ((close_price - open_price) / open_price) * 100
            candle_body = abs(close_price - open_price)
            candle_range = high_price - low_price
            body_to_range_ratio = candle_body / candle_range if candle_range > 0 else 0

            rsi = indicators.get("RSI", 50)
            sma20 = indicators.get("SMA20", close_price)
            ema50 = indicators.get("EMA50", close_price)

            price_above_sma = close_price > sma20
            price_above_ema = close_price > ema50

            if pattern_type == "bullish":
                conditions = [
                    current_change > min_growth,
                    body_to_range_ratio > 0.6,
                    price_above_sma,
                    45 < rsi < 80,
                    volume > 1000,
                ]
            elif pattern_type == "bearish":
                conditions = [
                    current_change < -min_growth,
                    body_to_range_ratio > 0.6,
                    not price_above_sma,
                    20 < rsi < 55,
                    volume > 1000,
                ]
            else:
                continue

            pattern_strength = sum(conditions)
            if pattern_strength < 3:
                continue

            metrics = compute_metrics(indicators)
            pattern_coins.append({
                "symbol": symbol,
                "price": round(close_price, 6),
                "current_change": round(current_change, 3),
                "candle_body_ratio": round(body_to_range_ratio, 3),
                "pattern_strength": pattern_strength,
                "volume": volume,
                "bollinger_rating": metrics.get("rating", 0) if metrics else 0,
                "rsi": round(rsi, 2),
                "price_levels": {
                    "open": round(open_price, 6),
                    "high": round(high_price, 6),
                    "low": round(low_price, 6),
                    "close": round(close_price, 6),
                },
                "momentum_signals": {
                    "above_sma20": price_above_sma,
                    "above_ema50": price_above_ema,
                    "strong_volume": volume > 5000,
                },
            })
        except Exception:
            continue

    if pattern_type == "bullish":
        pattern_coins.sort(key=lambda x: (x["pattern_strength"], x["current_change"]), reverse=True)
    else:
        pattern_coins.sort(key=lambda x: (x["pattern_strength"], -x["current_change"]), reverse=True)

    return {
        "exchange": exchange,
        "timeframe": timeframe,
        "pattern_type": pattern_type,
        "candle_count": candle_count,
        "min_growth": min_growth,
        "total_found": len(pattern_coins),
        "data": pattern_coins[:limit],
    }


# ── Advanced candle pattern (single-TF fallback) ──────────────────────────────

def scan_advanced_candle_patterns_single_tf(
    exchange: str,
    symbols: list[str],
    base_timeframe: str,
    pattern_length: int,
    min_size_increase: float,
    limit: int,
) -> dict:
    """
    Single-timeframe fallback for advanced candle pattern analysis.

    Used when tradingview-screener is unavailable and we fall back to tradingview_ta.
    """
    if not _TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    analysis = get_multiple_analysis(screener=screener, interval=base_timeframe, symbols=symbols)
    pattern_results: list[dict] = []

    for symbol, data in analysis.items():
        if data is None:
            continue
        try:
            indicators = data.indicators
            pattern_score = calculate_candle_pattern_score(indicators, pattern_length, min_size_increase)
            if pattern_score["detected"]:
                metrics = compute_metrics(indicators)
                pattern_results.append({
                    "symbol": symbol,
                    "pattern_score": pattern_score["score"],
                    "pattern_details": pattern_score["details"],
                    "current_price": pattern_score["price"],
                    "total_change": pattern_score["total_change"],
                    "volume": indicators.get("volume", 0),
                    "bollinger_rating": metrics.get("rating", 0) if metrics else 0,
                    "technical_strength": {
                        "rsi": round(indicators.get("RSI", 50), 2),
                        "momentum": "Strong" if abs(pattern_score["total_change"]) > min_size_increase else "Moderate",
                        "volume_trend": "High" if indicators.get("volume", 0) > 10000 else "Low",
                    },
                })
        except Exception:
            continue

    pattern_results.sort(key=lambda x: (x["pattern_score"], abs(x["total_change"])), reverse=True)
    return {
        "exchange": exchange,
        "base_timeframe": base_timeframe,
        "pattern_length": pattern_length,
        "min_size_increase": min_size_increase,
        "method": "enhanced-single-timeframe",
        "total_found": len(pattern_results),
        "data": pattern_results[:limit],
    }


# ── Multi-timeframe alignment analysis ─────────────────────────────────────────

def run_multi_timeframe_analysis(
    symbol: str,
    exchange: str,
) -> dict:
    """
    Multi-timeframe alignment analysis (Weekly → Daily → 4H → 1H → 15m).

    Runs analysis across 5 timeframes and computes a directional consensus.

    Args:
        symbol:   Full symbol string with exchange prefix (e.g. 'KUCOIN:BTCUSDT').
        exchange: Validated exchange identifier.

    Returns:
        Multi-timeframe analysis dict with per-TF breakdown, alignment status,
        and trading recommendation.
    """
    from tradingview_mcp.core.services.indicators import (
        extract_extended_indicators,
        analyze_timeframe_context,
    )

    if not _TA_AVAILABLE:
        return {"error": "tradingview_ta is missing; run `uv sync`."}

    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    timeframes = ["1W", "1D", "4h", "1h", "15m"]
    tf_labels = {
        "1W": "Weekly (Trend Bias)",
        "1D": "Daily (Swing Setup)",
        "4h": "4-Hour (Refinement)",
        "1h": "1-Hour (Entry Timing)",
        "15m": "15-Min (Execution)",
    }

    tf_results: dict = {}
    alignment_scores: list[int] = []

    for tf in timeframes:
        try:
            analysis = get_multiple_analysis(screener=screener, interval=tf, symbols=[symbol])
            if symbol not in analysis or analysis[symbol] is None:
                tf_results[tf] = {"error": f"No data for {tf}"}
                continue

            data = analysis[symbol]
            indicators = data.indicators
            metrics = compute_metrics(indicators)
            extended = extract_extended_indicators(indicators)
            tf_context = analyze_timeframe_context(indicators, tf)

            bias_num = 1 if tf_context["bias"] == "Bullish" else -1 if tf_context["bias"] == "Bearish" else 0
            alignment_scores.append(bias_num)

            tf_results[tf] = {
                "label": tf_labels.get(tf, tf),
                "bias": tf_context["bias"],
                "bias_reasons": tf_context["bias_reasons"],
                "key_indicators": tf_context["key_indicators_for_timeframe"],
                "advice": tf_context["advice"],
                "price": metrics.get("price") if metrics else None,
                "change_pct": metrics.get("change") if metrics else None,
                "rsi": extended["rsi"],
                "macd_crossover": extended["macd"]["crossover"],
                "ema_trend": {
                    "ema20": extended["ema"].get("ema20"),
                    "ema50": extended["ema"].get("ema50"),
                    "ema200": extended["ema"].get("ema200"),
                },
                "volume_signal": extended["volume"]["signal"],
                "market_structure": extended["market_structure"]["trend"],
                "trend_strength": extended["market_structure"]["trend_strength"],
                "momentum_aligned": extended["market_structure"]["momentum_aligned"],
            }
        except Exception as exc:
            tf_results[tf] = {"error": str(exc)}

    total_score = sum(alignment_scores)
    all_bullish = all(s > 0 for s in alignment_scores) if alignment_scores else False
    all_bearish = all(s < 0 for s in alignment_scores) if alignment_scores else False

    if all_bullish:
        alignment, confidence, action = "FULLY ALIGNED BULLISH", "Very High", "STRONG BUY - All timeframes bullish. Look for pullback entry on 1H/15m."
    elif all_bearish:
        alignment, confidence, action = "FULLY ALIGNED BEARISH", "Very High", "STRONG SELL - All timeframes bearish. Avoid longs."
    elif total_score >= 3:
        alignment, confidence, action = "MOSTLY BULLISH", "High", "BUY - Majority of timeframes bullish. Enter on 4H/1H pullback to support."
    elif total_score <= -3:
        alignment, confidence, action = "MOSTLY BEARISH", "High", "SELL - Majority of timeframes bearish. Avoid catching the falling knife."
    elif total_score > 0:
        alignment, confidence, action = "LEAN BULLISH", "Medium", "CAUTIOUS BUY - Some bullish signals but not fully aligned. Wait for better setup."
    elif total_score < 0:
        alignment, confidence, action = "LEAN BEARISH", "Medium", "CAUTIOUS SELL - Some bearish signals. Reduce position or wait."
    else:
        alignment, confidence, action = "MIXED/RANGING", "Low", "HOLD/NO TRADE - Timeframes conflict. Wait for alignment."

    higher_tf_bias = alignment_scores[0] if alignment_scores else 0
    divergent_tfs = [
        timeframes[i]
        for i, score in enumerate(alignment_scores)
        if score != 0 and score != higher_tf_bias and higher_tf_bias != 0
    ]

    return {
        "symbol": symbol,
        "exchange": exchange,
        "analysis_type": "Multi-Timeframe Alignment",
        "timeframes": tf_results,
        "alignment": {
            "status": alignment,
            "confidence": confidence,
            "net_score": total_score,
            "scores_by_tf": dict(zip(timeframes, alignment_scores)),
            "divergent_timeframes": divergent_tfs,
        },
        "recommendation": {
            "action": action,
            "entry_timeframe": "1H or 4H pullback" if total_score > 0 else "Wait for alignment",
            "rules": [
                "Weekly sets BIAS (direction only, not entries)",
                "Daily finds SETUP (swing level, confluence)",
                "4H refines entry zone",
                "1H/15m triggers entry with tight stop",
                "Never trade against Weekly + Daily combined direction",
            ],
        },
    }
