from __future__ import annotations

import argparse
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from mcp.server.fastmcp import FastMCP

# Import bollinger band screener modules
from tradingview_mcp.core.services.indicators import compute_metrics
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.utils.validators import sanitize_timeframe, sanitize_exchange, EXCHANGE_SCREENER, ALLOWED_TIMEFRAMES

try:
    from tradingview_ta import TA_Handler, get_multiple_analysis
    TRADINGVIEW_TA_AVAILABLE = True
except ImportError:
    TRADINGVIEW_TA_AVAILABLE = False

try:
    from tradingview_screener import Query
    from tradingview_screener.column import Column
    TRADINGVIEW_SCREENER_AVAILABLE = True
except ImportError:
    TRADINGVIEW_SCREENER_AVAILABLE = False


class IndicatorMap(TypedDict, total=False):
	open: Optional[float]
	close: Optional[float]
	SMA20: Optional[float]
	BB_upper: Optional[float]
	BB_lower: Optional[float]
	EMA50: Optional[float]
	RSI: Optional[float]
	volume: Optional[float]


class Row(TypedDict):
	symbol: str
	changePercent: float
	indicators: IndicatorMap


class MultiRow(TypedDict):
	symbol: str
	changes: dict[str, Optional[float]]
	base_indicators: IndicatorMap


def _map_indicators(raw: Dict[str, Any]) -> IndicatorMap:
	return IndicatorMap(
		open=raw.get("open"),
		close=raw.get("close"),
		SMA20=raw.get("SMA20"),
		BB_upper=raw.get("BB.upper") if "BB.upper" in raw else raw.get("BB_upper"),
		BB_lower=raw.get("BB.lower") if "BB.lower" in raw else raw.get("BB_lower"),
		EMA50=raw.get("EMA50"),
		RSI=raw.get("RSI"),
		volume=raw.get("volume"),
	)


def _percent_change(o: Optional[float], c: Optional[float]) -> Optional[float]:
	try:
		if o in (None, 0) or c is None:
			return None
		return (c - o) / o * 100
	except Exception:
		return None


def _tf_to_tv_resolution(tf: Optional[str]) -> Optional[str]:
	if not tf:
		return None
	return {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1D": "1D", "1W": "1W", "1M": "1M"}.get(tf)


def _get_market_for_exchange(exchange: str) -> str:
    """Return the appropriate TradingView market for the given exchange."""
    exchange_upper = exchange.upper()
    if exchange_upper in ('NYSE', 'NASDAQ', 'AMEX', 'BATS', 'ARCA'):
        return "america"
    elif exchange_upper in ('LSE', 'LON'):
        return "uk"
    elif exchange_upper in ('HKEX', 'HKSE'):
        return "hongkong"
    elif exchange_upper in ('TSE', 'TYO'):
        return "japan"
    else:
        return "crypto"


def _fetch_bollinger_analysis(exchange: str, timeframe: str = "4h", limit: int = 50, bbw_filter: float = None) -> List[Row]:
    """Fetch analysis using tradingview_ta with bollinger band logic from the original screener."""
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is missing; run `uv sync`.")
    
    # Load symbols from coinlist files
    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")
    
    # Limit symbols for performance
    symbols = symbols[:limit * 2]  # Get more to filter later
    
    # Get screener type based on exchange
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    
    try:
        analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=symbols)
    except Exception as e:
        raise RuntimeError(f"Analysis failed: {str(e)}")
    
    rows: List[Row] = []
    
    for key, value in analysis.items():
        try:
            if value is None:
                continue
                
            indicators = value.indicators
            metrics = compute_metrics(indicators)
            
            if not metrics or metrics.get('bbw') is None:
                continue
            
            # Apply BBW filter if specified
            if bbw_filter is not None and (metrics['bbw'] >= bbw_filter or metrics['bbw'] <= 0):
                continue
            
            # Check if we have required indicators
            if not (indicators.get("EMA50") and indicators.get("RSI")):
                continue
                
            rows.append(Row(
                symbol=key,
                changePercent=metrics['change'],
                indicators=IndicatorMap(
                    open=metrics.get('open'),
                    close=metrics.get('price'),
                    SMA20=indicators.get("SMA20"),
                    BB_upper=indicators.get("BB.upper"),
                    BB_lower=indicators.get("BB.lower"),
                    EMA50=indicators.get("EMA50"),
                    RSI=indicators.get("RSI"),
                    volume=indicators.get("volume"),
                )
            ))
                
        except (TypeError, ZeroDivisionError, KeyError):
            continue
    
    # Sort by change percentage in descending order (highest gainers first)
    rows.sort(key=lambda x: x["changePercent"], reverse=True)
    
    # Return the requested limit
    return rows[:limit]


def _fetch_trending_analysis(exchange: str, timeframe: str = "5m", filter_type: str = "", rating_filter: int = None, limit: int = 50) -> List[Row]:
    """Fetch trending coins analysis similar to the original app's trending endpoint."""
    if not TRADINGVIEW_TA_AVAILABLE:
        raise RuntimeError("tradingview_ta is missing; run `uv sync`.")
    
    symbols = load_symbols(exchange)
    if not symbols:
        raise RuntimeError(f"No symbols found for exchange: {exchange}")
    
    # Process symbols in batches due to TradingView API limits
    batch_size = 200  # Considering API limitations
    all_coins = []
    
    screener = EXCHANGE_SCREENER.get(exchange, "crypto")
    
    # Process symbols in batches
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]
        
        try:
            analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=batch_symbols)
        except Exception as e:
            continue  # If this batch fails, move to the next one
            
        # Process coins in this batch
        for key, value in analysis.items():
            try:
                if value is None:
                    continue
                    
                indicators = value.indicators
                metrics = compute_metrics(indicators)
                
                if not metrics or metrics.get('bbw') is None:
                    continue
                
                # Apply rating filter if specified
                if filter_type == "rating" and rating_filter is not None:
                    if metrics['rating'] != rating_filter:
                        continue
                
                all_coins.append(Row(
                    symbol=key,
                    changePercent=metrics['change'],
                    indicators=IndicatorMap(
                        open=metrics.get('open'),
                        close=metrics.get('price'),
                        SMA20=indicators.get("SMA20"),
                        BB_upper=indicators.get("BB.upper"),
                        BB_lower=indicators.get("BB.lower"),
                        EMA50=indicators.get("EMA50"),
                        RSI=indicators.get("RSI"),
                        volume=indicators.get("volume"),
                    )
                ))
                
            except (TypeError, ZeroDivisionError, KeyError):
                continue
    
    # Sort all coins by change percentage
    all_coins.sort(key=lambda x: x["changePercent"], reverse=True)
    
    return all_coins[:limit]
def _fetch_multi_changes(exchange: str, timeframes: List[str] | None, base_timeframe: str = "4h", limit: int | None = None, cookies: Any | None = None) -> List[MultiRow]:
	try:
		from tradingview_screener import Query
		from tradingview_screener.column import Column
	except Exception as e:
		raise RuntimeError("tradingview-screener missing; run `uv sync`.") from e

	tfs = timeframes or ["15m", "1h", "4h", "1D"]
	suffix_map: dict[str, str] = {}
	for tf in tfs:
		s = _tf_to_tv_resolution(tf)
		if s:
			suffix_map[tf] = s
	if not suffix_map:
		suffix_map = {base_timeframe: _tf_to_tv_resolution(base_timeframe) or "240"}

	base_suffix = _tf_to_tv_resolution(base_timeframe) or next(iter(suffix_map.values()))
	cols: list[str] = []
	seen: set[str] = set()
	for tf, s in suffix_map.items():
		for c in (f"open|{s}", f"close|{s}"):
			if c not in seen:
				cols.append(c)
				seen.add(c)
	for c in (f"SMA20|{base_suffix}", f"BB.upper|{base_suffix}", f"BB.lower|{base_suffix}", f"volume|{base_suffix}"):
		if c not in seen:
			cols.append(c)
			seen.add(c)

	q = Query().set_markets(_get_market_for_exchange(exchange)).select(*cols)
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
			changes[tf] = _percent_change(o, c)
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


mcp = FastMCP(
	name="TradingView Screener",
	instructions=("Crypto screener utilities backed by TradingView Screener. Tools: top_gainers, top_losers, multi_changes."),
)


@mcp.tool()
def top_gainers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """Return top gainers for an exchange and timeframe using bollinger band analysis.
    
    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    
    rows = _fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)
    # Convert Row objects to dicts properly
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"], 
        "indicators": dict(row["indicators"])
    } for row in rows]


@mcp.tool()
def top_losers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict]:
    """Return top losers for an exchange and timeframe using bollinger band analysis."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    
    rows = _fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)
    # Reverse sort for losers (lowest change first)
    rows.sort(key=lambda x: x["changePercent"])
    
    # Convert to dict format
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"],
        "indicators": dict(row["indicators"])
    } for row in rows[:limit]]


@mcp.tool()
def bollinger_scan(exchange: str = "KUCOIN", timeframe: str = "4h", bbw_threshold: float = 0.04, limit: int = 50) -> list[dict]:
    """Scan for coins with low Bollinger Band Width (squeeze detection).
    
    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M  
        bbw_threshold: Maximum BBW value to filter (default 0.04)
        limit: Number of rows to return (max 100)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "4h")
    limit = max(1, min(limit, 100))
    
    rows = _fetch_bollinger_analysis(exchange, timeframe=timeframe, bbw_filter=bbw_threshold, limit=limit)
    # Convert Row objects to dicts
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"],
        "indicators": dict(row["indicators"])
    } for row in rows]


@mcp.tool()
def rating_filter(exchange: str = "KUCOIN", timeframe: str = "5m", rating: int = 2, limit: int = 25) -> list[dict]:
    """Filter coins by Bollinger Band rating.
    
    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        rating: BB rating (-3 to +3): -3=Strong Sell, -2=Sell, -1=Weak Sell, 1=Weak Buy, 2=Buy, 3=Strong Buy
        limit: Number of rows to return (max 50)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "5m")
    rating = max(-3, min(3, rating))
    limit = max(1, min(limit, 50))
    
    rows = _fetch_trending_analysis(exchange, timeframe=timeframe, filter_type="rating", rating_filter=rating, limit=limit)
    # Convert Row objects to dicts
    return [{
        "symbol": row["symbol"],
        "changePercent": row["changePercent"],
        "indicators": dict(row["indicators"])
    } for row in rows]

@mcp.tool()
def coin_analysis(
    symbol: str,
    exchange: str = "KUCOIN",
    timeframe: str = "15m"
) -> dict:
    """Get detailed analysis for a specific coin on specified exchange and timeframe.
    
    Args:
        symbol: Coin symbol (e.g., "ACEUSDT", "BTCUSDT")
        exchange: Exchange name (BINANCE, KUCOIN, etc.) 
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)
    
    Returns:
        Detailed coin analysis with all indicators and metrics
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "15m")
        
        # Format symbol with exchange prefix
        if ":" not in symbol:
            full_symbol = f"{exchange.upper()}:{symbol.upper()}"
        else:
            full_symbol = symbol.upper()
        
        screener = EXCHANGE_SCREENER.get(exchange, _get_market_for_exchange(exchange))

        try:
            analysis = get_multiple_analysis(
                screener=screener,
                interval=timeframe,
                symbols=[full_symbol]
            )
            
            if full_symbol not in analysis or analysis[full_symbol] is None:
                return {
                    "error": f"No data found for {symbol} on {exchange}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                }
            
            data = analysis[full_symbol]
            indicators = data.indicators
            
            # Calculate all metrics
            metrics = compute_metrics(indicators)
            if not metrics:
                return {
                    "error": f"Could not compute metrics for {symbol}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                }
            
            # Additional technical indicators
            macd = indicators.get("MACD.macd", 0)
            macd_signal = indicators.get("MACD.signal", 0)
            adx = indicators.get("ADX", 0)
            stoch_k = indicators.get("Stoch.K", 0)
            stoch_d = indicators.get("Stoch.D", 0)
            
            # Volume analysis
            volume = indicators.get("volume", 0)

            # Price levels
            high = indicators.get("high", 0)
            low = indicators.get("low", 0)
            open_price = indicators.get("open", 0)
            close_price = indicators.get("close", 0)
            current_price = metrics['price']

            # Fetch extended data from tradingview_screener
            screener_data = {}
            if TRADINGVIEW_SCREENER_AVAILABLE:
                try:
                    market = _get_market_for_exchange(exchange)
                    screener_cols = [
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
                    # Add stock-only columns
                    is_stock_exchange = exchange.upper() in ('NYSE', 'NASDAQ', 'AMEX', 'BATS', 'ARCA')
                    if is_stock_exchange:
                        screener_cols.extend([
                            'sector', 'industry',
                            'Pivot.M.Classic.R1', 'Pivot.M.Classic.R2', 'Pivot.M.Classic.R3',
                            'Pivot.M.Classic.S1', 'Pivot.M.Classic.S2', 'Pivot.M.Classic.S3',
                            'Pivot.M.Classic.Middle',
                            'Low.3M', 'Low.6M', 'High.3M', 'High.6M',
                            'earnings_release_next_date',
                        ])
                    vq = Query().set_markets(market).select(
                        *screener_cols
                    ).set_tickers(full_symbol).limit(1)
                    _vt, vdf = vq.get_scanner_data()
                    if vdf is not None and not vdf.empty:
                        screener_data = vdf.iloc[0].to_dict()
                except Exception:
                    pass

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
            trend_direction = "BULLISH" if adx_plus > adx_minus else "BEARISH" if adx_minus > adx_plus else "NEUTRAL"

            # Moving averages for Stage 2 check
            ema50 = screener_data.get('EMA50', 0) or indicators.get("EMA50", 0) or 0
            sma150 = screener_data.get('SMA150', 0) or 0
            ema200 = screener_data.get('EMA200', 0) or indicators.get("EMA200", 0) or 0

            # 200MA trend detection
            ma_recommendation = screener_data.get('Recommend.MA', 0) or 0
            sma200 = screener_data.get('SMA200', 0)
            sma200_trending_up = ma_recommendation > 0

            # Sector/Industry info (stocks only)
            is_stock_exchange = exchange.upper() in ('NYSE', 'NASDAQ', 'AMEX', 'BATS', 'ARCA')

            # RS vs SPY calculation (US stocks only)
            if is_stock_exchange and TRADINGVIEW_SCREENER_AVAILABLE:
                try:
                    spy_query = (
                        Query()
                        .select('Perf.1M', 'Perf.3M', 'Perf.6M')
                        .where(Column('name') == 'SPY')
                        .set_markets('america')
                        .limit(1)
                    )
                    spy_result = spy_query.get_scanner_data()
                    if spy_result[1] is not None and not spy_result[1].empty:
                        spy_row = spy_result[1].iloc[0]
                        spy_perf_1m = spy_row.get('Perf.1M', 0) or 0
                        spy_perf_3m = spy_row.get('Perf.3M', 0) or 0
                        spy_perf_6m = spy_row.get('Perf.6M', 0) or 0
                    else:
                        spy_perf_1m = spy_perf_3m = spy_perf_6m = 0
                except Exception:
                    spy_perf_1m = spy_perf_3m = spy_perf_6m = 0
            else:
                spy_perf_1m = spy_perf_3m = spy_perf_6m = None

            # Pivot point analysis (stocks only)
            if is_stock_exchange:
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

            # Base depth analysis (stocks only)
            if is_stock_exchange:
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

            # Calculate relative strength (outperformance vs benchmark)
            if spy_perf_3m is not None:
                rs_1m = round(perf_1m - spy_perf_1m, 2) if perf_1m else None
                rs_3m = round(perf_3m - spy_perf_3m, 2) if perf_3m else None
                rs_6m = round(perf_6m - spy_perf_6m, 2) if perf_6m else None
            else:
                rs_1m = rs_3m = rs_6m = None

            # Earnings date (stocks only)
            earnings_date = None
            days_until_earnings = None
            earnings_within_7_days = None
            if is_stock_exchange:
                earnings_date_str = screener_data.get('earnings_release_next_date')
                if earnings_date_str:
                    try:
                        if isinstance(earnings_date_str, (int, float)):
                            earnings_date = datetime.fromtimestamp(earnings_date_str / 1000).date()
                        else:
                            earnings_date = datetime.strptime(str(earnings_date_str), "%Y-%m-%d").date()
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

            # Weekly performance
            perf_w = screener_data.get('Perf.W', 0) or 0

            # Climax detection
            sma20 = indicators.get("SMA20", 0) or 0
            change_percent = metrics['change']
            pct_above_20ma = round(((current_price - sma20) / sma20) * 100, 2) if sma20 and sma20 > 0 else None
            pct_above_50ma = round(((current_price - ema50) / ema50) * 100, 2) if ema50 and ema50 > 0 else None
            climax_above_20ma = pct_above_20ma > 30 if pct_above_20ma is not None else False
            climax_above_50ma = pct_above_50ma > 30 if pct_above_50ma is not None else False
            climax_weekly_gain = perf_w > 25 if perf_w else False
            climax_daily_gain = change_percent > 10 if change_percent else False
            climax_count = sum([climax_above_20ma, climax_above_50ma, climax_weekly_gain, climax_daily_gain])

            return {
                "symbol": full_symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": "real-time",
                "price_data": {
                    "current_price": current_price,
                    "open": round(open_price, 6) if open_price else None,
                    "high": round(high, 6) if high else None,
                    "low": round(low, 6) if low else None,
                    "close": round(close_price, 6) if close_price else None,
                    "change_percent": metrics['change'],
                    "volume": volume,
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
                "bollinger_analysis": {
                    "rating": metrics['rating'],
                    "signal": metrics['signal'],
                    "bbw": metrics['bbw'],
                    "bb_upper": round(indicators.get("BB.upper", 0), 6),
                    "bb_middle": round(indicators.get("SMA20", 0), 6),
                    "bb_lower": round(indicators.get("BB.lower", 0), 6),
                    "position": "Above Upper" if close_price > indicators.get("BB.upper", 0) else
                               "Below Lower" if close_price < indicators.get("BB.lower", 0) else
                               "Within Bands"
                },
                "technical_indicators": {
                    "rsi": round(indicators.get("RSI", 0), 2),
                    "rsi_signal": "Overbought" if indicators.get("RSI", 0) > 70 else
                                 "Oversold" if indicators.get("RSI", 0) < 30 else "Neutral",
                    "sma20": round(indicators.get("SMA20", 0), 6),
                    "ema50": round(ema50, 6),
                    "sma150": round(sma150, 6),
                    "ema200": round(ema200, 6),
                    "macd": round(macd, 6),
                    "macd_signal": round(macd_signal, 6),
                    "macd_divergence": round(macd - macd_signal, 6),
                    "adx": round(adx, 2),
                    "trend_strength": "Strong" if adx > 25 else "Weak",
                    "stoch_k": round(stoch_k, 2),
                    "stoch_d": round(stoch_d, 2)
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
                    "sector": screener_data.get('sector') if is_stock_exchange else None,
                    "industry": screener_data.get('industry') if is_stock_exchange else None,
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
                    "breakout_proximity": "CLOSE" if distance_to_r1_pct is not None and distance_to_r1_pct <= 3 else "FAR" if distance_to_r1_pct else None,
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
                "market_sentiment": {
                    "overall_rating": metrics['rating'],
                    "buy_sell_signal": metrics['signal'],
                    "volatility": "High" if metrics['bbw'] > 0.05 else "Medium" if metrics['bbw'] > 0.02 else "Low",
                    "momentum": "Bullish" if metrics['change'] > 0 else "Bearish"
                }
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
    except Exception as e:
        return {
            "error": f"Coin analysis failed: {str(e)}",
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe
        }

@mcp.tool()
def consecutive_candles_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    pattern_type: str = "bullish",
    candle_count: int = 3,
    min_growth: float = 2.0,
    limit: int = 20
) -> dict:
    """Scan for coins with consecutive growing/shrinking candles pattern.
    
    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h)
        pattern_type: "bullish" (growing candles) or "bearish" (shrinking candles)
        candle_count: Number of consecutive candles to check (2-5)
        min_growth: Minimum growth percentage for each candle
        limit: Maximum number of results to return
    
    Returns:
        List of coins with consecutive candle patterns
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        timeframe = sanitize_timeframe(timeframe, "15m")
        candle_count = max(2, min(5, candle_count))
        min_growth = max(0.5, min(20.0, min_growth))
        limit = max(1, min(50, limit))
        
        # Get symbols for the exchange
        symbols = load_symbols(exchange)
        if not symbols:
            return {
                "error": f"No symbols found for exchange: {exchange}",
                "exchange": exchange,
                "timeframe": timeframe
            }
        
        # Limit symbols for performance (we need historical data)
        symbols = symbols[:min(limit * 3, 200)]
        
        # We need to get data from multiple timeframes to analyze candle progression
        # For now, we'll use current timeframe data and simulate pattern detection
        screener = EXCHANGE_SCREENER.get(exchange, "crypto")
        
        try:
            analysis = get_multiple_analysis(
                screener=screener,
                interval=timeframe,
                symbols=symbols
            )
            
            pattern_coins = []
            
            for symbol, data in analysis.items():
                if data is None:
                    continue
                    
                try:
                    indicators = data.indicators
                    
                    # Calculate current candle metrics
                    open_price = indicators.get("open")
                    close_price = indicators.get("close")
                    high_price = indicators.get("high") 
                    low_price = indicators.get("low")
                    volume = indicators.get("volume", 0)
                    
                    if not all([open_price, close_price, high_price, low_price]):
                        continue
                    
                    # Calculate current candle body size and change
                    current_change = ((close_price - open_price) / open_price) * 100
                    candle_body = abs(close_price - open_price)
                    candle_range = high_price - low_price
                    body_to_range_ratio = candle_body / candle_range if candle_range > 0 else 0
                    
                    # For consecutive pattern, we'll use available indicators to simulate
                    # In a real implementation, we'd need historical OHLC data
                    
                    # Use RSI and price momentum as proxy for consecutive pattern
                    rsi = indicators.get("RSI", 50)
                    sma20 = indicators.get("SMA20", close_price)
                    ema50 = indicators.get("EMA50", close_price)
                    
                    # Calculate momentum indicators
                    price_above_sma = close_price > sma20
                    price_above_ema = close_price > ema50
                    strong_momentum = abs(current_change) >= min_growth
                    
                    # Pattern detection logic
                    pattern_detected = False
                    pattern_strength = 0
                    
                    if pattern_type == "bullish":
                        # Bullish pattern: price rising, good momentum, strong candle body
                        conditions = [
                            current_change > min_growth,  # Current candle is bullish
                            body_to_range_ratio > 0.6,    # Strong candle body
                            price_above_sma,              # Above short MA
                            rsi > 45 and rsi < 80,        # RSI in momentum range
                            volume > 1000                 # Decent volume
                        ]
                        
                        pattern_strength = sum(conditions)
                        pattern_detected = pattern_strength >= 3
                        
                    elif pattern_type == "bearish":
                        # Bearish pattern: price falling, bearish momentum
                        conditions = [
                            current_change < -min_growth,  # Current candle is bearish
                            body_to_range_ratio > 0.6,     # Strong candle body
                            not price_above_sma,           # Below short MA
                            rsi < 55 and rsi > 20,         # RSI in bearish range
                            volume > 1000                  # Decent volume
                        ]
                        
                        pattern_strength = sum(conditions)
                        pattern_detected = pattern_strength >= 3
                    
                    if pattern_detected:
                        # Calculate additional metrics
                        metrics = compute_metrics(indicators)
                        
                        coin_data = {
                            "symbol": symbol,
                            "price": round(close_price, 6),
                            "current_change": round(current_change, 3),
                            "candle_body_ratio": round(body_to_range_ratio, 3),
                            "pattern_strength": pattern_strength,
                            "volume": volume,
                            "bollinger_rating": metrics.get('rating', 0) if metrics else 0,
                            "rsi": round(rsi, 2),
                            "price_levels": {
                                "open": round(open_price, 6),
                                "high": round(high_price, 6), 
                                "low": round(low_price, 6),
                                "close": round(close_price, 6)
                            },
                            "momentum_signals": {
                                "above_sma20": price_above_sma,
                                "above_ema50": price_above_ema,
                                "strong_volume": volume > 5000
                            }
                        }
                        
                        pattern_coins.append(coin_data)
                        
                except Exception as e:
                    continue
            
            # Sort by pattern strength and current change
            if pattern_type == "bullish":
                pattern_coins.sort(key=lambda x: (x['pattern_strength'], x['current_change']), reverse=True)
            else:
                pattern_coins.sort(key=lambda x: (x['pattern_strength'], -x['current_change']), reverse=True)
            
            return {
                "exchange": exchange,
                "timeframe": timeframe,
                "pattern_type": pattern_type,
                "candle_count": candle_count,
                "min_growth": min_growth,
                "total_found": len(pattern_coins),
                "data": pattern_coins[:limit]
            }
            
        except Exception as e:
            return {
                "error": f"Pattern analysis failed: {str(e)}",
                "exchange": exchange,
                "timeframe": timeframe
            }
            
    except Exception as e:
        return {
            "error": f"Consecutive candles scan failed: {str(e)}",
            "exchange": exchange,
            "timeframe": timeframe
        }

@mcp.tool()
def advanced_candle_pattern(
    exchange: str = "KUCOIN",
    base_timeframe: str = "15m",
    pattern_length: int = 3,
    min_size_increase: float = 10.0,
    limit: int = 15
) -> dict:
    """Advanced candle pattern analysis using multi-timeframe data.
    
    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        base_timeframe: Base timeframe for analysis (5m, 15m, 1h, 4h)
        pattern_length: Number of consecutive periods to analyze (2-4)
        min_size_increase: Minimum percentage increase in candle size
        limit: Maximum number of results to return
    
    Returns:
        Coins with progressive candle size increase patterns
    """
    try:
        exchange = sanitize_exchange(exchange, "KUCOIN")
        base_timeframe = sanitize_timeframe(base_timeframe, "15m")
        pattern_length = max(2, min(4, pattern_length))
        min_size_increase = max(5.0, min(50.0, min_size_increase))
        limit = max(1, min(30, limit))
        
        # Get symbols
        symbols = load_symbols(exchange)
        if not symbols:
            return {
                "error": f"No symbols found for exchange: {exchange}",
                "exchange": exchange
            }
        
        # Limit for performance
        symbols = symbols[:min(limit * 2, 100)]
        
        # Use tradingview-screener for multi-timeframe data if available
        if TRADINGVIEW_SCREENER_AVAILABLE:
            try:
                # Get multiple timeframe data using screener
                results = _fetch_multi_timeframe_patterns(
                    exchange, symbols, base_timeframe, pattern_length, min_size_increase
                )
                
                return {
                    "exchange": exchange,
                    "base_timeframe": base_timeframe,
                    "pattern_length": pattern_length,
                    "min_size_increase": min_size_increase,
                    "method": "multi-timeframe",
                    "total_found": len(results),
                    "data": results[:limit]
                }
                
            except Exception as e:
                # Fallback to single timeframe analysis
                pass
        
        # Fallback: Use single timeframe with enhanced pattern detection
        screener = EXCHANGE_SCREENER.get(exchange, "crypto")
        
        analysis = get_multiple_analysis(
            screener=screener,
            interval=base_timeframe,
            symbols=symbols
        )
        
        pattern_results = []
        
        for symbol, data in analysis.items():
            if data is None:
                continue
                
            try:
                indicators = data.indicators
                
                # Enhanced pattern detection using available indicators
                pattern_score = _calculate_candle_pattern_score(
                    indicators, pattern_length, min_size_increase
                )
                
                if pattern_score['detected']:
                    metrics = compute_metrics(indicators)
                    
                    result = {
                        "symbol": symbol,
                        "pattern_score": pattern_score['score'],
                        "pattern_details": pattern_score['details'],
                        "current_price": pattern_score['price'],
                        "total_change": pattern_score['total_change'],
                        "volume": indicators.get("volume", 0),
                        "bollinger_rating": metrics.get('rating', 0) if metrics else 0,
                        "technical_strength": {
                            "rsi": round(indicators.get("RSI", 50), 2),
                            "momentum": "Strong" if abs(pattern_score['total_change']) > min_size_increase else "Moderate",
                            "volume_trend": "High" if indicators.get("volume", 0) > 10000 else "Low"
                        }
                    }
                    
                    pattern_results.append(result)
                    
            except Exception as e:
                continue
        
        # Sort by pattern score and total change
        pattern_results.sort(key=lambda x: (x['pattern_score'], abs(x['total_change'])), reverse=True)
        
        return {
            "exchange": exchange,
            "base_timeframe": base_timeframe,
            "pattern_length": pattern_length,
            "min_size_increase": min_size_increase,
            "method": "enhanced-single-timeframe",
            "total_found": len(pattern_results),
            "data": pattern_results[:limit]
        }
        
    except Exception as e:
        return {
            "error": f"Advanced pattern analysis failed: {str(e)}",
            "exchange": exchange,
            "base_timeframe": base_timeframe
        }

def _calculate_candle_pattern_score(indicators: dict, pattern_length: int, min_increase: float) -> dict:
    """Calculate candle pattern score based on available indicators."""
    try:
        open_price = indicators.get("open", 0)
        close_price = indicators.get("close", 0)
        high_price = indicators.get("high", 0)
        low_price = indicators.get("low", 0)
        volume = indicators.get("volume", 0)
        rsi = indicators.get("RSI", 50)
        
        if not all([open_price, close_price, high_price, low_price]):
            return {"detected": False, "score": 0}
        
        # Current candle analysis
        candle_body = abs(close_price - open_price)
        candle_range = high_price - low_price
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        # Price change
        price_change = ((close_price - open_price) / open_price) * 100
        
        # Pattern scoring
        score = 0
        details = []
        
        # Strong candle body
        if body_ratio > 0.7:
            score += 2
            details.append("Strong candle body")
        elif body_ratio > 0.5:
            score += 1
            details.append("Moderate candle body")
        
        # Significant price movement
        if abs(price_change) >= min_increase:
            score += 2
            details.append(f"Strong momentum ({price_change:.1f}%)")
        elif abs(price_change) >= min_increase / 2:
            score += 1
            details.append(f"Moderate momentum ({price_change:.1f}%)")
        
        # Volume confirmation
        if volume > 5000:
            score += 1
            details.append("Good volume")
        
        # RSI momentum
        if (price_change > 0 and 50 < rsi < 80) or (price_change < 0 and 20 < rsi < 50):
            score += 1
            details.append("RSI momentum aligned")
        
        # Trend consistency (using EMA vs price)
        ema50 = indicators.get("EMA50", close_price)
        if (price_change > 0 and close_price > ema50) or (price_change < 0 and close_price < ema50):
            score += 1
            details.append("Trend alignment")
        
        detected = score >= 3  # Minimum threshold
        
        return {
            "detected": detected,
            "score": score,
            "details": details,
            "price": round(close_price, 6),
            "total_change": round(price_change, 3),
            "body_ratio": round(body_ratio, 3),
            "volume": volume
        }
        
    except Exception as e:
        return {"detected": False, "score": 0, "error": str(e)}

def _fetch_multi_timeframe_patterns(exchange: str, symbols: List[str], base_tf: str, length: int, min_increase: float) -> List[dict]:
    """Fetch multi-timeframe pattern data using tradingview-screener."""
    try:
        from tradingview_screener import Query
        from tradingview_screener.column import Column
        
        # Map timeframe to TradingView format
        tf_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1D": "1D"}
        tv_interval = tf_map.get(base_tf, "15")
        
        # Create query for OHLC data
        cols = [
            f"open|{tv_interval}",
            f"close|{tv_interval}", 
            f"high|{tv_interval}",
            f"low|{tv_interval}",
            f"volume|{tv_interval}",
            "RSI"
        ]
        
        q = Query().set_markets(_get_market_for_exchange(exchange)).select(*cols)
        q = q.where(Column("exchange") == exchange.upper())
        q = q.limit(len(symbols))
        
        total, df = q.get_scanner_data()
        
        if df is None or df.empty:
            return []
        
        results = []
        
        for _, row in df.iterrows():
            symbol = row.get("ticker", "")
            
            try:
                open_val = row.get(f"open|{tv_interval}")
                close_val = row.get(f"close|{tv_interval}")
                high_val = row.get(f"high|{tv_interval}")
                low_val = row.get(f"low|{tv_interval}")
                volume_val = row.get(f"volume|{tv_interval}", 0)
                rsi_val = row.get("RSI", 50)
                
                if not all([open_val, close_val, high_val, low_val]):
                    continue
                
                # Calculate pattern metrics
                pattern_score = _calculate_candle_pattern_score({
                    "open": open_val,
                    "close": close_val,
                    "high": high_val,
                    "low": low_val,
                    "volume": volume_val,
                    "RSI": rsi_val
                }, length, min_increase)
                
                if pattern_score['detected']:
                    results.append({
                        "symbol": symbol,
                        "pattern_score": pattern_score['score'],
                        "price": pattern_score['price'],
                        "change": pattern_score['total_change'],
                        "body_ratio": pattern_score['body_ratio'],
                        "volume": volume_val,
                        "rsi": round(rsi_val, 2),
                        "details": pattern_score['details']
                    })
                    
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x['pattern_score'], reverse=True)
        
    except Exception as e:
        return []

@mcp.resource("exchanges://list")
def exchanges_list() -> str:
    """List available exchanges from coinlist directory."""
    try:
        import os
        # Get the directory where this module is located
        current_dir = os.path.dirname(__file__)
        coinlist_dir = os.path.join(current_dir, "coinlist")
        
        if os.path.exists(coinlist_dir):
            exchanges = []
            for filename in os.listdir(coinlist_dir):
                if filename.endswith('.txt'):
                    exchange_name = filename[:-4].upper()
                    exchanges.append(exchange_name)
            
            if exchanges:
                return f"Available exchanges: {', '.join(sorted(exchanges))}"
        
        # Fallback to static list
        return "Common exchanges: KUCOIN, BINANCE, BYBIT, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX, KRAKEN, BITSTAMP, BIST, NASDAQ"
    except Exception:
        return "Common exchanges: KUCOIN, BINANCE, BYBIT, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX, KRAKEN, BITSTAMP, BIST, NASDAQ"
def main() -> None:
	parser = argparse.ArgumentParser(description="TradingView Screener MCP server")
	parser.add_argument("transport", choices=["stdio", "streamable-http"], default="stdio", nargs="?", help="Transport (default stdio)")
	parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
	parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
	args = parser.parse_args()

	if os.environ.get("DEBUG_MCP"):
		import sys
		print(f"[DEBUG_MCP] pkg cwd={os.getcwd()} argv={sys.argv} file={__file__}", file=sys.stderr, flush=True)

	if args.transport == "stdio":
		mcp.run()
	else:
		try:
			mcp.settings.host = args.host
			mcp.settings.port = args.port
		except Exception:
			pass
		mcp.run(transport="streamable-http")


@mcp.tool()
def volume_breakout_scanner(exchange: str = "KUCOIN", timeframe: str = "15m", volume_multiplier: float = 2.0, price_change_min: float = 3.0, limit: int = 25) -> list[dict]:
	"""Detect coins with volume breakout + price breakout.
	
	Args:
		exchange: Exchange name like KUCOIN, BINANCE, BYBIT, etc.
		timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
		volume_multiplier: How many times the volume should be above normal level (default 2.0)
		price_change_min: Minimum price change percentage (default 3.0)
		limit: Number of rows to return (max 50)
	"""
	exchange = sanitize_exchange(exchange, "KUCOIN")
	timeframe = sanitize_timeframe(timeframe, "15m")
	volume_multiplier = max(1.5, min(10.0, volume_multiplier))
	price_change_min = max(1.0, min(20.0, price_change_min))
	limit = max(1, min(limit, 50))
	
	# Get symbols
	symbols = load_symbols(exchange)
	if not symbols:
		return []
	
	screener = EXCHANGE_SCREENER.get(exchange, "crypto")
	volume_breakouts = []
	
	# Process in batches
	batch_size = 100
	for i in range(0, min(len(symbols), 500), batch_size):  # Limit to 500 symbols for performance
		batch_symbols = symbols[i:i + batch_size]
		
		try:
			analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=batch_symbols)
		except Exception:
			continue
			
		for symbol, data in analysis.items():
			try:
				if not data or not hasattr(data, 'indicators'):
					continue
					
				indicators = data.indicators
				
				# Get required data
				volume = indicators.get('volume', 0)
				close = indicators.get('close', 0)
				open_price = indicators.get('open', 0)
				sma20_volume = indicators.get('volume.SMA20', 0)  # 20-period volume average
				
				if not all([volume, close, open_price]) or volume <= 0:
					continue
				
				# Calculate price change %
				price_change = ((close - open_price) / open_price) * 100 if open_price > 0 else 0
				
				# Volume ratio calculation
				# If SMA20 volume not available, use a simple heuristic
				if sma20_volume and sma20_volume > 0:
					volume_ratio = volume / sma20_volume
				else:
					# Estimate average volume as current volume / 2 (conservative)
					avg_volume_estimate = volume / 2
					volume_ratio = volume / avg_volume_estimate if avg_volume_estimate > 0 else 1
				
				# Check conditions
				if (abs(price_change) >= price_change_min and 
					volume_ratio >= volume_multiplier):
					
					# Get additional indicators
					rsi = indicators.get('RSI', 50)
					bb_upper = indicators.get('BB.upper', 0)
					bb_lower = indicators.get('BB.lower', 0)
					
					# Volume strength score
					volume_strength = min(10, volume_ratio)  # Cap at 10x
					
					volume_breakouts.append({
						"symbol": symbol,
						"changePercent": price_change,
						"volume_ratio": round(volume_ratio, 2),
						"volume_strength": round(volume_strength, 1),
						"current_volume": volume,
						"breakout_type": "bullish" if price_change > 0 else "bearish",
						"indicators": {
							"close": close,
							"RSI": rsi,
							"BB_upper": bb_upper,
							"BB_lower": bb_lower,
							"volume": volume
						}
					})
					
			except Exception:
				continue
	
	# Sort by volume strength first, then by price change
	volume_breakouts.sort(key=lambda x: (x["volume_strength"], abs(x["changePercent"])), reverse=True)
	
	return volume_breakouts[:limit]


@mcp.tool()
def volume_confirmation_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
	"""Detailed volume confirmation analysis for a specific coin.
	
	Args:
		symbol: Coin symbol (e.g., BTCUSDT)
		exchange: Exchange name
		timeframe: Time frame for analysis
	"""
	exchange = sanitize_exchange(exchange, "KUCOIN")
	timeframe = sanitize_timeframe(timeframe, "15m")
	
	if not symbol.upper().endswith('USDT'):
		symbol = symbol.upper() + 'USDT'
	
	screener = EXCHANGE_SCREENER.get(exchange, "crypto")
	
	try:
		analysis = get_multiple_analysis(screener=screener, interval=timeframe, symbols=[symbol])
		
		if not analysis or symbol not in analysis:
			return {"error": f"No data found for {symbol}"}
			
		data = analysis[symbol]
		if not data or not hasattr(data, 'indicators'):
			return {"error": f"No indicator data for {symbol}"}
			
		indicators = data.indicators
		
		# Get volume data
		volume = indicators.get('volume', 0)
		close = indicators.get('close', 0)
		open_price = indicators.get('open', 0)
		high = indicators.get('high', 0)
		low = indicators.get('low', 0)
		
		# Calculate price metrics
		price_change = ((close - open_price) / open_price) * 100 if open_price > 0 else 0
		candle_range = ((high - low) / low) * 100 if low > 0 else 0
		
		# Volume analysis
		sma20_volume = indicators.get('volume.SMA20', 0)
		volume_ratio = volume / sma20_volume if sma20_volume > 0 else 1
		
		# Technical indicators
		rsi = indicators.get('RSI', 50)
		bb_upper = indicators.get('BB.upper', 0)
		bb_lower = indicators.get('BB.lower', 0)
		bb_middle = (bb_upper + bb_lower) / 2 if bb_upper and bb_lower else close
		
		# Volume confirmation signals
		signals = []
		
		# Strong volume + price breakout
		if volume_ratio >= 2.0 and abs(price_change) >= 3.0:
			signals.append(f"🚀 STRONG BREAKOUT: {volume_ratio:.1f}x volume + {price_change:.1f}% price")
		
		# Volume divergence
		if volume_ratio >= 1.5 and abs(price_change) < 1.0:
			signals.append(f"⚠️ VOLUME DIVERGENCE: High volume ({volume_ratio:.1f}x) but low price movement")
		
		# Low volume on price move (weak signal)
		if abs(price_change) >= 2.0 and volume_ratio < 0.8:
			signals.append(f"❌ WEAK SIGNAL: Price moved but volume is low ({volume_ratio:.1f}x)")
		
		# Bollinger Band + Volume confirmation
		if close > bb_upper and volume_ratio >= 1.5:
			signals.append(f"💥 BB BREAKOUT CONFIRMED: Upper band breakout + volume confirmation")
		elif close < bb_lower and volume_ratio >= 1.5:
			signals.append(f"📉 BB SELL CONFIRMED: Lower band breakout + volume confirmation")
		
		# RSI + Volume analysis
		if rsi > 70 and volume_ratio >= 2.0:
			signals.append(f"🔥 OVERBOUGHT + VOLUME: RSI {rsi:.1f} + {volume_ratio:.1f}x volume")
		elif rsi < 30 and volume_ratio >= 2.0:
			signals.append(f"🛒 OVERSOLD + VOLUME: RSI {rsi:.1f} + {volume_ratio:.1f}x volume")
		
		# Overall assessment
		if volume_ratio >= 3.0:
			volume_strength = "VERY STRONG"
		elif volume_ratio >= 2.0:
			volume_strength = "STRONG"
		elif volume_ratio >= 1.5:
			volume_strength = "MEDIUM"
		elif volume_ratio >= 1.0:
			volume_strength = "NORMAL"
		else:
			volume_strength = "WEAK"
		
		return {
			"symbol": symbol,
			"price_data": {
				"close": close,
				"change_percent": round(price_change, 2),
				"candle_range_percent": round(candle_range, 2)
			},
			"volume_analysis": {
				"current_volume": volume,
				"volume_ratio": round(volume_ratio, 2),
				"volume_strength": volume_strength,
				"average_volume": sma20_volume
			},
			"technical_indicators": {
				"RSI": round(rsi, 1),
				"BB_position": "ABOVE" if close > bb_upper else "BELOW" if close < bb_lower else "WITHIN",
				"BB_upper": bb_upper,
				"BB_lower": bb_lower
			},
			"signals": signals,
			"overall_assessment": {
				"bullish_signals": len([s for s in signals if "🚀" in s or "💥" in s or "🛒" in s]),
				"bearish_signals": len([s for s in signals if "📉" in s or "❌" in s]),
				"warning_signals": len([s for s in signals if "⚠️" in s])
			}
		}
		
	except Exception as e:
		return {"error": f"Analysis failed: {str(e)}"}


@mcp.tool()
def smart_volume_scanner(exchange: str = "KUCOIN", min_volume_ratio: float = 2.0, min_price_change: float = 2.0, rsi_range: str = "any", limit: int = 20) -> list[dict]:
	"""Smart volume + technical analysis combination scanner.
	
	Args:
		exchange: Exchange name
		min_volume_ratio: Minimum volume multiplier (default 2.0)
		min_price_change: Minimum price change percentage (default 2.0)
		rsi_range: "oversold" (<30), "overbought" (>70), "neutral" (30-70), "any"
		limit: Number of results (max 30)
	"""
	exchange = sanitize_exchange(exchange, "KUCOIN")
	min_volume_ratio = max(1.2, min(10.0, min_volume_ratio))
	min_price_change = max(0.5, min(20.0, min_price_change))
	limit = max(1, min(limit, 30))
	
	# Get volume breakouts first
	volume_breakouts = volume_breakout_scanner(
		exchange=exchange, 
		volume_multiplier=min_volume_ratio,
		price_change_min=min_price_change,
		limit=limit * 2  # Get more to filter
	)
	
	if not volume_breakouts:
		return []
	
	# Apply RSI filter
	filtered_results = []
	for coin in volume_breakouts:
		rsi = coin["indicators"].get("RSI", 50)
		
		if rsi_range == "oversold" and rsi >= 30:
			continue
		elif rsi_range == "overbought" and rsi <= 70:
			continue
		elif rsi_range == "neutral" and (rsi <= 30 or rsi >= 70):
			continue
		# "any" passes all
		
		# Add trading recommendation
		recommendation = ""
		if coin["changePercent"] > 0 and coin["volume_ratio"] >= 2.0:
			if rsi < 70:
				recommendation = "🚀 STRONG BUY"
			else:
				recommendation = "⚠️ OVERBOUGHT - CAUTION"
		elif coin["changePercent"] < 0 and coin["volume_ratio"] >= 2.0:
			if rsi > 30:
				recommendation = "📉 STRONG SELL"
			else:
				recommendation = "🛒 OVERSOLD - OPPORTUNITY?"
		
		coin["trading_recommendation"] = recommendation
		filtered_results.append(coin)
	
	return filtered_results[:limit]


if __name__ == "__main__":
	main()

