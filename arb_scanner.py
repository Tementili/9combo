#!/usr/bin/env python3
"""
ULTIMATE ARBITRAGE SCANNER v3.2 - MEGA MULTI-COMBO + ALL MARKETS EDITION

NEW FEATURES:
- ✅ 200+ BETTING MARKETS supported
- ✅ 5-WAY up to 20-WAY arbitrage detection
- ✅ Horse racing (8-30+ horses, forecast/tricast with 1000+ combos)
- ✅ Tournament winners/outrights (20-100+ outcomes)
- ✅ Correct scores, goalscorers (15-22+ outcomes)
- ✅ Golf, motorsports, esports, cricket, and more
- ✅ Default 7-day scan window (was 48 hours)
- ✅ Exchange commission support

Usage:
  # Scan ALL markets for 7 days with up to 20-way arbs
  python arb_scanner.py --fetch-all --nway 20 --extra-markets --high-outcome-markets --all-books -v
  
  # Scan horse racing specifically
  python arb_scanner.py --sport horseracing_uk --nway 20 --extra-markets --high-outcome-markets --all-books -v
  
  # Continuous mega scanning
  python arb_scanner.py --fetch-all --nway 20 --extra-markets --high-outcome-markets --continuous --interval 10 -v
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import csv
import time
from pathlib import Path
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any
from itertools import combinations, product


def _load_dotenv() -> None:
    """Load ODDS_API_KEY from .env or .env.txt in script directory if not already set."""
    if os.environ.get("ODDS_API_KEY"):
        return
    script_dir = Path(__file__).resolve().parent
    placeholders = frozenset({"", "YOUR_KEY_HERE", "your_api_key", "your-key-here"})
    for name in (".env", ".env.txt"):
        path = script_dir / name
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, val = line.partition("=")
                            key, val = key.strip(), val.strip().strip('"\'')
                            if key and val and val.lower() not in placeholders:
                                os.environ[key] = val
            except OSError:
                pass


_load_dotenv()

STALE_MINUTES = 30

# Markets that the bulk /odds endpoint reliably supports.
BULK_MARKETS_ALLOWED = {"h2h", "totals", "spreads"}

# ============================================================
# ALL KNOWN MARKETS (COMPREHENSIVE - 200+ markets!)
# ============================================================
ALL_KNOWN_MARKETS = [
    # ============================================================
    # CORE MARKETS (supported by bulk endpoint)
    # ============================================================
    "h2h", "totals", "spreads",
    
    # ============================================================
    # PERIOD/QUARTER/HALF MARKETS
    # ============================================================
    "h2h_p1", "h2h_p2", "h2h_p3", "h2h_h1", "h2h_h2",
    "h2h_q1", "h2h_q2", "h2h_q3", "h2h_q4",
    "totals_h1", "totals_h2", "totals_q1", "totals_q2", "totals_q3", "totals_q4",
    "totals_p1", "totals_p2", "totals_p3",
    "spreads_h1", "spreads_h2", "spreads_q1", "spreads_q2", "spreads_q3", "spreads_q4",
    "spreads_p1", "spreads_p2", "spreads_p3",
    
    # ============================================================
    # ALTERNATE LINES
    # ============================================================
    "alternate_totals", "alternate_spreads",
    "alternate_totals_corners", "alternate_spreads_corners",
    "alternate_totals_cards", "alternate_spreads_cards",
    
    # ============================================================
    # TEAM TOTALS
    # ============================================================
    "team_totals", "team_totals_h1", "team_totals_h2",
    "team_totals_q1", "team_totals_q2", "team_totals_q3", "team_totals_q4",
    
    # ============================================================
    # SOCCER SPECIFIC MARKETS (HIGH-OUTCOME)
    # ============================================================
    "btts", "draw_no_bet", "double_chance",
    "correct_score",
    "halftime_fulltime",
    "anytime_goalscorer",
    "first_goalscorer",
    "last_goalscorer",
    "scorecast",
    "wincast",
    "team_to_score_first",
    "team_to_score_last",
    "both_teams_to_score_both_halves",
    "clean_sheet",
    "exact_goals",
    "odd_even_goals",
    "multi_goals",
    "goal_range",
    "winning_margin",
    "time_of_first_goal",
    "total_corners",
    "total_cards",
    "total_bookings",
    "asian_corners",
    "player_shots_on_target",
    "player_cards",
    "player_assists",
    "penalty_awarded",
    "own_goal",
    "hat_trick",
    
    # ============================================================
    # AMERICAN FOOTBALL MARKETS
    # ============================================================
    "first_td_scorer",
    "anytime_td_scorer",
    "last_td_scorer",
    "player_passing_yards",
    "player_rushing_yards",
    "player_receiving_yards",
    "player_receptions",
    "player_passing_tds",
    "player_rushing_tds",
    "longest_touchdown",
    "total_field_goals",
    "total_sacks",
    "total_turnovers",
    "safety_scored",
    "overtime",
    "first_drive_result",
    
    # ============================================================
    # BASKETBALL MARKETS
    # ============================================================
    "player_points",
    "player_rebounds",
    "player_threes",
    "player_steals",
    "player_blocks",
    "player_double_double",
    "player_triple_double",
    "first_basket",
    "race_to_points",
    "highest_scoring_quarter",
    "odd_even_total",
    
    # ============================================================
    # BASEBALL MARKETS
    # ============================================================
    "player_hits",
    "player_rbis",
    "player_runs",
    "player_home_runs",
    "player_strikeouts",
    "first_inning_result",
    "total_runs_odd_even",
    "first_team_to_score",
    "first_hit",
    "extra_innings",
    
    # ============================================================
    # ICE HOCKEY MARKETS
    # ============================================================
    "player_goals",
    "player_shots",
    "puck_line",
    "first_goal_scorer",
    "anytime_goal_scorer",
    "period_winner",
    "shootout",
    "total_goals_odd_even",
    
    # ============================================================
    # TENNIS MARKETS
    # ============================================================
    "set_winner",
    "set_betting",
    "total_games",
    "game_handicap",
    "set_correct_score",
    "player_to_win_first_set",
    "tiebreak_in_match",
    "total_sets",
    
    # ============================================================
    # GOLF MARKETS
    # ============================================================
    "tournament_winner",
    "top_5_finish",
    "top_10_finish",
    "top_20_finish",
    "make_cut",
    "first_round_leader",
    "hole_in_one",
    "playoff",
    "nationality_of_winner",
    
    # ============================================================
    # HORSE RACING MARKETS (MASSIVE OUTCOMES!)
    # ============================================================
    "horseracing_winner",
    "horseracing_place",
    "horseracing_show",
    "horseracing_forecast",
    "horseracing_tricast",
    "horseracing_reverse_forecast",
    "horseracing_combination_forecast",
    "horseracing_each_way",
    "horseracing_distance_winner",
    "horseracing_favourite",
    "horseracing_match_bet",
    
    # ============================================================
    # GREYHOUND RACING MARKETS
    # ============================================================
    "greyhound_winner",
    "greyhound_place",
    "greyhound_forecast",
    "greyhound_tricast",
    
    # ============================================================
    # MMA/BOXING MARKETS
    # ============================================================
    "method_of_victory",
    "round_betting",
    "fight_goes_distance",
    "total_rounds",
    "round_group_betting",
    "ko_tko_dq",
    "submission",
    "points_decision",
    
    # ============================================================
    # ESPORTS MARKETS
    # ============================================================
    "match_winner",
    "map_winner",
    "total_maps",
    "map_handicap",
    "first_blood",
    "first_tower",
    "first_dragon",
    "first_baron",
    "total_kills",
    "race_to_kills",
    "player_kills",
    
    # ============================================================
    # CRICKET MARKETS
    # ============================================================
    "innings_runs",
    "top_batsman",
    "top_bowler",
    "method_of_dismissal",
    "total_sixes",
    "total_wides",
    "man_of_match",
    "highest_opening_partnership",
    "century_scored",
    
    # ============================================================
    # RUGBY MARKETS
    # ============================================================
    "first_try_scorer",
    "anytime_try_scorer",
    "total_tries",
    "half_time_full_time",
    
    # ============================================================
    # MOTORSPORTS MARKETS (F1, NASCAR, etc.)
    # ============================================================
    "race_winner",
    "podium_finish",
    "points_finish",
    "fastest_lap",
    "first_retirement",
    "safety_car",
    "winning_constructor",
    "driver_matchup",
    
    # ============================================================
    # DARTS MARKETS
    # ============================================================
    "most_180s",
    "highest_checkout",
    "nine_dart_finish",
    "total_180s",
    
    # ============================================================
    # SNOOKER MARKETS
    # ============================================================
    "frame_winner",
    "century_break",
    "highest_break",
    "total_frames",
    
    # ============================================================
    # OUTRIGHT/FUTURES MARKETS (LONG-TERM BETS)
    # ============================================================
    "outrights",
    "outright_winner",
    "futures",
    "league_winner",
    "relegation",
    "top_4_finish",
    "top_6_finish",
    "top_goalscorer",
    "player_of_year",
    "manager_of_year",
    "to_reach_final",
    "to_win_group",
    "to_qualify",
    "tournament_stage_exit",
    "winning_conference",
    "winning_division",
    "mvp",
    "rookie_of_year",
    "defensive_player_year",
    "coach_of_year",
    "total_wins",
    "make_playoffs",
    "win_championship",
    
    # ============================================================
    # SPECIALS/NOVELTY MARKETS
    # ============================================================
    "specials",
    "entertainment",
    "politics",
    "tv_shows",
    "awards_ceremonies",
    "weather",
    "player_performance",
    "player_specials",
]

# ============================================================
# HIGH-OUTCOME MARKETS (5+ outcomes - PERFECT for mega n-way arbs!)
# ============================================================
HIGH_OUTCOME_MARKETS = [
    # ============================================================
    # MEGA OUTCOME MARKETS (50-1000+ outcomes!)
    # ============================================================
    "horseracing_forecast",
    "horseracing_tricast",
    "horseracing_combination_forecast",
    "scorecast",
    "tournament_winner",
    "top_goalscorer",
    
    # ============================================================
    # VERY HIGH OUTCOME MARKETS (20-50 outcomes)
    # ============================================================
    "horseracing_winner",
    "horseracing_place",
    "horseracing_show",
    "anytime_goalscorer",
    "first_goalscorer",
    "last_goalscorer",
    "first_td_scorer",
    "anytime_td_scorer",
    "last_td_scorer",
    "first_try_scorer",
    "anytime_try_scorer",
    "top_batsman",
    "top_bowler",
    "man_of_match",
    "race_winner",
    "podium_finish",
    "points_finish",
    "fastest_lap",
    "first_retirement",
    "outright_winner",
    "league_winner",
    "relegation",
    "top_4_finish",
    "top_6_finish",
    "player_of_year",
    "mvp",
    "rookie_of_year",
    "first_basket",
    "first_goal_scorer",
    "anytime_goal_scorer",
    "nationality_of_winner",
    
    # ============================================================
    # HIGH OUTCOME MARKETS (10-20 outcomes)
    # ============================================================
    "correct_score",
    "set_correct_score",
    "round_betting",
    "player_points",
    "player_rebounds",
    "player_assists",
    "exact_goals",
    "time_of_first_goal",
    "winning_margin",
    "winning_constructor",
    "greyhound_winner",
    "greyhound_forecast",
    "greyhound_tricast",
    "horseracing_distance_winner",
    "winning_division",
    "coach_of_year",
    "to_reach_final",
    
    # ============================================================
    # MODERATE OUTCOME MARKETS (5-10 outcomes)
    # ============================================================
    "halftime_fulltime",
    "h2h_3way",
    "method_of_victory",
    "multi_goals",
    "goal_range",
    "player_shots_on_target",
    "longest_touchdown",
    "first_drive_result",
    "set_betting",
    "round_group_betting",
    "method_of_dismissal",
    "tournament_stage_exit",
    "to_win_group",
    "total_maps",
    "highest_break",
    "highest_checkout",
    "highest_scoring_quarter",
    "map_winner",
    
    # ============================================================
    # WINCAST/SPECIAL COMBOS
    # ============================================================
    "wincast",
    "futures",
    "outrights",
    "specials",
]

# ✅ EXCHANGE COMMISSION RATES
EXCHANGE_COMMISSION = {
    "betfair_ex_uk": 0.05,
    "betfair_ex_eu": 0.05,
    "betfair_ex_au": 0.05,
    "smarkets": 0.02,
    "matchbook": 0.015,
    "betdaq": 0.05,
}

H2H_LIKE = frozenset({
    "h2h", "h2h_p1", "h2h_p2", "h2h_p3", "h2h_h1", "h2h_h2",
    "h2h_q1", "h2h_q2", "h2h_q3", "h2h_q4",
})

SPREADS_LIKE = frozenset({
    "spreads", "spreads_h1", "spreads_h2", "spreads_q1", "spreads_q2",
    "spreads_q3", "spreads_q4", "spreads_p1", "spreads_p2", "spreads_p3",
    "alternate_spreads", "alternate_spreads_corners", "alternate_spreads_cards",
})

TOTALS_LIKE = frozenset({
    "totals", "totals_h1", "totals_h2", "totals_q1", "totals_q2",
    "totals_q3", "totals_q4", "totals_p1", "totals_p2", "totals_p3",
    "alternate_totals", "alternate_totals_corners", "alternate_totals_cards",
})

ICE_HOCKEY_SPORTS = frozenset({
    "icehockey_nhl", "icehockey_sweden_hockey_league", "icehockey_finland_liiga",
    "icehockey_ohl", "icehockey_whl", "icehockey_qmjhl", "icehockey_ahl",
    "icehockey_ncaa_hockey",
})

SOCCER_SPORTS = frozenset({
    "soccer_epl", "soccer_spain_la_liga", "soccer_germany_bundesliga",
    "soccer_italy_serie_a", "soccer_france_ligue_one", "soccer_uefa_champs_league",
    "soccer_uefa_europa_league", "soccer_netherlands_eredivisie",
    "soccer_portugal_primeira_liga", "soccer_belgium_first_div",
    "soccer_turkey_super_league", "soccer_mls", "soccer_brazil_campeonato",
    "soccer_argentina_primera", "soccer_mexico_ligamx",
})

TRUSTED_BOOKMAKERS = frozenset({
    "bet365", "betway", "unibet", "bwin", "betsson", "pinnacle", "marathonbet",
    "betclic", "betfair", "williamhill", "betano", "sportingbet", "888sport",
    "sport888", "nordicbet", "coolbet", "betvictor", "betfair_sb_uk",
    "betfair_ex_eu", "betfair_ex_uk",
})

BETFAIR_EXCHANGES = frozenset({"betfair_ex_uk", "betfair_ex_eu", "betfair_ex_au"})

NON_BETFAIR_EXCHANGES = frozenset({
    "betopenly", "kalshi", "novig", "polymarket", "prophetx", "smarkets", "matchbook",
})

UNRELIABLE_BOOKMAKERS = frozenset({"williamhill", "unibet_uk"})

BLACKLIST_BOOKMAKERS = frozenset({"paddypower", "paddy_power"})

FINLAND_RESTRICTED = frozenset({
    "betmgm", "draftkings", "fanduel", "betrivers", "fanatics", "williamhill_us",
    "ballybet", "betparx", "espnbet", "hardrockbet", "fliff", "rebet",
    "pointsbetus", "barstool", "betfair_ex_au", "betr_au", "betright",
    "boombet", "ladbrokes_au", "neds", "playup", "pointsbetau", "sportsbet",
    "tab", "tabtouch", "gtbets", "bovada", "virginbet", "livescorebet",
    "coral", "ladbrokes_uk", "codere_it",
})

SPORT_PRESETS: dict[str, list[tuple[str, str]] | None] = {
    "soccer_all": [
        ("soccer_epl", "h2h,totals,spreads"),
        ("soccer_spain_la_liga", "h2h,totals,spreads"),
        ("soccer_germany_bundesliga", "h2h,totals,spreads"),
        ("soccer_italy_serie_a", "h2h,totals,spreads"),
        ("soccer_france_ligue_one", "h2h,totals,spreads"),
        ("soccer_uefa_champs_league", "h2h,totals,spreads"),
    ],
    "us_sports": [
        ("americanfootball_nfl", "h2h,totals,spreads"),
        ("basketball_nba", "h2h,totals,spreads"),
        ("icehockey_nhl", "h2h,totals,spreads"),
        ("baseball_mlb", "h2h,totals,spreads"),
    ],
    "mma_ufc": [("mma_mixed_martial_arts", "h2h,totals,spreads")],
    "all": [
        ("soccer_epl", "h2h,totals,spreads"),
        ("soccer_spain_la_liga", "h2h,totals,spreads"),
        ("soccer_germany_bundesliga", "h2h,totals,spreads"),
        ("basketball_nba", "h2h,totals,spreads"),
        ("basketball_ncaab", "h2h,totals,spreads"),
        ("americanfootball_nfl", "h2h,totals,spreads"),
        ("icehockey_nhl", "h2h,totals,spreads"),
        ("mma_mixed_martial_arts", "h2h,totals,spreads"),
    ],
    "safe_unified": [
        ("soccer_epl", "h2h,totals,spreads,btts,draw_no_bet,double_chance"),
        ("soccer_spain_la_liga", "h2h,totals,spreads,btts,draw_no_bet,double_chance"),
        ("soccer_germany_bundesliga", "h2h,totals,spreads,btts,draw_no_bet,double_chance"),
        ("soccer_italy_serie_a", "h2h,totals,spreads,btts,draw_no_bet,double_chance"),
        ("soccer_france_ligue_one", "h2h,totals,spreads,btts,draw_no_bet,double_chance"),
        ("soccer_uefa_champs_league", "h2h,totals,spreads,btts,draw_no_bet,double_chance"),
        ("basketball_nba", "h2h,totals,spreads"),
        ("americanfootball_nfl", "h2h,totals,spreads"),
        ("icehockey_nhl", "h2h,spreads"),
        ("icehockey_sweden_hockey_league", "h2h,spreads"),
        ("mma_mixed_martial_arts", "h2h"),
    ],
    "all_games": None,
    "two_way_h2h_all_games": None,
}

EXTRA_MARKETS = (
    "btts,draw_no_bet,double_chance,"
    "totals_h1,totals_h2,totals_q1,totals_q2,totals_q3,totals_q4,totals_p1,totals_p2,totals_p3,"
    "spreads_h1,spreads_h2,spreads_q1,spreads_q2,spreads_q3,spreads_q4,spreads_p1,spreads_p2,spreads_p3,"
    "h2h_p1,h2h_p2,h2h_p3,h2h_h1,h2h_h2,h2h_q1,h2h_q2,h2h_q3,h2h_q4,"
    "alternate_totals,alternate_spreads,"
    "alternate_totals_corners,alternate_spreads_corners,"
    "alternate_totals_cards,alternate_spreads_cards,"
    "team_totals,team_totals_h1,team_totals_h2,team_totals_q1,team_totals_q2,team_totals_q3,team_totals_q4"
)

ALL_EXTRA_MARKETS = ",".join([m for m in ALL_KNOWN_MARKETS if m not in BULK_MARKETS_ALLOWED])

EXTRA_MARKETS_SAFE = (
    "btts,draw_no_bet,double_chance,"
    "spreads_h1,spreads_h2,spreads_q1,spreads_q2,spreads_q3,spreads_q4,spreads_p1,spreads_p2,spreads_p3,"
    "h2h_p1,h2h_p2,h2h_p3,h2h_h1,h2h_h2,h2h_q1,h2h_q2,h2h_q3,h2h_q4,"
    "alternate_spreads,alternate_spreads_corners,alternate_spreads_cards"
)

BTTS = frozenset({"btts"})
TEAM_TOTALS = frozenset({
    "team_totals", "team_totals_h1", "team_totals_h2",
    "team_totals_q1", "team_totals_q2", "team_totals_q3", "team_totals_q4",
})
DRAW_NO_BET = frozenset({"draw_no_bet"})
DOUBLE_CHANCE = frozenset({"double_chance"})


def _is_stale(last_update: str | None) -> bool:
    """Return True if last_update is older than STALE_MINUTES."""
    if not last_update:
        return True
    try:
        ts = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts) > timedelta(minutes=STALE_MINUTES)
    except (ValueError, TypeError):
        return True


def adjust_odds_for_commission(odds: float, bookmaker_key: str) -> float:
    """Adjust displayed odds for exchange commission."""
    commission = EXCHANGE_COMMISSION.get(bookmaker_key, 0.0)
    
    if commission == 0.0:
        return odds
    
    effective_odds = 1.0 + (odds - 1.0) * (1.0 - commission)
    
    return effective_odds


def calc_arb_equal_payout(
    odds_a: float, 
    odds_b: float,
    bookmaker_a: str = "",
    bookmaker_b: str = ""
) -> tuple[float, float, float] | None:
    """Returns (stake_a, stake_b, total_stake) for equal payout arb, or None if no arb."""
    if odds_a is None or odds_b is None:
        return None
    if odds_a <= 1.0 or odds_b <= 1.0:
        return None
    
    effective_odds_a = adjust_odds_for_commission(odds_a, bookmaker_a)
    effective_odds_b = adjust_odds_for_commission(odds_b, bookmaker_b)
    
    inv = 1.0 / effective_odds_a + 1.0 / effective_odds_b
    if inv >= 1.0:
        return None
    
    total = 100.0
    stake_a = total / effective_odds_a / inv
    stake_b = total / effective_odds_b / inv
    return (stake_a, stake_b, stake_a + stake_b)


def calc_nway_arb(
    odds_list: list[float],
    bookmaker_keys: list[str] = None
) -> tuple[list[float], float] | None:
    """Calculate n-way arbitrage for any number of outcomes with commission support."""
    if not odds_list or any(o <= 1.0 for o in odds_list):
        return None
    
    if bookmaker_keys and len(bookmaker_keys) == len(odds_list):
        effective_odds = [adjust_odds_for_commission(o, bk) for o, bk in zip(odds_list, bookmaker_keys)]
    else:
        effective_odds = odds_list
    
    inv_sum = sum(1.0 / o for o in effective_odds)
    
    if inv_sum >= 1.0:
        return None
    
    total = 100.0
    stakes = [total / (o * inv_sum) for o in effective_odds]
    
    return (stakes, sum(stakes))


def _arb_profit_pct(
    odds_a: float, 
    odds_b: float,
    bookmaker_a: str = "",
    bookmaker_b: str = ""
) -> float:
    """Return profit % for a 2-way arb, accounting for commission."""
    effective_odds_a = adjust_odds_for_commission(odds_a, bookmaker_a)
    effective_odds_b = adjust_odds_for_commission(odds_b, bookmaker_b)
    
    inv = 1.0 / effective_odds_a + 1.0 / effective_odds_b
    return 100.0 * (1.0 / inv - 1.0)


def _nway_arb_profit_pct(
    odds_list: list[float],
    bookmaker_keys: list[str] = None
) -> float:
    """Return profit % for n-way arb with commission support."""
    if bookmaker_keys and len(bookmaker_keys) == len(odds_list):
        effective_odds = [adjust_odds_for_commission(o, bk) for o, bk in zip(odds_list, bookmaker_keys)]
    else:
        effective_odds = odds_list
    
    inv_sum = sum(1.0 / o for o in effective_odds)
    return 100.0 * (1.0 / inv_sum - 1.0)


def is_two_way_market(outcomes: list[dict]) -> bool:
    """True if outcomes are exactly 2 and none is 'Draw'."""
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return False
    names = {(o.get("name") or "").strip().lower() for o in outcomes}
    if "draw" in names:
        return False
    return True


def format_event(home: str, away: str, sport_key: str) -> str:
    """Format event string. US sports commonly displayed as Away @ Home."""
    if sport_key.startswith(("basketball_", "americanfootball_", "baseball_", "icehockey_")):
        return f"{away} @ {home}"
    return f"{home} v {away}"


# Global settings
_INCLUDE_UNRELIABLE = False
_USE_ALL_BOOKS = False
_INCLUDE_EXCHANGES = False
_FINLAND_ONLY = True
_TWO_WAY_ONLY = False
_ENABLE_3WAY = False
_ENABLE_4WAY = False
_MAX_NWAY = 4

# NEW: Allow excluding bookmaker keys via CLI (e.g. --exclude-books sportsbet)
_EXCLUDE_BOOKS: set[str] = set()


def _filter_bookmakers(bookmakers: list[dict]) -> list[dict]:
    out = []
    for bm in bookmakers:
        key = bm.get("key", "")
        # NEW: user-specified excludes
        if key and key.lower() in _EXCLUDE_BOOKS:
            continue
        if key in BLACKLIST_BOOKMAKERS:
            continue
        if key in NON_BETFAIR_EXCHANGES:
            continue
        if not _INCLUDE_EXCHANGES and key in BETFAIR_EXCHANGES:
            continue
        if _FINLAND_ONLY and key in FINLAND_RESTRICTED:
            continue
        if not _USE_ALL_BOOKS and key not in TRUSTED_BOOKMAKERS:
            continue
        if not _INCLUDE_UNRELIABLE and key in UNRELIABLE_BOOKMAKERS:
            continue
        out.append(bm)
    return out


def export_to_json(arbs: list[dict], filename: str) -> None:
    """Export arbitrage opportunities to JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(arbs, f, indent=2, default=str)
    print(f"✅ Saved {len(arbs)} arbs to {filename}", file=sys.stderr)


def export_to_csv(arbs: list[dict], filename: str) -> None:
    """Export arbitrage opportunities to CSV file."""
    if not arbs:
        return
    
    all_keys = set()
    for a in arbs:
        all_keys.update(a.keys())
    
    fieldnames = sorted(list(all_keys))
    
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(arbs)
    
    print(f"✅ Saved {len(arbs)} arbs to {filename}", file=sys.stderr)


def export_to_txt(arbs: list[dict], filename: str) -> None:
    """Export arbitrage opportunities to readable TXT file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write("ARBITRAGE OPPORTUNITIES - DETAILED REPORT\n")
        f.write("="*100 + "\n\n")
        
        for i, arb in enumerate(arbs, 1):
            f.write(f"\n--- ARB #{i} ---\n")
            f.write(f"Sport: {arb.get('sport_key', 'N/A')}\n")
            f.write(f"Event: {arb.get('home', 'N/A')} vs {arb.get('away', 'N/A')}\n")
            f.write(f"Market: {arb.get('market', 'N/A')}\n")
            f.write(f"Type: {arb.get('arb_type', 'N/A')}\n")
            f.write(f"Profit: {arb.get('profit_pct', 0):.2f}%\n")
            
            arb_type = arb.get('arb_type', '2-way')
            if arb_type.endswith('-way'):
                num_outcomes = int(arb_type.split('-')[0])
                for j in range(1, num_outcomes + 1):
                    f.write(f"  Outcome {j}: {arb.get(f'outcome_{j}', 'N/A')} @ {arb.get(f'odds_{j}', 0):.2f} ({arb.get(f'book_{j}', 'N/A')}) - ${arb.get(f'stake_{j}', 0):.2f}\n")
            
            f.write(f"Total Stake: ${arb.get('total_stake', 0):.2f}\n")
            f.write(f"Commence: {arb.get('commence', 'N/A')}\n")
            f.write("-"*100 + "\n")
        
        f.write(f"\n\nTOTAL ARBITRAGES FOUND: {len(arbs)}\n")
        f.write("="*100 + "\n")
    
    print(f"✅ Saved {len(arbs)} arbs to {filename}", file=sys.stderr)


def print_json_output(arbs: list[dict]) -> None:
    """Print arbitrage opportunities as JSON."""
    print(json.dumps(arbs, indent=2, default=str))


def print_csv_output(arbs: list[dict]) -> None:
    """Print arbitrage opportunities as CSV to stdout."""
    if not arbs:
        return
    
    all_keys = set()
    for a in arbs:
        all_keys.update(a.keys())
    
    fieldnames = sorted(list(all_keys))
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(arbs)


def scan_nway_dynamic(events: list[dict], sport_key: str, market_key: str, min_outcomes: int = 5, max_outcomes: int = 20) -> list[dict]:
    """
    Dynamic N-way arbitrage scanner for 5-20 way arbitrages.
    Scans markets with 5+ outcomes for profitable arbitrage combinations.
    """
    arbs = []
    
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        
        outcomes_data: dict[str, list] = defaultdict(list)
        
        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != market_key:
                    continue
                
                for o in mkt.get("outcomes", []):
                    name = (o.get("name") or "").strip()
                    point = o.get("point")
                    price = o.get("price")
                    
                    outcome_id = f"{name}_{point}" if point is not None else name
                    
                    if name and price is not None:
                        outcomes_data[outcome_id].append(
                            (bm.get("key", "?"), bm.get("title", bm.get("key", "?")), 
                             float(price), bm.get("last_update"), name, point)
                        )
        
        num_outcomes = len(outcomes_data)
        if num_outcomes < min_outcomes or num_outcomes > max_outcomes:
            continue
        
        outcome_names = list(outcomes_data.keys())
        
        best_odds = []
        best_books = []
        best_names = []
        best_updates = []
        
        for outcome_id in outcome_names:
            best_for_outcome = max(outcomes_data[outcome_id], key=lambda x: x[2])
            best_odds.append(best_for_outcome[2])
            best_books.append((best_for_outcome[0], best_for_outcome[1]))
            best_names.append(best_for_outcome[4])
            best_updates.append(best_for_outcome[3])
        
        bookmaker_keys = [bk[0] for bk in best_books]
        result = calc_nway_arb(best_odds, bookmaker_keys)
        
        if result:
            stakes, total_stake = result
            
            arb_dict = {
                "sport_key": sport_key,
                "event_id": eid,
                "home": home,
                "away": away,
                "commence": commence,
                "market": market_key,
                "line": None,
                "arb_type": f"{num_outcomes}-way",
                "total_stake": total_stake,
                "profit_pct": _nway_arb_profit_pct(best_odds, bookmaker_keys),
            }
            
            for i, (name, odds, stake, book, update) in enumerate(zip(best_names, best_odds, stakes, best_books, best_updates), 1):
                arb_dict[f"outcome_{i}"] = name
                arb_dict[f"odds_{i}"] = odds
                arb_dict[f"stake_{i}"] = stake
                arb_dict[f"book_{i}"] = book[1]
                arb_dict[f"book_{i}_key"] = book[0]
                arb_dict[f"last_update_{i}"] = update
            
            arbs.append(arb_dict)
    
    return arbs


def scan_asian_handicap(events: list[dict], sport_key: str) -> list[dict]:
    """Scan for Asian Handicap arbitrage opportunities."""
    arbs = []
    
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        
        handicap_lines: dict[float, list] = defaultdict(list)
        
        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") not in ("spreads", "alternate_spreads"):
                    continue
                
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    point = o.get("point")
                    price = o.get("price")
                    
                    if point is not None and price is not None:
                        line = float(point)
                        if abs(line % 0.5 - 0.25) < 0.01 or abs(line % 0.5 - 0.75) < 0.01:
                            handicap_lines[abs(line)].append(
                                (name, line, bm.get("key", "?"), bm.get("title", bm.get("key", "?")), 
                                 float(price), bm.get("last_update"))
                            )
        
        for line, outcomes in handicap_lines.items():
            positive = [o for o in outcomes if o[1] > 0]
            negative = [o for o in outcomes if o[1] < 0]
            
            if not positive or not negative:
                continue
            
            for (n1, p1, bk1, b1, o1, lu1) in negative:
                for (n2, p2, bk2, b2, o2, lu2) in positive:
                    if bk1 == bk2 or n1 == n2:
                        continue
                    
                    res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                    if res:
                        arbs.append({
                            "sport_key": sport_key,
                            "event_id": eid,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "market": "asian_handicap",
                            "line": line,
                            "arb_type": "2-way-asian",
                            "side_a": f"{n1} ({p1:+.2f})",
                            "side_b": f"{n2} ({p2:+.2f})",
                            "book_a": b1,
                            "book_b": b2,
                            "odds_a": o1,
                            "odds_b": o2,
                            "stake_a": res[0],
                            "stake_b": res[1],
                            "total_stake": res[2],
                            "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                            "last_update_a": lu1,
                            "last_update_b": lu2,
                        })
    
    return arbs


def scan_h2h_3way(events: list[dict], sport_key: str, market_key: str = "h2h") -> list[dict]:
    """Scan 3-way head-to-head markets (Home/Draw/Away)."""
    arbs = []
    
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        
        outcomes_data: dict[str, list] = defaultdict(list)
        
        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != market_key:
                    continue
                
                outs = mkt.get("outcomes", [])
                
                if len(outs) != 3:
                    continue
                
                has_draw = any((o.get("name") or "").strip().lower() == "draw" for o in outs)
                if not has_draw:
                    continue
                
                for o in outs:
                    name = (o.get("name") or "").strip()
                    price = o.get("price")
                    
                    if name and price is not None:
                        outcomes_data[name].append(
                            (bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update"))
                        )
        
        if len(outcomes_data) != 3:
            continue
        
        outcome_names = list(outcomes_data.keys())
        
        for (bk1, b1, o1, lu1) in outcomes_data[outcome_names[0]]:
            for (bk2, b2, o2, lu2) in outcomes_data[outcome_names[1]]:
                for (bk3, b3, o3, lu3) in outcomes_data[outcome_names[2]]:
                    if bk1 == bk2 or bk1 == bk3 or bk2 == bk3:
                        continue
                    
                    result = calc_nway_arb([o1, o2, o3], [bk1, bk2, bk3])
                    if result:
                        stakes, total_stake = result
                        arbs.append({
                            "sport_key": sport_key,
                            "event_id": eid,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "market": f"{market_key}_3way",
                            "line": None,
                            "arb_type": "3-way",
                            "outcome_1": outcome_names[0],
                            "outcome_2": outcome_names[1],
                            "outcome_3": outcome_names[2],
                            "book_1": b1,
                            "book_2": b2,
                            "book_3": b3,
                            "odds_1": o1,
                            "odds_2": o2,
                            "odds_3": o3,
                            "stake_1": stakes[0],
                            "stake_2": stakes[1],
                            "stake_3": stakes[2],
                            "total_stake": total_stake,
                            "profit_pct": _nway_arb_profit_pct([o1, o2, o3], [bk1, bk2, bk3]),
                            "last_update_1": lu1,
                            "last_update_2": lu2,
                            "last_update_3": lu3,
                        })
    
    return arbs


def scan_4way_combined_markets(events: list[dict], sport_key: str) -> list[dict]:
    """Scan for 4-way arbitrage by combining two 2-way markets."""
    arbs = []
    
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        
        h2h_odds: dict[str, list] = defaultdict(list)
        totals_odds: dict[tuple[float, str], list] = defaultdict(list)
        
        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                mk = mkt.get("key")
                
                if mk == "h2h":
                    outs = mkt.get("outcomes", [])
                    if len(outs) == 2 and not any((o.get("name") or "").lower() == "draw" for o in outs):
                        for o in outs:
                            name = o.get("name", "")
                            price = o.get("price")
                            if name and price:
                                h2h_odds[name].append(
                                    (bm.get("key"), bm.get("title", bm.get("key")), float(price), bm.get("last_update"))
                                )
                
                elif mk in ("totals", "alternate_totals"):
                    for o in mkt.get("outcomes", []):
                        name = o.get("name", "")
                        point = o.get("point")
                        price = o.get("price")
                        if name in ("Over", "Under") and point is not None and price is not None:
                            totals_odds[(float(point), name)].append(
                                (bm.get("key"), bm.get("title", bm.get("key")), float(price), bm.get("last_update"))
                            )
        
        h2h_names = list(h2h_odds.keys())
        if len(h2h_names) != 2:
            continue
        
        team1, team2 = h2h_names[0], h2h_names[1]
        
        lines_seen = set()
        for (line, side) in totals_odds.keys():
            if line in lines_seen:
                continue
            lines_seen.add(line)
            
            over_key = (line, "Over")
            under_key = (line, "Under")
            
            if over_key not in totals_odds or under_key not in totals_odds:
                continue
            
            for (bk_t1, bt1, ot1, lut1) in h2h_odds[team1]:
                for (bk_t2, bt2, ot2, lut2) in h2h_odds[team2]:
                    for (bk_o, bo, oo, luo) in totals_odds[over_key]:
                        for (bk_u, bu, ou, luu) in totals_odds[under_key]:
                            bookies = {bk_t1, bk_t2, bk_o, bk_u}
                            if len(bookies) < 4:
                                continue
                            
                            odds_t1_over = ot1 * oo
                            odds_t1_under = ot1 * ou
                            odds_t2_over = ot2 * oo
                            odds_t2_under = ot2 * ou
                            
                            bk_combo = [f"{bk_t1}+{bk_o}", f"{bk_t1}+{bk_u}", f"{bk_t2}+{bk_o}", f"{bk_t2}+{bk_u}"]
                            
                            result = calc_nway_arb(
                                [odds_t1_over, odds_t1_under, odds_t2_over, odds_t2_under],
                                bk_combo
                            )
                            
                            if result:
                                stakes, total_stake = result
                                
                                arbs.append({
                                    "sport_key": sport_key,
                                    "event_id": eid,
                                    "home": home,
                                    "away": away,
                                    "commence": commence,
                                    "market": f"4way_h2h+totals_{line}",
                                    "line": line,
                                    "arb_type": "4-way",
                                    "outcome_1": f"{team1} + Over {line}",
                                    "outcome_2": f"{team1} + Under {line}",
                                    "outcome_3": f"{team2} + Over {line}",
                                    "outcome_4": f"{team2} + Under {line}",
                                    "book_1": f"{bt1} + {bo}",
                                    "book_2": f"{bt1} + {bu}",
                                    "book_3": f"{bt2} + {bo}",
                                    "book_4": f"{bt2} + {bu}",
                                    "odds_1": odds_t1_over,
                                    "odds_2": odds_t1_under,
                                    "odds_3": odds_t2_over,
                                    "odds_4": odds_t2_under,
                                    "stake_1": stakes[0],
                                    "stake_2": stakes[1],
                                    "stake_3": stakes[2],
                                    "stake_4": stakes[3],
                                    "total_stake": total_stake,
                                    "profit_pct": _nway_arb_profit_pct(
                                        [odds_t1_over, odds_t1_under, odds_t2_over, odds_t2_under],
                                        bk_combo
                                    ),
                                })
    
    return arbs


def scan_totals(events: list[dict], sport_key: str, market_key: str = "totals") -> list[dict]:
    """Scan Over/Under markets."""
    arbs = []
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        by_line: dict[float, dict[str, list]] = defaultdict(lambda: {"Over": [], "Under": []})

        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != market_key:
                    continue
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    point = o.get("point")
                    price = o.get("price")
                    if name not in ("Over", "Under") or point is None or price is None:
                        continue
                    by_line[float(point)][name].append(
                        (bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update"))
                    )

        for line, sides in by_line.items():
            overs = sides["Over"]
            unders = sides["Under"]
            if not overs or not unders:
                continue
            for (bk1, b1, o1, lu1) in overs:
                for (bk2, b2, o2, lu2) in unders:
                    if bk1 == bk2:
                        continue
                    res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                    if res:
                        arbs.append({
                            "sport_key": sport_key,
                            "event_id": eid,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "market": market_key,
                            "line": line,
                            "arb_type": "2-way",
                            "side_a": "Over",
                            "side_b": "Under",
                            "book_a": b1,
                            "book_b": b2,
                            "odds_a": o1,
                            "odds_b": o2,
                            "stake_a": res[0],
                            "stake_b": res[1],
                            "total_stake": res[2],
                            "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                            "last_update_a": lu1,
                            "last_update_b": lu2,
                        })
    return arbs


def scan_alternate_totals_enhanced(events: list[dict], sport_key: str) -> list[dict]:
    """Enhanced alternate totals scanner."""
    arbs = []
    
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        
        all_lines: dict[float, dict[str, list]] = defaultdict(lambda: {"Over": [], "Under": []})
        
        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "alternate_totals":
                    continue
                    
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    point = o.get("point")
                    price = o.get("price")
                    
                    if name not in ("Over", "Under") or point is None or price is None:
                        continue
                    
                    line_value = float(point)
                    all_lines[line_value][name].append(
                        (bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update"))
                    )
        
        for line, sides in all_lines.items():
            overs = sides["Over"]
            unders = sides["Under"]
            
            if not overs or not unders:
                continue
            
            for (bk1, b1, o1, lu1) in overs:
                for (bk2, b2, o2, lu2) in unders:
                    if bk1 == bk2:
                        continue
                    
                    res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                    if res:
                        arbs.append({
                            "sport_key": sport_key,
                            "event_id": eid,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "market": "alternate_totals",
                            "line": line,
                            "arb_type": "2-way",
                            "side_a": "Over",
                            "side_b": "Under",
                            "book_a": b1,
                            "book_b": b2,
                            "odds_a": o1,
                            "odds_b": o2,
                            "stake_a": res[0],
                            "stake_b": res[1],
                            "total_stake": res[2],
                            "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                            "last_update_a": lu1,
                            "last_update_b": lu2,
                        })
    
    return arbs


def scan_cross_line_opportunities(events: list[dict], sport_key: str) -> list[dict]:
    """Advanced cross-line middle scanner."""
    arbs = []
    
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        
        all_overs = []
        all_unders = []
        
        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") not in ("totals", "alternate_totals"):
                    continue
                
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    point = o.get("point")
                    price = o.get("price")
                    
                    if point is None or price is None:
                        continue
                    
                    line = float(point)
                    odds = float(price)
                    bk = bm.get("key", "?")
                    title = bm.get("title", bk)
                    lu = bm.get("last_update")
                    
                    if name == "Over":
                        all_overs.append((line, odds, bk, title, lu))
                    elif name == "Under":
                        all_unders.append((line, odds, bk, title, lu))
        
        for (line_over, odds_over, bk_over, title_over, lu_over) in all_overs:
            for (line_under, odds_under, bk_under, title_under, lu_under) in all_unders:
                if bk_over == bk_under:
                    continue
                
                if line_under <= line_over:
                    continue
                
                res = calc_arb_equal_payout(odds_over, odds_under, bk_over, bk_under)
                if res:
                    gap = line_under - line_over
                    arbs.append({
                        "sport_key": sport_key,
                        "event_id": eid,
                        "home": home,
                        "away": away,
                        "commence": commence,
                        "market": "cross_line_totals",
                        "line": f"{line_over}/{line_under}",
                        "arb_type": "2-way-middle",
                        "side_a": f"Over {line_over}",
                        "side_b": f"Under {line_under}",
                        "book_a": title_over,
                        "book_b": title_under,
                        "odds_a": odds_over,
                        "odds_b": odds_under,
                        "stake_a": res[0],
                        "stake_b": res[1],
                        "total_stake": res[2],
                        "profit_pct": _arb_profit_pct(odds_over, odds_under, bk_over, bk_under),
                        "last_update_a": lu_over,
                        "last_update_b": lu_under,
                        "gap": gap,
                        "type": "middle" if gap > 1 else "scalp"
                    })
    
    return arbs


def scan_alternate_spreads_enhanced(events: list[dict], sport_key: str) -> list[dict]:
    """Enhanced alternate spreads scanner."""
    arbs = []
    
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        
        by_line: dict[float, list] = defaultdict(list)
        
        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "alternate_spreads":
                    continue
                
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    point = o.get("point")
                    price = o.get("price")
                    
                    if name is None or point is None or price is None:
                        continue
                    
                    pt = float(point)
                    abs_line = abs(pt)
                    
                    by_line[abs_line].append(
                        (name, pt, bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update"))
                    )
        
        for line, outcomes in by_line.items():
            neg = [(n, p, bk, b, o, lu) for n, p, bk, b, o, lu in outcomes if p < 0]
            pos = [(n, p, bk, b, o, lu) for n, p, bk, b, o, lu in outcomes if p > 0]
            
            if not neg or not pos:
                continue
            
            for (n1, p1, bk1, b1, o1, lu1) in neg:
                for (n2, p2, bk2, b2, o2, lu2) in pos:
                    if bk1 == bk2:
                        continue
                    
                    if n1 == n2:
                        continue
                    
                    res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                    if res:
                        arbs.append({
                            "sport_key": sport_key,
                            "event_id": eid,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "market": "alternate_spreads",
                            "line": line,
                            "arb_type": "2-way",
                            "side_a": f"{n1} ({p1:+.1f})",
                            "side_b": f"{n2} ({p2:+.1f})",
                            "book_a": b1,
                            "book_b": b2,
                            "odds_a": o1,
                            "odds_b": o2,
                            "stake_a": res[0],
                            "stake_b": res[1],
                            "total_stake": res[2],
                            "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                            "last_update_a": lu1,
                            "last_update_b": lu2,
                        })
                        break
    
    return arbs


def scan_spreads(events: list[dict], sport_key: str, market_keys: frozenset[str]) -> list[dict]:
    """Scan point spread / handicap markets."""
    arbs = []
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        by_line: dict[tuple[str, float], list] = defaultdict(list)

        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                mk = mkt.get("key")
                if mk not in market_keys:
                    continue
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    point = o.get("point")
                    price = o.get("price")
                    if name is None or point is None or price is None:
                        continue
                    pt = float(point)
                    by_line[(mk, abs(pt))].append(
                        (name, pt, bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update"))
                    )

        for (mk, line), outcomes in by_line.items():
            neg = [(n, p, bk, b, o, lu) for n, p, bk, b, o, lu in outcomes if p < 0]
            pos = [(n, p, bk, b, o, lu) for n, p, bk, b, o, lu in outcomes if p > 0]
            if not neg or not pos:
                continue

            for (n1, p1, bk1, b1, o1, lu1) in neg:
                for (n2, p2, bk2, b2, o2, lu2) in pos:
                    if bk1 == bk2:
                        continue
                    
                    if n1 == n2:
                        continue
                    
                    res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                    if res:
                        arbs.append({
                            "sport_key": sport_key,
                            "event_id": eid,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "market": mk,
                            "line": line,
                            "arb_type": "2-way",
                            "side_a": f"{n1} ({p1:+.1f})",
                            "side_b": f"{n2} ({p2:+.1f})",
                            "book_a": b1,
                            "book_b": b2,
                            "odds_a": o1,
                            "odds_b": o2,
                            "stake_a": res[0],
                            "stake_b": res[1],
                            "total_stake": res[2],
                            "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                            "last_update_a": lu1,
                            "last_update_b": lu2,
                        })
                        break
    return arbs


def scan_team_totals(events: list[dict], sport_key: str, market_key: str = "team_totals") -> list[dict]:
    """Scan team totals."""
    arbs = []
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        by_team_line: dict[tuple[str, float], dict[str, list]] = defaultdict(lambda: {"Over": [], "Under": []})

        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != market_key:
                    continue
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    desc = (o.get("description") or "").strip()
                    point = o.get("point")
                    price = o.get("price")
                    if name not in ("Over", "Under") or point is None or price is None:
                        continue
                    team = desc or "Team"
                    key = (team, float(point))
                    by_team_line[key][name].append(
                        (bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update"))
                    )

        for (team, line), sides in by_team_line.items():
            overs = sides["Over"]
            unders = sides["Under"]
            if not overs or not unders:
                continue
            for (bk1, b1, o1, lu1) in overs:
                for (bk2, b2, o2, lu2) in unders:
                    if bk1 == bk2:
                        continue
                    res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                    if res:
                        arbs.append({
                            "sport_key": sport_key,
                            "event_id": eid,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "market": f"{market_key}({team})",
                            "line": line,
                            "arb_type": "2-way",
                            "side_a": "Over",
                            "side_b": "Under",
                            "book_a": b1,
                            "book_b": b2,
                            "odds_a": o1,
                            "odds_b": o2,
                            "stake_a": res[0],
                            "stake_b": res[1],
                            "total_stake": res[2],
                            "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                            "last_update_a": lu1,
                            "last_update_b": lu2,
                        })
    return arbs


def scan_double_chance(events: list[dict], sport_key: str) -> list[dict]:
    """Scan double chance."""
    arbs = []
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        by_outcome: dict[str, list] = defaultdict(list)

        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "double_chance":
                    continue
                for o in mkt.get("outcomes", []):
                    name = (o.get("name", "") or "").replace(" ", "").lower()
                    price = o.get("price")
                    if not name or price is None:
                        continue
                    by_outcome[name].append((bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update")))

        pairs = [("homeordraw", "away"), ("awayordraw", "home"), ("homeoraway", "draw")]
        for n1, n2 in pairs:
            if n1 not in by_outcome or n2 not in by_outcome:
                continue
            for (bk1, b1, o1, lu1) in by_outcome[n1]:
                for (bk2, b2, o2, lu2) in by_outcome[n2]:
                    if bk1 == bk2:
                        continue
                    res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                    if res:
                        arbs.append({
                            "sport_key": sport_key,
                            "event_id": eid,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "market": "double_chance",
                            "line": f"{n1} vs {n2}",
                            "arb_type": "2-way",
                            "side_a": n1,
                            "side_b": n2,
                            "book_a": b1,
                            "book_b": b2,
                            "odds_a": o1,
                            "odds_b": o2,
                            "stake_a": res[0],
                            "stake_b": res[1],
                            "total_stake": res[2],
                            "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                            "last_update_a": lu1,
                            "last_update_b": lu2,
                        })
    return arbs


def scan_btts(events: list[dict], sport_key: str) -> list[dict]:
    """Scan Both Teams to Score."""
    arbs = []
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        yes_odds = []
        no_odds = []

        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "btts":
                    continue
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    price = o.get("price")
                    if price is None:
                        continue
                    if name and name.lower() == "yes":
                        yes_odds.append((bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update")))
                    elif name and name.lower() == "no":
                        no_odds.append((bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update")))

        for (bk1, b1, o1, lu1) in yes_odds:
            for (bk2, b2, o2, lu2) in no_odds:
                if bk1 == bk2:
                    continue
                res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                if res:
                    arbs.append({
                        "sport_key": sport_key,
                        "event_id": eid,
                        "home": home,
                        "away": away,
                        "commence": commence,
                        "market": "btts",
                        "line": None,
                        "arb_type": "2-way",
                        "side_a": "Yes",
                        "side_b": "No",
                        "book_a": b1,
                        "book_b": b2,
                        "odds_a": o1,
                        "odds_b": o2,
                        "stake_a": res[0],
                        "stake_b": res[1],
                        "total_stake": res[2],
                        "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                        "last_update_a": lu1,
                        "last_update_b": lu2,
                    })
    return arbs


def scan_draw_no_bet(events: list[dict], sport_key: str) -> list[dict]:
    """Scan Draw No Bet."""
    arbs = []
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))
        home_odds = []
        away_odds = []

        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "draw_no_bet":
                    continue
                for o in mkt.get("outcomes", []):
                    name = o.get("name", "")
                    price = o.get("price")
                    if price is None:
                        continue
                    n = (name or "").lower()
                    if n == "home" or (home and n == home.lower()):
                        home_odds.append((bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update")))
                    elif n == "away" or (away and n == away.lower()):
                        away_odds.append((bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update")))

        for (bk1, b1, o1, lu1) in home_odds:
            for (bk2, b2, o2, lu2) in away_odds:
                if bk1 == bk2:
                    continue
                res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                if res:
                    arbs.append({
                        "sport_key": sport_key,
                        "event_id": eid,
                        "home": home,
                        "away": away,
                        "commence": commence,
                        "market": "draw_no_bet",
                        "line": None,
                        "arb_type": "2-way",
                        "side_a": "Home",
                        "side_b": "Away",
                        "book_a": b1,
                        "book_b": b2,
                        "odds_a": o1,
                        "odds_b": o2,
                        "stake_a": res[0],
                        "stake_b": res[1],
                        "total_stake": res[2],
                        "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                        "last_update_a": lu1,
                        "last_update_b": lu2,
                    })
    return arbs


def scan_h2h(events: list[dict], sport_key: str, market_keys: frozenset[str]) -> list[dict]:
    """Scan 2-way h2h (match/period winner)."""
    arbs = []
    for ev in events:
        eid = ev.get("id", "")
        home = ev.get("home_team", "Home")
        away = ev.get("away_team", "Away")
        commence = ev.get("commence_time", "")
        bookmakers = _filter_bookmakers(ev.get("bookmakers", []))

        for mkt_key in market_keys:
            outcomes_by_side: dict[str, list] = defaultdict(list)
            for bm in bookmakers:
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != mkt_key:
                        continue
                    outs = mkt.get("outcomes", [])
                    if _TWO_WAY_ONLY:
                        if not is_two_way_market(outs):
                            continue
                    else:
                        if len(outs) != 2:
                            continue
                    for o in outs:
                        name = o.get("name", "")
                        price = o.get("price")
                        if name and price is not None:
                            outcomes_by_side[name].append(
                                (bm.get("key", "?"), bm.get("title", bm.get("key", "?")), float(price), bm.get("last_update"))
                            )

            sides = list(outcomes_by_side.keys())
            if len(sides) != 2:
                continue
            a_side, b_side = sides[0], sides[1]

            for (bk1, b1, o1, lu1) in outcomes_by_side[a_side]:
                for (bk2, b2, o2, lu2) in outcomes_by_side[b_side]:
                    if bk1 == bk2:
                        continue
                    res = calc_arb_equal_payout(o1, o2, bk1, bk2)
                    if res:
                        arbs.append({
                            "sport_key": sport_key,
                            "event_id": eid,
                            "home": home,
                            "away": away,
                            "commence": commence,
                            "market": mkt_key,
                            "line": None,
                            "arb_type": "2-way",
                            "side_a": a_side,
                            "side_b": b_side,
                            "book_a": b1,
                            "book_b": b2,
                            "odds_a": o1,
                            "odds_b": o2,
                            "stake_a": res[0],
                            "stake_b": res[1],
                            "total_stake": res[2],
                            "profit_pct": _arb_profit_pct(o1, o2, bk1, bk2),
                            "last_update_a": lu1,
                            "last_update_b": lu2,
                        })
                        break
    return arbs


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_future_days(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_next_48h() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_sports(api_key: str) -> list[dict]:
    """Fetch in-season sports from The Odds API."""
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        return data if isinstance(data, list) else []
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err = json.loads(body) if body else {}
            msg = err.get("message", err.get("detail", body or str(e)))
        except json.JSONDecodeError:
            msg = body or str(e)
        raise SystemExit(f"API error {e.code} fetching sports: {msg}") from e


def fetch_odds(
    api_key: str,
    sport_key: str,
    markets: str,
    regions: str = "eu,uk,se",
    days: int = 0,
    *,
    skip_on_error: bool = False,
) -> list[dict]:
    """Fetch odds from The Odds API bulk /odds endpoint."""
    requested = [m.strip() for m in markets.split(",") if m.strip()]
    bulk_markets = [m for m in requested if m in BULK_MARKETS_ALLOWED]
    if not bulk_markets:
        bulk_markets = ["h2h"]
    markets_param = ",".join(bulk_markets)

    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        f"?apiKey={api_key}&regions={regions}&markets={markets_param}&oddsFormat=decimal"
    )
    
    # ✅ CHANGED: Default to 7 days instead of 48 hours
    if days is not None and days > 0:
        url += f"&commenceTimeFrom={_iso_now()}&commenceTimeTo={_iso_future_days(days)}"
    else:
        url += f"&commenceTimeFrom={_iso_now()}&commenceTimeTo={_iso_future_days(7)}"

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
        return data if isinstance(data, list) else []
    except urllib.error.HTTPError as e:
        if skip_on_error:
            body = e.read().decode() if e.fp else ""
            try:
                err = json.loads(body) if body else {}
                msg = err.get("message", err.get("detail", body or str(e)))
            except json.JSONDecodeError:
                msg = body or str(e)
            print(f"[{sport_key}] skipped ({e.code}): {msg}", file=sys.stderr)
            return []
        body = e.read().decode() if e.fp else ""
        try:
            err = json.loads(body) if body else {}
            msg = err.get("message", err.get("detail", body or str(e)))
        except json.JSONDecodeError:
            msg = body or str(e)
        raise SystemExit(
            f"API error {e.code} for {sport_key} (markets={markets_param}, requested={markets}): {msg}\n"
            f"Tip: bulk /odds supports only h2h, spreads, totals. Extras require /events/{{id}}/odds."
        ) from e


def fetch_event_odds(api_key: str, sport_key: str, event_id: str, markets: str, regions: str) -> dict | None:
    """Fetch odds for a single event via event-odds endpoint."""
    markets_param = markets.replace(" ", "")
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds/"
        f"?apiKey={api_key}&regions={regions}&markets={markets_param}&oddsFormat=decimal"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        return data if isinstance(data, dict) and data.get("id") else None
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _merge_event_markets(ev: dict, extra_ev: dict) -> None:
    """Merge extra markets from extra_ev into ev's bookmakers."""
    bm_by_key: dict[str, dict] = {bm.get("key", ""): bm for bm in ev.get("bookmakers", [])}
    existing_mk: dict[str, set[str]] = {
        bm.get("key", ""): {m.get("key") for m in bm.get("markets", []) if m.get("key")}
        for bm in ev.get("bookmakers", [])
    }
    for bm in extra_ev.get("bookmakers", []):
        key = bm.get("key", "")
        if not key:
            continue
        if key not in bm_by_key:
            ev.setdefault("bookmakers", []).append(bm)
            continue
        target = bm_by_key[key]
        target_markets = target.setdefault("markets", [])
        existing = existing_mk.get(key, set())
        for mkt in bm.get("markets", []):
            mk = mkt.get("key")
            if mk and mk not in existing:
                target_markets.append(mkt)
                existing.add(mk)


def enrich_events_with_extra_markets(
    events: list[dict],
    sport_key: str,
    api_key: str,
    regions: str,
    verbose: bool = False,
    *,
    extra_markets: str | None = None,
) -> None:
    """Fetch extra markets per event and merge into events."""
    if not events:
        return
    mkts = extra_markets or EXTRA_MARKETS
    n = len(events)
    for i, ev in enumerate(events):
        eid = ev.get("id")
        if not eid:
            continue
        extra = fetch_event_odds(api_key, sport_key, eid, mkts, regions)
        if extra:
            _merge_event_markets(ev, extra)
        if verbose and (i + 1) % 10 == 0:
            print(f"  enriched {i + 1}/{n} events", file=sys.stderr)


def run_scanners(
    events: list[dict],
    markets: set[str],
    sport_key: str = "",
    safe_only: bool = False,
) -> list[dict]:
    """Run all applicable scanners based on requested markets."""
    all_arbs = []
    mkt = frozenset(markets)

    if _ENABLE_3WAY and "h2h" in mkt:
        all_arbs.extend(scan_h2h_3way(events, sport_key, "h2h"))
    
    h2h_mkts = mkt & H2H_LIKE
    if h2h_mkts:
        all_arbs.extend(scan_h2h(events, sport_key, h2h_mkts))

    skip_totals = safe_only and sport_key in ICE_HOCKEY_SPORTS
    if not skip_totals:
        for mk in TOTALS_LIKE:
            if mk in mkt:
                if mk == "alternate_totals":
                    all_arbs.extend(scan_alternate_totals_enhanced(events, sport_key))
                    all_arbs.extend(scan_cross_line_opportunities(events, sport_key))
                else:
                    all_arbs.extend(scan_totals(events, sport_key, mk))

    if not skip_totals:
        for mk in TEAM_TOTALS:
            if mk in mkt:
                all_arbs.extend(scan_team_totals(events, sport_key, mk))

    spreads_mkts = mkt & SPREADS_LIKE
    if spreads_mkts:
        alternate = spreads_mkts & {"alternate_spreads", "alternate_spreads_corners", "alternate_spreads_cards"}
        regular = spreads_mkts - alternate
        
        if regular:
            all_arbs.extend(scan_spreads(events, sport_key, regular))
        if "alternate_spreads" in alternate:
            all_arbs.extend(scan_alternate_spreads_enhanced(events, sport_key))
            all_arbs.extend(scan_asian_handicap(events, sport_key))

    if "btts" in mkt:
        all_arbs.extend(scan_btts(events, sport_key))
    if "draw_no_bet" in mkt:
        all_arbs.extend(scan_draw_no_bet(events, sport_key))
    if "double_chance" in mkt:
        all_arbs.extend(scan_double_chance(events, sport_key))
    
    if _ENABLE_4WAY:
        all_arbs.extend(scan_4way_combined_markets(events, sport_key))
    
    # 🚀 MEGA N-WAY DYNAMIC SCANNING (5-way up to 20-way!)
    if _MAX_NWAY >= 5:
        for mk in sorted(mkt):
            all_arbs.extend(scan_nway_dynamic(events, sport_key, mk, 5, min(_MAX_NWAY, 20)))

    return all_arbs


def main() -> None:
    ap = argparse.ArgumentParser(description="Ultimate Arb Scanner v3.2 - ALL MARKETS + 7-DAY DEFAULT")
    ap.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY"), help="API key")
    ap.add_argument("--sport", default="soccer_epl", help="Sport key")
    ap.add_argument("--markets", default="h2h,totals,spreads", help="Comma-separated markets")
    ap.add_argument("--preset", choices=list(SPORT_PRESETS), help="Use preset sport+markets")
    ap.add_argument("--regions", default="eu,uk,se", help="Regions")
    ap.add_argument("--days", type=int, default=0, help="Days ahead (default: 7)")
    ap.add_argument("--max-stake", type=float, default=100, help="Scale display stakes")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    ap.add_argument("--include-unreliable", action="store_true", help="Include unreliable books")
    ap.add_argument("--all-books", action="store_true", default=False, help="Use all bookmakers")
    ap.add_argument("--no-exchanges", action="store_true", help="Exclude Betfair exchanges")
    ap.add_argument("--no-finland-filter", action="store_true", help="Include US/AU bookmakers")

    ap.add_argument("--two-way-only", action="store_true", default=False, help="Only 2-outcome markets")
    ap.add_argument("--3way", action="store_true", help="Enable 3-way arb scanning")
    ap.add_argument("--4way", action="store_true", help="Enable 4-way arb scanning")
    
    ap.add_argument("--nway", type=int, default=4, help="Maximum n-way arbitrage to scan (4-20+)")
    ap.add_argument("--asian-handicap", action="store_true", help="Enable Asian handicap arbitrage scanning")
    ap.add_argument("--high-outcome-markets", action="store_true", help="Include high-outcome markets")
    
    ap.add_argument("--include-hockey-two-way", action="store_true", default=False, help="Include ice hockey")

    ap.add_argument("--fetch-all", action="store_true", default=False, help="FETCH ALL")
    ap.add_argument("--extra-markets", action="store_true", default=False, help="Fetch extra markets per event")
    ap.add_argument("--safe-only", action="store_true", default=False, help="Use only unified markets")
    ap.add_argument("--no-dedupe", action="store_true", default=False, help="Disable deduplication")
    
    ap.add_argument("--min-profit", type=float, default=0.0, help="Minimum profit percentage")
    ap.add_argument("--max-profit", type=float, default=100.0, help="Maximum profit percentage")
    ap.add_argument("--output", choices=["table", "json", "csv", "txt"], default="table", help="Output format")
    ap.add_argument("--save-to", type=str, help="Save results to file")
    ap.add_argument("--continuous", action="store_true", help="Run continuously")
    ap.add_argument("--interval", type=int, default=5, help="Minutes between scans")
    ap.add_argument("--max-scans", type=int, default=0, help="Maximum scans")

    # NEW: exclude specific bookmaker keys without removing regions
    ap.add_argument(
        "--exclude-books",
        default="",
        help="Comma-separated bookmaker keys to exclude (e.g. sportsbet,1xbet)",
    )

    args = ap.parse_args()

    # NEW: populate global exclude set
    global _EXCLUDE_BOOKS
    _EXCLUDE_BOOKS = {b.strip().lower() for b in args.exclude_books.split(",") if b.strip()}

    if args.fetch_all:
        print("🚀 FETCH-ALL MODE ENABLED (200+ markets, 7-day default) 🚀", file=sys.stderr)
        print("⚠️  WARNING: This will consume SIGNIFICANT API quota! ⚠️\n", file=sys.stderr)
        
        args.all_books = True
        args.include_unreliable = True
        args.no_finland_filter = True
        args.extra_markets = True
        args.no_dedupe = True
        args.preset = "all_games"
        args.regions = "eu,uk,us,au,se"
        args.safe_only = False
        args.nway = 20
        args.high_outcome_markets = True

    global _INCLUDE_UNRELIABLE, _USE_ALL_BOOKS, _INCLUDE_EXCHANGES, _FINLAND_ONLY, _TWO_WAY_ONLY, _ENABLE_3WAY, _ENABLE_4WAY, _MAX_NWAY
    _INCLUDE_UNRELIABLE = args.include_unreliable
    _USE_ALL_BOOKS = args.all_books
    _INCLUDE_EXCHANGES = not args.no_exchanges or args.fetch_all
    _FINLAND_ONLY = not args.no_finland_filter
    _TWO_WAY_ONLY = args.two_way_only
    _ENABLE_3WAY = getattr(args, '3way', False) or args.fetch_all
    _ENABLE_4WAY = getattr(args, '4way', False) or args.fetch_all
    
    _MAX_NWAY = min(max(args.nway, 4), 20)

    api_key = args.api_key
    if not api_key:
        print("Set ODDS_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    if _ENABLE_3WAY:
        print("✅ 3-WAY ARB SCANNING ENABLED", file=sys.stderr)
    if _ENABLE_4WAY:
        print("✅ 4-WAY ARB SCANNING ENABLED", file=sys.stderr)
    if _MAX_NWAY > 4:
        print(f"✅ {_MAX_NWAY}-WAY ARB SCANNING ENABLED (MEGA MODE!)", file=sys.stderr)
    if args.asian_handicap:
        print("✅ ASIAN HANDICAP ARB SCANNING ENABLED", file=sys.stderr)
    if args.high_outcome_markets:
        print("✅ HIGH-OUTCOME MARKETS ENABLED (horse racing, correct scores, outrights, etc.)", file=sys.stderr)
    if _INCLUDE_EXCHANGES:
        print("✅ EXCHANGE COMMISSION ADJUSTMENT ENABLED", file=sys.stderr)
    
    print(f"✅ DEFAULT SCAN WINDOW: 7 DAYS", file=sys.stderr)

    if args.fetch_all:
        markets_to_request = ",".join(ALL_KNOWN_MARKETS)
        extra_mk_str = ALL_EXTRA_MARKETS
        if args.high_outcome_markets:
            extra_mk_str += "," + ",".join(HIGH_OUTCOME_MARKETS)
    else:
        markets_to_request = args.markets
        extra_mk_str = EXTRA_MARKETS_SAFE if args.safe_only else EXTRA_MARKETS
        if args.high_outcome_markets:
            extra_mk_str += "," + ",".join(HIGH_OUTCOME_MARKETS)

    if args.preset == "all_games":
        sports = fetch_sports(api_key)
        configs = [
            (s["key"], markets_to_request if args.fetch_all else "h2h,totals,spreads")
            for s in sports
            if s.get("active")
            and "_winner" not in s.get("key", "").lower()
            and "outright" not in s.get("key", "").lower()
        ]
        if args.verbose:
            print(f"all_games: scanning {len(configs)} in-season sports", file=sys.stderr)

    elif args.preset == "two_way_h2h_all_games":
        sports = fetch_sports(api_key)
        allowed_prefixes = ("tennis_", "basketball_", "mma_", "baseball_", "americanfootball_")
        if args.include_hockey_two_way:
            allowed_prefixes = allowed_prefixes + ("icehockey_",)
        configs = [
            (s["key"], markets_to_request if args.fetch_all else "h2h,spreads")
            for s in sports
            if s.get("active")
            and s.get("key", "").startswith(allowed_prefixes)
            and "_winner" not in s.get("key", "").lower()
            and "outright" not in s.get("key", "").lower()
        ]
        if args.verbose:
            print(f"two_way_h2h_all_games: scanning {len(configs)} in-season sports", file=sys.stderr)

    elif args.preset:
        configs = SPORT_PRESETS[args.preset] or []
    else:
        configs = [(args.sport, markets_to_request if args.fetch_all else args.markets)]

    extra_mk_set = set(m.strip() for m in extra_mk_str.split(",") if m.strip())

    all_arbs: list[dict] = []
    total_events = 0
    total_api_calls = 0
    
    skip_on_error = args.preset is not None or len(configs) > 1 or args.fetch_all

    for sport_key, markets_str in configs:
        events = fetch_odds(api_key, sport_key, markets_str, args.regions, args.days, skip_on_error=skip_on_error)
        total_events += len(events)
        total_api_calls += 1

        if events and args.extra_markets:
            if args.verbose:
                print(f"[{sport_key}] fetching extra markets for {len(events)} events...", file=sys.stderr)
            enrich_events_with_extra_markets(
                events, sport_key, api_key, args.regions, args.verbose,
                extra_markets=extra_mk_str if args.safe_only or args.fetch_all else None,
            )
            total_api_calls += len(events)

        markets_set = set(m.strip() for m in markets_str.split(",") if m.strip())
        if args.extra_markets:
            markets_set |= extra_mk_set

        arbs = run_scanners(events, markets_set, sport_key=sport_key, safe_only=args.safe_only)
        all_arbs.extend(arbs)

        if args.verbose:
            print(f"[{sport_key}] {len(events)} events, {len(arbs)} arbs", file=sys.stderr)

    if args.days <= 0:
        end = datetime.fromisoformat(_iso_future_days(7).replace("Z", "+00:00"))
    else:
        end = datetime.now(timezone.utc) + timedelta(days=args.days)

    def _commence_ok(a: dict) -> bool:
        try:
            c = a.get("commence") or ""
            return datetime.fromisoformat(c.replace("Z", "+00:00")) <= end
        except (ValueError, TypeError):
            return True

    all_arbs = [a for a in all_arbs if _commence_ok(a)]

    all_arbs = [
        a for a in all_arbs 
        if args.min_profit <= a.get("profit_pct", 0) <= args.max_profit
    ]

    if not args.no_dedupe:
        by_key: dict[tuple[str, str, Any, str], dict] = {}
        for a in all_arbs:
            key = (a["event_id"], a["market"], a.get("line"), a.get("arb_type", "2-way"))
            if key not in by_key or a["profit_pct"] > by_key[key]["profit_pct"]:
                by_key[key] = a
        all_arbs = list(by_key.values())

    scale = args.max_stake / 100.0 if args.max_stake else 1.0
    
    for a in all_arbs:
        arb_type = a.get("arb_type", "2-way")
        if arb_type.endswith("-way"):
            num_outcomes = int(arb_type.split('-')[0])
            for i in range(1, num_outcomes + 1):
                a[f"stake_{i}"] = a.get(f"stake_{i}", 0) * scale
            a["total_stake"] = a.get("total_stake", 0) * scale

    all_arbs = sorted(all_arbs, key=lambda x: x.get("profit_pct", 0), reverse=True)

    if args.output == "json":
        print_json_output(all_arbs)
    elif args.output == "csv":
        print_csv_output(all_arbs)
    elif args.output == "txt":
        for i, arb in enumerate(all_arbs, 1):
            print(f"\n--- ARB #{i} ---")
            print(f"Profit: {arb.get('profit_pct', 0):.2f}%")
            print(f"Event: {arb.get('home', 'N/A')} vs {arb.get('away', 'N/A')}")
            print(f"Market: {arb.get('market', 'N/A')}")
            arb_type = arb.get('arb_type', '2-way')
            if arb_type.endswith('-way'):
                num_outcomes = int(arb_type.split('-')[0])
                for j in range(1, num_outcomes + 1):
                    print(f"  Bet {j}: {arb.get(f'outcome_{j}', 'N/A')} @ {arb.get(f'odds_{j}', 0):.2f} - ${arb.get(f'stake_{j}', 0):.2f}")
            print("-" * 80)
    else:
        if all_arbs:
            print("=" * 200)
            print("⚠️  WARNING: ALWAYS verify odds on bookmaker sites before placing bets!")
            print("=" * 200)
            print()
            
            header = (
                f"{'EVENT':<38} {'MARKET':<22} {'LN':<6} "
                f"{'SIDE A':<28} {'ODDS':<7} {'BOOK A':<16} {'STAKE A':<11} "
                f"{'SIDE B':<28} {'ODDS':<7} {'BOOK B':<16} {'STAKE B':<11} "
                f"{'PROFIT':<9}"
            )
            
            separator = "=" * 200
            
            print(separator)
            print(header)
            print(separator)
            
            for a in all_arbs[:100]:
                ev_str = format_event(a["home"], a["away"], a.get("sport_key", ""))
                ev = (ev_str[:36] + "..") if len(ev_str) > 38 else ev_str
                
                mkt = a.get('market', '')[:20]
                line_val = f"{a.get('line', '-')}" if a.get('line') is not None else "-"
                line_val = (line_val[:4] + "..") if len(line_val) > 6 else line_val
                
                arb_type = a.get("arb_type", "2-way")
                
                if arb_type in ("2-way", "2-way-middle", "2-way-asian"):
                    side_a = str(a.get('side_a', '?'))
                    side_a = (side_a[:26] + "..") if len(side_a) > 28 else side_a
                    
                    side_b = str(a.get('side_b', '?'))
                    side_b = (side_b[:26] + "..") if len(side_b) > 28 else side_b
                    
                    odds_a = f"{a.get('odds_a', 0):.2f}"
                    odds_b = f"{a.get('odds_b', 0):.2f}"
                    
                    book_a = str(a.get('book_a', '?'))[:14]
                    book_b = str(a.get('book_b', '?'))[:14]
                    
                    stake_a = f"${a.get('stake_a', 0):.2f}"
                    stake_b = f"${a.get('stake_b', 0):.2f}"
                    
                    profit = f"{a.get('profit_pct', 0):.2f}%"
                    
                    stale_marker = " *" if _is_stale(a.get("last_update_a")) or _is_stale(a.get("last_update_b")) else ""
                    warning = " ⚠" if a.get('profit_pct', 0) > 10 else ""
                    
                    exchange_marker = ""
                    if any(a.get(f"book_{side}_key", "").startswith(("betfair_ex", "smarkets", "matchbook")) for side in ["a", "b"]):
                        exchange_marker = " 📊"
                    
                    row = (
                        f"{ev:<38} {mkt:<22} {line_val:<6} "
                        f"{side_a:<28} {odds_a:<7} {book_a:<16} {stake_a:<11} "
                        f"{side_b:<28} {odds_b:<7} {book_b:<16} {stake_b:<11} "
                        f"{profit:<9}{stale_marker}{warning}{exchange_marker}"
                    )
                    
                    print(row)
                
                elif arb_type.endswith("-way"):
                    profit = f"{a.get('profit_pct', 0):.2f}%"
                    total = f"${a.get('total_stake', 0):.2f}"
                    num_outcomes = int(arb_type.split('-')[0])
                    
                    if num_outcomes <= 8:
                        emoji = "🚀"
                    else:
                        emoji = "🔥💥🚀"
                    
                    print(f"{ev:<38} {mkt:<22} {line_val:<6} **{emoji} {num_outcomes}-WAY MEGA ARB {emoji}** Profit: {profit} Total: {total}")
                    for i in range(1, num_outcomes + 1):
                        outcome = a.get(f'outcome_{i}', '?')
                        odds = a.get(f'odds_{i}', 0)
                        stake = a.get(f'stake_{i}', 0)
                        book = a.get(f'book_{i}', '?')
                        print(f"  🎯 Bet {i}: {outcome:<45} @ {odds:.2f} ({book}) - ${stake:.2f}")
                    print()
            
            print(separator)
            print("\nLEGEND:")
            print("  * = Stale odds (>30 min old) - VERIFY before betting!")
            print("  ⚠ = Suspicious profit >10% - likely error or stale data")
            print("  📊 = Exchange bet - commission already accounted for in profit")
            print("  🚀 = Mega multi-way arbitrage (5-8 outcomes)")
            print("  🔥💥🚀 = ULTRA MEGA arbitrage (9+ outcomes)")
            print()
            
        else:
            print("No arbs found.")
    
    if args.save_to:
        if args.save_to.endswith(".json"):
            export_to_json(all_arbs, args.save_to)
        elif args.save_to.endswith(".csv"):
            export_to_csv(all_arbs, args.save_to)
        elif args.save_to.endswith(".txt"):
            export_to_txt(all_arbs, args.save_to)
        else:
            export_to_txt(all_arbs, args.save_to)

    print(f"\n✅ DONE!", file=sys.stderr)
    print(f"📊 Total events: {total_events}", file=sys.stderr)
    print(f"🎯 Total arbs: {len(all_arbs)}", file=sys.stderr)
    print(f"📞 API calls: ~{total_api_calls}", file=sys.stderr)
    
    if _ENABLE_3WAY or _ENABLE_4WAY or _MAX_NWAY > 4:
        by_type = defaultdict(int)
        for a in all_arbs:
            by_type[a.get("arb_type", "2-way")] += 1
        print("\n📈 Arbs by type:", file=sys.stderr)
        for t, count in sorted(by_type.items()):
            n = int(t.split('-')[0]) if t.endswith("-way") else 0
            if n > 8:
                emoji = "🔥💥🚀"
            elif n > 4:
                emoji = "🚀"
            else:
                emoji = ""
            print(f"   {emoji}{t}: {count}", file=sys.stderr)


if __name__ == "__main__":
    parser_check = argparse.ArgumentParser(add_help=False)
    parser_check.add_argument("--continuous", action="store_true")
    parser_check.add_argument("--interval", type=int, default=5)
    parser_check.add_argument("--max-scans", type=int, default=0)
    args_check, _ = parser_check.parse_known_args()
    
    if args_check.continuous:
        scan_count = 0
        print(f"🔄 CONTINUOUS MODE: Scanning every {args_check.interval} minutes", file=sys.stderr)
        print(f"   Press Ctrl+C to stop\n", file=sys.stderr)
        
        try:
            while True:
                scan_count += 1
                print(f"\n{'='*80}", file=sys.stderr)
                print(f"🔍 SCAN #{scan_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
                
                main()
                
                if args_check.max_scans > 0 and scan_count >= args_check.max_scans:
                    print(f"\n✅ Completed {scan_count} scans", file=sys.stderr)
                    break
                
                print(f"\n⏳ Waiting {args_check.interval} minutes...", file=sys.stderr)
                time.sleep(args_check.interval * 60)
                
        except KeyboardInterrupt:
            print(f"\n\n⛔ Stopped after {scan_count} scans", file=sys.stderr)
            sys.exit(0)
    else:
        main()