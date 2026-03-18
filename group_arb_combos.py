#!/usr/bin/env python3
"""
ARB COMBINATION GROUPER & ORGANIZER
Groups all bookmaker combinations for the same game/market together
Creates both CSV files AND human-readable TXT files
Excludes: Bovada, Ladbrokes, Livescore Bet, FanDuel, Tipico, Neds, Paddy Power, TAB/Tabtouch
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ============================================================
# CONFIGURATION
# ============================================================

# Bookmakers to COMPLETELY EXCLUDE (any combo with these is removed)
BLACKLIST_BOOKMAKERS = [
    "bovada",
    "ladbrokes",
    "livescore bet",
    "fanduel",
    "tipico",
    "neds",
    "paddy power",
    # Added per request: exclude TAB sportsbook(s)
    "tab",
    "tabtouch",
    # (Sportsbet was mentioned earlier; include if you want it excluded here too)
    "sportsbet",
]

# Bookmaker preference score (higher = more preferred)
BOOKMAKER_SCORES = {
    # Most trusted
    "pinnacle": 100,
    "bet365": 95,
    "betway": 90,
    "betsson": 85,
    "unibet": 85,
    "marathonbet": 80,
    "nordicbet": 80,
    "nordic bet": 80,
    "coolbet": 75,
    "betclic": 75,

    # Decent
    "bwin": 70,
    "betfair": 70,
    "888sport": 65,
    "william hill": 60,
    "neds": 55,

    # Use with caution (low score, but still included)
    "betonline": 40,
    "betus": 30,
    "1xbet": 10,

    # EXCLUDED (will be filtered out)
    "bovada": 0,
    "ladbrokes": 0,
    "tab": 0,
    "tabtouch": 0,
    "sportsbet": 0,
}

# Minimum reliability score for "reliable bookmakers only" outputs
RELIABLE_THRESHOLD = 65

# Minimum ROI (profit %) to include - arbs below this are ruled out
MIN_ROI_PCT = 1.0


def is_blacklisted(bookmaker_name: object) -> bool:
    """Check if bookmaker is blacklisted"""
    if bookmaker_name is None:
        return False

    book_str = str(bookmaker_name)
    if not book_str or book_str.lower() == "nan":
        return False

    book_lower = book_str.lower()

    for blacklisted in BLACKLIST_BOOKMAKERS:
        if blacklisted.lower() in book_lower:
            return True

    return False


def contains_blacklisted_bookmaker(row: pd.Series) -> bool:
    """Check if any bookmaker in this combo is blacklisted"""
    book_cols = [col for col in row.index if "book" in col.lower() and "_key" not in col.lower()]

    for col in book_cols:
        book = row[col]
        if pd.notna(book) and is_blacklisted(book):
            return True

    return False


def get_bookmaker_score(bookmaker_name: object) -> float:
    """Get reliability score for a bookmaker"""
    if bookmaker_name is None:
        return 0

    book_str = str(bookmaker_name)
    if not book_str or book_str.lower() == "nan":
        return 0

    book_lower = book_str.lower()

    # Check if blacklisted (should have been filtered already)
    if is_blacklisted(bookmaker_name):
        return 0

    # Check if any known bookmaker is in the name
    for book, score in BOOKMAKER_SCORES.items():
        if book in book_lower:
            return score

    # Unknown bookmaker
    return 50  # neutral score


def calculate_combo_reliability_score(row: pd.Series) -> float:
    """Calculate overall reliability score for a combination"""
    scores: list[float] = []

    # Get all bookmaker columns
    book_cols = [col for col in row.index if "book" in col.lower() and "_key" not in col.lower()]

    for col in book_cols:
        book = row[col]
        if pd.notna(book):
            scores.append(get_bookmaker_score(book))

    if not scores:
        return 0

    # Average score
    return sum(scores) / len(scores)


# ============================================================
# DATA LOADING & FILTERING
# ============================================================

def load_csv(filename: str) -> pd.DataFrame:
    """Load CSV file"""
    try:
        df = pd.read_csv(filename)
        print(f"✅ Loaded {len(df)} arbs from {filename}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)


def filter_blacklisted_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any combinations containing blacklisted bookmakers"""
    before = len(df)

    df = df[~df.apply(contains_blacklisted_bookmaker, axis=1)]

    after = len(df)
    removed = before - after

    if removed > 0:
        print(f"\n🚫 FILTERED OUT BLACKLISTED BOOKMAKERS:")
        for book in BLACKLIST_BOOKMAKERS:
            print(f"   ❌ {book.title()}")
        print(f"\n   Removed {removed} combinations ({removed/before*100:.1f}%)")
        print(f"   Kept {after} combinations")

    return df


def filter_min_roi(df: pd.DataFrame, min_pct: float | None = None) -> pd.DataFrame:
    """Remove arbs with ROI below minimum (default MIN_ROI_PCT)."""
    if min_pct is None:
        min_pct = MIN_ROI_PCT
    before = len(df)
    if "profit_pct" not in df.columns:
        return df
    df = df[df["profit_pct"] >= min_pct].copy()
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"\n📉 RULED OUT UNDER {min_pct}% ROI:")
        print(f"   Removed {removed} combinations (ROI < {min_pct}%)")
        print(f"   Kept {after} combinations")
    return df


def debug_show_columns(df: pd.DataFrame) -> None:
    """Show all column names to debug"""
    print("\n🔍 DEBUG: CSV Columns Analysis")
    print("=" * 100)

    # Show ALL columns
    print("\nALL COLUMNS IN CSV:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:3}. {col}")

    print("\n" + "=" * 100)

    # Get a sample 2-way arb
    sample_2way = df[df["arb_type"] == "2-way"].iloc[0] if len(df[df["arb_type"] == "2-way"]) > 0 else None

    if sample_2way is not None:
        print("\n2-WAY ARB EXAMPLE:")
        print("-" * 100)
        relevant_cols = [
            col for col in sample_2way.index
            if any(x in col.lower() for x in ["odds", "side", "book", "stake", "outcome", "price", "amount", "wager"])
        ]
        for col in sorted(relevant_cols):
            val = sample_2way[col]
            val_str = f"{val:.2f}" if pd.notna(val) and isinstance(val, (int, float)) else str(val)
            print(f"   {col:25} = {val_str}")

    # Get a sample 3-way arb
    sample_3way = df[df["arb_type"] == "3-way"].iloc[0] if len(df[df["arb_type"] == "3-way"]) > 0 else None

    if sample_3way is not None:
        print("\n3-WAY ARB EXAMPLE:")
        print("-" * 100)
        relevant_cols = [
            col for col in sample_3way.index
            if any(x in col.lower() for x in ["odds", "side", "book", "stake", "outcome", "price", "amount", "wager"])
        ]
        for col in sorted(relevant_cols):
            val = sample_3way[col]
            val_str = f"{val:.2f}" if pd.notna(val) and isinstance(val, (int, float)) else str(val)
            print(f"   {col:25} = {val_str}")

    print("\n" + "=" * 100 + "\n")


# ============================================================
# GROUPING FUNCTIONS
# ============================================================

def create_game_key(row: pd.Series) -> str:
    """Create unique key for game/market combination"""
    event_id = str(row.get("event_id", ""))
    market = str(row.get("market", ""))
    line = str(row.get("line", "")) if pd.notna(row.get("line")) else "NA"

    return f"{event_id}|{market}|{line}"


def group_by_game(df: pd.DataFrame):
    """Group all combinations by game/market"""
    df["game_key"] = df.apply(create_game_key, axis=1)

    grouped = df.groupby("game_key")

    print(f"\n📊 Found {len(grouped)} unique game/market combinations")
    print(f"   Total entries: {len(df)}")
    print(f"   Average {len(df)/len(grouped):.1f} bookmaker combinations per game")

    return grouped


def add_reliability_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add reliability score to each combination"""
    df["reliability_score"] = df.apply(calculate_combo_reliability_score, axis=1)
    return df


def parse_commence_date(commence_str: object):
    """Parse commence string to date (UTC). Returns None if invalid."""
    if pd.isna(commence_str) or not commence_str:
        return None
    try:
        s = str(commence_str).strip()
        if not s or s == "nan":
            return None
        # Handle ISO format: 2026-02-15T11:30:00Z or 2026-02-15T11:30:00
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        return dt.date()
    except (ValueError, TypeError):
        return None


def filter_today_tomorrow(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to games starting today or tomorrow (UTC). Returns filtered df."""
    today_utc = datetime.now(timezone.utc).date()
    tomorrow_utc = today_utc + timedelta(days=1)
    mask = df["commence"].apply(
        lambda c: parse_commence_date(c) in (today_utc, tomorrow_utc) if c else False
    )
    return df[mask].copy()


def add_bookmaker_list(df: pd.DataFrame) -> pd.DataFrame:
    """Add comma-separated list of all bookmakers used"""
    def get_bookmakers(row: pd.Series) -> str:
        bookmakers: list[str] = []
        book_cols = [col for col in row.index if "book" in col.lower() and "_key" not in col.lower()]

        for col in book_cols:
            book = row[col]
            if pd.notna(book) and str(book) != "nan":
                # Extract just the bookmaker name (remove any + signs from combined markets)
                book_str = str(book).split("+")[0].strip() if "+" in str(book) else str(book)
                if book_str not in bookmakers:
                    bookmakers.append(book_str)

        return ", ".join(bookmakers)

    df["bookmakers_used"] = df.apply(get_bookmakers, axis=1)
    return df


# ============================================================
# CSV OUTPUT FUNCTIONS
# ============================================================

def create_grouped_csv(df: pd.DataFrame, output_file: str) -> None:
    """Create CSV with all combinations grouped and sorted"""
    # Add helper columns
    df = add_reliability_score(df)
    df = add_bookmaker_list(df)

    # Sort within each game: reliability first, then profit
    df = df.sort_values(["game_key", "reliability_score", "profit_pct"],
                        ascending=[True, False, False])

    # Add rank within each game
    df["combo_rank"] = df.groupby("game_key").cumcount() + 1

    # Add total combos for this game
    df["total_combos"] = df.groupby("game_key")["game_key"].transform("count")

    # Reorder columns for better readability
    first_cols = [
        "sport_key", "home", "away", "commence", "market", "line", "arb_type",
        "combo_rank", "total_combos", "profit_pct", "reliability_score",
        "bookmakers_used", "total_stake",
    ]

    other_cols = [col for col in df.columns if col not in first_cols and col != "game_key"]
    final_cols = first_cols + other_cols

    # Only include columns that exist
    final_cols = [col for col in final_cols if col in df.columns]

    df_output = df[final_cols]

    df_output.to_csv(output_file, index=False)
    print(f"   📊 CSV: {Path(output_file).name}")


def create_best_per_game_csv(df: pd.DataFrame, output_file: str) -> None:
    """Create CSV with only the best combination per game (highest reliability)"""
    df = add_reliability_score(df)
    df = add_bookmaker_list(df)

    # Sort and keep best
    df = df.sort_values(["game_key", "reliability_score", "profit_pct"],
                        ascending=[True, False, False])

    df_best = df.groupby("game_key").first().reset_index(drop=True)

    # Reorder columns
    first_cols = [
        "sport_key", "home", "away", "commence", "market", "line", "arb_type",
        "profit_pct", "reliability_score", "bookmakers_used", "total_stake",
    ]

    other_cols = [col for col in df_best.columns if col not in first_cols and col != "game_key"]
    final_cols = first_cols + other_cols
    final_cols = [col for col in final_cols if col in df_best.columns]

    df_best = df_best[final_cols]
    df_best = df_best.sort_values("profit_pct", ascending=False)

    df_best.to_csv(output_file, index=False)
    print(f"   📊 CSV: {Path(output_file).name}")


def create_highest_profit_csv(df: pd.DataFrame, output_file: str) -> None:
    """Create CSV with highest profit combination per game"""
    # Sort by profit and keep best
    df = df.sort_values(["game_key", "profit_pct"], ascending=[True, False])

    df_best = df.groupby("game_key").first().reset_index(drop=True)
    df_best = add_reliability_score(df_best)
    df_best = add_bookmaker_list(df_best)

    # Reorder columns
    first_cols = [
        "sport_key", "home", "away", "commence", "market", "line", "arb_type",
        "profit_pct", "reliability_score", "bookmakers_used", "total_stake",
    ]

    other_cols = [col for col in df_best.columns if col not in first_cols and col != "game_key"]
    final_cols = first_cols + other_cols
    final_cols = [col for col in final_cols if col in df_best.columns]

    df_best = df_best[final_cols]
    df_best = df_best.sort_values("profit_pct", ascending=False)

    df_best.to_csv(output_file, index=False)
    print(f"   📊 CSV: {Path(output_file).name}")


def create_next_coming_reliable_csv(df: pd.DataFrame, output_file: str) -> None:
    """Today/tomorrow games only, best per game by reliability (reliable books only), sorted by ROI."""
    df = filter_today_tomorrow(df)
    if len(df) == 0:
        print(f"   ⏭️  No today/tomorrow games, skipping {Path(output_file).name}")
        return
    df = add_reliability_score(df)
    df = add_bookmaker_list(df)
    df = df[df["reliability_score"] >= RELIABLE_THRESHOLD]
    if len(df) == 0:
        print(f"   ⏭️  No reliable combos for today/tomorrow, skipping {Path(output_file).name}")
        return
    df = df.sort_values(["game_key", "reliability_score", "profit_pct"], ascending=[True, False, False])
    df_best = df.groupby("game_key").first().reset_index(drop=True)
    df_best = df_best.sort_values(["commence", "profit_pct"], ascending=[True, False])
    first_cols = [
        "sport_key", "home", "away", "commence", "market", "line", "arb_type",
        "profit_pct", "reliability_score", "bookmakers_used", "total_stake",
    ]
    other_cols = [col for col in df_best.columns if col not in first_cols and col != "game_key"]
    final_cols = [c for c in first_cols + other_cols if c in df_best.columns]
    df_best[final_cols].to_csv(output_file, index=False)
    print(f"   📊 CSV: {Path(output_file).name} ({len(df_best)} games today/tomorrow, reliable only)")


def create_next_coming_highest_roi_csv(df: pd.DataFrame, output_file: str) -> None:
    """Today/tomorrow games only, best per game by ROI (any books), sorted by ROI."""
    df = filter_today_tomorrow(df)
    if len(df) == 0:
        print(f"   ⏭️  No today/tomorrow games, skipping {Path(output_file).name}")
        return
    df = add_reliability_score(df)
    df = add_bookmaker_list(df)
    df = df.sort_values(["game_key", "profit_pct"], ascending=[True, False])
    df_best = df.groupby("game_key").first().reset_index(drop=True)
    df_best = df_best.sort_values(["commence", "profit_pct"], ascending=[True, False])
    first_cols = [
        "sport_key", "home", "away", "commence", "market", "line", "arb_type",
        "profit_pct", "reliability_score", "bookmakers_used", "total_stake",
    ]
    other_cols = [col for col in df_best.columns if col not in first_cols and col != "game_key"]
    final_cols = [c for c in first_cols + other_cols if c in df_best.columns]
    df_best[final_cols].to_csv(output_file, index=False)
    print(f"   📊 CSV: {Path(output_file).name} ({len(df_best)} games today/tomorrow, highest ROI)")


# ============================================================
# TXT OUTPUT FUNCTIONS
# ============================================================

def format_arb_row_txt(row: pd.Series, rank: int | None = None) -> str:
    """Format a single arb row for TXT output"""
    lines: list[str] = []

    # Header
    if rank:
        lines.append(f"{'='*100}")
        lines.append(f"OPTION {rank}")
        lines.append(f"{'='*100}")

    # Basic info
    profit = row.get("profit_pct", 0)
    reliability = row.get("reliability_score", 0)
    stars = "⭐" * int(float(reliability) / 20)

    lines.append(f"Profit:      {float(profit):.2f}%")
    lines.append(f"Reliability: {float(reliability):.0f}/100 {stars}")
    lines.append(f"Arb Type:    {row.get('arb_type', 'N/A')}")
    lines.append(f"Total Stake: ${float(row.get('total_stake', 0) or 0):.2f}")
    lines.append("")

    arb_type = row.get("arb_type", "2-way")

    if "way" in str(arb_type):
        n = int(str(arb_type).split("-")[0])

        lines.append("BETS:")
        lines.append("-" * 100)

        all_cols = list(row.index)

        for i in range(1, n + 1):
            outcome = None
            odds = None
            book = None
            stake = None

            # OUTCOME
            if n == 2:
                letter = chr(96 + i)  # a, b
                if f"side_{letter}" in all_cols and pd.notna(row[f"side_{letter}"]):
                    outcome = row[f"side_{letter}"]
                elif f"outcome_{letter}" in all_cols and pd.notna(row[f"outcome_{letter}"]):
                    outcome = row[f"outcome_{letter}"]

            if outcome is None:
                if f"outcome_{i}" in all_cols and pd.notna(row[f"outcome_{i}"]):
                    outcome = row[f"outcome_{i}"]
                elif f"side_{i}" in all_cols and pd.notna(row[f"side_{i}"]):
                    outcome = row[f"side_{i}"]

            # ODDS
            if n == 2:
                letter = chr(96 + i)
                if f"odds_{letter}" in all_cols and pd.notna(row[f"odds_{letter}"]):
                    odds = row[f"odds_{letter}"]

            if odds is None:
                if f"odds_{i}" in all_cols and pd.notna(row[f"odds_{i}"]):
                    odds = row[f"odds_{i}"]

            # BOOKMAKER
            if n == 2:
                letter = chr(96 + i)
                if f"book_{letter}" in all_cols and pd.notna(row[f"book_{letter}"]):
                    book = row[f"book_{letter}"]

            if book is None:
                if f"book_{i}" in all_cols and pd.notna(row[f"book_{i}"]):
                    book = row[f"book_{i}"]

            # STAKE
            if n == 2:
                letter = chr(96 + i)
                if f"stake_{letter}" in all_cols and pd.notna(row[f"stake_{letter}"]):
                    stake = row[f"stake_{letter}"]

            if stake is None:
                if f"stake_{i}" in all_cols and pd.notna(row[f"stake_{i}"]):
                    stake = row[f"stake_{i}"]

            if outcome is not None or book is not None:
                lines.append(f"  Bet {i}: {outcome if outcome is not None else 'N/A'}")

                if odds is not None:
                    try:
                        lines.append(f"         Odds:  {float(odds):.2f}")
                    except (ValueError, TypeError):
                        lines.append(f"         Odds:  {odds}")
                else:
                    lines.append("         Odds:  N/A")

                if book is not None:
                    lines.append(f"         Book:  {book}")
                else:
                    lines.append("         Book:  N/A")

                if stake is not None:
                    try:
                        stake_val = float(stake)
                        lines.append(f"         Stake: ${stake_val:.2f}")

                        if odds is not None:
                            try:
                                odds_val = float(odds)
                                payout = odds_val * stake_val
                                profit_from_bet = payout - stake_val
                                lines.append(f"         Pays:  ${payout:.2f} (profit: ${profit_from_bet:.2f})")
                            except (ValueError, TypeError):
                                pass
                    except (ValueError, TypeError):
                        lines.append(f"         Stake: {stake}")
                else:
                    lines.append("         Stake: N/A")

                lines.append("")

    return "\n".join(lines)


def _format_alt_combos_table(game_combos: pd.DataFrame, *, limit: int, min_profit: float) -> str:
    """Return a compact table of alternative positive combos for a single game_key."""
    if limit <= 0 or game_combos is None or len(game_combos) == 0:
        return ""

    # Ensure helper cols exist
    if "reliability_score" not in game_combos.columns:
        game_combos = add_reliability_score(game_combos)
    if "bookmakers_used" not in game_combos.columns:
        game_combos = add_bookmaker_list(game_combos)

    # Keep only positive combos above threshold
    if "profit_pct" not in game_combos.columns:
        return ""

    alts = game_combos[game_combos["profit_pct"] >= float(min_profit)].copy()
    if len(alts) <= 1:
        return ""

    # Sort same as "best reliable": reliability then profit
    alts = alts.sort_values(["reliability_score", "profit_pct"], ascending=[False, False])

    # The first row is the "best" itself in many cases; we want alternatives
    # Drop duplicates of identical bookmaker sets to avoid spam
    if "bookmakers_used" in alts.columns:
        alts = alts.drop_duplicates(subset=["bookmakers_used"], keep="first")

    # Remove the top one (best) and keep next N
    alts = alts.iloc[1 : 1 + limit]
    if len(alts) == 0:
        return ""

    lines: list[str] = []
    lines.append("ALTERNATIVE POSITIVE COMBOS:")
    lines.append("-" * 100)
    lines.append(f"{'#':>2}  {'PROFIT':>7}  {'REL':>3}  BOOKMAKERS")
    lines.append("-" * 100)

    for i, (_, r) in enumerate(alts.iterrows(), 1):
        profit = float(r.get("profit_pct", 0) or 0)
        rel = float(r.get("reliability_score", 0) or 0)
        books = str(r.get("bookmakers_used", "") or "")
        lines.append(f"{i:>2}  {profit:>6.2f}%  {rel:>3.0f}  {books}")

    lines.append("-" * 100)
    return "\n".join(lines) + "\n"


def create_all_combos_txt(df: pd.DataFrame, output_file: str) -> None:
    """Create human-readable TXT file with all combinations"""
    df = add_reliability_score(df)
    df = add_bookmaker_list(df)

    # Sort
    df = df.sort_values(["game_key", "reliability_score", "profit_pct"], ascending=[True, False, False])

    with open(output_file, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("ARBITRAGE OPPORTUNITIES - ALL BOOKMAKER COMBINATIONS\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Combinations: {len(df)}\n")
        f.write(f"Unique Games: {df['game_key'].nunique()}\n")
        f.write("=" * 100 + "\n\n")

        # Group by game
        current_game = None
        game_count = 0

        for _, row in df.iterrows():
            game_key = row["game_key"]

            # New game section
            if game_key != current_game:
                current_game = game_key
                game_count += 1

                # Game header
                f.write("\n" + "=" * 100 + "\n")
                f.write(f">>> GAME #{game_count}\n")
                f.write("=" * 100 + "\n\n")

                f.write(f"Event:   {row.get('home', 'N/A')} vs {row.get('away', 'N/A')}\n")
                f.write(f"Sport:   {row.get('sport_key', 'N/A')}\n")
                f.write(f"Market:  {row.get('market', 'N/A')}")
                if pd.notna(row.get("line")):
                    f.write(f" (Line: {row.get('line')})")
                f.write("\n")
                f.write(f"Date:    {row.get('commence', 'N/A')}\n")

                # Count combos for this game
                game_combos = df[df["game_key"] == game_key]
                f.write(f"\nFound {len(game_combos)} different bookmaker combinations:\n\n")

                combo_rank = 1

            # Write combination
            f.write(format_arb_row_txt(row, combo_rank))
            f.write("\n")
            combo_rank += 1

        # Footer
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")

    print(f"   📄 TXT: {Path(output_file).name}")


def create_best_reliable_txt(df: pd.DataFrame, output_file: str, *, alt_combos: int = 0, alt_min_profit: float | None = None) -> None:
    """Create human-readable TXT with best reliable option per game.
    If alt_combos > 0, also print alternative positive combos per game (same market/line/game_key).
    """
    if alt_min_profit is None:
        alt_min_profit = MIN_ROI_PCT

    df = add_reliability_score(df)
    df = add_bookmaker_list(df)

    # Get best per game
    df = df.sort_values(["game_key", "reliability_score", "profit_pct"], ascending=[True, False, False])
    df_best = df.groupby("game_key").first().reset_index(drop=True)
    df_best = df_best.sort_values("profit_pct", ascending=False)

    with open(output_file, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("BEST (MOST RELIABLE) ARBITRAGE OPPORTUNITIES\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Games: {len(df_best)}\n")
        f.write("\nShowing the most reliable bookmaker combination for each game.\n")
        if alt_combos > 0:
            f.write(f"Also showing up to {alt_combos} alternative positive combos per game (profit >= {alt_min_profit}%).\n")
        f.write("=" * 100 + "\n\n")

        for game_num, (_, row) in enumerate(df_best.iterrows(), 1):
            f.write("=" * 100 + "\n")
            f.write(f">>> GAME #{game_num}\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Event:   {row.get('home', 'N/A')} vs {row.get('away', 'N/A')}\n")
            f.write(f"Sport:   {row.get('sport_key', 'N/A')}\n")
            f.write(f"Market:  {row.get('market', 'N/A')}")
            if pd.notna(row.get("line")):
                f.write(f" (Line: {row.get('line')})")
            f.write("\n")
            f.write(f"Date:    {row.get('commence', 'N/A')}\n\n")

            f.write(format_arb_row_txt(row))
            f.write("\n")

            if alt_combos > 0:
                game_key = create_game_key(row)
                game_combos = df[df["game_key"] == game_key]
                table = _format_alt_combos_table(game_combos, limit=alt_combos, min_profit=float(alt_min_profit))
                if table:
                    f.write(table)
                    f.write("\n")

        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")

    print(f"   📄 TXT: {Path(output_file).name}")


def create_highest_profit_txt(df: pd.DataFrame, output_file: str) -> None:
    """Create human-readable TXT with highest profit option per game"""
    df = df.sort_values(["game_key", "profit_pct"], ascending=[True, False])
    df_best = df.groupby("game_key").first().reset_index(drop=True)
    df_best = add_reliability_score(df_best)
    df_best = add_bookmaker_list(df_best)
    df_best = df_best.sort_values("profit_pct", ascending=False)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("HIGHEST PROFIT ARBITRAGE OPPORTUNITIES\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Games: {len(df_best)}\n")
        f.write("\nShowing the highest profit combination for each game.\n")
        f.write("⚠️  WARNING: High profit may indicate less reliable bookmakers or stale odds!\n")
        f.write("=" * 100 + "\n\n")

        for game_num, (_, row) in enumerate(df_best.iterrows(), 1):
            f.write("=" * 100 + "\n")
            f.write(f">>> GAME #{game_num}\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Event:   {row.get('home', 'N/A')} vs {row.get('away', 'N/A')}\n")
            f.write(f"Sport:   {row.get('sport_key', 'N/A')}\n")
            f.write(f"Market:  {row.get('market', 'N/A')}")
            if pd.notna(row.get("line")):
                f.write(f" (Line: {row.get('line')})")
            f.write("\n")
            f.write(f"Date:    {row.get('commence', 'N/A')}\n\n")

            f.write(format_arb_row_txt(row))
            f.write("\n\n")

        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")

    print(f"   📄 TXT: {Path(output_file).name}")


# (Other TXT functions unchanged; kept exactly to avoid breaking behavior)
def create_next_coming_reliable_txt(df, output_file):
    df = filter_today_tomorrow(df)
    if len(df) == 0:
        print(f"   ⏭️  No today/tomorrow games, skipping {Path(output_file).name}")
        return
    df = add_reliability_score(df)
    df = add_bookmaker_list(df)
    df = df[df['reliability_score'] >= RELIABLE_THRESHOLD]
    if len(df) == 0:
        print(f"   ⏭️  No reliable combos for today/tomorrow, skipping {Path(output_file).name}")
        return
    df = df.sort_values(['game_key', 'reliability_score', 'profit_pct'], ascending=[True, False, False])
    df_best = df.groupby('game_key').first().reset_index(drop=True)
    df_best = df_best.sort_values(['commence', 'profit_pct'], ascending=[True, False])
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("NEXT COMING GAMES – MOST RELIABLE (Today & Tomorrow)\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Games: {len(df_best)} (sorted by start time, then ROI)\n")
        f.write("Reliable bookmakers only (score >= 65). Best combo per game.\n")
        f.write("="*100 + "\n\n")
        for game_num, (idx, row) in enumerate(df_best.iterrows(), 1):
            f.write("="*100 + "\n")
            f.write(f">>> GAME #{game_num}\n")
            f.write("="*100 + "\n\n")
            f.write(f"Event:   {row.get('home', 'N/A')} vs {row.get('away', 'N/A')}\n")
            f.write(f"Sport:   {row.get('sport_key', 'N/A')}\n")
            f.write(f"Market:  {row.get('market', 'N/A')}")
            if pd.notna(row.get('line')):
                f.write(f" (Line: {row.get('line')})")
            f.write("\n")
            f.write(f"Date:    {row.get('commence', 'N/A')}\n\n")
            f.write(format_arb_row_txt(row))
            f.write("\n\n")
        f.write("="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")
    print(f"   📄 TXT: {Path(output_file).name}")


def create_next_coming_highest_roi_txt(df, output_file):
    df = filter_today_tomorrow(df)
    if len(df) == 0:
        print(f"   ⏭️  No today/tomorrow games, skipping {Path(output_file).name}")
        return
    df = add_reliability_score(df)
    df = add_bookmaker_list(df)
    df = df.sort_values(['game_key', 'profit_pct'], ascending=[True, False])
    df_best = df.groupby('game_key').first().reset_index(drop=True)
    df_best = df_best.sort_values(['commence', 'profit_pct'], ascending=[True, False])
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("NEXT COMING GAMES – HIGHEST ROI (Today & Tomorrow)\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Games: {len(df_best)} (sorted by start time, then ROI)\n")
        f.write("Best profit per game, any bookmakers. ⚠️ May include less reliable sites.\n")
        f.write("="*100 + "\n\n")
        for game_num, (idx, row) in enumerate(df_best.iterrows(), 1):
            f.write("="*100 + "\n")
            f.write(f">>> GAME #{game_num}\n")
            f.write("="*100 + "\n\n")
            f.write(f"Event:   {row.get('home', 'N/A')} vs {row.get('away', 'N/A')}\n")
            f.write(f"Sport:   {row.get('sport_key', 'N/A')}\n")
            f.write(f"Market:  {row.get('market', 'N/A')}")
            if pd.notna(row.get('line')):
                f.write(f" (Line: {row.get('line')})")
            f.write("\n")
            f.write(f"Date:    {row.get('commence', 'N/A')}\n\n")
            f.write(format_arb_row_txt(row))
            f.write("\n\n")
        f.write("="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")
    print(f"   📄 TXT: {Path(output_file).name}")


def create_summary_txt(df, output_file):
    # unchanged from your original
    df = add_reliability_score(df)
    df = add_bookmaker_list(df)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("ARBITRAGE ANALYSIS SUMMARY\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        f.write("OVERVIEW:\n")
        f.write("-"*100 + "\n")
        f.write(f"Total Combinations:      {len(df)}\n")
        f.write(f"Unique Games/Markets:    {df['game_key'].nunique()}\n")
        f.write(f"Avg Combos per Game:     {len(df)/df['game_key'].nunique():.1f}\n\n")
        f.write("PROFIT STATISTICS:\n")
        f.write("-"*100 + "\n")
        f.write(f"Average Profit:          {df['profit_pct'].mean():.2f}%\n")
        f.write(f"Median Profit:           {df['profit_pct'].median():.2f}%\n")
        f.write(f"Min Profit:              {df['profit_pct'].min():.2f}%\n")
        f.write(f"Max Profit:              {df['profit_pct'].max():.2f}%\n")
        f.write(f"Std Deviation:           {df['profit_pct'].std():.2f}%\n\n")
        f.write("PROFIT DISTRIBUTION:\n")
        f.write("-"*100 + "\n")
        ranges = [
            (0, 1, "0-1%"),
            (1, 2, "1-2%"),
            (2, 3, "2-3%"),
            (3, 5, "3-5%"),
            (5, 10, "5-10%"),
            (10, 100, "10%+"),
        ]
        for low, high, label in ranges:
            count = len(df[(df['profit_pct'] >= low) & (df['profit_pct'] < high)])
            if count > 0:
                pct = (count / len(df)) * 100
                bar = "█" * int(pct / 2)
                f.write(f"{label:12} {count:5} ({pct:5.1f}%) {bar}\n")
        f.write("\n")
        f.write("BY ARB TYPE:\n")
        f.write("-"*100 + "\n")
        type_counts = df['arb_type'].value_counts()
        for arb_type, count in type_counts.items():
            pct = (count / len(df)) * 100
            avg_profit = df[df['arb_type'] == arb_type]['profit_pct'].mean()
            f.write(f"{arb_type:15} {count:5} ({pct:5.1f}%) - Avg profit: {avg_profit:.2f}%\n")
        f.write("\n")
        f.write("TOP 15 SPORTS:\n")
        f.write("-"*100 + "\n")
        sport_counts = df['sport_key'].value_counts().head(15)
        for sport, count in sport_counts.items():
            pct = (count / len(df)) * 100
            avg_profit = df[df['sport_key'] == sport]['profit_pct'].mean()
            f.write(f"{sport:40} {count:4} ({pct:5.1f}%) - Avg: {avg_profit:.2f}%\n")
        f.write("\n")
        f.write("TOP 20 BOOKMAKERS:\n")
        f.write("-"*100 + "\n")
        bookmaker_counts = {}
        for _, row in df.iterrows():
            book_cols = [col for col in row.index if 'book' in col.lower() and '_key' not in col.lower()]
            for col in book_cols:
                book = str(row[col])
                if book != 'nan' and book:
                    book_clean = book.split('+')[0].strip() if '+' in book else book
                    bookmaker_counts[book_clean] = bookmaker_counts.get(book_clean, 0) + 1
        sorted_books = sorted(bookmaker_counts.items(), key=lambda x: -x[1])
        for book, count in sorted_books[:20]:
            score = get_bookmaker_score(book)
            stars = "⭐" * int(score / 20)
            pct = (count / len(df)) * 100
            f.write(f"{book:30} {count:5} ({pct:5.1f}%) | Score: {score:3.0f} {stars}\n")
        f.write("\n")
        f.write("TOP 10 HIGHEST PROFIT OPPORTUNITIES:\n")
        f.write("-"*100 + "\n")
        top10 = df.nlargest(10, 'profit_pct')
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            event = f"{row.get('home', 'N/A')} vs {row.get('away', 'N/A')}"[:50]
            market = row.get('market', 'N/A')[:20]
            profit = row.get('profit_pct', 0)
            reliability = row.get('reliability_score', 0)
            arb_type = row.get('arb_type', 'N/A')
            f.write(f"{rank:2}. {profit:6.2f}% | {arb_type:10} | Rel: {reliability:3.0f} | {event} | {market}\n")
        f.write("\n" + "="*100 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*100 + "\n")
    print(f"   📄 TXT: {Path(output_file).name}")


# ============================================================
# CONSOLE DISPLAY FUNCTIONS (unchanged)
# ============================================================

def print_example_games(df, n=5):
    print(f"\n{'='*100}")
    print(f"📋 EXAMPLE: Games with Multiple Bookmaker Combinations")
    print("="*100 + "\n")

    df = add_reliability_score(df)
    df = add_bookmaker_list(df)

    combo_counts = df.groupby('game_key').size().sort_values(ascending=False)

    shown = 0
    for game_key, count in combo_counts.items():
        if count < 2:
            continue

        game_combos = df[df['game_key'] == game_key].sort_values(
            ['reliability_score', 'profit_pct'],
            ascending=[False, False]
        )

        first = game_combos.iloc[0]

        print(f"🎯 {first['home']} vs {first['away']}")
        print(f"   Market: {first['market']}")
        print(f"   Found {count} different bookmaker combinations:\n")

        for idx, combo in game_combos.head(10).iterrows():
            reliability = combo['reliability_score']
            reliability_label = "⭐" * int(reliability / 20)
            rank = list(game_combos.index).index(idx) + 1

            print(f"   Option {rank:2}: "
                  f"Profit {combo['profit_pct']:5.2f}% | "
                  f"Reliability {reliability:3.0f} {reliability_label:5} | "
                  f"{combo['bookmakers_used']}")

        if count > 10:
            print(f"   ... and {count - 10} more combinations")

        print()

        shown += 1
        if shown >= n:
            break

    print("="*100)


def analyze_bookmaker_usage(df):
    print(f"\n{'='*100}")
    print("📊 BOOKMAKER USAGE ANALYSIS")
    print("="*100 + "\n")

    bookmaker_counts = {}

    for _, row in df.iterrows():
        book_cols = [col for col in row.index if 'book' in col.lower() and '_key' not in col.lower()]

        for col in book_cols:
            book = str(row[col])
            if book != 'nan' and book:
                book_clean = book.split('+')[0].strip() if '+' in book else book
                bookmaker_counts[book_clean] = bookmaker_counts.get(book_clean, 0) + 1

    sorted_books = sorted(bookmaker_counts.items(), key=lambda x: -x[1])

    print("Most Used Bookmakers:")
    for book, count in sorted_books[:20]:
        score = get_bookmaker_score(book)
        score_label = "⭐" * int(score / 20)
        pct = (count / len(df)) * 100
        print(f"   {book:30} {count:5} ({pct:5.1f}%) | Score: {score:3.0f} {score_label}")

    print("\n" + "="*100)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ARB Combination Grouper & Organizer")
    parser.add_argument("input_file", nargs="?", default="results.csv", help="Input CSV from arb_scanner.py")
    parser.add_argument("--alt-combos", type=int, default=0, help="Show up to N alternative positive combos per game in BEST_RELIABLE.txt (default: 0)")
    parser.add_argument("--alt-min-profit", type=float, default=MIN_ROI_PCT, help=f"Min profit_pct for alternative combos (default: {MIN_ROI_PCT})")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug column dump")
    args = parser.parse_args()

    print("="*100)
    print("🔧 ARB COMBINATION GROUPER & ORGANIZER")
    print("="*100 + "\n")

    df = load_csv(args.input_file)

    if len(df) == 0:
        print("❌ No data in CSV")
        return

    if not args.no_debug:
        debug_show_columns(df)

    df = filter_blacklisted_combinations(df)

    if len(df) == 0:
        print("❌ No combinations left after filtering blacklisted bookmakers")
        return

    df = filter_min_roi(df)

    if len(df) == 0:
        print("❌ No combinations left after filtering by minimum ROI")
        return

    grouped = group_by_game(df)

    analyze_bookmaker_usage(df)
    print_example_games(df, n=5)

    print(f"\n{'='*100}")
    print("💾 CREATING OUTPUT FILES")
    print("="*100 + "\n")

    input_path = Path(args.input_file)
    base_name = input_path.parent / input_path.stem

    # CSV Files
    file1_csv = f"{base_name}_ALL_COMBOS.csv"
    file2_csv = f"{base_name}_BEST_RELIABLE.csv"
    file3_csv = f"{base_name}_HIGHEST_PROFIT.csv"
    file4_csv = f"{base_name}_NEXT_COMING_RELIABLE.csv"
    file5_csv = f"{base_name}_NEXT_COMING_HIGHEST_ROI.csv"

    create_grouped_csv(df.copy(), file1_csv)
    create_best_per_game_csv(df.copy(), file2_csv)
    create_highest_profit_csv(df.copy(), file3_csv)
    create_next_coming_reliable_csv(df.copy(), file4_csv)
    create_next_coming_highest_roi_csv(df.copy(), file5_csv)

    # TXT Files
    file1_txt = f"{base_name}_ALL_COMBOS.txt"
    file2_txt = f"{base_name}_BEST_RELIABLE.txt"
    file3_txt = f"{base_name}_HIGHEST_PROFIT.txt"
    file4_txt = f"{base_name}_SUMMARY.txt"
    file5_txt = f"{base_name}_NEXT_COMING_RELIABLE.txt"
    file6_txt = f"{base_name}_NEXT_COMING_HIGHEST_ROI.txt"

    create_all_combos_txt(df.copy(), file1_txt)
    create_best_reliable_txt(df.copy(), file2_txt, alt_combos=args.alt_combos, alt_min_profit=args.alt_min_profit)
    create_highest_profit_txt(df.copy(), file3_txt)
    create_summary_txt(df.copy(), file4_txt)
    create_next_coming_reliable_txt(df.copy(), file5_txt)
    create_next_coming_highest_roi_txt(df.copy(), file6_txt)

    print(f"\n{'='*100}")
    print("✅ COMPLETE!")
    print("="*100)

    unique_games = len(grouped)
    total_combos = len(df)
    avg_per_game = total_combos / unique_games

    print(f"\n   📊 Statistics:")
    print(f"      Unique games/markets: {unique_games}")
    print(f"      Total combinations: {total_combos}")
    print(f"      Avg per game: {avg_per_game:.1f}")

    print(f"\n   🚫 Excluded bookmakers: {', '.join(sorted(set(b.title() for b in BLACKLIST_BOOKMAKERS)))}")
    print(f"   📉 Ruled out: arbs with ROI < {MIN_ROI_PCT}%")

    print("\n" + "="*100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)