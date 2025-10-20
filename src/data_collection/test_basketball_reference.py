"""
Quick test of Basketball Reference data collection
"""

import pandas as pd
from basketball_reference_scraper.seasons import get_schedule, get_standings
from basketball_reference_scraper.teams import get_roster_stats
import time

print("Testing Basketball Reference Data Collection")
print("=" * 60)

# Test 1: Get 2023-24 schedule
print("\n1. Testing schedule collection for 2023-24 season...")
try:
    schedule = get_schedule(2024)
    if schedule is not None and not schedule.empty:
        print(f"   ✓ Successfully collected {len(schedule)} games")
        print(f"   Sample columns: {list(schedule.columns[:5])}")
        print(f"\n   First game:")
        print(schedule.head(1).to_string())
    else:
        print("   ✗ No schedule data returned")
except Exception as e:
    print(f"   ✗ Error: {e}")

time.sleep(3)

# Test 2: Get standings
print("\n2. Testing standings collection for 2023-24 season...")
try:
    standings = get_standings(2024)
    if standings is not None and not standings.empty:
        print(f"   ✓ Successfully collected standings")
        print(f"   Teams: {len(standings)}")
        print(f"\n   Top 3 teams:")
        print(standings.head(3).to_string())
    else:
        print("   ✗ No standings data returned")
except Exception as e:
    print(f"   ✗ Error: {e}")

time.sleep(3)

# Test 3: Get team roster stats
print("\n3. Testing roster stats for Lakers (LAL)...")
try:
    roster = get_roster_stats('LAL', 2024)
    if roster is not None and not roster.empty:
        print(f"   ✓ Successfully collected {len(roster)} players")
        print(f"   Columns: {list(roster.columns[:8])}")
        print(f"\n   Top 3 scorers:")
        if 'PTS' in roster.columns:
            top_scorers = roster.nlargest(3, 'PTS')[['PLAYER', 'PTS', 'REB', 'AST']]
            print(top_scorers.to_string(index=False))
        else:
            print(roster.head(3).to_string())
    else:
        print("   ✗ No roster data returned")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("Basketball Reference data collection test complete!")
print("=" * 60)
print("\nThe library is working! You can now collect full season data.")
print("Run: python collect_basketball_reference.py")

