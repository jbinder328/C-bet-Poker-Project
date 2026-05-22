"""
parse_hands.py

Parse PokerStars hand history .txt files into a per-player-per-hand CSV.
"""

import re
import glob
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Regex patterns (from spec)
# ---------------------------------------------------------------------------
HAND_HEADER  = re.compile(
    r"PokerStars Hand #(\d+):.*?No Limit \(\$([0-9.]+)/\$([0-9.]+).*?\) - "
    r"\d{4}/\d{2}/\d{2} \d{1,2}:\d{2}:\d{2}.*?\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) ET\]"
)
BUTTON_SEAT  = re.compile(r"Seat #(\d+) is the button")
SEAT_CHIPS   = re.compile(r"Seat (\d+): (.+?) \(\$([0-9.]+) in chips\)")
SMALL_BLIND  = re.compile(r"(.+?): posts small blind \$([0-9.]+)")
BIG_BLIND    = re.compile(r"(.+?): posts big blind \$([0-9.]+)")
FLOP_CARDS   = re.compile(r"\*\*\* FLOP \*\*\* \[(\w+) (\w+) (\w+)\]")
RAISE_ACTION = re.compile(r"(.+?): raises .+ to \$([0-9.]+)")
BET_ACTION   = re.compile(r"(.+?): bets \$([0-9.]+)")
CALL_ACTION  = re.compile(r"(.+?): calls \$([0-9.]+)")
FOLD_ACTION  = re.compile(r"(.+?): folds")
COLLECTED    = re.compile(r"(.+?) collected \$([0-9.]+) from pot")
RAKE_LINE    = re.compile(r"Total pot \$[0-9.]+ \| Rake \$([0-9.]+)")
TOTAL_POT    = re.compile(r"Total pot \$([0-9.]+)")
SUMMARY_SEAT = re.compile(r"Seat \d+: (.+?) (?:\(.+?\) )?(collected|showed|folded|mucked)")
UNCALLED     = re.compile(r"Uncalled bet \(\$([0-9.]+)\) returned to (.+)")
TABLE_NAME   = re.compile(r"Table '(.+?)' ")
CASHED_OUT   = re.compile(r"(.+?) cashed out the hand for \$([0-9.]+)")
BOTH_BLINDS  = re.compile(r"(.+?): posts small & big blinds \$([0-9.]+)")


# ---------------------------------------------------------------------------
# Position assignment
# ---------------------------------------------------------------------------
def assign_position(seat_number, button_seat, all_seat_numbers):
    seat_nums = sorted(all_seat_numbers)
    n = len(seat_nums)

    if n == 0:
        return "UNK"

    btn_idx = seat_nums.index(button_seat) if button_seat in seat_nums else 0

    # Order from left of button (SB) around to button (BTN)
    ordered = seat_nums[btn_idx + 1:] + seat_nums[:btn_idx + 1]

    position_labels = {
        2: ["BB", "BTN"],
        3: ["SB", "BB", "BTN"],
        4: ["UTG", "SB", "BB", "BTN"],
        5: ["UTG", "CO", "SB", "BB", "BTN"],
        6: ["UTG", "HJ", "CO", "SB", "BB", "BTN"],
        7: ["UTG", "MP", "HJ", "CO", "SB", "BB", "BTN"],
        8: ["UTG", "UTG+1", "MP", "HJ", "CO", "SB", "BB", "BTN"],
        9: ["UTG", "UTG+1", "MP", "MP+1", "HJ", "CO", "SB", "BB", "BTN"],
    }

    labels = position_labels.get(n, [f"P{i}" for i in range(n)])

    try:
        idx = ordered.index(seat_number)
        return labels[idx] if idx < len(labels) else "UNK"
    except ValueError:
        return "UNK"


# ---------------------------------------------------------------------------
# Hand splitter
# ---------------------------------------------------------------------------
def split_hands(file_text):
    hands = re.split(r'\n{2,}', file_text.strip())
    return [h for h in hands if h.strip().startswith("PokerStars Hand #")]


# ---------------------------------------------------------------------------
# Single hand parser
# ---------------------------------------------------------------------------
def parse_single_hand(hand_text):
    """Return a list of dicts, one per active player in the hand."""
    lines = [l.rstrip('\r') for l in hand_text.split('\n')]

    # --- Header ---
    header_m = HAND_HEADER.search(hand_text)
    if not header_m:
        return []

    hand_id   = header_m.group(1)
    small_blind_val = float(header_m.group(2))
    big_blind_val   = float(header_m.group(3))
    timestamp = header_m.group(4)

    # --- Table name ---
    table_name = ""
    for line in lines[:5]:
        tm = TABLE_NAME.search(line)
        if tm:
            table_name = tm.group(1)
            break

    # --- Button seat ---
    button_seat = None
    for line in lines[:5]:
        bm = BUTTON_SEAT.search(line)
        if bm:
            button_seat = int(bm.group(1))
            break

    # --- Seats (active players only — must have "in chips") ---
    seats = {}   # seat_number -> {name, starting_stack}
    for line in lines:
        if '*** HOLE CARDS ***' in line:
            break
        sm = SEAT_CHIPS.match(line.strip())
        if sm:
            seat_num = int(sm.group(1))
            name     = sm.group(2).strip()
            stack    = float(sm.group(3))
            # Skip players that are sitting out (line ends with "is sitting out")
            # We still include them if they post blinds, but PokerStars marks
            # sitting-out players separately; active seats are just "Seat X: Name ($Y in chips)"
            # without "is sitting out" appended
            if 'sitting out' not in line:
                seats[seat_num] = {'name': name, 'starting_stack': stack}

    if not seats:
        return []

    all_seat_numbers = list(seats.keys())
    if button_seat is None:
        button_seat = all_seat_numbers[0]

    # --- Parse streets ---
    # We'll walk the lines and track street transitions
    STREETS = ('preflop', 'flop', 'turn', 'river', 'showdown', 'summary')
    street = 'preflop'

    # Per-player state
    folded    = set()   # players who have folded
    # cumulative investment per player per street (to handle raise amounts correctly)
    street_committed  = {name_dict['name']: 0.0 for name_dict in seats.values()}
    total_invested    = {name_dict['name']: 0.0 for name_dict in seats.values()}
    uncalled_returned = {name_dict['name']: 0.0 for name_dict in seats.values()}
    amount_collected  = {name_dict['name']: 0.0 for name_dict in seats.values()}

    posted_blind  = {name_dict['name']: 0 for name_dict in seats.values()}
    vpip          = {name_dict['name']: 0 for name_dict in seats.values()}
    pfr           = {name_dict['name']: 0 for name_dict in seats.values()}
    preflop_raise_amount = {name_dict['name']: 0.0 for name_dict in seats.values()}

    saw_flop  = {name_dict['name']: 0 for name_dict in seats.values()}
    saw_turn  = {name_dict['name']: 0 for name_dict in seats.values()}
    saw_river = {name_dict['name']: 0 for name_dict in seats.values()}

    flop_bet_made = {name_dict['name']: 0 for name_dict in seats.values()}
    flop_bet_size = {name_dict['name']: 0.0 for name_dict in seats.values()}

    went_to_showdown = 0
    preflop_raises_count = 0

    flop_card_1 = flop_card_2 = flop_card_3 = None
    total_pot_val = 0.0
    rake_val = 0.0
    pot_size_at_flop = 0.0

    # Helper: name lookup from regex match (names can have spaces)
    active_names = set(d['name'] for d in seats.values())

    def is_active(name):
        return name in active_names

    def reset_street_committed():
        for n in street_committed:
            street_committed[n] = 0.0

    # Process lines
    for line in lines:
        line_stripped = line.strip()

        # --- Street transitions ---
        if '*** HOLE CARDS ***' in line_stripped:
            street = 'preflop'
            reset_street_committed()
            # Carry over blind commitments into street_committed
            for n, amt in total_invested.items():
                street_committed[n] = amt
            continue

        if '*** FLOP ***' in line_stripped:
            # pot_size_at_flop = sum of all preflop investments
            pot_size_at_flop = sum(total_invested.values())
            # Determine who saw the flop
            for n in active_names:
                if n not in folded:
                    saw_flop[n] = 1
            street = 'flop'
            reset_street_committed()
            # Parse flop cards
            fm = FLOP_CARDS.search(line_stripped)
            if fm:
                flop_card_1 = fm.group(1)
                flop_card_2 = fm.group(2)
                flop_card_3 = fm.group(3)
            continue

        if '*** TURN ***' in line_stripped:
            for n in active_names:
                if n not in folded:
                    saw_turn[n] = 1
            street = 'turn'
            reset_street_committed()
            continue

        if '*** RIVER ***' in line_stripped:
            for n in active_names:
                if n not in folded:
                    saw_river[n] = 1
            street = 'river'
            reset_street_committed()
            continue

        if '*** SHOW DOWN ***' in line_stripped or '*** SHOWDOWN ***' in line_stripped:
            went_to_showdown = 1
            street = 'showdown'
            continue

        if '*** SUMMARY ***' in line_stripped:
            street = 'summary'
            continue

        # --- Summary section ---
        if street == 'summary':
            # Total pot and rake
            rm = RAKE_LINE.search(line_stripped)
            if rm:
                rake_val = float(rm.group(1))
                tp = TOTAL_POT.search(line_stripped)
                if tp:
                    total_pot_val = float(tp.group(1))
                continue
            tp = TOTAL_POT.search(line_stripped)
            if tp:
                total_pot_val = float(tp.group(1))
            continue

        # --- Blind posts ---
        # Check "posts small & big blinds" first (before individual blind checks)
        bothm = BOTH_BLINDS.match(line_stripped)
        if bothm:
            name = bothm.group(1).strip()
            amt  = float(bothm.group(2))
            if is_active(name):
                total_invested[name] += amt
                street_committed[name] = amt
                posted_blind[name] = 1
            continue

        sbm = SMALL_BLIND.match(line_stripped)
        if sbm:
            name = sbm.group(1).strip()
            amt  = float(sbm.group(2))
            if is_active(name):
                total_invested[name] += amt
                street_committed[name] = amt
                posted_blind[name] = 1
            continue

        bbm = BIG_BLIND.match(line_stripped)
        if bbm:
            name = bbm.group(1).strip()
            amt  = float(bbm.group(2))
            if is_active(name):
                # Player may post BB on top of SB or after someone else
                total_invested[name] += amt
                street_committed[name] = total_invested[name]
                posted_blind[name] = 1
            continue

        # --- Uncalled bet returned ---
        ucm = UNCALLED.match(line_stripped)
        if ucm:
            amt  = float(ucm.group(1))
            name = ucm.group(2).strip()
            if is_active(name):
                uncalled_returned[name] += amt
            continue

        # --- Collected from pot ---
        cm = COLLECTED.search(line_stripped)
        if cm:
            name = cm.group(1).strip()
            amt  = float(cm.group(2))
            if is_active(name):
                amount_collected[name] += amt
            continue

        # --- Cashed out (PokerStars Cash Out feature) ---
        com = CASHED_OUT.match(line_stripped)
        if com:
            name = com.group(1).strip()
            amt  = float(com.group(2))
            if is_active(name):
                amount_collected[name] += amt
            continue

        # --- Actions ---
        if street in ('preflop', 'flop', 'turn', 'river'):
            # Raise
            ram = RAISE_ACTION.match(line_stripped)
            if ram:
                name    = ram.group(1).strip()
                to_amt  = float(ram.group(2))
                if is_active(name):
                    additional = to_amt - street_committed[name]
                    if additional > 0:
                        total_invested[name] += additional
                    street_committed[name] = to_amt
                    if street == 'preflop':
                        pfr[name] = 1
                        vpip[name] = 1
                        preflop_raise_amount[name] = to_amt
                        preflop_raises_count += 1
                continue

            # Bet
            betm = BET_ACTION.match(line_stripped)
            if betm:
                name = betm.group(1).strip()
                amt  = float(betm.group(2))
                # "and is all-in" may be in the line; BET_ACTION matches up to the amount
                if is_active(name):
                    additional = amt - street_committed[name]
                    if additional > 0:
                        total_invested[name] += additional
                    street_committed[name] = amt
                    if street == 'flop':
                        flop_bet_made[name] = 1
                        flop_bet_size[name] = amt
                continue

            # Call
            callm = CALL_ACTION.match(line_stripped)
            if callm:
                name = callm.group(1).strip()
                amt  = float(callm.group(2))
                if is_active(name):
                    total_invested[name] += amt
                    street_committed[name] += amt
                    if street == 'preflop':
                        vpip[name] = 1
                continue

            # Fold
            foldm = FOLD_ACTION.match(line_stripped)
            if foldm:
                name = foldm.group(1).strip()
                if is_active(name):
                    folded.add(name)
                continue

    # --- players_to_flop ---
    players_to_flop = sum(saw_flop.values())

    # --- Build rows ---
    rows = []
    for seat_num, info in seats.items():
        name = info['name']
        pos  = assign_position(seat_num, button_seat, all_seat_numbers)

        # Adjust total_invested for uncalled bets
        net_invested = total_invested[name] - uncalled_returned[name]
        if net_invested < 0:
            net_invested = 0.0

        net_profit = amount_collected[name] - net_invested

        row = {
            'hand_id':               hand_id,
            'timestamp':             timestamp,
            'small_blind':           small_blind_val,
            'big_blind':             big_blind_val,
            'table_name':            table_name,
            'num_players_dealt':     len(seats),
            'flop_card_1':           flop_card_1,
            'flop_card_2':           flop_card_2,
            'flop_card_3':           flop_card_3,
            'total_pot':             total_pot_val,
            'rake':                  rake_val,
            'player_name':           name,
            'seat_number':           seat_num,
            'starting_stack':        info['starting_stack'],
            'position':              pos,
            'posted_blind':          posted_blind[name],
            'vpip':                  vpip[name],
            'pfr':                   pfr[name],
            'preflop_raise_amount':  preflop_raise_amount[name],
            'saw_flop':              saw_flop[name],
            'players_to_flop':       players_to_flop,
            'flop_bet_made':         flop_bet_made[name],
            'flop_bet_size':         flop_bet_size[name],
            'pot_size_at_flop':      pot_size_at_flop,
            'preflop_raises_count':  preflop_raises_count,
            'saw_turn':              saw_turn[name],
            'saw_river':             saw_river[name],
            'went_to_showdown':      went_to_showdown,
            'amount_collected':      amount_collected[name],
            'total_invested':        net_invested,
            'net_profit':            net_profit,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def parse_all_files(txt_dir, output_csv):
    all_rows = []
    files = glob.glob(f"{txt_dir}/*.txt")

    print(f"Parsing {len(files)} files...")

    for i, filepath in enumerate(files):
        if i % 100 == 0:
            print(f"  Processing file {i}/{len(files)}...")

        try:
            with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
                content = f.read()

            hands = split_hands(content)
            for hand_text in hands:
                try:
                    rows = parse_single_hand(hand_text)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"  Error in hand: {e}")
                    continue
        except Exception as e:
            print(f"  Error in {filepath}: {e}")
            continue

    df = pd.DataFrame(all_rows)
    before = len(df)
    df = df.drop_duplicates(subset=["hand_id", "player_name"])
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} duplicate rows from repeated hands in source files")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df):,} rows ({df['hand_id'].nunique():,} hands) to {output_csv}")
    return df


if __name__ == "__main__":
    df = parse_all_files(
        "poker_model/data/raw/",
        "poker_model/outputs/all_hands.csv"
    )
    print(df.shape)
    print(df.dtypes)
    print(df.isnull().sum())
    print(df.head(3))
