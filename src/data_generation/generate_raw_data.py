"""
BNPL Synthetic Data Generator (Optimized)

Generates ~50K customers, ~80K orders, ~320K installments across 8 normalized
tables with realistic distributions, fraud patterns, and intentional data
quality issues. Writes everything to a SQLite database.

Usage:
    python -m src.data_generation.generate_raw_data
"""

import os
import random
import sqlite3
import string
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.data_generation.config import (
    BROWSERS, BROWSER_WEIGHTS, CHANNELS, CHANNEL_WEIGHTS,
    CREDIT_DECISION_RULES, CUSTOMER_SEGMENTS, DATA_END_DATE,
    DATA_QUALITY_ISSUES, DATA_START_DATE, DEVICE_TYPES, DEVICE_WEIGHTS,
    DISPUTE_REASON_WEIGHTS, DISPUTE_REASONS, DISPUTE_RESOLUTION_WEIGHTS,
    DISPUTE_RESOLUTIONS, EMPLOYMENT_STATUSES, EMPLOYMENT_WEIGHTS,
    FRAUD_PATTERNS, MERCHANT_CATEGORIES, NUM_CUSTOMERS, NUM_MERCHANTS,
    OPERATING_SYSTEMS, OS_WEIGHTS, PAYMENT_METHODS, PAYMENT_METHOD_WEIGHTS,
    PLAN_TYPES, RISK_TIER_FRAUD_MULTIPLIER, SEED,
    STATE_POPULATION_WEIGHTS, US_STATES,
)

np.random.seed(SEED)
random.seed(SEED)

# ─── Name / address pools ───

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Dorothy", "Paul", "Kimberly", "Andrew", "Emily", "Joshua", "Donna",
    "Kenneth", "Michelle", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Emma", "Scott", "Nicole", "Brandon", "Helen",
    "Benjamin", "Samantha", "Samuel", "Katherine", "Raymond", "Christine", "Gregory", "Debra",
    "Frank", "Rachel", "Alexander", "Carolyn", "Patrick", "Janet", "Jack", "Catherine",
    "Aiden", "Sofia", "Liam", "Olivia", "Noah", "Isabella", "Ethan", "Mia",
    "Mason", "Charlotte", "Logan", "Amelia", "Lucas", "Harper", "Elijah", "Evelyn",
    "Raj", "Priya", "Wei", "Mei", "Carlos", "Maria", "Ahmed", "Fatima",
    "Hiroshi", "Yuki", "Dmitri", "Natasha", "Kwame", "Ama", "Pedro", "Ana",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
    "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell",
    "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales",
    "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson",
    "Bailey", "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward",
    "Richardson", "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray",
    "Mendoza", "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel",
    "Myers", "Long", "Ross", "Foster", "Jimenez", "Powell", "Jenkins", "Perry",
    "Chen", "Wang", "Singh", "Kumar", "Ali", "Shah", "Park", "Nakamura",
]

STREET_NAMES = [
    "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine St", "Elm St", "Washington Blvd",
    "Park Ave", "Lake Dr", "Hill Rd", "River Rd", "Forest Ave", "Sunset Blvd",
    "Broadway", "Church St", "Spring St", "Highland Ave", "Meadow Ln", "Valley Rd",
]

CITIES_BY_STATE = {
    "CA": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Sacramento"],
    "TX": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth"],
    "FL": ["Miami", "Orlando", "Tampa", "Jacksonville", "Fort Lauderdale"],
    "NY": ["New York", "Buffalo", "Rochester", "Albany", "Syracuse"],
    "PA": ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Reading"],
    "IL": ["Chicago", "Aurora", "Naperville", "Rockford", "Joliet"],
    "OH": ["Columbus", "Cleveland", "Cincinnati", "Toledo", "Akron"],
    "GA": ["Atlanta", "Augusta", "Savannah", "Athens", "Macon"],
    "NC": ["Charlotte", "Raleigh", "Durham", "Greensboro", "Winston-Salem"],
    "MI": ["Detroit", "Grand Rapids", "Ann Arbor", "Lansing", "Flint"],
}
DEFAULT_CITIES = ["Springfield", "Franklin", "Clinton", "Madison", "Georgetown"]

ZIP_RANGES = {
    "CA": (90000, 96699), "TX": (73301, 79999), "FL": (32004, 34997),
    "NY": (10001, 14975), "PA": (15001, 19640), "IL": (60001, 62999),
    "OH": (43001, 45999), "GA": (30001, 31999), "NC": (27006, 28909),
    "MI": (48001, 49971), "NJ": (7001, 8989), "VA": (20040, 24658),
    "WA": (98001, 99403), "AZ": (85001, 86556), "MA": (1001, 2791),
    "TN": (37010, 38589), "IN": (46001, 47997), "MO": (63001, 65899),
    "MD": (20601, 21930), "WI": (53001, 54990), "CO": (80001, 81658),
    "MN": (55001, 56763), "SC": (29001, 29948), "AL": (35004, 36925),
    "LA": (70001, 71497), "KY": (40003, 42788), "OR": (97001, 97920),
    "OK": (73001, 74966), "CT": (6001, 6928), "UT": (84001, 84784),
}

LAT_LON = {
    "CA": (36.7, -119.4), "TX": (31.0, -99.5), "FL": (27.6, -81.5),
    "NY": (42.1, -74.0), "PA": (40.8, -77.8), "IL": (40.6, -89.3),
    "OH": (40.4, -82.7), "GA": (32.2, -83.4), "NC": (35.5, -79.9),
    "MI": (44.3, -84.5),
}
DEFAULT_LAT_LON = (38.0, -97.0)

MERCHANT_NAME_PREFIXES = [
    "Global", "Prime", "NextGen", "Urban", "Elite", "Mega", "Super", "Value",
    "Smart", "Quick", "Express", "Digital", "Modern", "Classic", "Nova",
    "Peak", "First", "Premier", "Pro", "Max", "Ultra", "Apex", "Core",
]
MERCHANT_NAME_SUFFIXES = {
    "electronics": ["Tech", "Electronics", "Devices", "Gadgets", "Computing"],
    "fashion": ["Fashion", "Apparel", "Style", "Wear", "Clothing"],
    "home_furniture": ["Home", "Furniture", "Living", "Decor", "Interiors"],
    "health_beauty": ["Beauty", "Wellness", "Health", "Care", "Skincare"],
    "travel": ["Travel", "Getaways", "Trips", "Adventures", "Voyages"],
    "gaming": ["Games", "Gaming", "Play", "Arcade", "Entertainment"],
    "grocery": ["Grocery", "Market", "Foods", "Fresh", "Pantry"],
    "jewelry_luxury": ["Jewels", "Luxury", "Gems", "Fine Goods", "Bijoux"],
    "sports_outdoor": ["Sports", "Outdoor", "Athletics", "Fitness", "Active"],
    "education": ["Learning", "Education", "Academy", "Courses", "Study"],
}


def _days_in_range() -> int:
    return (DATA_END_DATE - DATA_START_DATE).days


# ──────────────────────────────────────────
# 1. Customers  (vectorized)
# ──────────────────────────────────────────

def generate_customers() -> pd.DataFrame:
    print("Generating customers...", flush=True)
    n = NUM_CUSTOMERS
    segments = list(CUSTOMER_SEGMENTS.keys())
    seg_weights = [CUSTOMER_SEGMENTS[s]["weight"] for s in segments]
    assigned = np.random.choice(segments, size=n, p=seg_weights)

    states_list = list(US_STATES.keys())
    chosen_states = np.random.choice(states_list, size=n, p=STATE_POPULATION_WEIGHTS)

    total_days = _days_in_range() - 60
    signup_offsets = np.random.randint(0, max(total_days, 1), size=n)
    dob_start = date(1955, 1, 1)
    dob_range = (date(2005, 12, 31) - dob_start).days
    dob_offsets = np.random.randint(0, dob_range, size=n)

    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]
    first_names = np.random.choice(FIRST_NAMES, size=n)
    last_names = np.random.choice(LAST_NAMES, size=n)
    emp_statuses = np.random.choice(EMPLOYMENT_STATUSES, size=n, p=EMPLOYMENT_WEIGHTS)

    rows = []
    for i in range(n):
        seg = assigned[i]
        cfg = CUSTOMER_SEGMENTS[seg]
        st = chosen_states[i]
        fn, ln = first_names[i], last_names[i]

        city_opts = CITIES_BY_STATE.get(st, DEFAULT_CITIES)
        city = city_opts[i % len(city_opts)]
        lo, hi = ZIP_RANGES.get(st, (10000, 99999))

        cs = np.random.randint(cfg["credit_score_range"][0], cfg["credit_score_range"][1] + 1)
        inc = round(np.random.uniform(*cfg["income_range"]), 2)

        sep = [".", "_", ""][i % 3]
        dom = domains[i % len(domains)]
        email = f"{fn.lower()}{sep}{ln.lower()}{i % 100}@{dom}"

        rows.append((
            i + 1, fn, ln, email,
            f"({np.random.randint(200,999)}) {np.random.randint(100,999)}-{np.random.randint(1000,9999)}",
            f"{np.random.randint(100,9999)} {STREET_NAMES[i % len(STREET_NAMES)]}",
            city, st, str(np.random.randint(lo, hi + 1)).zfill(5),
            (dob_start + timedelta(days=int(dob_offsets[i]))).isoformat(),
            (DATA_START_DATE + timedelta(days=int(signup_offsets[i]))).isoformat(),
            f"{np.random.randint(0,9999):04d}",
            inc, int(cs), emp_statuses[i], seg,
        ))

    cols = [
        "customer_id", "first_name", "last_name", "email", "phone",
        "address", "city", "state", "zip_code", "dob", "signup_date",
        "ssn_last4", "annual_income", "credit_score", "employment_status", "segment",
    ]
    df = pd.DataFrame(rows, columns=cols)
    print(f"  Customers: {len(df):,}", flush=True)
    return df


# ──────────────────────────────────────────
# 2. Merchants  (vectorized)
# ──────────────────────────────────────────

def generate_merchants() -> pd.DataFrame:
    print("Generating merchants...", flush=True)
    cats = list(MERCHANT_CATEGORIES.keys())
    cat_w = [MERCHANT_CATEGORIES[c]["weight"] for c in cats]
    assigned_cats = np.random.choice(cats, size=NUM_MERCHANTS, p=cat_w)

    rows, used = [], set()
    for i in range(NUM_MERCHANTS):
        cat = assigned_cats[i]
        cfg = MERCHANT_CATEGORIES[cat]
        attempts = 0
        while True:
            prefix = random.choice(MERCHANT_NAME_PREFIXES)
            suffix = random.choice(MERCHANT_NAME_SUFFIXES[cat])
            if attempts > 50:
                name = f"{prefix} {suffix} {i}"
            else:
                name = f"{prefix} {suffix}"
            if name not in used:
                used.add(name)
                break
            attempts += 1
        onb = DATA_START_DATE - timedelta(days=np.random.randint(0, 365))
        rows.append((i + 1, name, cat, cfg["risk_tier"], onb.isoformat()))

    return pd.DataFrame(rows, columns=["merchant_id", "merchant_name", "category", "risk_tier", "onboarding_date"])


# ──────────────────────────────────────────
# 3. Orders + downstream (batch-per-customer)
# ──────────────────────────────────────────

def generate_orders_and_downstream(customers: pd.DataFrame, merchants: pd.DataFrame):
    print("Generating orders & downstream tables...", flush=True)

    cats = list(MERCHANT_CATEGORIES.keys())
    cat_w = np.array([MERCHANT_CATEGORIES[c]["weight"] for c in cats])
    cat_w = cat_w / cat_w.sum()

    merch_by_cat = {}
    for _, m in merchants.iterrows():
        merch_by_cat.setdefault(m["category"], []).append(m["merchant_id"])

    fraud_device_pool = [f"SHARED_DEV_{j}" for j in range(15)]
    fraud_ip_pool = [f"{np.random.randint(1,223)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,254)}" for _ in range(10)]

    plan_cfg_map = PLAN_TYPES
    high_value_cats = FRAUD_PATTERNS["high_value"]["preferred_categories"]

    orders, plans, insts, pays = [], [], [], []
    decisions, devices, disputes = [], [], []
    order_fraud_labels = []
    oid, pid, iid, payid, did, fid, dispid = 0, 0, 0, 0, 0, 0, 0

    seg_arr = customers["segment"].values
    cid_arr = customers["customer_id"].values
    cs_arr = customers["credit_score"].values
    signup_arr = customers["signup_date"].values
    state_arr = customers["state"].values

    total = len(customers)
    pct_step = max(total // 10, 1)

    for idx in range(total):
        if idx % pct_step == 0:
            print(f"  Progress: {idx:,}/{total:,} customers ({100*idx//total}%)", flush=True)

        seg = seg_arr[idx]
        cfg = CUSTOMER_SEGMENTS[seg]
        cust_id = int(cid_arr[idx])
        credit_score = cs_arr[idx]
        try:
            signup_dt = date.fromisoformat(str(signup_arr[idx]))
        except Exception:
            signup_dt = DATA_START_DATE

        st = str(state_arr[idx])
        lat_base, lon_base = LAT_LON.get(st, DEFAULT_LAT_LON)

        is_fraudster = seg == "fraudster"
        fraud_pattern = None
        if is_fraudster:
            fp_keys = list(FRAUD_PATTERNS.keys())
            fp_w = [FRAUD_PATTERNS[k]["weight"] for k in fp_keys]
            fraud_pattern = np.random.choice(fp_keys, p=fp_w)

        num_orders = np.random.randint(cfg["orders_range"][0], cfg["orders_range"][1] + 1)

        deterioration_start = None
        if seg == "slow_deterioration":
            deterioration_start = max(1, np.random.randint(1, num_orders))

        first_default_trigger = None
        if seg == "first_time_defaulter":
            first_default_trigger = max(1, num_orders - 1)

        for oi in range(num_orders):
            oid += 1

            # Order date
            if is_fraudster and fraud_pattern == "bust_out":
                odate = signup_dt + timedelta(days=np.random.randint(0, 30))
            else:
                earliest = signup_dt + timedelta(days=oi * 30)
                latest = min(earliest + timedelta(days=120), DATA_END_DATE)
                if earliest >= DATA_END_DATE:
                    break
                odate = earliest + timedelta(days=np.random.randint(0, max(1, (latest - earliest).days)))

            if odate > DATA_END_DATE:
                odate = DATA_END_DATE

            odt = datetime(odate.year, odate.month, odate.day,
                           np.random.randint(6, 23), np.random.randint(0, 59), np.random.randint(0, 59))

            # Category / merchant
            if is_fraudster and fraud_pattern == "high_value":
                cat = random.choice(high_value_cats)
            else:
                cat = np.random.choice(cats, p=cat_w)
            cat_cfg = MERCHANT_CATEGORIES[cat]
            m_ids = merch_by_cat.get(cat, merch_by_cat[cats[0]])
            merchant_id = random.choice(m_ids)

            # Amount
            if is_fraudster and fraud_pattern == "high_value":
                amount = round(np.random.uniform(600, 2500), 2)
            else:
                amount = round(max(20, np.random.normal(cat_cfg["avg_order"], cat_cfg["order_std"])), 2)

            channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)

            # Credit decision
            did += 1
            if credit_score is not None and not np.isnan(credit_score):
                cs_val = int(credit_score)
            else:
                cs_val = 600
            if cs_val >= CREDIT_DECISION_RULES["auto_approve_min_score"]:
                decision = "approved"
                decline_reason = None
            elif cs_val <= CREDIT_DECISION_RULES["auto_decline_max_score"]:
                decision = "declined"
                decline_reason = random.choice(CREDIT_DECISION_RULES["decline_reasons"])
            else:
                decision = "approved" if np.random.random() < CREDIT_DECISION_RULES["approval_rate_overall"] else "declined"
                decline_reason = random.choice(CREDIT_DECISION_RULES["decline_reasons"]) if decision == "declined" else None

            decisions.append((did, cust_id, oid, decision,
                              round(amount, 2) if decision == "approved" else None,
                              odt.isoformat(), decline_reason))

            order_status = "completed" if decision == "approved" else "declined"
            orders.append([oid, cust_id, merchant_id, amount, odt.isoformat(), order_status, channel])

            # Probabilistic fraud label per order (not per customer).
            # Fraudsters don't commit fraud on every order; legitimate
            # customers occasionally commit opportunistic fraud.
            if is_fraudster:
                if fraud_pattern == "bust_out":
                    fraud_prob = 0.70
                elif fraud_pattern == "high_value":
                    fraud_prob = 0.80
                elif fraud_pattern == "synthetic_identity":
                    fraud_prob = 0.65
                elif fraud_pattern == "friendly_fraud":
                    fraud_prob = 0.50
                else:
                    fraud_prob = 0.60
            elif seg == "slow_deterioration":
                fraud_prob = 0.04
            elif seg == "first_time_defaulter":
                fraud_prob = 0.03
            elif seg == "occasional_late":
                fraud_prob = 0.015
            else:
                fraud_prob = 0.005
            is_fraud_order = 1 if np.random.random() < fraud_prob else 0
            order_fraud_labels.append((oid, cust_id, seg, is_fraud_order))

            if decision == "declined":
                continue

            # Payment plan
            pid += 1
            plan_prefs = cfg["plan_preference"]
            pt_keys = list(plan_prefs.keys())
            pt_w = [plan_prefs[k] for k in pt_keys]
            chosen_pt = np.random.choice(pt_keys, p=pt_w)
            pcfg = plan_cfg_map[chosen_pt]
            n_inst = pcfg["num_installments"]
            freq = pcfg["frequency_days"]
            apr = pcfg["apr"]
            total_amt = round(amount * (1 + apr), 2)
            inst_amt = round(total_amt / n_inst, 2)
            start_dt = odate
            end_dt = start_dt + timedelta(days=freq * n_inst)

            is_deterio = seg == "slow_deterioration" and deterioration_start is not None and oi >= deterioration_start
            is_ftd = seg == "first_time_defaulter" and first_default_trigger is not None and oi >= first_default_trigger

            # Pre-compute per-order parameters before installment loop
            bust_trust_phase = np.random.randint(0, 4) if (is_fraudster and fraud_pattern == "bust_out") else 0

            plan_inst_statuses = []
            for inst_num in range(1, n_inst + 1):
                iid += 1
                due = start_dt + timedelta(days=freq * inst_num)
                dpd = 0

                if due <= DATA_END_DATE:
                    if is_fraudster:
                        if fraud_pattern == "bust_out":
                            trust_phase = bust_trust_phase
                            if inst_num <= trust_phase:
                                st_val = "paid" if np.random.random() < 0.70 else "late"
                                dpd = np.random.randint(1, 10) if st_val == "late" else 0
                            elif np.random.random() < 0.12:
                                st_val = "late"
                                dpd = np.random.randint(15, 60)
                            else:
                                st_val = "missed"
                                dpd = np.random.randint(30, 150)
                        elif fraud_pattern == "synthetic_identity":
                            # Irregular but overlaps with struggling customers.
                            # ~30% on-time (lower than occasional_late's 70%).
                            if np.random.random() < 0.30:
                                st_val = "paid"
                            elif np.random.random() < 0.45:
                                st_val = "late"
                                dpd = np.random.randint(5, 50)
                            else:
                                st_val = "missed"
                                dpd = np.random.randint(20, 100)
                        elif fraud_pattern == "high_value":
                            # Most miss early but ~25% make a first payment.
                            # Some sporadic late payments after that.
                            if inst_num == 1 and np.random.random() < 0.25:
                                st_val = "paid"
                            elif inst_num > 1 and np.random.random() < 0.08:
                                st_val = "late"
                                dpd = np.random.randint(10, 45)
                            else:
                                st_val = "missed"
                                dpd = np.random.randint(45, 150)
                        elif fraud_pattern == "friendly_fraud":
                            # Nearly indistinguishable from occasional_late.
                            # Slightly worse on-time rate and higher DPD.
                            if np.random.random() < 0.55:
                                st_val = "paid"
                            elif np.random.random() < 0.5:
                                st_val = "late"
                                dpd = np.random.randint(5, 55)
                            else:
                                st_val = "missed"
                                dpd = np.random.randint(20, 75)
                        else:
                            st_val = "missed"
                            dpd = np.random.randint(30, 120)
                    elif is_deterio:
                        # Gradual decline with noise: sometimes recovers
                        # briefly before worsening again.
                        order_decay = min(0.65, 0.15 + 0.10 * max(0, oi - deterioration_start))
                        inst_decay = 0.06 * (inst_num - 1)
                        prob_trouble = min(0.80, order_decay + inst_decay)
                        # Occasional recovery even mid-decline
                        if np.random.random() < 0.08:
                            st_val = "paid"
                        elif np.random.random() > prob_trouble:
                            st_val = "paid"
                        elif np.random.random() < 0.5:
                            st_val = "late"
                            dpd = np.random.randint(3, 50)
                        else:
                            st_val = "missed"
                            dpd = np.random.randint(10, 70)
                    elif is_ftd:
                        if inst_num <= 2:
                            st_val = "paid" if np.random.random() < 0.75 else "late"
                            dpd = np.random.randint(1, 20) if st_val == "late" else 0
                        elif np.random.random() < 0.10:
                            st_val = "late"
                            dpd = np.random.randint(10, 40)
                        else:
                            st_val = "missed"
                            dpd = np.random.randint(25, 80)
                    else:
                        # Good payers and occasional late — add realistic
                        # life-event noise so they aren't perfectly clean.
                        base_ontime = cfg["on_time_prob"]
                        # ~5% of good payers hit a "rough patch" per order
                        # where their on-time rate drops temporarily
                        hit_rough_patch = (seg == "good_payer" and
                                           np.random.random() < 0.05)
                        if hit_rough_patch:
                            base_ontime = max(0.40, base_ontime - 0.35)

                        if np.random.random() < base_ontime:
                            st_val = "paid"
                        elif np.random.random() < 0.55:
                            st_val = "late"
                            dpd = np.random.randint(1, 40)
                        else:
                            st_val = "missed"
                            dpd = np.random.randint(15, 75)
                else:
                    st_val = "pending"

                plan_inst_statuses.append(st_val)
                insts.append((iid, pid, inst_num, inst_amt, due.isoformat(), st_val, dpd))

                if st_val in ("paid", "late"):
                    payid += 1
                    if st_val == "paid":
                        pay_dt = due - timedelta(days=np.random.randint(0, 3))
                    else:
                        pay_dt = due + timedelta(days=dpd)
                    pdt = datetime(pay_dt.year, pay_dt.month, pay_dt.day,
                                   np.random.randint(6, 23), np.random.randint(0, 59), np.random.randint(0, 59))
                    pays.append((payid, iid, inst_amt, pdt.isoformat(),
                                 np.random.choice(PAYMENT_METHODS, p=PAYMENT_METHOD_WEIGHTS), "success"))

            # Plan status
            if all(s == "paid" for s in plan_inst_statuses):
                pstatus = "completed"
            elif sum(1 for s in plan_inst_statuses if s == "missed") >= 3:
                pstatus = "defaulted"
                orders[-1][5] = "defaulted"
            elif any(s == "missed" for s in plan_inst_statuses):
                pstatus = "delinquent"
            elif any(s == "pending" for s in plan_inst_statuses):
                pstatus = "active"
            else:
                pstatus = "completed"

            plans.append((pid, oid, chosen_pt, n_inst, inst_amt, total_amt, apr,
                          start_dt.isoformat(), end_dt.isoformat(), pstatus))

            # Device fingerprint — signals overlap between segments.
            # Legitimate users increasingly use VPNs and multiple devices.
            fid += 1
            if is_fraudster and fraud_pattern == "synthetic_identity":
                dev_id = random.choice(fraud_device_pool) if np.random.random() < 0.6 else f"DEV_{cust_id}_{np.random.randint(1,4)}"
                ip = random.choice(fraud_ip_pool) if np.random.random() < 0.5 else f"{np.random.randint(1,223)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,254)}"
                vpn = np.random.random() < 0.45
            elif is_fraudster and fraud_pattern == "bust_out":
                dev_id = f"DEV_{cust_id}_{np.random.randint(1,5)}"
                ip = f"{np.random.randint(1,223)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,254)}"
                vpn = np.random.random() < 0.35
            elif is_fraudster:
                dev_id = f"DEV_{cust_id}_{np.random.randint(1,4)}"
                ip = f"{np.random.randint(1,223)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,254)}"
                vpn = np.random.random() < 0.20
            else:
                dev_id = f"DEV_{cust_id}_{np.random.randint(1,3)}"
                ip = f"{np.random.randint(1,223)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,254)}"
                vpn = np.random.random() < 0.08

            if is_fraudster and fraud_pattern == "bust_out":
                lat = round(lat_base + np.random.uniform(-10, 10), 4)
                lon = round(lon_base + np.random.uniform(-10, 10), 4)
            elif is_fraudster:
                lat = round(lat_base + np.random.uniform(-5, 5), 4)
                lon = round(lon_base + np.random.uniform(-5, 5), 4)
            else:
                lat = round(lat_base + np.random.uniform(-3, 3), 4)
                lon = round(lon_base + np.random.uniform(-3, 3), 4)

            devices.append((fid, cust_id, oid,
                            np.random.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS),
                            dev_id, ip,
                            np.random.choice(BROWSERS, p=BROWSER_WEIGHTS),
                            np.random.choice(OPERATING_SYSTEMS, p=OS_WEIGHTS),
                            vpn, lat, lon))

            # Dispute
            dprob = cfg["dispute_prob"]
            if is_fraudster and fraud_pattern == "friendly_fraud":
                dprob = 0.45
            if np.random.random() < dprob:
                dispid += 1
                filed = odate + timedelta(days=np.random.randint(7, 90))
                disputes.append((dispid, oid,
                                 np.random.choice(DISPUTE_REASONS, p=DISPUTE_REASON_WEIGHTS),
                                 round(amount * np.random.uniform(0.5, 1.0), 2),
                                 filed.isoformat(),
                                 np.random.choice(DISPUTE_RESOLUTIONS, p=DISPUTE_RESOLUTION_WEIGHTS)))

    print(f"  Orders: {len(orders):,}", flush=True)
    print(f"  Plans: {len(plans):,}", flush=True)
    print(f"  Installments: {len(insts):,}", flush=True)
    print(f"  Payments: {len(pays):,}", flush=True)
    print(f"  Decisions: {len(decisions):,}", flush=True)
    print(f"  Devices: {len(devices):,}", flush=True)
    print(f"  Disputes: {len(disputes):,}", flush=True)

    orders_df = pd.DataFrame(orders, columns=["order_id", "customer_id", "merchant_id", "order_amount", "order_date", "order_status", "channel"])
    plans_df = pd.DataFrame(plans, columns=["plan_id", "order_id", "plan_type", "num_installments", "installment_amount", "total_amount", "apr", "start_date", "end_date", "plan_status"])
    insts_df = pd.DataFrame(insts, columns=["installment_id", "plan_id", "installment_number", "amount_due", "due_date", "status", "days_past_due"])
    pays_df = pd.DataFrame(pays, columns=["payment_id", "installment_id", "amount_paid", "payment_date", "payment_method", "payment_status"])
    decisions_df = pd.DataFrame(decisions, columns=["decision_id", "customer_id", "order_id", "decision", "approved_amount", "decision_timestamp", "decline_reason"])
    devices_df = pd.DataFrame(devices, columns=["fingerprint_id", "customer_id", "order_id", "device_type", "device_id", "ip_address", "browser", "os", "is_vpn", "latitude", "longitude"])
    disputes_df = pd.DataFrame(disputes, columns=["dispute_id", "order_id", "dispute_reason", "dispute_amount", "filed_date", "resolution"])
    fraud_labels_df = pd.DataFrame(order_fraud_labels, columns=["order_id", "customer_id", "segment", "is_fraud"])

    return orders_df, plans_df, insts_df, pays_df, decisions_df, devices_df, disputes_df, fraud_labels_df


# ──────────────────────────────────────────
# 4. Data Quality Issues
# ──────────────────────────────────────────

def inject_data_quality_issues(cust, orders, insts, pays):
    print("Injecting data quality issues...", flush=True)
    dq = DATA_QUALITY_ISSUES
    cust = cust.copy()
    orders = orders.copy()
    insts = insts.copy()
    pays = pays.copy()

    n = len(cust)

    # Missing values
    for col, rate_key in [("email", "missing_email_rate"), ("phone", "missing_phone_rate"),
                           ("annual_income", "missing_income_rate"), ("credit_score", "missing_credit_score_rate"),
                           ("dob", "missing_dob_rate"), ("employment_status", "missing_employment_rate")]:
        mask = np.random.random(n) < dq[rate_key]
        cust.loc[mask, col] = None
        print(f"  Missing {col}: {mask.sum():,}", flush=True)

    # Near-duplicates
    n_dupes = int(n * dq["duplicate_customer_rate"])
    dupe_idx = np.random.choice(n, size=n_dupes, replace=False)
    dupes = cust.iloc[dupe_idx].copy()
    dupes["customer_id"] = range(n + 1, n + 1 + n_dupes)
    for i, idx in enumerate(dupes.index):
        choice = i % 3
        if choice == 0:
            name = str(dupes.at[idx, "first_name"])
            if len(name) > 2:
                pos = np.random.randint(1, len(name))
                dupes.at[idx, "first_name"] = name[:pos] + random.choice("aeiou") + name[pos + 1:]
        elif choice == 1:
            addr = str(dupes.at[idx, "address"])
            dupes.at[idx, "address"] = addr.replace("St", "Street").replace("Ave", "Avenue")
        else:
            dupes.at[idx, "email"] = f"alt_{dupes.at[idx, 'email']}"
    cust = pd.concat([cust, dupes], ignore_index=True)
    print(f"  Near-duplicates added: {n_dupes:,}", flush=True)

    # State format inconsistency
    n_cust = len(cust)
    mask_state = np.random.random(n_cust) < dq["inconsistent_state_format_rate"]
    state_vals = cust["state"].values
    for idx in np.where(mask_state)[0]:
        abbr = str(state_vals[idx])
        if abbr in US_STATES:
            state_vals[idx] = US_STATES[abbr]
    cust["state"] = state_vals
    print(f"  State format changes: {mask_state.sum():,}", flush=True)

    # Date format inconsistency in signup_date
    mask_date = np.random.random(n_cust) < dq["inconsistent_date_format_rate"]
    signup_vals = cust["signup_date"].values.copy()
    for idx in np.where(mask_date)[0]:
        val = str(signup_vals[idx])
        if len(val) == 10 and "-" in val:
            try:
                dt = date.fromisoformat(val)
                if idx % 2 == 0:
                    signup_vals[idx] = dt.strftime("%m/%d/%Y")
                else:
                    signup_vals[idx] = dt.strftime("%b %d, %Y")
            except ValueError:
                pass
    cust["signup_date"] = signup_vals

    # Zip code leading zero loss
    mask_zip = np.random.random(n_cust) < dq["leading_zero_zip_loss_rate"]
    zip_vals = cust["zip_code"].values.copy()
    for idx in np.where(mask_zip)[0]:
        z = str(zip_vals[idx])
        if z.startswith("0"):
            zip_vals[idx] = str(int(z))
    cust["zip_code"] = zip_vals

    # Dollar signs in order amounts
    n_ord = len(orders)
    orders["order_amount"] = orders["order_amount"].astype(object)
    mask_dollar = np.random.random(n_ord) < dq["dollar_sign_in_amount_rate"]
    for idx in np.where(mask_dollar)[0]:
        orders.iat[idx, orders.columns.get_loc("order_amount")] = f"${orders.iat[idx, orders.columns.get_loc('order_amount')]}"
    print(f"  Dollar-sign amounts: {mask_dollar.sum():,}", flush=True)

    # Negative amounts
    mask_neg = np.random.random(n_ord) < dq["negative_amount_rate"]
    amt_col = orders.columns.get_loc("order_amount")
    for idx in np.where(mask_neg)[0]:
        v = orders.iat[idx, amt_col]
        if isinstance(v, (int, float)):
            orders.iat[idx, amt_col] = -abs(v)

    # Absurd amounts
    mask_abs = np.random.random(n_ord) < dq["absurd_order_amount_rate"]
    for idx in np.where(mask_abs)[0]:
        orders.iat[idx, amt_col] = round(np.random.uniform(50000, 99999), 2)

    # Future DOBs
    mask_fdob = np.random.random(n_cust) < dq["future_dob_rate"]
    dob_vals = cust["dob"].values.copy()
    for idx in np.where(mask_fdob)[0]:
        future = date.today() + timedelta(days=np.random.randint(100, 3000))
        dob_vals[idx] = future.isoformat()
    cust["dob"] = dob_vals

    # Payments before orders
    if len(pays) > 0:
        n_pay = len(pays)
        mask_pbf = np.random.random(n_pay) < dq["payment_before_order_rate"]
        pdt_col = pays.columns.get_loc("payment_date")
        for idx in np.where(mask_pbf)[0]:
            val = str(pays.iat[idx, pdt_col])
            try:
                dt = datetime.fromisoformat(val)
                shifted = dt - timedelta(days=np.random.randint(30, 180))
                pays.iat[idx, pdt_col] = shifted.isoformat()
            except ValueError:
                pass

        # Overpayments
        mask_over = np.random.random(n_pay) < dq["overpayment_rate"]
        ap_col = pays.columns.get_loc("amount_paid")
        for idx in np.where(mask_over)[0]:
            pays.iat[idx, ap_col] = round(float(pays.iat[idx, ap_col]) * np.random.uniform(1.5, 3.0), 2)

    # Orphaned installments
    if len(insts) > 0:
        n_orphans = int(len(insts) * dq["orphaned_installment_rate"])
        orphan_idx = np.random.choice(len(insts), size=n_orphans, replace=False)
        max_pid = insts["plan_id"].max()
        pid_col = insts.columns.get_loc("plan_id")
        for idx in orphan_idx:
            insts.iat[idx, pid_col] = max_pid + np.random.randint(1000, 9999)
        print(f"  Orphaned installments: {n_orphans:,}", flush=True)

    # Inconsistent date formats in due_date
    n_inst = len(insts)
    mask_idate = np.random.random(n_inst) < dq["inconsistent_date_format_rate"] * 0.5
    dd_col = insts.columns.get_loc("due_date")
    insts["due_date"] = insts["due_date"].astype(object)
    for idx in np.where(mask_idate)[0]:
        val = str(insts.iat[idx, dd_col])
        if len(val) == 10 and "-" in val:
            try:
                dt = date.fromisoformat(val)
                insts.iat[idx, dd_col] = dt.strftime("%m/%d/%Y")
            except ValueError:
                pass

    return cust, orders, insts, pays


# ──────────────────────────────────────────
# 5. Write to SQLite
# ──────────────────────────────────────────

def write_to_sqlite(db_path, customers, merchants, orders, plans, insts, pays, decisions, devices, disputes, fraud_labels):
    print(f"Writing to {db_path}...", flush=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    seg_truth = customers[["customer_id", "segment"]].copy()
    cust_write = customers.drop(columns=["segment"], errors="ignore")

    cust_write.to_sql("customers", conn, index=False)
    merchants.to_sql("merchants", conn, index=False)
    orders.to_sql("orders", conn, index=False)
    plans.to_sql("payment_plans", conn, index=False)
    insts.to_sql("installments", conn, index=False)
    pays.to_sql("payments", conn, index=False)
    decisions.to_sql("credit_decisions", conn, index=False)
    devices.to_sql("device_fingerprints", conn, index=False)
    disputes.to_sql("disputes", conn, index=False)
    seg_truth.to_sql("customer_segments_ground_truth", conn, index=False)
    fraud_labels.to_sql("order_fraud_labels", conn, index=False)

    for idx_sql in [
        "CREATE INDEX idx_orders_cust ON orders(customer_id)",
        "CREATE INDEX idx_orders_merch ON orders(merchant_id)",
        "CREATE INDEX idx_plans_order ON payment_plans(order_id)",
        "CREATE INDEX idx_inst_plan ON installments(plan_id)",
        "CREATE INDEX idx_pay_inst ON payments(installment_id)",
        "CREATE INDEX idx_dev_cust ON device_fingerprints(customer_id)",
        "CREATE INDEX idx_disp_order ON disputes(order_id)",
        "CREATE INDEX idx_dec_order ON credit_decisions(order_id)",
    ]:
        conn.execute(idx_sql)

    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print("\nDatabase tables:")
    for (tbl,) in cur.fetchall():
        cur.execute(f"SELECT COUNT(*) FROM [{tbl}]")
        cnt = cur.fetchone()[0]
        print(f"  {tbl}: {cnt:,} rows")

    conn.close()
    print("\nDone!", flush=True)


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bnpl.db")
    db_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    customers = generate_customers()
    merchants = generate_merchants()

    orders, plans, insts, pays, decisions, devices, disputes, fraud_labels = \
        generate_orders_and_downstream(customers, merchants)

    cust_d, orders_d, insts_d, pays_d = \
        inject_data_quality_issues(customers, orders, insts, pays)

    write_to_sqlite(db_path, cust_d, merchants, orders_d, plans, insts_d, pays_d, decisions, devices, disputes, fraud_labels)


if __name__ == "__main__":
    main()
