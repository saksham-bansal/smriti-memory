"""
Relative-to-Absolute Temporal Expression Resolver.

Addresses Failure Mode #2 of standard RAG: turns containing relative time markers
('yesterday', 'last week', '4 years ago', 'last Friday', 'next month') cannot be
retrieved or reasoned over accurately unless anchored to the session's timestamp at ingestion.
"""
import calendar
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_NUMBER_WORDS = {
    "one": 1, "a": 1, "an": 1, "two": 2, "couple": 2, "three": 3, "few": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}


def _word_to_int(val: str) -> Optional[int]:
    val = val.strip().lower()
    if val.isdigit():
        return int(val)
    return _NUMBER_WORDS.get(val)


class TemporalNormalizer:
    """Parses session timestamps and resolves relative time mentions in dialogue turns."""

    @staticmethod
    def parse_session_timestamp(ts_str: str) -> Optional[datetime]:
        if not ts_str:
            return None
        s = ts_str.strip()
        # Format 1: LoCoMo style "1:56 pm on 8 May, 2023" or "8 May, 2023"
        m = re.search(
            r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", s
        )
        if m:
            day = int(m.group(1))
            mon = _MONTHS.get(m.group(2).lower())
            year = int(m.group(3))
            if mon:
                try:
                    return datetime(year, mon, day)
                except ValueError:
                    pass

        # Format 2: "May 8, 2023"
        m2 = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
        if m2:
            mon = _MONTHS.get(m2.group(1).lower())
            day = int(m2.group(2))
            year = int(m2.group(3))
            if mon:
                try:
                    return datetime(year, mon, day)
                except ValueError:
                    pass

        # Format 3: ISO "2023-05-08"
        m3 = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m3:
            try:
                return datetime(int(m3.group(1)), int(m3.group(2)), int(m3.group(3)))
            except ValueError:
                pass
        return None

    @classmethod
    def extract_temporal_anchors(
        cls, text: str, session_timestamp: str
    ) -> Tuple[List[str], Optional[str]]:
        """
        Returns (human_readable_anchors, primary_normalized_date_str) resolved
        against `session_timestamp`.
        """
        dt = cls.parse_session_timestamp(session_timestamp)
        if dt is None:
            return [], None

        session_date_pretty = f"{dt.day} {_MONTH_NAMES[dt.month]} {dt.year}"
        anchors: List[str] = [f"session_date={session_date_pretty} ({dt.strftime('%Y-%m-%d')})"]
        primary_norm: Optional[str] = session_date_pretty
        lower = text.lower()

        # 1. "yesterday" / "the day before yesterday" / "tomorrow"
        if "day before yesterday" in lower:
            target = dt - timedelta(days=2)
            p = f"{target.day} {_MONTH_NAMES[target.month]} {target.year}"
            anchors.append(f"'the day before yesterday' -> {p} ({target.strftime('%Y-%m-%d')})")
            primary_norm = p
        elif re.search(r"\byesterday\b", lower):
            target = dt - timedelta(days=1)
            p = f"{target.day} {_MONTH_NAMES[target.month]} {target.year}"
            anchors.append(f"'yesterday' -> {p} ({target.strftime('%Y-%m-%d')})")
            primary_norm = p
        if re.search(r"\btomorrow\b", lower):
            target = dt + timedelta(days=1)
            p = f"{target.day} {_MONTH_NAMES[target.month]} {target.year}"
            anchors.append(f"'tomorrow' -> {p} ({target.strftime('%Y-%m-%d')})")
            primary_norm = p

        # 2. "N days/weeks/months/years ago"
        for m in re.finditer(
            r"\b(\d+|one|two|couple|three|few|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s+(day|days|week|weeks|month|months|year|years)\s+ago\b",
            lower,
        ):
            n = _word_to_int(m.group(1))
            unit = m.group(2)
            if n is not None:
                if unit.startswith("day"):
                    t = dt - timedelta(days=n)
                    p = f"{t.day} {_MONTH_NAMES[t.month]} {t.year}"
                    anchors.append(f"'{m.group(0)}' -> {p}")
                    primary_norm = p
                elif unit.startswith("week"):
                    t = dt - timedelta(days=7 * n)
                    p = f"week of {t.day} {_MONTH_NAMES[t.month]} {t.year} ({_MONTH_NAMES[t.month]} {t.year})"
                    anchors.append(f"'{m.group(0)}' -> {p}")
                    primary_norm = f"{_MONTH_NAMES[t.month]} {t.year}"
                elif unit.startswith("month"):
                    total_m = (dt.year * 12 + (dt.month - 1)) - n
                    y, m_idx = divmod(total_m, 12)
                    p = f"{_MONTH_NAMES[m_idx + 1]} {y}"
                    anchors.append(f"'{m.group(0)}' -> {p}")
                    primary_norm = p
                elif unit.startswith("year"):
                    y = dt.year - n
                    anchors.append(f"'{m.group(0)}' -> {y} ({n} years before {dt.year})")
                    primary_norm = str(y)

        # 3. "last year" / "next year" / "this year"
        if re.search(r"\blast year\b", lower):
            y = dt.year - 1
            anchors.append(f"'last year' -> {y}")
            primary_norm = str(y)
        if re.search(r"\bnext year\b", lower):
            y = dt.year + 1
            anchors.append(f"'next year' -> {y}")
            primary_norm = str(y)

        # 4. "last month" / "next month"
        if re.search(r"\blast month\b", lower):
            total_m = (dt.year * 12 + (dt.month - 1)) - 1
            y, m_idx = divmod(total_m, 12)
            p = f"{_MONTH_NAMES[m_idx + 1]} {y}"
            anchors.append(f"'last month' -> {p}")
            primary_norm = p
        if re.search(r"\bnext month\b", lower):
            total_m = (dt.year * 12 + (dt.month - 1)) + 1
            y, m_idx = divmod(total_m, 12)
            p = f"{_MONTH_NAMES[m_idx + 1]} {y}"
            anchors.append(f"'next month' -> {p}")
            primary_norm = p

        # 5. "last week" / "next week" / "last weekend" / "two weekends ago"
        for m_wknd in re.finditer(
            r"\b(\d+|one|two|couple|three|few)\s+weekends?\s+ago\b", lower
        ):
            n_w = _word_to_int(m_wknd.group(1)) or 2
            p = f"{m_wknd.group(1)} weekends before {session_date_pretty}"
            anchors.append(f"'{m_wknd.group(0)}' -> {p}")
            primary_norm = p

        if re.search(r"\blast week\b", lower):
            t = dt - timedelta(days=7)
            p = f"The week before {session_date_pretty} (around {t.day} {_MONTH_NAMES[t.month]} {t.year})"
            anchors.append(f"'last week' -> {p}")
            primary_norm = f"The week before {session_date_pretty}"
        if re.search(r"\blast weekend\b", lower):
            days_since_sun = (dt.weekday() + 1) % 7
            if days_since_sun == 0:
                days_since_sun = 7
            sun = dt - timedelta(days=days_since_sun)
            sat = sun - timedelta(days=1)
            p = f"the weekend before {session_date_pretty} ({sat.day}-{sun.day} {_MONTH_NAMES[sun.month]} {sun.year})"
            anchors.append(f"'last weekend' -> {p}")
            primary_norm = p
        if re.search(r"\bnext week\b", lower):
            t = dt + timedelta(days=7)
            p = f"the week after {session_date_pretty} (around {t.day} {_MONTH_NAMES[t.month]} {t.year})"
            anchors.append(f"'next week' -> {p}")

        # 6. "(last|this past|on) <weekday>" (e.g., "last Friday", "on Tuesday")
        for m in re.finditer(
            r"\b(last|this past|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            lower,
        ):
            w_name = m.group(2).capitalize()
            w_target = _WEEKDAYS[m.group(2)]
            diff = (dt.weekday() - w_target) % 7
            if diff == 0:
                diff = 7
            t = dt - timedelta(days=diff)
            p = f"{t.day} {_MONTH_NAMES[t.month]} {t.year}"
            anchors.append(
                f"'{m.group(0)}' -> {p} ({t.strftime('%Y-%m-%d')}; the {w_name} before {session_date_pretty})"
            )
            primary_norm = p

        # 7. Duration expressions: "for N years", "N years now", "since YYYY"
        for m in re.finditer(
            r"\b(?:for\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s+(year|years|month|months)(?:\s+now|\s+so\s+far|\s+ago)?\b",
            lower,
        ):
            full_match = m.group(0)
            if full_match.endswith("ago"):
                continue
            n = _word_to_int(m.group(1))
            unit = m.group(2)
            if n is not None and unit.startswith("year"):
                start_y = dt.year - n
                anchors.append(
                    f"'{full_match}' -> {n} years / Since {start_y} (as of {dt.year})"
                )
                if primary_norm == session_date_pretty:
                    primary_norm = f"{n} years (Since {start_y})"

        # 8. Explicit month mentions without year (e.g. "in June", "on May 20")
        for m in re.finditer(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
            lower,
        ):
            mon_num = _MONTHS[m.group(1)]
            day_num = int(m.group(2))
            p = f"{day_num} {_MONTH_NAMES[mon_num]} {dt.year}"
            anchors.append(f"'{m.group(0)}' -> {p}")
            primary_norm = p

        return anchors, primary_norm
