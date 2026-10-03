import pandas as pd

from agent.instruments import get
from agent.macro import BiasFile, CalendarSurprise
from agent.news.calendar import Event, load_history
from agent.news.guard import NewsGuard

CPI = pd.Timestamp("2024-03-12 12:30", tz="UTC")
EV = [Event(CPI, "USD", "High", "CPI m/m", "0.4%", "0.3%", "0.3%")]


def test_flatten_30_minutes_before():
    g = NewsGuard(EV)
    eur, gold, gbpjpy = get("EURUSD"), get("GOLDUSDC"), get("GBPJPY")
    assert g.must_flatten(eur, CPI - pd.Timedelta("29min"))
    assert g.must_flatten(gold, CPI - pd.Timedelta("30min"))
    assert not g.must_flatten(eur, CPI - pd.Timedelta("31min"))
    assert not g.must_flatten(gbpjpy, CPI - pd.Timedelta("10min"))   # no USD in GBPJPY


def test_no_entries_until_30_minutes_after():
    g = NewsGuard(EV)
    eur = get("EURUSD")
    assert g.blocks_entry(eur, CPI + pd.Timedelta("29min"))
    assert not g.blocks_entry(eur, CPI + pd.Timedelta("31min"))
    assert g.blocks_entry(eur, CPI - pd.Timedelta("20min"))


def test_correlated_markets_flatten_too():
    t = pd.Timestamp("2024-03-07 13:15", tz="UTC")
    ecb = [Event(t, "EUR", "High", "Main Refinancing Rate")]
    china = [Event(t, "CNY", "High", "GDP q/y")]
    soon = t - pd.Timedelta("10min")
    assert NewsGuard(ecb).must_flatten(get("GBPUSD"), soon)            # GBP moves with EUR
    assert not NewsGuard(ecb, include_correlated=False).must_flatten(get("GBPUSD"), soon)
    assert NewsGuard(china).must_flatten(get("AUDUSD"), soon)
    assert not NewsGuard(ecb).must_flatten(get("USDJPY"), soon)


def test_surprise_events_are_not_known_in_advance():
    from agent.news.calendar import scheduled_only
    ev = [Event(pd.Timestamp("2020-03-15 21:00", tz="UTC"), "USD", "High", "Federal Funds Rate"),
          Event(pd.Timestamp("2020-03-17 15:40", tz="UTC"), "USD", "High", "President Speaks"),
          Event(pd.Timestamp("2024-03-10 21:45", tz="UTC"), "NZD", "High", "GDP q/q"),
          EV[0]]
    kept = [e.title for e in scheduled_only(ev)]
    assert kept == ["GDP q/q", "CPI m/m"]


def test_load_forexfactory_csv(tmp_path):
    p = tmp_path / "ff.csv"
    p.write_text("date,time,currency,impact,event,actual,forecast,previous\n"
                 "Tue Mar 12 2024,12:30,USD,High,CPI m/m,0.4%,0.3%,0.3%\n"
                 "Tue Mar 12 2024,All Day,USD,High,Holiday,,,\n"
                 "Tue Mar 12 2024,9:00,EUR,Low,Something,,,\n", encoding="utf-8-sig")
    ev = load_history(p)
    assert len(ev) == 1 and ev[0].time == CPI and ev[0].currency == "USD"


def test_surprise_veto_has_no_lookahead():
    events = [Event(CPI + pd.Timedelta(days=i), "USD", "High", f"x{i}", "2", "1") for i in range(3)]
    s = CalendarSurprise(events, threshold=3)
    eur = get("EURUSD")
    # Before the 3rd release only 2 surprises are known -> no veto yet.
    assert s.veto(eur, 1, CPI + pd.Timedelta(days=1, hours=1)) is None
    # After all 3 strong USD beats, buying EUR (selling USD) is vetoed.
    assert s.veto(eur, 1, CPI + pd.Timedelta(days=2, hours=1))
    assert s.veto(eur, -1, CPI + pd.Timedelta(days=2, hours=1)) is None


def test_unemployment_lower_is_better():
    e = [Event(CPI, "USD", "High", "Unemployment Rate", "3.5%", "3.8%")] * 3
    assert CalendarSurprise(e).score("USD", CPI) == 3


def test_bias_file(tmp_path):
    f = tmp_path / "bias.json"
    f.write_text('{"risk_mode": "off", "avoid": ["SOLUSDC"], "bias": {"GOLDUSDC": 1}}')
    b = BiasFile(f)
    assert b.veto(get("SOLUSDC"), -1, CPI)
    assert b.veto(get("GOLDUSDC"), -1, CPI)
    assert b.veto(get("GOLDUSDC"), 1, CPI) is None
    assert b.veto(get("BTCUSDC"), 1, CPI)          # risk-off: no crypto longs
    assert b.veto(get("BTCUSDC"), -1, CPI) is None
