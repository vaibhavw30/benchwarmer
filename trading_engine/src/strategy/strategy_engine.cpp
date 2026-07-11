#include "strategy/strategy_engine.hpp"
#include "strategy/pricing.hpp"
#include "strategy/arb.hpp"
#include "strategy/edge_taker.hpp"
#include "strategy/market_maker.hpp"
namespace te {

void StrategyEngine::on_book_update(const Ticker& t, const OrderBook& b, long now_ms) {
  auto fv = fv_.fair_value(t);
  bool stale = fv_.is_stale(t, now_ms, c_.fair_value_max_age_secs);
  bool crossed = b.crossed();
  if (!fv || stale || crossed) {
    tel_.event("skip", {{"ticker", t},
                         {"has_fv", fv.has_value()},
                         {"stale", stale},
                         {"crossed", crossed}});
    return;
  }

  Cents fair = fair_price_cents(fv->p_yes);
  int thr = edge_threshold_cents(fv->confidence, c_);

  // 1) Arb
  if (auto s = detect_arb(b, c_); s.present) {
    // LIMITATION(v1): places only the YES leg; PaperVenue models signed-YES
    // only, so the NO leg of the lock is not executed. True two-legged arb
    // execution is deferred (final review / M5). This is a naked directional
    // fill, NOT a riskless lock.
    Order o{t, Side::Yes, s.action, s.yes_price, s.qty};
    auto d = risk_.check(o, stale, crossed);
    if (d.allow) {
      o.qty = d.approved_qty;
      venue_.place_against(b, o);
      risk_.set_position(t, venue_.position(t));
      tel_.event("arb", {{"ticker", t},
                          {"qty", d.approved_qty},
                          {"yes", s.yes_price},
                          {"no", s.no_price},
                          {"leg", "yes_only"}});
    }
    return;
  }

  // 2) Edge-take
  if (auto s = detect_take(b, fair, thr, c_); s.present) {
    Order o{t, s.side, s.action, s.price, s.size};
    auto d = risk_.check(o, stale, crossed);
    if (d.allow) {
      o.qty = d.approved_qty;
      venue_.place_against(b, o);
      risk_.set_position(t, venue_.position(t));
      tel_.event("take", {{"ticker", t}, {"price", s.price}, {"qty", d.approved_qty}});
    }
    return;
  }

  // 3) Market-make (log the intended quote; paper resting-order sim is deferred)
  auto q = make_quote(fair, fv->confidence, venue_.position(t), c_);
  tel_.event("quote", {{"ticker", t}, {"bid", q.bid}, {"ask", q.ask}, {"size", q.size}, {"fair", fair}});
}
}  // namespace te
