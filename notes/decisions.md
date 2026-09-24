# Research decisions log

## Mode-group taxonomy (WP2/WP4), decided 2026-09-20

**Source of truth:** BadODD paper (arXiv 2401.10659), Table 2, which defines each class by
power source and size rather than local name:

| Class | Paper's definition |
|---|---|
| `three_wheeler` | "3 Wheeler, Paddle, Small" — pedal-powered, i.e. the cycle-rickshaw |
| `auto_rickshaw` | "3 Wheeler, Gas/Electric, Medium" — CNG **and** battery-powered ("tomtom"/easy-bike) three-wheelers are both this one class, split from `three_wheeler` purely by power source |
| `cart_vehicle` | "3 or 2 Wheeler, Human/Animal, No paddle, Small" — push-cart / animal cart |
| `priority_vehicle` | "Siren vehicle, fossil fuel or paddle, any size" — ambulance/police/fire; fossil-fuel in practice |
| `construction_vehicle` | "Construction vehicle, diesel/fossil fuel, medium/big" |

This directly resolves a naming confusion: CNG autorickshaws, battery rickshaws and "tomtom"
(all colloquially "three-wheelers" in Chattogram) are **not** split between classes by local
name — the dataset splits them by power source. Every motorised three-wheeler (CNG or battery)
is `auto_rickshaw`; every pedal-powered three-wheeler is `three_wheeler`.

**Decision: 4 mode groups**, matching `AGENT_BRIEF.md` §8's original framing (chosen over a
5-group split, since only 7 classes have enough Chattogram test instances to report
individually — 5 groups would over-fragment that):

| Group | Classes | Notes |
|---|---|---|
| Pedestrian | `person` | headline |
| Non-motorised | `three_wheeler` (cycle-rickshaw), `bicycle`, `cart_vehicle` | `three_wheeler` is headline (1,037 CTG instances); `bicycle`/`cart_vehicle` are footnote-only (<30 instances) |
| Informal motorised | `auto_rickshaw` (CNG + battery/tomtom) | headline, sole member |
| Formal motorised | `car`, `bus`, `truck`, `motorbike` | all headline; `priority_vehicle`, `construction_vehicle` also assigned here (fossil-fuel, footnote-only) |

Mode-group AP/recall is aggregated weighted by each class's instance count, so the
footnote-only classes contribute negligibly rather than being silently dropped.
