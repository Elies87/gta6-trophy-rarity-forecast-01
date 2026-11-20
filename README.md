# GTA VI Trophy Rarity Forecast

Entry for the make.fun bounty **gta6-trophy-rarity-forecast-01**: forecast,
for every PlayStation 5 trophy of Grand Theft Auto VI, the percent of the
players Exophase tracks for the game who will have earned it at the case's
target time — as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy, scored
by pinball loss on the logit scale.

## The program

`main.py` (Python 3.13, standard library only, no network) reads every case
file `*.in` in the input directory (`$MAKEFUN_INPUT_DIR`, else `/input`) and
writes one `<id>.out` per case to the output directory
(`$MAKEFUN_OUTPUT_DIR`, else `/output`). A case carries the trophies and the
house's recorded history of the game's Exophase page: every read's
players-tracked count and every trophy's earn rate. The output is
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}` — one row
per trophy, quantiles non-decreasing, in percent. Runs are deterministic:
the same case always produces the same bytes.

## The model

An earn rate is a ratio — earners over players tracked — and the two sides
move for different reasons, so the program forecasts them separately and
divides at the end:

1. **States, not reads.** Exophase refreshes its counts every few hours, so
   a run of identical reads is one state of the page. The history is
   compressed into distinct states; all dynamics live at refresh boundaries.
2. **Pool forecast.** Players tracked is extrapolated on a log-time axis
   (u = ln(1 + hours since the first read)) with a least-squares fit,
   cross-checked against the recent half's growth rate, and never allowed
   to shrink.
3. **Earner forecast.** Each trophy's implied earners (rate × pool / 100)
   get their own log-time fit; the growth slope is shrunk toward the pooled
   slope of the trophy's tier group (platinum rides with gold), because a
   slope read off two or three states is mostly noise. Earners are floored
   just under their last observed count.
4. **Median.** 100 × forecast earners / forecast pool, clamped to
   [0.01, 99.99]. A trophy earned by every tracked player at every read
   stays a hair under 100.
5. **Spread.** On the logit scale, σ combines a random-walk floor
   (0.5·√horizon-days), the earner fit's residual scatter grown by the
   extrapolation distance, and a refresh-risk term that stays large while
   the observed history is still flat — the launch-night lesson: an
   un-refreshed page can jump, diluting almost every rate at once.
6. **Early-pool prior.** While no refresh has been observed and the game is
   days old, the tracked pool is still the self-selected first wave of
   players, who earn trophies faster than later joiners; the median is
   shifted down 0.4 logit units to allow for that dilution.

## Running it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

where `input/` holds the case files (the kit's practice cases, or the final
case at grading time). Each run takes well under a second per case.
