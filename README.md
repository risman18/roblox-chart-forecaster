# Roblox Up-and-Coming Forecaster

A forecaster for Roblox's Up-and-Coming chart, built for make.fun's
roblox-up-and-coming-forecast-01 bounty.

`main.py` reads the chart's snapshot history at `/input` (history.ndjson +
task.json) and writes one forecast to `/output/forecast.json`: for each of
the top 10 places, a probability for each game named being at that place
across the scored 72-hour window, plus the guessed top 10 in order.

Model: each game gets a latent rank path — last rank, a damped drift from
its rank velocity and player-count growth, and noise that widens with the
horizon (with the chart's 28-day age-out applied where a creation date is
known). A seeded Monte Carlo rolls the paths over the window; a place's
probabilities are the time-averaged occupancy, shrunk toward a persistence
prior and a diffuse top-30 prior. Deterministic, no network, standard
library only.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
