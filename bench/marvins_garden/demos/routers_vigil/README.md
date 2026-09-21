# The Router's Vigil — graphical realm defense, played by Marvin

Request packets fall from the sky; Marvin routes each into one of eight
serving portals whose specs (RAM / context / vision) are the trained policy
constraints. Tool turns pass the Oracle gate; injection wraiths are judged
by the Warden. Wrong verdicts breach the realm's hearts.

```bash
python server.py            # loads Bonsai-27B + v15c adapter, serves the game
open http://localhost:8765  # ▶ Watch Marvin (replay a real round) · ⚔ defend yourself
```

- Scenarios come from `bench/marvins_garden` generator samplers, so every
  event is in-distribution and scored by the policy pickers — never by the
  model judging itself.
- `run_round.py` records a full round to JSON; `make_gameplay_video.py`
  renders it to mp4 (frame-accurate replay of real decisions).
