# CFR for Kuhn Poker

## Description

Simple implementation of the Counterfactual Regret Minimization (CFR) algorithm to solve Kuhn Poker (ante 1 / raise 1 / no duplicate cards).

### Kuhn Poker Rules

Kuhn Poker is a simplified version of poker played with a deck of three cards: King (K), Queen (Q), and Jack (J). The game is played between two players, and the rules are as follows (see [Kuhn Poker on Wikipedia](https://en.wikipedia.org/wiki/Kuhn_poker) for more details):

1. Each player antes 1 chip to the pot.
2. Each player is dealt one card from the deck, which they keep hidden from their opponent.
3. The first player (Player 0) can either bet 1 chip or check.
4. If Player 0 checks, Player 1 can either bet 1 chip or check.
5. If Player 0 bets, Player 1 can either call (match the bet) or fold.
6. If Player 1 bets after Player 0 checks, Player 0 can either call or fold.
7. If both players check, the player with the higher card wins the pot (2 chips).
8. If a player folds, the other player wins the pot.

### CFR Algorithm

Take a look at [https://modelai.gettysburg.edu/2013/cfr/cfr.pdf](https://modelai.gettysburg.edu/2013/cfr/cfr.pdf) for more details on the Counterfactual Regret Minimization (CFR) algorithm.

## Usage

The project requires `jax` and `numpy`.

```bash
python run main.py
```

## Example Output

Below is the output after running the CFR algorithm for 100,000 iterations and testing the resulting strategies over 10,000 games:

```plaintext
Player 0 average strategy:
  Infoset 0: 0.885 0.115
  Infoset 1: 1.000 0.000
  Infoset 2: 1.000 0.000
  Infoset 3: 0.557 0.443
  Infoset 4: 0.676 0.324
  Infoset 5: 0.000 1.000
Player 1 average strategy:
  Infoset 0: 0.666 0.334
  Infoset 1: 1.000 0.000
  Infoset 2: 1.000 0.000
  Infoset 3: 0.666 0.334
  Infoset 4: 0.000 1.000
  Infoset 5: 0.000 1.000
Average Payoff over 10000 games: Player 0: -0.3602, Player 1: 0.3602
```
