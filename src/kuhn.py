from typing import NamedTuple
from jax import numpy as jnp
import numpy as np


class GameState(NamedTuple):
    player_hands: tuple[int, int]
    bets: tuple[int, int]
    active_player: int
    pot: int


class KuhnPoker:
    def __init__(self):
        self.num_players = 2
        self.players = (0, 1)
        self.actions = {0: "p/c", 1: "bet"}
        self.num_actions = len(self.actions)
        self.cards = [1, 2, 3]

    def reset(self) -> GameState:
        # Reset the game state for a new round
        return GameState(
            player_hands=(0, 0),
            bets=(1, 1),
            active_player=0,
            pot=2,
        )

    def deal_cards(self, state: GameState) -> GameState:
        cards = np.random.permutation(self.cards)
        return state._replace(player_hands=(cards[0], cards[1]))

    def step(self, state: GameState, action: int) -> tuple[GameState, bool]:
        player = state.active_player
        bets = list(state.bets)
        pot = state.pot

        if action == 1:
            bets[player] += 1
            pot += 1

        terminal = False
        if action == 0 and bets[0] != bets[1]:
            # If the player checks but the bets are unequal, treat it as a fold
            terminal = True
        elif action == 1 and bets[0] == bets[1]:
            # If the player bets and the bets are equal, treat it as a call
            terminal = True
        elif player == 1 and action == 0:
            # If player 1 checks after player 0 checked, the round ends
            terminal = True

        # Switch active player
        next_player = 1 - player if not terminal else player

        return (
            GameState(
                player_hands=state.player_hands,
                bets=(bets[0], bets[1]),
                active_player=next_player,
                pot=pot,
            ),
            terminal,
        )

    def get_payoffs(self, state: GameState) -> tuple[int, int]:
        player1_win = 0
        if state.bets[0] != state.bets[1]:
            player1_win = 1 if state.bets[0] > state.bets[1] else -1
        else:
            player1_win = 1 if state.player_hands[0] > state.player_hands[1] else -1

        if player1_win > 0:
            return (state.pot - state.bets[0], -state.bets[1])
        else:
            return (-state.bets[0], state.pot - state.bets[1])

    def get_infoset(self, state: GameState, player: int) -> int:
        hand = state.player_hands[player]
        opponent_bet = state.bets[1 - player]
        return (hand - 1) * 2 + (opponent_bet - 1)  # [0, 1, 2] * 2 + [0, 1]


class PlayerStrategy:
    def __init__(self):
        self.num_actions = 2
        self.num_infosets = 6  # 3 cards * 2 betting histories
        self.regret_sum = jnp.zeros((self.num_infosets, self.num_actions))
        self.strategy_sum = jnp.zeros((self.num_infosets, self.num_actions))

    def get_strategy(self, infoset: int) -> jnp.ndarray:
        positive_regrets = jnp.maximum(self.regret_sum[infoset], 0)
        normalizing_sum = jnp.sum(positive_regrets)
        strategy = jnp.where(
            normalizing_sum > 0,
            positive_regrets / normalizing_sum,  # Avoid division by zero
            jnp.ones(self.num_actions) / self.num_actions,
        )
        return strategy

    def update_regrets(self, infoset: int, regrets: jnp.ndarray, reach_prob: float):
        self.regret_sum = self.regret_sum.at[infoset].add(regrets * reach_prob)

    def get_average_strategy(self) -> jnp.ndarray:
        normalizing_sum = jnp.sum(self.strategy_sum, axis=1, keepdims=True)
        avg_strategy = jnp.where(
            normalizing_sum > 0,
            self.strategy_sum / normalizing_sum,
            jnp.ones((self.num_infosets, self.num_actions)) / self.num_actions,
        )
        return avg_strategy

    def update_strategy_sum(
        self, infoset: int, strategy: jnp.ndarray, reach_prob: float
    ):
        self.strategy_sum = self.strategy_sum.at[infoset].add(strategy * reach_prob)


def cfr(
    game: KuhnPoker,
    state: GameState,
    i: int,
    strategies: list[PlayerStrategy],
    terminal: bool,
    p1: float,
    p2: float,
) -> float:
    if terminal:
        payoffs = game.get_payoffs(state)
        return payoffs[i]

    player = state.active_player
    infoset = game.get_infoset(state, player)
    strategy = strategies[player].get_strategy(infoset)

    util = jnp.zeros(game.num_actions)
    node_util = 0.0

    for a in range(game.num_actions):
        action_prob: float = strategy[a].item()
        new_state, terminal = game.step(state, a)

        if player == 0:
            new_p1 = p1 * action_prob
            new_p2 = p2
        else:
            new_p1 = p1
            new_p2 = p2 * action_prob

        util_a = cfr(game, new_state, i, strategies, terminal, new_p1, new_p2)
        util = util.at[a].set(util_a)
        node_util += action_prob * util_a

    if player == i:
        for a in range(game.num_actions):
            regret = (util[a] - node_util) * (p2 if i == 0 else p1)
            strategies[player].update_regrets(
                infoset,
                jnp.array([0 if j != a else regret for j in range(game.num_actions)]),
                1.0,
            )
            strategies[player].update_strategy_sum(
                infoset, strategy, p1 if player == 0 else p2
            )

    return node_util


def solve(K: int):
    game = KuhnPoker()
    strategies = [PlayerStrategy(), PlayerStrategy()]

    for t in range(K):
        for i in range(2):
            state = game.reset()
            state = game.deal_cards(state)
            cfr(game, state, i, strategies, False, 1.0, 1.0)

        if (t + 1) % (K // 10) == 0:
            print(f"Iteration {t + 1}/{K} completed.")

    return strategies


def test_kuhn(strategies: list[PlayerStrategy], num_tests: int = 10000):
    game = KuhnPoker()
    total_payoff = [0, 0]

    for _ in range(num_tests):
        state = game.reset()
        state = game.deal_cards(state)
        terminal = False
        p1, p2 = 1.0, 1.0

        while not terminal:
            player = state.active_player
            infoset = game.get_infoset(state, player)
            strategy = strategies[player].get_strategy(infoset)
            action = np.random.choice(game.num_actions, p=strategy)

            state, terminal = game.step(state, action)

        payoffs = game.get_payoffs(state)
        total_payoff[0] += payoffs[0]
        total_payoff[1] += payoffs[1]

    avg_payoff = [total / num_tests for total in total_payoff]
    print(
        f"Average Payoff over {num_tests} games: Player 0: {avg_payoff[0]}, Player 1: {avg_payoff[1]}"
    )
