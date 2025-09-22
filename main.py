from src.kuhn import solve, test_kuhn
import numpy as np

if __name__ == "__main__":
    np.random.seed(42)
    strategies = solve(10000)
    for i, strategy in enumerate(strategies):
        avg_strategy = strategy.get_average_strategy()
        print(f"Player {i} average strategy:")
        for infoset, strat in enumerate(avg_strategy):
            print(f"  Infoset {infoset}: {strat[0].item():.3f} {strat[1].item():.3f}")

    test_kuhn(strategies, 10000)
