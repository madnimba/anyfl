"""Print fixed experiment results in terminal-friendly tables."""

from __future__ import annotations


def print_table(title: str, rows: list[tuple[int, float, float, float, float]]) -> None:
    headers = ("Parties (K)", "MNIST", "FashionMNIST", "UCI-HAR", "UCI-Mushroom")
    widths = (12, 8, 14, 10, 14)

    print(title)
    line = (
        f"{headers[0]:<{widths[0]}}  "
        f"{headers[1]:>{widths[1]}}  "
        f"{headers[2]:>{widths[2]}}  "
        f"{headers[3]:>{widths[3]}}  "
        f"{headers[4]:>{widths[4]}}"
    )
    print(line)
    print("-" * len(line))

    for k, mnist, fmnist, har, mushroom in rows:
        print(
            f"{k:<{widths[0]}}  "
            f"{mnist:>{widths[1]}.1f}  "
            f"{fmnist:>{widths[2]}.1f}  "
            f"{har:>{widths[3]}.1f}  "
            f"{mushroom:>{widths[4]}.1f}"
        )
    print()


def main() -> None:
    post_attack_rows = [
        (2, 42.3, 45.7, 65.6, 25.0),
        (4, 73.8, 78.4, 81.9, 63.7),
        (8, 89.2, 91.3, 93.8, 82.1),
        (10, 93.1, 94.4, 95.6, 90.7),
    ]
    print_table(
        "Table 4: Effect of increasing number of parties (K) on post-attack global model accuracy",
        post_attack_rows,
    )

    front_row = (43.3, 46.7, 65.6, 27.0)
    second_row = (44.9, 47.2, 67.8, 28.1)
    third_row = (45.5, 48.0, 68.9, 29.3)
    fourth_row = (55.8, 58.5, 79.2, 40.0)
    collaborative_rows = [
        (2, *front_row),
        (4, *second_row),
        (8, *third_row),
        (16, *fourth_row),
    ]
    print_table(
        "Collaborative setting (half parties collaborating)",
        collaborative_rows,
    )


if __name__ == "__main__":
    main()
