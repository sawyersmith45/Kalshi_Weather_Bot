from kalshi_client import KalshiClient


def main():
    c = KalshiClient()
    print("Base URL:", c.base_url)

    bal = c.get_portfolio_balance()
    print("Balance:", bal)

    try:
        pos = c.get_positions(limit=200)
        print("Positions:", pos)
    except Exception as e:
        print("Positions error:", repr(e))

    try:
        orders = c.get_orders(limit=50)
        print("Orders:", orders)
    except Exception as e:
        print("Orders error:", repr(e))


if __name__ == "__main__":
    main()
