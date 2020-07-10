import matplotlib.pyplot as plt
from tabulate import tabulate


def visualize_price_history(data_frame):
    # Visualize the closing price history
    plt.figure(figsize=(16, 8))
    plt.title('Close Price History')
    plt.plot(data_frame['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()


def pretty_print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))
